using System.Linq;
using Microsoft.Extensions.Logging;
using SaAgentOS.Models;
using SaAgentOS.Tools;
using SharpAIKit.Agent;
using SharpAIKit.LLM;
using SharpAIKit.Memory;
using SharpAIKit.Common;

namespace SaAgentOS.Agents;

/// <summary>
/// SA-RAG Agent: Core agent class that orchestrates knowledge retrieval and reasoning
/// Built on top of SharpAIKit's AiAgent for seamless integration
/// Supports conversation history for multi-turn dialogue
/// </summary>
public class SaRagAgent
{
    private readonly AiAgent _agent;
    private readonly KnowledgeRetrievalTool _knowledgeTool;
    private readonly DeepSeekReasoningTool _reasoningTool;
    private readonly MemoryTool _memoryTool;
    private readonly IMemory _conversationMemory;
    private readonly ILogger<SaRagAgent>? _logger;

    public string Name => "SA-RAG Agent";
    public string Description => "AI Agent with knowledge retrieval and reasoning capabilities";

    public SaRagAgent(
        ILLMClient llmClient,
        KnowledgeRetrievalTool knowledgeTool,
        DeepSeekReasoningTool reasoningTool,
        MemoryTool memoryTool,
        ILogger<SaRagAgent>? logger = null)
    {
        _knowledgeTool = knowledgeTool;
        _reasoningTool = reasoningTool;
        _memoryTool = memoryTool;
        _logger = logger;

        // Create conversation memory for multi-turn dialogue
        _conversationMemory = new BufferMemory
        {
            MaxMessages = 20  // Keep last 20 messages
        };

        // Create SharpAIKit agent
        _agent = new AiAgent(llmClient)
        {
            SystemPrompt = """
                You are an intelligent AI assistant with access to a knowledge base (SA-RAG), reasoning capabilities (DeepSeek), and long-term memory.
                
                You have access to the following tools:
                - knowledge_retrieval: Search the knowledge base for information
                - deepseek_reasoning: Perform complex reasoning tasks
                - add_memory: Store important information in long-term memory
                - retrieve_memory: Recall past information from memory
                
                When answering questions:
                1. First, try to retrieve relevant knowledge using knowledge_retrieval
                2. If needed, use deepseek_reasoning to synthesize the information
                3. Store important information in memory for future reference
                4. Provide clear, comprehensive answers based on the retrieved knowledge
                
                You can remember previous conversation context and refer to it when answering follow-up questions.
                Always be helpful, accurate, and cite your sources when possible.
                """
        };

        // Add tools to agent
        _agent.AddTool(_knowledgeTool);
        _agent.AddTool(_reasoningTool);
        _agent.AddTool(_memoryTool);
    }

    /// <summary>
    /// Process a user query using SharpAIKit agent with conversation history
    /// </summary>
    public async Task<AgentResponse> ProcessQueryAsync(
        string query,
        CancellationToken cancellationToken = default)
    {
        _logger?.LogInformation("Processing query: '{Query}'", query);

        try
        {
            // Get conversation history for context
            var historyMessages = await _conversationMemory.GetMessagesAsync(query);
            
            // Build query with conversation context if available
            string contextualQuery = query;
            if (historyMessages.Count > 0)
            {
                var contextString = await _conversationMemory.GetContextStringAsync(query);
                if (!string.IsNullOrEmpty(contextString))
                {
                    contextualQuery = $"{contextString}\n\nCurrent question: {query}";
                    _logger?.LogDebug("Using conversation context: {ContextLength} messages", historyMessages.Count);
                }
            }

            // Use SharpAIKit agent to process the query
            var agentResult = await _agent.RunAsync(contextualQuery, cancellationToken);

            // Store conversation in memory
            await _conversationMemory.AddExchangeAsync(query, agentResult.Answer);

            // Convert SharpAIKit result to our response format
            var response = new AgentResponse
            {
                Query = query,
                FinalAnswer = agentResult.Answer,
                Steps = agentResult.Steps.Select((step, index) => new ReasoningStep
                {
                    StepId = index + 1,
                    StepType = step.Type,
                    ToolName = step.ToolName,
                    Input = step.ToolArgs != null ? string.Join(", ", step.ToolArgs.Select(kv => $"{kv.Key}={kv.Value}")) : "",
                    Output = step.Result ?? "",
                    TimestampMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                    Metadata = new Dictionary<string, object>
                    {
                        ["thought"] = step.Thought ?? "",
                        ["history_length"] = historyMessages.Count
                    }
                }).ToList()
            };

            _logger?.LogInformation("Query processed successfully: Answer length={AnswerLength}, Steps={StepCount}, History={HistoryCount}",
                agentResult.Answer.Length, agentResult.Steps.Count, historyMessages.Count);

            return response;
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Error processing query");
            return new AgentResponse
            {
                Query = query,
                FinalAnswer = $"I encountered an error: {ex.Message}",
                Steps = new List<ReasoningStep>
                {
                    new ReasoningStep
                    {
                        StepId = 1,
                        StepType = "error",
                        Input = query,
                        Output = $"Error: {ex.Message}",
                        TimestampMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
                    }
                }
            };
        }
    }

    /// <summary>
    /// Clear conversation history
    /// </summary>
    public async Task ClearHistoryAsync()
    {
        await _conversationMemory.ClearAsync();
        _logger?.LogInformation("Conversation history cleared");
    }

    /// <summary>
    /// Get conversation history count
    /// </summary>
    public async Task<int> GetHistoryCountAsync()
    {
        return await _conversationMemory.GetCountAsync();
    }
}

/// <summary>
/// Agent response
/// </summary>
public class AgentResponse
{
    public string Query { get; set; } = string.Empty;
    public List<ReasoningStep> Steps { get; set; } = new();
    public string FinalAnswer { get; set; } = string.Empty;
    public ExecutionGraph? ExecutionGraph { get; set; }
}
