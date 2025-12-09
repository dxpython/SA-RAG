using System.Text;
using System.Text.Json;
using SharpAIKit.Common;
using SharpAIKit.LLM;

namespace SharpAIKit.Agent;

/// <summary>
/// Represents the result of an agent execution.
/// </summary>
public class AgentResult
{
    /// <summary>
    /// Gets or sets the final answer.
    /// </summary>
    public string Answer { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the execution steps.
    /// </summary>
    public List<AgentStep> Steps { get; set; } = new();

    /// <summary>
    /// Gets or sets whether the execution was successful.
    /// </summary>
    public bool Success { get; set; }
}

/// <summary>
/// Represents a single step in the agent's execution.
/// </summary>
public class AgentStep
{
    /// <summary>
    /// Gets or sets the step type.
    /// </summary>
    public string Type { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the tool name (if applicable).
    /// </summary>
    public string? ToolName { get; set; }

    /// <summary>
    /// Gets or sets the tool arguments (if applicable).
    /// </summary>
    public Dictionary<string, object?>? ToolArgs { get; set; }

    /// <summary>
    /// Gets or sets the execution result.
    /// </summary>
    public string? Result { get; set; }

    /// <summary>
    /// Gets or sets the thought process.
    /// </summary>
    public string? Thought { get; set; }
}

/// <summary>
/// Lightweight AI Agent that supports tool calling and automatic planning.
/// </summary>
public class AiAgent
{
    private readonly ILLMClient _llmClient;
    private readonly Dictionary<string, ToolDefinition> _tools = new();

    /// <summary>
    /// Gets or sets the maximum number of execution steps.
    /// </summary>
    public int MaxSteps { get; set; } = 10;

    /// <summary>
    /// Gets or sets the system prompt template.
    /// </summary>
    public string SystemPrompt { get; set; } = """
        You are an intelligent assistant that can use tools to complete user tasks.

        Available tools:
        {tools}

        Please respond in the following JSON format:
        1. If you need to call a tool:
        {"action": "tool", "tool": "tool_name", "args": {"param_name": "param_value"}, "thought": "your reasoning"}

        2. If you can answer the user directly:
        {"action": "answer", "content": "your answer", "thought": "your reasoning"}

        Note: You can only perform one action at a time. Please respond strictly in JSON format.
        """;

    /// <summary>
    /// Creates a new AI Agent instance.
    /// </summary>
    /// <param name="llmClient">The LLM client to use.</param>
    public AiAgent(ILLMClient llmClient)
    {
        _llmClient = llmClient ?? throw new ArgumentNullException(nameof(llmClient));
    }

    /// <summary>
    /// Adds a tool to the agent.
    /// </summary>
    /// <param name="tool">The tool instance.</param>
    public void AddTool(ToolBase tool)
    {
        var definitions = tool.GetToolDefinitions();
        foreach (var def in definitions)
        {
            _tools[def.Name] = def;
        }
    }

    /// <summary>
    /// Adds a tool definition to the agent.
    /// </summary>
    /// <param name="definition">The tool definition.</param>
    public void AddTool(ToolDefinition definition)
    {
        _tools[definition.Name] = definition;
    }

    /// <summary>
    /// Runs the agent to complete a task.
    /// </summary>
    /// <param name="task">The task description.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The execution result.</returns>
    public async Task<AgentResult> RunAsync(string task, CancellationToken cancellationToken = default)
    {
        var result = new AgentResult();
        var messages = new List<ChatMessage>
        {
            ChatMessage.System(SystemPrompt), // We don't inject tools into system prompt if using native tools
            ChatMessage.User(task)
        };

        // Prepare tools for native calling
        var toolDefinitions = _tools.Values.Select(t => new SharpAIKit.LLM.ToolDefinition
        {
            Type = "function",
            Function = new SharpAIKit.LLM.FunctionDefinition
            {
                Name = t.Name,
                Description = t.Description,
                Parameters = new
                {
                    type = "object",
                    properties = t.Parameters.ToDictionary(
                        p => p.Name,
                        p => new
                        {
                            type = p.Type,
                            description = p.Description
                        }
                    ),
                    required = t.Parameters.Where(p => p.Required).Select(p => p.Name).ToList()
                }
            }
        }).ToList();

        var chatOptions = new ChatOptions
        {
            Tools = toolDefinitions,
            ToolChoice = "auto"
        };

        // If no tools, we don't need special handling
        if (toolDefinitions.Count == 0)
        {
            chatOptions.Tools = null;
            chatOptions.ToolChoice = null;
        }
        else
        {
            // If using native tools, we might want to adjust system prompt to NOT include manual tool descriptions
            // But for compatibility with models that don't support native tools well, we might keep it.
            // For now, let's assume we rely on native tools if available.
            // However, the current SystemPrompt has a placeholder {tools}. We should handle that.
            if (SystemPrompt.Contains("{tools}"))
            {
                 // If we are using native tools, we might clear the manual tool instructions or keep them as backup.
                 // Let's keep them for now but maybe simplify.
                 messages[0] = ChatMessage.System(BuildSystemPrompt());
            }
        }

        for (var step = 0; step < MaxSteps; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Get next action from LLM
            var response = await _llmClient.ChatAsync(messages, chatOptions, cancellationToken);

            // Parse the response
            // It might be a direct tool call (JSON) or a text response
            var action = ParseAction(response);
            
            // If parse failed, try to see if it's just a direct answer
            if (action == null)
            {
                // If the response is not a JSON tool call, treat it as an answer
                // But we need to be careful. If the model was supposed to call a tool but failed to format JSON,
                // we might want to error. But often it just talks.
                action = new AgentStep
                {
                    Type = "answer",
                    Result = response,
                    Thought = "Direct response"
                };
            }

            result.Steps.Add(action);

            // If this is the final answer
            if (action.Type == "answer")
            {
                result.Answer = action.Result ?? string.Empty;
                result.Success = true;
                return result;
            }

            // Execute tool
            if (action.Type == "tool" && !string.IsNullOrEmpty(action.ToolName))
            {
                var toolResult = await ExecuteToolAsync(action.ToolName, action.ToolArgs ?? new());
                action.Result = toolResult;

                // Add tool execution result to conversation
                // Note: For native tool calls, we should ideally add a Tool message.
                // But our ChatMessage structure might need updates to support Tool role properly.
                // For now, we simulate it with User role or Assistant role.
                // Let's stick to the existing pattern: Assistant (Tool Call) -> User (Result)
                messages.Add(ChatMessage.Assistant(response));
                messages.Add(ChatMessage.User($"Tool execution result:\n{toolResult}"));
            }
        }

        result.Answer = "Maximum execution steps reached";
        result.Success = false;
        return result;
    }

    /// <summary>
    /// Runs the agent and returns only the answer.
    /// </summary>
    /// <param name="task">The task description.</param>
    /// <returns>The final answer.</returns>
    public async Task<string> RunAsync(string task)
    {
        var result = await RunAsync(task, CancellationToken.None);
        return result.Answer;
    }

    /// <summary>
    /// Builds the system prompt with tool descriptions.
    /// </summary>
    private string BuildSystemPrompt()
    {
        var toolsDescription = new StringBuilder();
        foreach (var tool in _tools.Values)
        {
            toolsDescription.AppendLine($"- {tool.Name}: {tool.Description}");
            foreach (var param in tool.Parameters)
            {
                var required = param.Required ? "(required)" : "(optional)";
                toolsDescription.AppendLine($"    - {param.Name} ({param.Type}): {param.Description} {required}");
            }
        }

        return SystemPrompt.Replace("{tools}", toolsDescription.ToString());
    }

    /// <summary>
    /// Parses an LLM response into an agent step.
    /// </summary>
    /// <summary>
    /// Parses an LLM response into an agent step.
    /// Handles both JSON blocks and native tool calls (which come as JSON).
    /// </summary>
    private AgentStep? ParseAction(string response)
    {
        try
        {
            var jsonStr = ExtractJson(response);
            if (string.IsNullOrEmpty(jsonStr)) return null;

            using var doc = JsonDocument.Parse(jsonStr);
            var root = doc.RootElement;

            // Check for native tool calls format (OpenAI style)
            if (root.TryGetProperty("tool_calls", out var toolCalls) && toolCalls.GetArrayLength() > 0)
            {
                var firstCall = toolCalls[0];
                var function = firstCall.GetProperty("function");
                var name = function.GetProperty("name").GetString();
                var argsStr = function.GetProperty("arguments").GetString();
                
                var args = new Dictionary<string, object?>();
                if (!string.IsNullOrEmpty(argsStr))
                {
                    using var argsDoc = JsonDocument.Parse(argsStr);
                    foreach (var prop in argsDoc.RootElement.EnumerateObject())
                    {
                        args[prop.Name] = prop.Value.Clone();
                    }
                }

                return new AgentStep
                {
                    Type = "tool",
                    ToolName = name,
                    ToolArgs = args,
                    Thought = "Native tool call"
                };
            }

            // Check for our custom JSON format
            if (root.TryGetProperty("action", out var actionProp))
            {
                var action = actionProp.GetString();
                var thought = root.TryGetProperty("thought", out var t) ? t.GetString() : null;

                if (action == "answer")
                {
                    return new AgentStep
                    {
                        Type = "answer",
                        Result = root.TryGetProperty("content", out var c) ? c.GetString() : string.Empty,
                        Thought = thought
                    };
                }
                else if (action == "tool")
                {
                    var toolName = root.GetProperty("tool").GetString();
                    var args = new Dictionary<string, object?>();

                    if (root.TryGetProperty("args", out var argsElement))
                    {
                        foreach (var prop in argsElement.EnumerateObject())
                        {
                            args[prop.Name] = prop.Value.Clone();
                        }
                    }

                    return new AgentStep
                    {
                        Type = "tool",
                        ToolName = toolName,
                        ToolArgs = args,
                        Thought = thought
                    };
                }
            }
        }
        catch
        {
            // Ignore parse errors
        }

        return null;
    }

    /// <summary>
    /// Extracts JSON from a string, handling Markdown code blocks.
    /// </summary>
    private string? ExtractJson(string text)
    {
        text = text.Trim();
        
        // Handle markdown code blocks
        if (text.Contains("```json"))
        {
            var start = text.IndexOf("```json") + 7;
            var end = text.IndexOf("```", start);
            if (end > start)
            {
                return text.Substring(start, end - start).Trim();
            }
        }
        else if (text.Contains("```"))
        {
            var start = text.IndexOf("```") + 3;
            var end = text.IndexOf("```", start);
            if (end > start)
            {
                return text.Substring(start, end - start).Trim();
            }
        }

        // Handle raw JSON
        var jsonStart = text.IndexOf('{');
        var jsonEnd = text.LastIndexOf('}');

        if (jsonStart >= 0 && jsonEnd > jsonStart)
        {
            return text.Substring(jsonStart, jsonEnd - jsonStart + 1);
        }

        return null;
    }

    /// <summary>
    /// Executes a tool with the given arguments.
    /// </summary>
    private async Task<string> ExecuteToolAsync(string toolName, Dictionary<string, object?> args)
    {
        if (!_tools.TryGetValue(toolName, out var tool))
        {
            return $"Error: Tool '{toolName}' not found";
        }

        if (tool.Execute == null)
        {
            return $"Error: Tool '{toolName}' has no execution method defined";
        }

        try
        {
            return await tool.Execute(args);
        }
        catch (Exception ex)
        {
            return $"Tool execution error: {ex.Message}";
        }
    }
}

