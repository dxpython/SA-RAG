using Microsoft.Extensions.Logging;

namespace SaAgentOS.Logging;

/// <summary>
/// Agent-specific logger wrapper
/// </summary>
public class AgentLogger
{
    private readonly ILogger _logger;

    public AgentLogger(ILogger logger)
    {
        _logger = logger;
    }

    public void LogQuery(string query)
    {
        _logger.LogInformation("🔍 Query: {Query}", query);
    }

    public void LogRetrieval(int nodeCount, double topScore)
    {
        _logger.LogInformation("📚 Retrieval: Found {NodeCount} nodes, Top score: {TopScore:F2}",
            nodeCount, topScore);
    }

    public void LogReasoning(string reasoning)
    {
        _logger.LogInformation("🧠 Reasoning: {Reasoning}",
            reasoning.Substring(0, Math.Min(100, reasoning.Length)));
    }

    public void LogMemory(int memoryCount)
    {
        _logger.LogInformation("💾 Memory: Retrieved {MemoryCount} relevant memories", memoryCount);
    }

    public void LogAnswer(string answer)
    {
        _logger.LogInformation("✅ Answer: {Answer}",
            answer.Substring(0, Math.Min(200, answer.Length)));
    }

    public void LogError(string error, Exception? ex = null)
    {
        _logger.LogError(ex, "❌ Error: {Error}", error);
    }

    public void LogStep(string stepType, string toolName, string input, string output)
    {
        _logger.LogDebug("Step [{StepType}] {ToolName}: Input={Input}, Output={Output}",
            stepType, toolName, input, output);
    }
}

