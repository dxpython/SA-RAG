using System.Text.Json.Serialization;

namespace SaAgentOS.Models;

/// <summary>
/// Reasoning trace for agent execution
/// </summary>
public class ReasoningTrace
{
    [JsonPropertyName("query")]
    public string Query { get; set; } = string.Empty;

    [JsonPropertyName("steps")]
    public List<ReasoningStep> Steps { get; set; } = new();

    [JsonPropertyName("final_answer")]
    public string FinalAnswer { get; set; } = string.Empty;

    [JsonPropertyName("execution_time_ms")]
    public double ExecutionTimeMs { get; set; }

    [JsonPropertyName("execution_graph")]
    public ExecutionGraph? ExecutionGraph { get; set; }
}

/// <summary>
/// Single reasoning step
/// </summary>
public class ReasoningStep
{
    [JsonPropertyName("step_id")]
    public int StepId { get; set; }

    [JsonPropertyName("step_type")]
    public string StepType { get; set; } = string.Empty; // "planning", "retrieval", "reasoning", "memory", "reflection"

    [JsonPropertyName("tool_name")]
    public string? ToolName { get; set; }

    [JsonPropertyName("input")]
    public string Input { get; set; } = string.Empty;

    [JsonPropertyName("output")]
    public string Output { get; set; } = string.Empty;

    [JsonPropertyName("timestamp_ms")]
    public long TimestampMs { get; set; }

    [JsonPropertyName("metadata")]
    public Dictionary<string, object> Metadata { get; set; } = new();
}

