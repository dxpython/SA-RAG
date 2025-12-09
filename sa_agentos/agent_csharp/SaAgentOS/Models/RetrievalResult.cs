using System.Text.Json.Serialization;

namespace SaAgentOS.Models;

/// <summary>
/// Retrieval result from SA-RAG
/// </summary>
public class RetrievalResult
{
    [JsonPropertyName("id")]
    public int Id { get; set; }

    [JsonPropertyName("text")]
    public string Text { get; set; } = string.Empty;

    [JsonPropertyName("score")]
    public double Score { get; set; }

    [JsonPropertyName("source")]
    public string Source { get; set; } = "unknown";

    [JsonPropertyName("node_type")]
    public string NodeType { get; set; } = "text";
}

/// <summary>
/// Execution graph node
/// </summary>
public class ExecutionGraphNode
{
    [JsonPropertyName("node_id")]
    public string NodeId { get; set; } = string.Empty;

    [JsonPropertyName("node_type")]
    public string NodeType { get; set; } = string.Empty;

    [JsonPropertyName("description")]
    public string Description { get; set; } = string.Empty;

    [JsonPropertyName("execution_time_ms")]
    public double ExecutionTimeMs { get; set; }
}

/// <summary>
/// Execution graph edge
/// </summary>
public class ExecutionGraphEdge
{
    [JsonPropertyName("from_node")]
    public string FromNode { get; set; } = string.Empty;

    [JsonPropertyName("to_node")]
    public string ToNode { get; set; } = string.Empty;

    [JsonPropertyName("edge_type")]
    public string EdgeType { get; set; } = "data_flow";

    [JsonPropertyName("weight")]
    public double Weight { get; set; } = 1.0;
}

/// <summary>
/// Execution graph structure
/// </summary>
public class ExecutionGraph
{
    [JsonPropertyName("query")]
    public string Query { get; set; } = string.Empty;

    [JsonPropertyName("nodes")]
    public List<ExecutionGraphNode> Nodes { get; set; } = new();

    [JsonPropertyName("edges")]
    public List<ExecutionGraphEdge> Edges { get; set; } = new();

    [JsonPropertyName("execution_trace")]
    public List<string> ExecutionTrace { get; set; } = new();

    [JsonPropertyName("total_time_ms")]
    public double TotalTimeMs { get; set; }
}

/// <summary>
/// RAG query response
/// </summary>
public class RAGQueryResponse
{
    [JsonPropertyName("answer")]
    public string Answer { get; set; } = string.Empty;

    [JsonPropertyName("nodes")]
    public List<RetrievalResult> Nodes { get; set; } = new();

    [JsonPropertyName("execution_graph")]
    public ExecutionGraph? ExecutionGraph { get; set; }

    [JsonPropertyName("query")]
    public string Query { get; set; } = string.Empty;

    [JsonPropertyName("top_k")]
    public int TopK { get; set; }
}

/// <summary>
/// RAG query request
/// </summary>
public class RAGQueryRequest
{
    [JsonPropertyName("query")]
    public string Query { get; set; } = string.Empty;

    [JsonPropertyName("use_graph")]
    public bool UseGraph { get; set; } = true;

    [JsonPropertyName("use_memory")]
    public bool UseMemory { get; set; } = true;

    [JsonPropertyName("top_k")]
    public int TopK { get; set; } = 6;
}

