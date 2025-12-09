using System.Text.Json.Serialization;

namespace SaAgentOS.Models;

/// <summary>
/// Agent configuration model
/// </summary>
public class AgentConfig
{
    [JsonPropertyName("DeepSeek")]
    public DeepSeekConfig DeepSeek { get; set; } = new();

    [JsonPropertyName("SaRag")]
    public SaRagConfig SaRag { get; set; } = new();

    [JsonPropertyName("Memory")]
    public MemoryConfig Memory { get; set; } = new();

    [JsonPropertyName("Logging")]
    public LoggingConfig Logging { get; set; } = new();
}

public class DeepSeekConfig
{
    [JsonPropertyName("ApiKey")]
    public string ApiKey { get; set; } = string.Empty;

    [JsonPropertyName("BaseUrl")]
    public string BaseUrl { get; set; } = "https://api.deepseek.com";

    [JsonPropertyName("Model")]
    public string Model { get; set; } = "deepseek-chat";
}

public class SaRagConfig
{
    [JsonPropertyName("Endpoint")]
    public string Endpoint { get; set; } = "http://localhost:8000/rag/query";

    [JsonPropertyName("UseGraph")]
    public bool UseGraph { get; set; } = true;

    [JsonPropertyName("UseMemory")]
    public bool UseMemory { get; set; } = true;

    [JsonPropertyName("TopK")]
    public int TopK { get; set; } = 6;
}

public class MemoryConfig
{
    [JsonPropertyName("StoragePath")]
    public string StoragePath { get; set; } = "./agent_memory.json";

    [JsonPropertyName("UseSaRagMemory")]
    public bool UseSaRagMemory { get; set; } = true;
}

public class LoggingConfig
{
    [JsonPropertyName("LogLevel")]
    public Dictionary<string, string> LogLevel { get; set; } = new();
}

