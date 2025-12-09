using System.Text.Json.Serialization;

namespace SharpAIKit.LLM;

/// <summary>
/// Options for chat completion requests.
/// </summary>
public class ChatOptions
{
    /// <summary>
    /// The model name to use (overrides the client default).
    /// </summary>
    public string? Model { get; set; }

    /// <summary>
    /// The sampling temperature (0.0 to 2.0).
    /// </summary>
    public double? Temperature { get; set; }

    /// <summary>
    /// The maximum number of tokens to generate.
    /// </summary>
    public int? MaxTokens { get; set; }

    /// <summary>
    /// The list of tools available to the model.
    /// </summary>
    public List<ToolDefinition>? Tools { get; set; }

    /// <summary>
    /// Controls which (if any) tool is called by the model.
    /// "auto" means the model can pick between generating a message or calling a tool.
    /// "none" means the model will not call a tool and instead generates a message.
    /// "required" means the model must call a tool.
    /// </summary>
    public object? ToolChoice { get; set; }

    /// <summary>
    /// Whether to enforce JSON output.
    /// </summary>
    public bool JsonMode { get; set; }

    /// <summary>
    /// Additional parameters to pass to the API.
    /// </summary>
    public Dictionary<string, object>? AdditionalParameters { get; set; }
}

/// <summary>
/// Definition of a tool that the model can call.
/// </summary>
public class ToolDefinition
{
    /// <summary>
    /// The type of the tool. Currently only "function" is supported.
    /// </summary>
    [JsonPropertyName("type")]
    public string Type { get; set; } = "function";

    /// <summary>
    /// The function definition.
    /// </summary>
    [JsonPropertyName("function")]
    public FunctionDefinition Function { get; set; } = new();
}

/// <summary>
/// Definition of a function that the model can call.
/// </summary>
public class FunctionDefinition
{
    /// <summary>
    /// The name of the function to be called.
    /// </summary>
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// A description of what the function does.
    /// </summary>
    [JsonPropertyName("description")]
    public string? Description { get; set; }

    /// <summary>
    /// The parameters the functions accepts, described as a JSON Schema object.
    /// </summary>
    [JsonPropertyName("parameters")]
    public object? Parameters { get; set; }
}
