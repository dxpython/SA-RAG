namespace SharpAIKit.Common;

/// <summary>
/// Enumeration of supported LLM provider types.
/// </summary>
public enum LLMProviderType
{
    /// <summary>
    /// OpenAI-compatible API.
    /// Works with: OpenAI, DeepSeek, Qwen, Mistral, Yi, Groq, LM Studio, and any other OpenAI-compatible service.
    /// </summary>
    OpenAI,

    /// <summary>
    /// Ollama local service.
    /// Supports: Llama, Qwen, Phi, Mistral, Gemma, DeepSeek, Yi, etc.
    /// </summary>
    Ollama
}
