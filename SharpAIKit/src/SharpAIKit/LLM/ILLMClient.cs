namespace SharpAIKit.LLM;

/// <summary>
/// Unified interface for LLM clients.
/// Supports OpenAI, Ollama, LM Studio and other backends.
/// </summary>
public interface ILLMClient : IDisposable
{
    /// <summary>
    /// Sends a chat request and returns the complete response.
    /// </summary>
    /// <param name="prompt">The user's input prompt.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The model's generated response text.</returns>
    Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default);

    /// <summary>
    /// Sends a chat request with conversation history.
    /// </summary>
    /// <param name="messages">The conversation history.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The model's generated response text.</returns>
    Task<string> ChatAsync(IEnumerable<Common.ChatMessage> messages, CancellationToken cancellationToken = default);

    /// <summary>
    /// Sends a chat request with options (tools, json mode, etc).
    /// </summary>
    /// <param name="messages">The conversation history.</param>
    /// <param name="options">The chat options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The model's generated response text.</returns>
    Task<string> ChatAsync(IEnumerable<Common.ChatMessage> messages, ChatOptions options, CancellationToken cancellationToken = default);

    /// <summary>
    /// Streams a chat response token by token.
    /// </summary>
    /// <param name="prompt">The user's input prompt.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>An async enumerable of response chunks.</returns>
    IAsyncEnumerable<string> ChatStreamAsync(string prompt, CancellationToken cancellationToken = default);

    /// <summary>
    /// Streams a chat response with conversation history.
    /// </summary>
    /// <param name="messages">The conversation history.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>An async enumerable of response chunks.</returns>
    IAsyncEnumerable<string> ChatStreamAsync(IEnumerable<Common.ChatMessage> messages, CancellationToken cancellationToken = default);

    /// <summary>
    /// Streams a chat response with options.
    /// </summary>
    /// <param name="messages">The conversation history.</param>
    /// <param name="options">The chat options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>An async enumerable of response chunks.</returns>
    IAsyncEnumerable<string> ChatStreamAsync(IEnumerable<Common.ChatMessage> messages, ChatOptions options, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates an embedding vector for the given text.
    /// </summary>
    /// <param name="text">The text to generate embeddings for.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The embedding vector as a float array.</returns>
    Task<float[]> EmbeddingAsync(string text, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates embedding vectors for multiple texts.
    /// </summary>
    /// <param name="texts">The texts to generate embeddings for.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A list of embedding vectors.</returns>
    Task<List<float[]>> EmbeddingBatchAsync(IEnumerable<string> texts, CancellationToken cancellationToken = default);
}

