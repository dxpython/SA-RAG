using System.Runtime.CompilerServices;
using System.Text;
using SharpAIKit.Common;

using Microsoft.Extensions.Logging;
using SharpAIKit.Common;

namespace SharpAIKit.LLM;

/// <summary>
/// Base class for LLM clients.
/// Provides common HTTP request and retry logic.
/// </summary>
public abstract class BaseLLMClient : ILLMClient
{
    /// <summary>
    /// The HTTP client instance.
    /// </summary>
    protected readonly HttpClient HttpClient;

    /// <summary>
    /// The configuration options.
    /// </summary>
    protected readonly LLMOptions Options;

    /// <summary>
    /// The logger instance.
    /// </summary>
    protected readonly ILogger Logger;

    /// <summary>
    /// Whether the client has been disposed.
    /// </summary>
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the BaseLLMClient.
    /// </summary>
    /// <param name="httpClient">The HTTP client to use.</param>
    /// <param name="options">The configuration options.</param>
    /// <summary>
    /// Initializes a new instance of the BaseLLMClient.
    /// </summary>
    /// <param name="httpClient">The HTTP client to use.</param>
    /// <param name="options">The configuration options.</param>
    /// <param name="logger">The logger instance.</param>
    protected BaseLLMClient(HttpClient httpClient, LLMOptions options, ILogger? logger = null)
    {
        HttpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
        Options = options ?? throw new ArgumentNullException(nameof(options));
        Logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance;

        // Set timeout
        HttpClient.Timeout = TimeSpan.FromSeconds(options.TimeoutSeconds);
    }

    /// <inheritdoc />
    public abstract Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default);

    /// <inheritdoc />
    public abstract Task<string> ChatAsync(IEnumerable<ChatMessage> messages, CancellationToken cancellationToken = default);

    /// <inheritdoc />
    public abstract Task<string> ChatAsync(IEnumerable<ChatMessage> messages, ChatOptions options, CancellationToken cancellationToken = default);

    /// <inheritdoc />
    public abstract IAsyncEnumerable<string> ChatStreamAsync(string prompt, CancellationToken cancellationToken = default);

    /// <inheritdoc />
    public abstract IAsyncEnumerable<string> ChatStreamAsync(IEnumerable<ChatMessage> messages, CancellationToken cancellationToken = default);

    /// <inheritdoc />
    public abstract IAsyncEnumerable<string> ChatStreamAsync(IEnumerable<ChatMessage> messages, ChatOptions options, CancellationToken cancellationToken = default);

    /// <inheritdoc />
    public abstract Task<float[]> EmbeddingAsync(string text, CancellationToken cancellationToken = default);

    /// <inheritdoc />
    public virtual async Task<List<float[]>> EmbeddingBatchAsync(IEnumerable<string> texts, CancellationToken cancellationToken = default)
    {
        // Use parallelism to speed up batch embeddings
        var tasks = texts.Select(text => EmbeddingAsync(text, cancellationToken));
        var results = await Task.WhenAll(tasks);
        return results.ToList();
    }

    /// <summary>
    /// Sends a POST request and returns the response body.
    /// </summary>
    /// <param name="endpoint">The API endpoint.</param>
    /// <param name="payload">The request payload.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The response body as a string.</returns>
    protected async Task<string> PostAsync(string endpoint, object payload, CancellationToken cancellationToken)
    {
        var json = JsonHelper.Serialize(payload);
        var content = new StringContent(json, Encoding.UTF8, "application/json");

        var response = await HttpClient.PostAsync(endpoint, content, cancellationToken);
        response.EnsureSuccessStatusCode();

        return await response.Content.ReadAsStringAsync(cancellationToken);
    }

    /// <summary>
    /// Sends a streaming POST request and yields response chunks.
    /// </summary>
    /// <param name="endpoint">The API endpoint.</param>
    /// <param name="payload">The request payload.</param>
    /// <param name="parseChunk">Function to parse each chunk.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>An async enumerable of response chunks.</returns>
    protected async IAsyncEnumerable<string> PostStreamAsync(
        string endpoint,
        object payload,
        Func<string, string?> parseChunk,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var json = JsonHelper.Serialize(payload);
        var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var request = new HttpRequestMessage(HttpMethod.Post, endpoint) { Content = content };
        using var response = await HttpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();

        using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var reader = new StreamReader(stream);

        while (!reader.EndOfStream && !cancellationToken.IsCancellationRequested)
        {
            var line = await reader.ReadLineAsync(cancellationToken);
            if (string.IsNullOrEmpty(line)) continue;

            var chunk = parseChunk(line);
            if (chunk != null)
            {
                yield return chunk;
            }
        }
    }

    /// <summary>
    /// Releases all resources used by the client.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases the unmanaged resources and optionally releases the managed resources.
    /// </summary>
    /// <param name="disposing">true to release both managed and unmanaged resources.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing)
        {
            // HttpClient is managed externally, do not dispose here
        }
        _disposed = true;
    }
}

