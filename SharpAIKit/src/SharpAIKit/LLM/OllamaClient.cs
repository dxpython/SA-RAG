using System.Runtime.CompilerServices;
using System.Text.Json;
using SharpAIKit.Common;

namespace SharpAIKit.LLM;

/// <summary>
/// Ollama local service client implementation.
/// Default connection: http://localhost:11434
/// </summary>
public class OllamaClient : BaseLLMClient
{
    private readonly string _chatEndpoint;
    private readonly string _embeddingEndpoint;

    /// <summary>
    /// Creates a new Ollama client instance.
    /// </summary>
    /// <param name="httpClient">HTTP client to use.</param>
    /// <param name="options">Configuration options.</param>
    /// <param name="logger">Logger instance.</param>
    public OllamaClient(HttpClient httpClient, LLMOptions options, Microsoft.Extensions.Logging.ILogger? logger = null) : base(httpClient, options, logger)
    {
        var baseUrl = string.IsNullOrEmpty(options.BaseUrl)
            ? "http://localhost:11434"
            : options.BaseUrl.TrimEnd('/');

        _chatEndpoint = $"{baseUrl}/api/chat";
        _embeddingEndpoint = $"{baseUrl}/api/embeddings";

        // Set default model if not specified
        if (string.IsNullOrEmpty(Options.Model))
        {
            Options.Model = "llama3.2";
        }
    }

    /// <inheritdoc />
    public override async Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default)
    {
        return await ChatAsync(new[] { ChatMessage.User(prompt) }, cancellationToken);
    }

    /// <inheritdoc />
    public override async Task<string> ChatAsync(IEnumerable<ChatMessage> messages, CancellationToken cancellationToken = default)
    {
        return await ChatAsync(messages, new ChatOptions(), cancellationToken);
    }

    /// <inheritdoc />
    public override async Task<string> ChatAsync(IEnumerable<ChatMessage> messages, ChatOptions options, CancellationToken cancellationToken = default)
    {
        var payload = new Dictionary<string, object>
        {
            ["model"] = options.Model ?? Options.Model,
            ["messages"] = messages.Select(m => new { role = m.Role, content = m.Content }),
            ["stream"] = false,
            ["options"] = new Dictionary<string, object>
            {
                ["temperature"] = options.Temperature ?? Options.Temperature,
                ["num_predict"] = options.MaxTokens ?? Options.MaxTokens
            }
        };

        if (options.JsonMode)
        {
            payload["format"] = "json";
        }
        
        // Ollama tools support (basic mapping)
        if (options.Tools != null && options.Tools.Any())
        {
             payload["tools"] = options.Tools;
        }

        if (options.AdditionalParameters != null)
        {
            foreach (var kvp in options.AdditionalParameters)
            {
                payload[kvp.Key] = kvp.Value;
            }
        }

        var responseJson = await PostAsync(_chatEndpoint, payload, cancellationToken);
        using var doc = JsonDocument.Parse(responseJson);

        return doc.RootElement
            .GetProperty("message")
            .GetProperty("content")
            .GetString() ?? string.Empty;
    }

    /// <inheritdoc />
    public override IAsyncEnumerable<string> ChatStreamAsync(string prompt, CancellationToken cancellationToken = default)
    {
        return ChatStreamAsync(new[] { ChatMessage.User(prompt) }, new ChatOptions(), cancellationToken);
    }

    /// <inheritdoc />
    public override IAsyncEnumerable<string> ChatStreamAsync(
        IEnumerable<ChatMessage> messages,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        return ChatStreamAsync(messages, new ChatOptions(), cancellationToken);
    }

    /// <inheritdoc />
    public override async IAsyncEnumerable<string> ChatStreamAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions options,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var payload = new Dictionary<string, object>
        {
            ["model"] = options.Model ?? Options.Model,
            ["messages"] = messages.Select(m => new { role = m.Role, content = m.Content }),
            ["stream"] = true,
            ["options"] = new Dictionary<string, object>
            {
                ["temperature"] = options.Temperature ?? Options.Temperature,
                ["num_predict"] = options.MaxTokens ?? Options.MaxTokens
            }
        };

        if (options.JsonMode)
        {
            payload["format"] = "json";
        }

        if (options.AdditionalParameters != null)
        {
            foreach (var kvp in options.AdditionalParameters)
            {
                payload[kvp.Key] = kvp.Value;
            }
        }

        await foreach (var chunk in PostStreamAsync(_chatEndpoint, payload, ParseStreamChunk, cancellationToken))
        {
            yield return chunk;
        }
    }

    /// <summary>
    /// Parses an Ollama streaming response chunk.
    /// </summary>
    /// <param name="line">The response line to parse.</param>
    /// <returns>The extracted content, or null if not applicable.</returns>
    private static string? ParseStreamChunk(string line)
    {
        if (string.IsNullOrWhiteSpace(line)) return null;

        try
        {
            using var doc = JsonDocument.Parse(line);
            if (doc.RootElement.TryGetProperty("message", out var message) &&
                message.TryGetProperty("content", out var content))
            {
                return content.GetString();
            }
        }
        catch
        {
            // Ignore parse errors
        }

        return null;
    }

    /// <inheritdoc />
    public override async Task<float[]> EmbeddingAsync(string text, CancellationToken cancellationToken = default)
    {
        var payload = new
        {
            model = Options.Model,
            prompt = text
        };

        var responseJson = await PostAsync(_embeddingEndpoint, payload, cancellationToken);
        using var doc = JsonDocument.Parse(responseJson);

        var embeddingArray = doc.RootElement.GetProperty("embedding");
        var result = new float[embeddingArray.GetArrayLength()];
        var index = 0;
        foreach (var value in embeddingArray.EnumerateArray())
        {
            result[index++] = value.GetSingle();
        }

        return result;
    }
}

