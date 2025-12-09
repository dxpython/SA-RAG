using System.Runtime.CompilerServices;
using System.Text.Json;
using SharpAIKit.Common;

namespace SharpAIKit.LLM;

/// <summary>
/// OpenAI API client implementation.
/// Supports Chat Completion and Embedding APIs.
/// </summary>
public class OpenAIClient : BaseLLMClient
{
    private readonly string _chatEndpoint;
    private readonly string _embeddingEndpoint;
    private readonly string _embeddingModel;

    /// <summary>
    /// Creates a new OpenAI client instance.
    /// </summary>
    /// <param name="httpClient">HTTP client with pre-configured Authorization header.</param>
    /// <param name="options">Configuration options.</param>
    /// <param name="logger">Logger instance.</param>
    public OpenAIClient(HttpClient httpClient, LLMOptions options, Microsoft.Extensions.Logging.ILogger? logger = null) : base(httpClient, options, logger)
    {
        var baseUrl = string.IsNullOrEmpty(options.BaseUrl)
            ? "https://api.openai.com/v1"
            : options.BaseUrl.TrimEnd('/');

        _chatEndpoint = $"{baseUrl}/chat/completions";
        _embeddingEndpoint = $"{baseUrl}/embeddings";
        _embeddingModel = "text-embedding-3-small";

        // Set default model if not specified
        if (string.IsNullOrEmpty(Options.Model))
        {
            Options.Model = "gpt-3.5-turbo";
        }

        // Set Authorization header
        if (!string.IsNullOrEmpty(options.ApiKey))
        {
            HttpClient.DefaultRequestHeaders.Authorization =
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", options.ApiKey);
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
            ["temperature"] = options.Temperature ?? Options.Temperature,
            ["max_tokens"] = options.MaxTokens ?? Options.MaxTokens
        };

        if (options.JsonMode)
        {
            payload["response_format"] = new { type = "json_object" };
        }

        if (options.Tools != null && options.Tools.Any())
        {
            payload["tools"] = options.Tools;
            if (options.ToolChoice != null)
            {
                payload["tool_choice"] = options.ToolChoice;
            }
        }

        if (options.AdditionalParameters != null)
        {
            foreach (var kvp in options.AdditionalParameters)
            {
                payload[kvp.Key] = kvp.Value;
            }
        }

        var responseJson = await PostAsync(_chatEndpoint, payload, cancellationToken);
        
        // Log raw response if debug enabled
        // Logger.LogDebug("OpenAI Response: {Response}", responseJson);

        using var doc = JsonDocument.Parse(responseJson);
        var choice = doc.RootElement.GetProperty("choices")[0];
        var message = choice.GetProperty("message");

        // Handle tool calls
        if (message.TryGetProperty("tool_calls", out var toolCalls))
        {
            // For now, we just return the raw JSON of the message including tool_calls
            // The Agent will need to parse this. 
            // Alternatively, we could return a structured object, but the interface returns string.
            // So we return the JSON representation of the message content or the tool calls.
            // If content is null but tool_calls exists, we return the whole message JSON to let the agent handle it.
            return message.GetRawText(); 
        }

        return message.GetProperty("content").GetString() ?? string.Empty;
    }

    /// <inheritdoc />
    public override IAsyncEnumerable<string> ChatStreamAsync(string prompt, CancellationToken cancellationToken = default)
    {
        return ChatStreamAsync(new[] { ChatMessage.User(prompt) }, cancellationToken);
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
            ["temperature"] = options.Temperature ?? Options.Temperature,
            ["max_tokens"] = options.MaxTokens ?? Options.MaxTokens,
            ["stream"] = true
        };

        if (options.JsonMode)
        {
            payload["response_format"] = new { type = "json_object" };
        }

        if (options.Tools != null && options.Tools.Any())
        {
            payload["tools"] = options.Tools;
            if (options.ToolChoice != null)
            {
                payload["tool_choice"] = options.ToolChoice;
            }
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
    /// Parses a Server-Sent Events (SSE) data chunk.
    /// </summary>
    /// <param name="line">The SSE line to parse.</param>
    /// <returns>The extracted content, or null if not applicable.</returns>
    private static string? ParseStreamChunk(string line)
    {
        if (!line.StartsWith("data: ")) return null;
        var data = line[6..];
        if (data == "[DONE]") return null;

        try
        {
            using var doc = JsonDocument.Parse(data);
            var delta = doc.RootElement
                .GetProperty("choices")[0]
                .GetProperty("delta");

            if (delta.TryGetProperty("content", out var content))
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
            model = _embeddingModel,
            input = text
        };

        var responseJson = await PostAsync(_embeddingEndpoint, payload, cancellationToken);
        using var doc = JsonDocument.Parse(responseJson);

        var embeddingArray = doc.RootElement
            .GetProperty("data")[0]
            .GetProperty("embedding");

        var result = new float[embeddingArray.GetArrayLength()];
        var index = 0;
        foreach (var value in embeddingArray.EnumerateArray())
        {
            result[index++] = value.GetSingle();
        }

        return result;
    }
}

