using System.Runtime.CompilerServices;
using SharpAIKit.LLM;
using SharpAIKit.Common;

namespace SharpAIKit.Chain;

/// <summary>
/// A chain that invokes an LLM for generation.
/// This is the core building block for LLM-powered chains.
/// </summary>
public class LLMChain : ChainBase
{
    private readonly ILLMClient _client;
    private readonly string? _systemPrompt;
    private readonly ChatOptions? _options;

    /// <summary>
    /// Gets or sets the input key to read from context.
    /// </summary>
    public string InputKey { get; set; } = "input";

    /// <summary>
    /// Gets or sets the output key to write to context.
    /// </summary>
    public string OutputKey { get; set; } = "output";

    /// <summary>
    /// Creates a new LLM chain.
    /// </summary>
    /// <param name="client">The LLM client to use.</param>
    /// <param name="systemPrompt">Optional system prompt.</param>
    /// <param name="options">Optional chat options.</param>
    public LLMChain(ILLMClient client, string? systemPrompt = null, ChatOptions? options = null)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
        _systemPrompt = systemPrompt;
        _options = options;
    }

    /// <inheritdoc/>
    protected override async Task<ChainContext> ExecuteAsync(ChainContext context, CancellationToken cancellationToken)
    {
        var input = context.Get<string>(InputKey) ?? context.Input;
        var messages = BuildMessages(input);
        
        var response = _options != null 
            ? await _client.ChatAsync(messages, _options, cancellationToken)
            : await _client.ChatAsync(messages, cancellationToken);

        context.Set(OutputKey, response);
        context.Output = response;
        return context;
    }

    /// <inheritdoc/>
    public override async IAsyncEnumerable<ChainContext> StreamAsync(
        ChainContext context, 
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var input = context.Get<string>(InputKey) ?? context.Input;
        var messages = BuildMessages(input);
        var accumulated = string.Empty;

        var stream = _options != null
            ? _client.ChatStreamAsync(messages, _options, cancellationToken)
            : _client.ChatStreamAsync(messages, cancellationToken);

        await foreach (var chunk in stream)
        {
            accumulated += chunk;
            var streamContext = context.Clone();
            streamContext.Set(OutputKey, accumulated);
            streamContext.Output = accumulated;
            streamContext.Set("chunk", chunk);
            streamContext.Set("streaming", true);
            yield return streamContext;
        }
    }

    private List<ChatMessage> BuildMessages(string input)
    {
        var messages = new List<ChatMessage>();
        
        if (!string.IsNullOrEmpty(_systemPrompt))
        {
            messages.Add(ChatMessage.System(_systemPrompt));
        }
        
        messages.Add(ChatMessage.User(input));
        return messages;
    }
}

/// <summary>
/// A chain that generates embeddings for text.
/// </summary>
public class EmbeddingChain : ChainBase
{
    private readonly ILLMClient _client;

    /// <summary>
    /// Gets or sets the input key to read from context.
    /// </summary>
    public string InputKey { get; set; } = "input";

    /// <summary>
    /// Gets or sets the output key for the embedding.
    /// </summary>
    public string OutputKey { get; set; } = "embedding";

    /// <summary>
    /// Creates a new embedding chain.
    /// </summary>
    public EmbeddingChain(ILLMClient client)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
    }

    /// <inheritdoc/>
    protected override async Task<ChainContext> ExecuteAsync(ChainContext context, CancellationToken cancellationToken)
    {
        var input = context.Get<string>(InputKey) ?? context.Input;
        var embedding = await _client.EmbeddingAsync(input, cancellationToken);
        
        context.Set(OutputKey, embedding);
        return context;
    }
}

/// <summary>
/// A chain for conversational interactions with memory support.
/// </summary>
public class ConversationChain : ChainBase
{
    private readonly ILLMClient _client;
    private readonly List<ChatMessage> _history = new();
    private readonly string? _systemPrompt;

    /// <summary>
    /// Gets or sets the maximum number of messages to keep in history.
    /// </summary>
    public int MaxHistoryLength { get; set; } = 20;

    /// <summary>
    /// Creates a new conversation chain.
    /// </summary>
    public ConversationChain(ILLMClient client, string? systemPrompt = null)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
        _systemPrompt = systemPrompt;
    }

    /// <inheritdoc/>
    protected override async Task<ChainContext> ExecuteAsync(ChainContext context, CancellationToken cancellationToken)
    {
        var input = context.Input;
        
        // Build messages with history
        var messages = new List<ChatMessage>();
        if (!string.IsNullOrEmpty(_systemPrompt))
        {
            messages.Add(ChatMessage.System(_systemPrompt));
        }
        messages.AddRange(_history);
        messages.Add(ChatMessage.User(input));

        var response = await _client.ChatAsync(messages, cancellationToken);

        // Update history
        _history.Add(ChatMessage.User(input));
        _history.Add(ChatMessage.Assistant(response));

        // Trim history if needed
        while (_history.Count > MaxHistoryLength * 2)
        {
            _history.RemoveAt(0);
            _history.RemoveAt(0);
        }

        context.Output = response;
        context.Set("history_length", _history.Count);
        return context;
    }

    /// <summary>
    /// Clears the conversation history.
    /// </summary>
    public void ClearHistory() => _history.Clear();

    /// <summary>
    /// Gets the current conversation history.
    /// </summary>
    public IReadOnlyList<ChatMessage> History => _history.AsReadOnly();
}

