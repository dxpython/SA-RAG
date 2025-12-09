using System.Text;
using SharpAIKit.Common;

namespace SharpAIKit.Memory;

/// <summary>
/// Simple buffer memory that stores the last N messages.
/// This is the most basic memory implementation.
/// </summary>
public class BufferMemory : IMemory
{
    private readonly List<ChatMessage> _messages = new();
    private readonly object _lock = new();

    /// <summary>
    /// Gets or sets the maximum number of messages to keep.
    /// </summary>
    public int MaxMessages { get; set; } = 20;

    /// <summary>
    /// Gets or sets the human prefix for formatting.
    /// </summary>
    public string HumanPrefix { get; set; } = "Human";

    /// <summary>
    /// Gets or sets the AI prefix for formatting.
    /// </summary>
    public string AiPrefix { get; set; } = "AI";

    /// <inheritdoc/>
    public Task AddMessageAsync(ChatMessage message)
    {
        lock (_lock)
        {
            _messages.Add(message);
            TrimMessages();
        }
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task AddExchangeAsync(string userMessage, string assistantMessage)
    {
        lock (_lock)
        {
            _messages.Add(ChatMessage.User(userMessage));
            _messages.Add(ChatMessage.Assistant(assistantMessage));
            TrimMessages();
        }
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task<List<ChatMessage>> GetMessagesAsync(string? query = null)
    {
        lock (_lock)
        {
            return Task.FromResult(_messages.ToList());
        }
    }

    /// <inheritdoc/>
    public Task<string> GetContextStringAsync(string? query = null)
    {
        lock (_lock)
        {
            var sb = new StringBuilder();
            foreach (var msg in _messages)
            {
                var prefix = msg.Role == "user" ? HumanPrefix : AiPrefix;
                sb.AppendLine($"{prefix}: {msg.Content}");
            }
            return Task.FromResult(sb.ToString());
        }
    }

    /// <inheritdoc/>
    public Task ClearAsync()
    {
        lock (_lock)
        {
            _messages.Clear();
        }
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task<int> GetCountAsync()
    {
        lock (_lock)
        {
            return Task.FromResult(_messages.Count);
        }
    }

    private void TrimMessages()
    {
        while (_messages.Count > MaxMessages)
        {
            _messages.RemoveAt(0);
        }
    }
}

/// <summary>
/// Window buffer memory that keeps the last N exchanges (user + assistant pairs).
/// </summary>
public class WindowBufferMemory : IMemory
{
    private readonly List<ChatMessage> _messages = new();
    private readonly object _lock = new();

    /// <summary>
    /// Gets or sets the number of conversation turns (exchanges) to keep.
    /// </summary>
    public int WindowSize { get; set; } = 5;

    /// <inheritdoc/>
    public Task AddMessageAsync(ChatMessage message)
    {
        lock (_lock)
        {
            _messages.Add(message);
            TrimToWindow();
        }
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task AddExchangeAsync(string userMessage, string assistantMessage)
    {
        lock (_lock)
        {
            _messages.Add(ChatMessage.User(userMessage));
            _messages.Add(ChatMessage.Assistant(assistantMessage));
            TrimToWindow();
        }
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task<List<ChatMessage>> GetMessagesAsync(string? query = null)
    {
        lock (_lock)
        {
            return Task.FromResult(_messages.ToList());
        }
    }

    /// <inheritdoc/>
    public Task<string> GetContextStringAsync(string? query = null)
    {
        lock (_lock)
        {
            var sb = new StringBuilder();
            foreach (var msg in _messages)
            {
                var prefix = msg.Role == "user" ? "Human" : "AI";
                sb.AppendLine($"{prefix}: {msg.Content}");
            }
            return Task.FromResult(sb.ToString());
        }
    }

    /// <inheritdoc/>
    public Task ClearAsync()
    {
        lock (_lock)
        {
            _messages.Clear();
        }
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task<int> GetCountAsync()
    {
        lock (_lock)
        {
            return Task.FromResult(_messages.Count);
        }
    }

    private void TrimToWindow()
    {
        // Each exchange = 2 messages (user + assistant)
        var maxMessages = WindowSize * 2;
        while (_messages.Count > maxMessages)
        {
            // Remove oldest pair
            _messages.RemoveAt(0);
            if (_messages.Count > 0)
                _messages.RemoveAt(0);
        }
    }
}

/// <summary>
/// Summary memory that summarizes older conversations.
/// Uses an LLM to create summaries when the conversation exceeds a threshold.
/// </summary>
public class SummaryMemory : IMemory
{
    private readonly LLM.ILLMClient _llmClient;
    private readonly List<ChatMessage> _recentMessages = new();
    private string _summary = string.Empty;
    private readonly object _lock = new();

    /// <summary>
    /// Gets or sets the number of recent messages to keep in full.
    /// </summary>
    public int RecentMessagesCount { get; set; } = 6;

    /// <summary>
    /// Gets or sets the threshold at which to create a summary.
    /// </summary>
    public int SummarizeThreshold { get; set; } = 10;

    /// <summary>
    /// Gets or sets the summary prompt template.
    /// </summary>
    public string SummaryPrompt { get; set; } = """
        Progressively summarize the conversation below, adding onto the existing summary.
        
        Current summary:
        {summary}
        
        New conversation:
        {new_lines}
        
        New summary:
        """;

    /// <summary>
    /// Creates a new summary memory.
    /// </summary>
    public SummaryMemory(LLM.ILLMClient llmClient)
    {
        _llmClient = llmClient ?? throw new ArgumentNullException(nameof(llmClient));
    }

    /// <inheritdoc/>
    public async Task AddMessageAsync(ChatMessage message)
    {
        lock (_lock)
        {
            _recentMessages.Add(message);
        }
        await TrySummarizeAsync();
    }

    /// <inheritdoc/>
    public async Task AddExchangeAsync(string userMessage, string assistantMessage)
    {
        lock (_lock)
        {
            _recentMessages.Add(ChatMessage.User(userMessage));
            _recentMessages.Add(ChatMessage.Assistant(assistantMessage));
        }
        await TrySummarizeAsync();
    }

    /// <inheritdoc/>
    public Task<List<ChatMessage>> GetMessagesAsync(string? query = null)
    {
        lock (_lock)
        {
            var result = new List<ChatMessage>();
            if (!string.IsNullOrEmpty(_summary))
            {
                result.Add(ChatMessage.System($"Previous conversation summary: {_summary}"));
            }
            result.AddRange(_recentMessages);
            return Task.FromResult(result);
        }
    }

    /// <inheritdoc/>
    public Task<string> GetContextStringAsync(string? query = null)
    {
        lock (_lock)
        {
            var sb = new StringBuilder();
            if (!string.IsNullOrEmpty(_summary))
            {
                sb.AppendLine($"Summary of earlier conversation: {_summary}");
                sb.AppendLine();
            }
            foreach (var msg in _recentMessages)
            {
                var prefix = msg.Role == "user" ? "Human" : "AI";
                sb.AppendLine($"{prefix}: {msg.Content}");
            }
            return Task.FromResult(sb.ToString());
        }
    }

    /// <inheritdoc/>
    public Task ClearAsync()
    {
        lock (_lock)
        {
            _recentMessages.Clear();
            _summary = string.Empty;
        }
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task<int> GetCountAsync()
    {
        lock (_lock)
        {
            return Task.FromResult(_recentMessages.Count);
        }
    }

    private async Task TrySummarizeAsync()
    {
        List<ChatMessage> messagesToSummarize;
        
        lock (_lock)
        {
            if (_recentMessages.Count < SummarizeThreshold)
                return;

            // Keep the most recent messages, summarize the rest
            var countToSummarize = _recentMessages.Count - RecentMessagesCount;
            if (countToSummarize <= 0)
                return;

            messagesToSummarize = _recentMessages.Take(countToSummarize).ToList();
            _recentMessages.RemoveRange(0, countToSummarize);
        }

        // Format messages for summarization
        var newLines = new StringBuilder();
        foreach (var msg in messagesToSummarize)
        {
            var prefix = msg.Role == "user" ? "Human" : "AI";
            newLines.AppendLine($"{prefix}: {msg.Content}");
        }

        // Generate summary
        var prompt = SummaryPrompt
            .Replace("{summary}", _summary)
            .Replace("{new_lines}", newLines.ToString());

        var newSummary = await _llmClient.ChatAsync(prompt);

        lock (_lock)
        {
            _summary = newSummary;
        }
    }
}

