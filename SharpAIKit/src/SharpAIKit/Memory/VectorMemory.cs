using System.Text;
using SharpAIKit.Common;
using SharpAIKit.LLM;
using SharpAIKit.RAG;

namespace SharpAIKit.Memory;

/// <summary>
/// Vector-based memory that retrieves relevant past conversations using semantic search.
/// This is useful for long conversations where you want to recall relevant context.
/// </summary>
public class VectorMemory : IMemory
{
    private readonly ILLMClient _llmClient;
    private readonly IVectorStore _vectorStore;
    private readonly List<ChatMessage> _allMessages = new();
    private readonly object _lock = new();

    /// <summary>
    /// Gets or sets the number of relevant messages to retrieve.
    /// </summary>
    public int TopK { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to include recent messages regardless of relevance.
    /// </summary>
    public int RecentMessagesToAlwaysInclude { get; set; } = 4;

    /// <summary>
    /// Creates a new vector memory.
    /// </summary>
    public VectorMemory(ILLMClient llmClient, IVectorStore? vectorStore = null)
    {
        _llmClient = llmClient ?? throw new ArgumentNullException(nameof(llmClient));
        _vectorStore = vectorStore ?? new MemoryVectorStore();
    }

    /// <inheritdoc/>
    public async Task AddMessageAsync(ChatMessage message)
    {
        lock (_lock)
        {
            _allMessages.Add(message);
        }

        // Generate embedding and store
        var embedding = await _llmClient.EmbeddingAsync(message.Content);
        var doc = new VectorDocument
        {
            Content = $"{message.Role}: {message.Content}",
            Embedding = embedding,
            Metadata = new Dictionary<string, string>
            {
                ["role"] = message.Role,
                ["index"] = (_allMessages.Count - 1).ToString()
            }
        };
        await _vectorStore.AddAsync(doc);
    }

    /// <inheritdoc/>
    public async Task AddExchangeAsync(string userMessage, string assistantMessage)
    {
        await AddMessageAsync(ChatMessage.User(userMessage));
        await AddMessageAsync(ChatMessage.Assistant(assistantMessage));
    }

    /// <inheritdoc/>
    public async Task<List<ChatMessage>> GetMessagesAsync(string? query = null)
    {
        List<ChatMessage> allMessages;
        lock (_lock)
        {
            allMessages = _allMessages.ToList();
        }

        if (string.IsNullOrEmpty(query) || allMessages.Count <= TopK + RecentMessagesToAlwaysInclude)
        {
            // Return recent messages if no query or few messages
            return allMessages.TakeLast(TopK + RecentMessagesToAlwaysInclude).ToList();
        }

        // Search for relevant messages
        var queryEmbedding = await _llmClient.EmbeddingAsync(query);
        var searchResults = await _vectorStore.SearchAsync(queryEmbedding, TopK);

        // Get indices of relevant messages
        var relevantIndices = searchResults
            .Select(r => int.TryParse(r.Document.Metadata.GetValueOrDefault("index"), out var i) ? i : -1)
            .Where(i => i >= 0)
            .ToHashSet();

        // Always include recent messages
        var recentStartIndex = Math.Max(0, allMessages.Count - RecentMessagesToAlwaysInclude);
        for (var i = recentStartIndex; i < allMessages.Count; i++)
        {
            relevantIndices.Add(i);
        }

        // Build result maintaining chronological order
        var result = new List<ChatMessage>();
        foreach (var index in relevantIndices.OrderBy(i => i))
        {
            if (index < allMessages.Count)
            {
                result.Add(allMessages[index]);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public async Task<string> GetContextStringAsync(string? query = null)
    {
        var messages = await GetMessagesAsync(query);
        var sb = new StringBuilder();
        foreach (var msg in messages)
        {
            var prefix = msg.Role == "user" ? "Human" : "AI";
            sb.AppendLine($"{prefix}: {msg.Content}");
        }
        return sb.ToString();
    }

    /// <inheritdoc/>
    public async Task ClearAsync()
    {
        lock (_lock)
        {
            _allMessages.Clear();
        }
        await _vectorStore.ClearAsync();
    }

    /// <inheritdoc/>
    public Task<int> GetCountAsync()
    {
        lock (_lock)
        {
            return Task.FromResult(_allMessages.Count);
        }
    }
}

/// <summary>
/// Entity memory that extracts and tracks entities mentioned in conversations.
/// </summary>
public class EntityMemory : IMemory
{
    private readonly ILLMClient _llmClient;
    private readonly Dictionary<string, string> _entities = new();
    private readonly List<ChatMessage> _recentMessages = new();
    private readonly object _lock = new();

    /// <summary>
    /// Gets or sets the number of recent messages to keep.
    /// </summary>
    public int RecentMessagesCount { get; set; } = 6;

    /// <summary>
    /// Gets or sets the entity extraction prompt.
    /// </summary>
    public string EntityExtractionPrompt { get; set; } = """
        Extract any entities (people, places, things, concepts) mentioned in this conversation and their descriptions.
        Format as JSON: {"entity_name": "description", ...}
        Only include entities that are clearly defined or described.
        
        Conversation:
        {conversation}
        
        Entities (JSON):
        """;

    /// <summary>
    /// Creates a new entity memory.
    /// </summary>
    public EntityMemory(ILLMClient llmClient)
    {
        _llmClient = llmClient ?? throw new ArgumentNullException(nameof(llmClient));
    }

    /// <inheritdoc/>
    public async Task AddMessageAsync(ChatMessage message)
    {
        lock (_lock)
        {
            _recentMessages.Add(message);
            while (_recentMessages.Count > RecentMessagesCount)
            {
                _recentMessages.RemoveAt(0);
            }
        }

        // Periodically extract entities
        if (_recentMessages.Count >= 2)
        {
            await ExtractEntitiesAsync();
        }
    }

    /// <inheritdoc/>
    public async Task AddExchangeAsync(string userMessage, string assistantMessage)
    {
        await AddMessageAsync(ChatMessage.User(userMessage));
        await AddMessageAsync(ChatMessage.Assistant(assistantMessage));
    }

    /// <inheritdoc/>
    public Task<List<ChatMessage>> GetMessagesAsync(string? query = null)
    {
        lock (_lock)
        {
            var result = new List<ChatMessage>();
            
            // Add entity context as system message
            if (_entities.Count > 0)
            {
                var entityContext = string.Join("\n", _entities.Select(e => $"- {e.Key}: {e.Value}"));
                result.Add(ChatMessage.System($"Known entities:\n{entityContext}"));
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
            
            if (_entities.Count > 0)
            {
                sb.AppendLine("Known entities:");
                foreach (var entity in _entities)
                {
                    sb.AppendLine($"- {entity.Key}: {entity.Value}");
                }
                sb.AppendLine();
            }
            
            sb.AppendLine("Recent conversation:");
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
            _entities.Clear();
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

    /// <summary>
    /// Gets all tracked entities.
    /// </summary>
    public IReadOnlyDictionary<string, string> GetEntities()
    {
        lock (_lock)
        {
            return new Dictionary<string, string>(_entities);
        }
    }

    private async Task ExtractEntitiesAsync()
    {
        string conversation;
        lock (_lock)
        {
            var sb = new StringBuilder();
            foreach (var msg in _recentMessages)
            {
                var prefix = msg.Role == "user" ? "Human" : "AI";
                sb.AppendLine($"{prefix}: {msg.Content}");
            }
            conversation = sb.ToString();
        }

        var prompt = EntityExtractionPrompt.Replace("{conversation}", conversation);
        
        try
        {
            var response = await _llmClient.ChatAsync(prompt);
            
            // Parse JSON response
            var jsonStart = response.IndexOf('{');
            var jsonEnd = response.LastIndexOf('}');
            if (jsonStart >= 0 && jsonEnd > jsonStart)
            {
                var json = response.Substring(jsonStart, jsonEnd - jsonStart + 1);
                var extracted = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, string>>(json);
                
                if (extracted != null)
                {
                    lock (_lock)
                    {
                        foreach (var entity in extracted)
                        {
                            _entities[entity.Key] = entity.Value;
                        }
                    }
                }
            }
        }
        catch
        {
            // Ignore extraction errors
        }
    }
}

