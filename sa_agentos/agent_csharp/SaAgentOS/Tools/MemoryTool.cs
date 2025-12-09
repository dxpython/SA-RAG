using System.Text.Json;
using Microsoft.Extensions.Logging;
using SaAgentOS.Models;
using SharpAIKit.Agent;

namespace SaAgentOS.Tools;

/// <summary>
/// Tool for managing agent memory
/// Inherits from SharpAIKit's ToolBase for seamless integration
/// </summary>
public class MemoryTool : ToolBase
{
    private readonly MemoryConfig _config;
    private readonly HttpClient? _httpClient;
    private readonly ILogger<MemoryTool>? _logger;
    private readonly List<MemoryEntry> _localMemory;

    public MemoryTool(
        MemoryConfig config,
        HttpClient? httpClient = null,
        ILogger<MemoryTool>? logger = null)
    {
        _config = config;
        _httpClient = httpClient;
        _logger = logger;
        _localMemory = new List<MemoryEntry>();

        // Load existing memory if file exists
        LoadMemory();
    }

    /// <summary>
    /// Memory entry
    /// </summary>
    public class MemoryEntry
    {
        public string Text { get; set; } = string.Empty;
        public double Importance { get; set; } = 0.5;
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        public Dictionary<string, object> Metadata { get; set; } = new();
    }

    /// <summary>
    /// Add memory entry (SharpAIKit tool method)
    /// </summary>
    [Tool("add_memory", "Adds a memory entry to the agent's long-term memory. Use this to store important information for future reference.")]
    public async Task<string> AddMemory(
        [Parameter("The memory content to store")] string text,
        [Parameter("Importance score from 0.0 to 1.0 (default: 0.5)")] double importance = 0.5)
    {
        await AddMemoryAsync(text, importance);
        return $"Memory added successfully: {text.Substring(0, Math.Min(50, text.Length))}...";
    }

    /// <summary>
    /// Add memory entry (internal method)
    /// </summary>
    public async Task AddMemoryAsync(
        string text,
        double importance = 0.5,
        Dictionary<string, object>? metadata = null,
        CancellationToken cancellationToken = default)
    {
        var entry = new MemoryEntry
        {
            Text = text,
            Importance = importance,
            Timestamp = DateTime.UtcNow,
            Metadata = metadata ?? new Dictionary<string, object>()
        };

        _localMemory.Add(entry);

        // If configured, also add to SA-RAG memory
        if (_config.UseSaRagMemory && _httpClient != null)
        {
            try
            {
                var memoryUrl = _config.StoragePath.Replace("agent_memory.json", "").TrimEnd('/');
                if (string.IsNullOrEmpty(memoryUrl))
                {
                    memoryUrl = "http://localhost:8000";
                }
                memoryUrl = memoryUrl.TrimEnd('/') + "/rag/memory";

                var response = await _httpClient.PostAsync(
                    $"{memoryUrl}?text={Uri.EscapeDataString(text)}&importance={importance}",
                    null,
                    cancellationToken);

                if (response.IsSuccessStatusCode)
                {
                    _logger?.LogInformation("Memory added to SA-RAG: Text='{Text}', Importance={Importance}",
                        text, importance);
                }
            }
            catch (Exception ex)
            {
                _logger?.LogWarning(ex, "Failed to add memory to SA-RAG, using local storage only");
            }
        }

        // Save to local file
        SaveMemory();

        _logger?.LogInformation("Memory added: Text='{Text}', Importance={Importance}",
            text, importance);
    }

    /// <summary>
    /// Retrieve relevant memories (SharpAIKit tool method)
    /// </summary>
    [Tool("retrieve_memory", "Retrieves relevant memories from the agent's long-term memory based on a query. Use this to recall past information.")]
    public string RetrieveMemory(
        [Parameter("The query to search for in memories")] string query,
        [Parameter("Maximum number of results to return (default: 5)")] int maxResults = 5)
    {
        var memories = RetrieveRelevant(query, maxResults);
        if (memories.Count == 0)
        {
            return "No relevant memories found.";
        }

        var result = $"Found {memories.Count} relevant memories:\n\n";
        foreach (var memory in memories)
        {
            result += $"- [{memory.Importance:F2}] {memory.Text}\n";
        }
        return result;
    }

    /// <summary>
    /// Retrieve relevant memories (internal method)
    /// </summary>
    public List<MemoryEntry> RetrieveRelevant(
        string query,
        int maxResults = 5)
    {
        // Simple keyword-based retrieval (can be enhanced with embeddings)
        var relevant = _localMemory
            .Where(m => m.Text.Contains(query, StringComparison.OrdinalIgnoreCase) ||
                       query.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                           .Any(word => m.Text.Contains(word, StringComparison.OrdinalIgnoreCase)))
            .OrderByDescending(m => m.Importance)
            .ThenByDescending(m => m.Timestamp)
            .Take(maxResults)
            .ToList();

        _logger?.LogInformation("Retrieved {Count} relevant memories for query: '{Query}'",
            relevant.Count, query);

        return relevant;
    }

    /// <summary>
    /// Load memory from file
    /// </summary>
    private void LoadMemory()
    {
        try
        {
            if (File.Exists(_config.StoragePath))
            {
                var json = File.ReadAllText(_config.StoragePath);
                var entries = JsonSerializer.Deserialize<List<MemoryEntry>>(json);
                if (entries != null)
                {
                    _localMemory.AddRange(entries);
                    _logger?.LogInformation("Loaded {Count} memory entries from {Path}",
                        _localMemory.Count, _config.StoragePath);
                }
            }
        }
        catch (Exception ex)
        {
            _logger?.LogWarning(ex, "Failed to load memory from {Path}", _config.StoragePath);
        }
    }

    /// <summary>
    /// Save memory to file
    /// </summary>
    private void SaveMemory()
    {
        try
        {
            var directory = Path.GetDirectoryName(_config.StoragePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var json = JsonSerializer.Serialize(_localMemory, new JsonSerializerOptions
            {
                WriteIndented = true
            });
            File.WriteAllText(_config.StoragePath, json);
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Failed to save memory to {Path}", _config.StoragePath);
        }
    }
}

