using System.Text.Json;
using SharpAIKit.Common;

namespace SharpAIKit.RAG;

/// <summary>
/// A simple file-based vector store for persistence.
/// Stores documents and embeddings in a local JSON file.
/// </summary>
public class FileVectorStore : IVectorStore
{
    private readonly string _filePath;
    private readonly List<VectorDocument> _documents = new();
    private readonly object _lock = new();

    /// <summary>
    /// Creates a new file vector store.
    /// </summary>
    /// <param name="filePath">The path to the JSON file.</param>
    public FileVectorStore(string filePath)
    {
        _filePath = filePath;
        LoadAsync().Wait();
    }

    /// <summary>
    /// Loads the index from disk.
    /// </summary>
    private async Task LoadAsync()
    {
        if (!File.Exists(_filePath)) return;

        try
        {
            var json = await File.ReadAllTextAsync(_filePath);
            var docs = JsonSerializer.Deserialize<List<VectorDocument>>(json);
            if (docs != null)
            {
                lock (_lock)
                {
                    _documents.Clear();
                    _documents.AddRange(docs);
                }
            }
        }
        catch
        {
            // Ignore load errors (start fresh)
        }
    }

    /// <summary>
    /// Saves the index to disk.
    /// </summary>
    private async Task SaveAsync()
    {
        string json;
        lock (_lock)
        {
            json = JsonSerializer.Serialize(_documents);
        }
        await File.WriteAllTextAsync(_filePath, json);
    }

    /// <inheritdoc />
    public async Task AddAsync(VectorDocument document)
    {
        lock (_lock)
        {
            _documents.Add(document);
        }
        await SaveAsync();
    }

    /// <inheritdoc />
    public async Task AddRangeAsync(IEnumerable<VectorDocument> documents)
    {
        lock (_lock)
        {
            _documents.AddRange(documents);
        }
        await SaveAsync();
    }

    /// <inheritdoc />
    public Task<List<SearchResult>> SearchAsync(float[] queryEmbedding, int topK = 3)
    {
        List<VectorDocument> docs;
        lock (_lock)
        {
            docs = _documents.ToList();
        }

        var results = docs
            .Select(doc => new SearchResult
            {
                Document = doc,
                Score = Similarity.Cosine(queryEmbedding, doc.Embedding)
            })
            .OrderByDescending(r => r.Score)
            .Take(topK)
            .ToList();

        return Task.FromResult(results);
    }

    /// <inheritdoc />
    public async Task DeleteAsync(string documentId)
    {
        lock (_lock)
        {
            _documents.RemoveAll(d => d.Id == documentId);
        }
        await SaveAsync();
    }

    /// <inheritdoc />
    public async Task ClearAsync()
    {
        lock (_lock)
        {
            _documents.Clear();
        }
        await SaveAsync();
    }

    /// <inheritdoc />
    public Task<int> CountAsync()
    {
        lock (_lock)
        {
            return Task.FromResult(_documents.Count);
        }
    }
}
