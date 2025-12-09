namespace SharpAIKit.RAG;

/// <summary>
/// In-memory vector store implementation.
/// Suitable for small-scale data and rapid prototyping.
/// </summary>
public class MemoryVectorStore : IVectorStore
{
    private readonly List<VectorDocument> _documents = new();
    private readonly object _lock = new();

    /// <summary>
    /// Gets or sets the similarity function to use.
    /// Default is cosine similarity.
    /// </summary>
    public Func<float[], float[], float> SimilarityFunction { get; set; } = Similarity.Cosine;

    /// <inheritdoc />
    public Task AddAsync(VectorDocument document)
    {
        lock (_lock)
        {
            _documents.Add(document);
        }
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public Task AddRangeAsync(IEnumerable<VectorDocument> documents)
    {
        lock (_lock)
        {
            _documents.AddRange(documents);
        }
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public Task<List<SearchResult>> SearchAsync(float[] queryEmbedding, int topK = 5)
    {
        List<SearchResult> results;

        lock (_lock)
        {
            results = _documents
                .Select(doc => new SearchResult
                {
                    Document = doc,
                    Score = SimilarityFunction(queryEmbedding, doc.Embedding)
                })
                .OrderByDescending(r => r.Score)
                .Take(topK)
                .ToList();
        }

        return Task.FromResult(results);
    }

    /// <inheritdoc />
    public Task DeleteAsync(string documentId)
    {
        lock (_lock)
        {
            _documents.RemoveAll(d => d.Id == documentId);
        }
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public Task ClearAsync()
    {
        lock (_lock)
        {
            _documents.Clear();
        }
        return Task.CompletedTask;
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

