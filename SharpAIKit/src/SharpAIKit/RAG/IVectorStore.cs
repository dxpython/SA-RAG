namespace SharpAIKit.RAG;

/// <summary>
/// Represents a document with its embedding vector.
/// </summary>
public class VectorDocument
{
    /// <summary>
    /// Gets or sets the unique document identifier.
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the original text content.
    /// </summary>
    public string Content { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the embedding vector.
    /// </summary>
    public float[] Embedding { get; set; } = Array.Empty<float>();

    /// <summary>
    /// Gets or sets the document metadata.
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}

/// <summary>
/// Represents a search result with similarity score.
/// </summary>
public class SearchResult
{
    /// <summary>
    /// Gets or sets the matched document.
    /// </summary>
    public VectorDocument Document { get; set; } = new();

    /// <summary>
    /// Gets or sets the similarity score.
    /// </summary>
    public float Score { get; set; }
}

/// <summary>
/// Interface for vector storage backends.
/// </summary>
public interface IVectorStore
{
    /// <summary>
    /// Adds a document to the store.
    /// </summary>
    /// <param name="document">The document to add.</param>
    Task AddAsync(VectorDocument document);

    /// <summary>
    /// Adds multiple documents to the store.
    /// </summary>
    /// <param name="documents">The documents to add.</param>
    Task AddRangeAsync(IEnumerable<VectorDocument> documents);

    /// <summary>
    /// Searches for similar documents.
    /// </summary>
    /// <param name="queryEmbedding">The query embedding vector.</param>
    /// <param name="topK">Number of results to return.</param>
    /// <returns>A list of search results ordered by similarity.</returns>
    Task<List<SearchResult>> SearchAsync(float[] queryEmbedding, int topK = 5);

    /// <summary>
    /// Deletes a document by its ID.
    /// </summary>
    /// <param name="documentId">The document ID to delete.</param>
    Task DeleteAsync(string documentId);

    /// <summary>
    /// Clears all documents from the store.
    /// </summary>
    Task ClearAsync();

    /// <summary>
    /// Gets the total number of documents in the store.
    /// </summary>
    /// <returns>The document count.</returns>
    Task<int> CountAsync();
}

