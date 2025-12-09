using SharpAIKit.LLM;

namespace SharpAIKit.RAG;

/// <summary>
/// Local mini RAG (Retrieval-Augmented Generation) engine.
/// Provides document indexing and intelligent Q and A capabilities.
/// </summary>
public class RagEngine
{
    private readonly ILLMClient _llmClient;
    private readonly IVectorStore _vectorStore;
    private readonly TextSplitter _textSplitter;

    /// <summary>
    /// Gets or sets the number of documents to retrieve for context.
    /// </summary>
    public int TopK { get; set; } = 3;

    /// <summary>
    /// Gets or sets the system prompt template.
    /// Use {context} as placeholder for the retrieved context.
    /// </summary>
    public string SystemPromptTemplate { get; set; } = """
        You are a professional AI assistant. Please answer the user's question based on the following reference materials.
        If the reference materials don't contain relevant information, please inform the user honestly.

        Reference Materials:
        {context}
        """;

    /// <summary>
    /// Creates a new RAG engine instance.
    /// </summary>
    /// <param name="llmClient">The LLM client for embeddings and generation.</param>
    /// <param name="vectorStore">The vector store (optional, defaults to in-memory store).</param>
    /// <param name="textSplitter">The text splitter (optional).</param>
    public RagEngine(
        ILLMClient llmClient,
        IVectorStore? vectorStore = null,
        TextSplitter? textSplitter = null)
    {
        _llmClient = llmClient ?? throw new ArgumentNullException(nameof(llmClient));
        _vectorStore = vectorStore ?? new MemoryVectorStore();
        _textSplitter = textSplitter ?? new TextSplitter();
    }

    /// <summary>
    /// Indexes document content.
    /// </summary>
    /// <param name="content">The document content to index.</param>
    /// <param name="metadata">Optional metadata for the document.</param>
    public async Task IndexContentAsync(string content, Dictionary<string, string>? metadata = null)
    {
        // Split text into chunks
        var chunks = _textSplitter.SplitBySentences(content);

        // Generate embeddings in batch
        var embeddings = await _llmClient.EmbeddingBatchAsync(chunks);

        // Store in vector database
        var documents = chunks.Zip(embeddings, (text, embedding) => new VectorDocument
        {
            Content = text,
            Embedding = embedding,
            Metadata = metadata ?? new Dictionary<string, string>()
        });

        await _vectorStore.AddRangeAsync(documents);
    }

    /// <summary>
    /// Indexes a document from a file.
    /// </summary>
    /// <param name="filePath">Path to the file to index.</param>
    public async Task IndexDocumentAsync(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"File not found: {filePath}");

        var content = await File.ReadAllTextAsync(filePath);
        var metadata = new Dictionary<string, string>
        {
            ["source"] = filePath,
            ["filename"] = Path.GetFileName(filePath)
        };

        await IndexContentAsync(content, metadata);
    }

    /// <summary>
    /// Indexes all text files in a directory.
    /// </summary>
    /// <param name="directoryPath">Path to the directory.</param>
    /// <param name="searchPattern">File search pattern (default: *.txt).</param>
    public async Task IndexDirectoryAsync(string directoryPath, string searchPattern = "*.txt")
    {
        if (!Directory.Exists(directoryPath))
            throw new DirectoryNotFoundException($"Directory not found: {directoryPath}");

        var files = Directory.GetFiles(directoryPath, searchPattern, SearchOption.AllDirectories);
        foreach (var file in files)
        {
            await IndexDocumentAsync(file);
        }
    }

    /// <summary>
    /// Answers a question using RAG.
    /// </summary>
    /// <param name="question">The user's question.</param>
    /// <returns>The AI-generated answer.</returns>
    public async Task<string> AskAsync(string question)
    {
        // Generate embedding for the question
        var queryEmbedding = await _llmClient.EmbeddingAsync(question);

        // Search for relevant documents
        var searchResults = await _vectorStore.SearchAsync(queryEmbedding, TopK);

        // Build context from search results
        var context = string.Join("\n\n", searchResults.Select((r, i) =>
            $"[{i + 1}] {r.Document.Content}"));

        // Build prompt with context
        var systemPrompt = SystemPromptTemplate.Replace("{context}", context);
        var messages = new[]
        {
            Common.ChatMessage.System(systemPrompt),
            Common.ChatMessage.User(question)
        };

        // Generate answer using LLM
        return await _llmClient.ChatAsync(messages);
    }

    /// <summary>
    /// Answers a question using RAG with streaming response.
    /// </summary>
    /// <param name="question">The user's question.</param>
    /// <returns>An async enumerable of response chunks.</returns>
    public async IAsyncEnumerable<string> AskStreamAsync(string question)
    {
        // Generate embedding for the question
        var queryEmbedding = await _llmClient.EmbeddingAsync(question);

        // Search for relevant documents
        var searchResults = await _vectorStore.SearchAsync(queryEmbedding, TopK);

        // Build context from search results
        var context = string.Join("\n\n", searchResults.Select((r, i) =>
            $"[{i + 1}] {r.Document.Content}"));

        // Build prompt with context
        var systemPrompt = SystemPromptTemplate.Replace("{context}", context);
        var messages = new[]
        {
            Common.ChatMessage.System(systemPrompt),
            Common.ChatMessage.User(question)
        };

        // Stream answer from LLM
        await foreach (var chunk in _llmClient.ChatStreamAsync(messages))
        {
            yield return chunk;
        }
    }

    /// <summary>
    /// Retrieves relevant documents without generating an answer.
    /// </summary>
    /// <param name="query">The search query.</param>
    /// <param name="topK">Number of results to return (optional, uses default if not specified).</param>
    /// <returns>A list of search results.</returns>
    public async Task<List<SearchResult>> RetrieveAsync(string query, int? topK = null)
    {
        var queryEmbedding = await _llmClient.EmbeddingAsync(query);
        return await _vectorStore.SearchAsync(queryEmbedding, topK ?? TopK);
    }

    /// <summary>
    /// Clears all indexed documents.
    /// </summary>
    public Task ClearIndexAsync() => _vectorStore.ClearAsync();

    /// <summary>
    /// Gets the count of indexed documents.
    /// </summary>
    /// <returns>The number of indexed document chunks.</returns>
    public Task<int> GetDocumentCountAsync() => _vectorStore.CountAsync();
}

