namespace SharpAIKit.DocumentLoader;

/// <summary>
/// Represents a loaded document with content and metadata.
/// </summary>
public class Document
{
    /// <summary>
    /// Gets or sets the document content.
    /// </summary>
    public string Content { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the document metadata.
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();

    /// <summary>
    /// Creates a document with content only.
    /// </summary>
    public static Document FromContent(string content) => new() { Content = content };

    /// <summary>
    /// Creates a document with content and source.
    /// </summary>
    public static Document FromContent(string content, string source) => new()
    {
        Content = content,
        Metadata = new() { ["source"] = source }
    };
}

/// <summary>
/// Interface for document loaders.
/// </summary>
public interface IDocumentLoader
{
    /// <summary>
    /// Loads documents from a source.
    /// </summary>
    Task<List<Document>> LoadAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Lazily loads documents one at a time.
    /// </summary>
    IAsyncEnumerable<Document> LazyLoadAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Base class for file-based document loaders.
/// </summary>
public abstract class FileLoaderBase : IDocumentLoader
{
    /// <summary>
    /// Gets the file path.
    /// </summary>
    protected string FilePath { get; }

    /// <summary>
    /// Creates a new file loader.
    /// </summary>
    protected FileLoaderBase(string filePath)
    {
        FilePath = filePath ?? throw new ArgumentNullException(nameof(filePath));
    }

    /// <inheritdoc/>
    public abstract Task<List<Document>> LoadAsync(CancellationToken cancellationToken = default);

    /// <inheritdoc/>
    public virtual async IAsyncEnumerable<Document> LazyLoadAsync(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var documents = await LoadAsync(cancellationToken);
        foreach (var doc in documents)
        {
            yield return doc;
        }
    }

    /// <summary>
    /// Creates base metadata for the file.
    /// </summary>
    protected Dictionary<string, string> CreateBaseMetadata()
    {
        return new Dictionary<string, string>
        {
            ["source"] = FilePath,
            ["filename"] = Path.GetFileName(FilePath),
            ["extension"] = Path.GetExtension(FilePath)
        };
    }
}

/// <summary>
/// Base class for directory loaders.
/// </summary>
public abstract class DirectoryLoaderBase : IDocumentLoader
{
    /// <summary>
    /// Gets the directory path.
    /// </summary>
    protected string DirectoryPath { get; }

    /// <summary>
    /// Gets the search pattern.
    /// </summary>
    protected string SearchPattern { get; }

    /// <summary>
    /// Gets whether to search recursively.
    /// </summary>
    protected bool Recursive { get; }

    /// <summary>
    /// Creates a new directory loader.
    /// </summary>
    protected DirectoryLoaderBase(string directoryPath, string searchPattern = "*.*", bool recursive = true)
    {
        DirectoryPath = directoryPath ?? throw new ArgumentNullException(nameof(directoryPath));
        SearchPattern = searchPattern;
        Recursive = recursive;
    }

    /// <inheritdoc/>
    public abstract Task<List<Document>> LoadAsync(CancellationToken cancellationToken = default);

    /// <inheritdoc/>
    public virtual async IAsyncEnumerable<Document> LazyLoadAsync(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var documents = await LoadAsync(cancellationToken);
        foreach (var doc in documents)
        {
            yield return doc;
        }
    }

    /// <summary>
    /// Gets all files matching the pattern.
    /// </summary>
    protected IEnumerable<string> GetFiles()
    {
        var searchOption = Recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
        return Directory.GetFiles(DirectoryPath, SearchPattern, searchOption);
    }
}

