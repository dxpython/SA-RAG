using System.Text;

namespace SharpAIKit.DocumentLoader;

/// <summary>
/// Loads plain text files.
/// </summary>
public class TextFileLoader : FileLoaderBase
{
    private readonly Encoding _encoding;

    /// <summary>
    /// Creates a new text file loader.
    /// </summary>
    public TextFileLoader(string filePath, Encoding? encoding = null) : base(filePath)
    {
        _encoding = encoding ?? Encoding.UTF8;
    }

    /// <inheritdoc/>
    public override async Task<List<Document>> LoadAsync(CancellationToken cancellationToken = default)
    {
        var content = await File.ReadAllTextAsync(FilePath, _encoding, cancellationToken);
        var metadata = CreateBaseMetadata();
        metadata["encoding"] = _encoding.WebName;
        metadata["char_count"] = content.Length.ToString();

        return new List<Document>
        {
            new Document { Content = content, Metadata = metadata }
        };
    }
}

/// <summary>
/// Loads all text files from a directory.
/// </summary>
public class TextDirectoryLoader : DirectoryLoaderBase
{
    private readonly Encoding _encoding;

    /// <summary>
    /// Creates a new text directory loader.
    /// </summary>
    public TextDirectoryLoader(
        string directoryPath, 
        string searchPattern = "*.txt",
        bool recursive = true,
        Encoding? encoding = null) 
        : base(directoryPath, searchPattern, recursive)
    {
        _encoding = encoding ?? Encoding.UTF8;
    }

    /// <inheritdoc/>
    public override async Task<List<Document>> LoadAsync(CancellationToken cancellationToken = default)
    {
        var documents = new List<Document>();
        
        foreach (var file in GetFiles())
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            var content = await File.ReadAllTextAsync(file, _encoding, cancellationToken);
            documents.Add(new Document
            {
                Content = content,
                Metadata = new Dictionary<string, string>
                {
                    ["source"] = file,
                    ["filename"] = Path.GetFileName(file),
                    ["extension"] = Path.GetExtension(file),
                    ["encoding"] = _encoding.WebName
                }
            });
        }
        
        return documents;
    }

    /// <inheritdoc/>
    public override async IAsyncEnumerable<Document> LazyLoadAsync(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        foreach (var file in GetFiles())
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            var content = await File.ReadAllTextAsync(file, _encoding, cancellationToken);
            yield return new Document
            {
                Content = content,
                Metadata = new Dictionary<string, string>
                {
                    ["source"] = file,
                    ["filename"] = Path.GetFileName(file),
                    ["extension"] = Path.GetExtension(file)
                }
            };
        }
    }
}

/// <summary>
/// Loads CSV files with column awareness.
/// </summary>
public class CsvLoader : FileLoaderBase
{
    private readonly Encoding _encoding;
    private readonly char _delimiter;
    private readonly bool _hasHeader;

    /// <summary>
    /// Gets or sets whether to create one document per row.
    /// </summary>
    public bool OneDocumentPerRow { get; set; } = false;

    /// <summary>
    /// Gets or sets the content column(s) to use (null = all columns).
    /// </summary>
    public string[]? ContentColumns { get; set; }

    /// <summary>
    /// Gets or sets the metadata columns.
    /// </summary>
    public string[]? MetadataColumns { get; set; }

    /// <summary>
    /// Creates a new CSV loader.
    /// </summary>
    public CsvLoader(
        string filePath, 
        char delimiter = ',',
        bool hasHeader = true,
        Encoding? encoding = null) : base(filePath)
    {
        _delimiter = delimiter;
        _hasHeader = hasHeader;
        _encoding = encoding ?? Encoding.UTF8;
    }

    /// <inheritdoc/>
    public override async Task<List<Document>> LoadAsync(CancellationToken cancellationToken = default)
    {
        var lines = await File.ReadAllLinesAsync(FilePath, _encoding, cancellationToken);
        if (lines.Length == 0) return new List<Document>();

        var documents = new List<Document>();
        var headers = _hasHeader 
            ? ParseLine(lines[0]) 
            : Enumerable.Range(0, ParseLine(lines[0]).Length).Select(i => $"column_{i}").ToArray();
        
        var startIndex = _hasHeader ? 1 : 0;

        if (OneDocumentPerRow)
        {
            for (var i = startIndex; i < lines.Length; i++)
            {
                var values = ParseLine(lines[i]);
                var rowData = headers.Zip(values, (h, v) => (h, v)).ToDictionary(x => x.h, x => x.v);
                
                // Build content
                var contentCols = ContentColumns ?? headers;
                var content = string.Join("\n", contentCols
                    .Where(c => rowData.ContainsKey(c))
                    .Select(c => $"{c}: {rowData[c]}"));

                // Build metadata
                var metadata = CreateBaseMetadata();
                metadata["row"] = i.ToString();
                
                if (MetadataColumns != null)
                {
                    foreach (var col in MetadataColumns.Where(c => rowData.ContainsKey(c)))
                    {
                        metadata[col] = rowData[col];
                    }
                }

                documents.Add(new Document { Content = content, Metadata = metadata });
            }
        }
        else
        {
            // Single document with all content
            var sb = new StringBuilder();
            sb.AppendLine(string.Join(_delimiter, headers));
            for (var i = startIndex; i < lines.Length; i++)
            {
                sb.AppendLine(lines[i]);
            }

            var metadata = CreateBaseMetadata();
            metadata["row_count"] = (lines.Length - startIndex).ToString();
            metadata["column_count"] = headers.Length.ToString();
            
            documents.Add(new Document { Content = sb.ToString(), Metadata = metadata });
        }

        return documents;
    }

    private string[] ParseLine(string line)
    {
        var result = new List<string>();
        var inQuotes = false;
        var current = new StringBuilder();

        foreach (var c in line)
        {
            if (c == '"')
            {
                inQuotes = !inQuotes;
            }
            else if (c == _delimiter && !inQuotes)
            {
                result.Add(current.ToString().Trim());
                current.Clear();
            }
            else
            {
                current.Append(c);
            }
        }
        result.Add(current.ToString().Trim());

        return result.ToArray();
    }
}

/// <summary>
/// Loads Markdown files with optional section splitting.
/// </summary>
public class MarkdownLoader : FileLoaderBase
{
    /// <summary>
    /// Gets or sets whether to split by headers.
    /// </summary>
    public bool SplitByHeaders { get; set; } = false;

    /// <summary>
    /// Gets or sets the minimum header level to split on (1-6).
    /// </summary>
    public int SplitHeaderLevel { get; set; } = 2;

    /// <summary>
    /// Creates a new Markdown loader.
    /// </summary>
    public MarkdownLoader(string filePath) : base(filePath) { }

    /// <inheritdoc/>
    public override async Task<List<Document>> LoadAsync(CancellationToken cancellationToken = default)
    {
        var content = await File.ReadAllTextAsync(FilePath, cancellationToken);
        var metadata = CreateBaseMetadata();
        metadata["format"] = "markdown";

        if (!SplitByHeaders)
        {
            return new List<Document>
            {
                new Document { Content = content, Metadata = metadata }
            };
        }

        // Split by headers
        var documents = new List<Document>();
        var lines = content.Split('\n');
        var currentSection = new StringBuilder();
        var currentHeader = string.Empty;
        var headerPattern = new string('#', SplitHeaderLevel);

        foreach (var line in lines)
        {
            if (line.StartsWith(headerPattern) && !line.StartsWith(headerPattern + "#"))
            {
                // Save previous section
                if (currentSection.Length > 0)
                {
                    var sectionMeta = new Dictionary<string, string>(metadata)
                    {
                        ["section"] = currentHeader
                    };
                    documents.Add(new Document
                    {
                        Content = currentSection.ToString().Trim(),
                        Metadata = sectionMeta
                    });
                    currentSection.Clear();
                }
                currentHeader = line.TrimStart('#').Trim();
            }
            currentSection.AppendLine(line);
        }

        // Add last section
        if (currentSection.Length > 0)
        {
            var sectionMeta = new Dictionary<string, string>(metadata)
            {
                ["section"] = currentHeader
            };
            documents.Add(new Document
            {
                Content = currentSection.ToString().Trim(),
                Metadata = sectionMeta
            });
        }

        return documents;
    }
}

/// <summary>
/// Loads JSON files.
/// </summary>
public class JsonLoader : FileLoaderBase
{
    /// <summary>
    /// Gets or sets the JSON path to extract content from.
    /// </summary>
    public string? ContentPath { get; set; }

    /// <summary>
    /// Gets or sets whether to pretty-print the JSON.
    /// </summary>
    public bool PrettyPrint { get; set; } = true;

    /// <summary>
    /// Creates a new JSON loader.
    /// </summary>
    public JsonLoader(string filePath) : base(filePath) { }

    /// <inheritdoc/>
    public override async Task<List<Document>> LoadAsync(CancellationToken cancellationToken = default)
    {
        var content = await File.ReadAllTextAsync(FilePath, cancellationToken);
        var metadata = CreateBaseMetadata();
        metadata["format"] = "json";

        if (PrettyPrint)
        {
            try
            {
                using var doc = System.Text.Json.JsonDocument.Parse(content);
                content = System.Text.Json.JsonSerializer.Serialize(doc, new System.Text.Json.JsonSerializerOptions 
                { 
                    WriteIndented = true 
                });
            }
            catch
            {
                // Keep original if parsing fails
            }
        }

        return new List<Document>
        {
            new Document { Content = content, Metadata = metadata }
        };
    }
}

/// <summary>
/// Loads content from a URL.
/// </summary>
public class WebLoader : IDocumentLoader
{
    private readonly string _url;
    private readonly HttpClient _httpClient;

    /// <summary>
    /// Creates a new web loader.
    /// </summary>
    public WebLoader(string url, HttpClient? httpClient = null)
    {
        _url = url ?? throw new ArgumentNullException(nameof(url));
        _httpClient = httpClient ?? new HttpClient();
    }

    /// <inheritdoc/>
    public async Task<List<Document>> LoadAsync(CancellationToken cancellationToken = default)
    {
        var content = await _httpClient.GetStringAsync(_url, cancellationToken);
        
        return new List<Document>
        {
            new Document
            {
                Content = content,
                Metadata = new Dictionary<string, string>
                {
                    ["source"] = _url,
                    ["type"] = "web"
                }
            }
        };
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<Document> LazyLoadAsync(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var documents = await LoadAsync(cancellationToken);
        foreach (var doc in documents)
        {
            yield return doc;
        }
    }
}

