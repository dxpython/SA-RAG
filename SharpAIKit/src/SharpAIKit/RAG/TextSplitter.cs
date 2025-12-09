using System.Text.RegularExpressions;

namespace SharpAIKit.RAG;

/// <summary>
/// Text splitter for chunking documents.
/// Supports splitting by sentences, fixed character count, or paragraphs.
/// </summary>
public class TextSplitter
{
    /// <summary>
    /// Gets or sets the maximum number of characters per chunk.
    /// </summary>
    public int ChunkSize { get; set; } = 500;

    /// <summary>
    /// Gets or sets the number of overlapping characters between chunks.
    /// </summary>
    public int ChunkOverlap { get; set; } = 50;

    /// <summary>
    /// Regex pattern for sentence boundaries.
    /// </summary>
    private static readonly Regex SentenceRegex = new(@"(?<=[。！？.!?])\s*", RegexOptions.Compiled);

    /// <summary>
    /// Creates a new text splitter instance.
    /// </summary>
    /// <param name="chunkSize">Maximum chunk size in characters.</param>
    /// <param name="chunkOverlap">Overlap size in characters.</param>
    public TextSplitter(int chunkSize = 500, int chunkOverlap = 50)
    {
        ChunkSize = chunkSize;
        ChunkOverlap = chunkOverlap;
    }

    /// <summary>
    /// Splits text by fixed character count with overlap.
    /// </summary>
    /// <param name="text">The text to split.</param>
    /// <returns>A list of text chunks.</returns>
    public List<string> SplitByCharacters(string text)
    {
        if (string.IsNullOrEmpty(text)) return new List<string>();

        var chunks = new List<string>();
        var start = 0;

        while (start < text.Length)
        {
            var length = Math.Min(ChunkSize, text.Length - start);
            var chunk = text.Substring(start, length);
            chunks.Add(chunk.Trim());

            start += ChunkSize - ChunkOverlap;
            if (start < 0) start = 0;
        }

        return chunks;
    }

    /// <summary>
    /// Splits text by sentence boundaries.
    /// </summary>
    /// <param name="text">The text to split.</param>
    /// <returns>A list of text chunks.</returns>
    public List<string> SplitBySentences(string text)
    {
        if (string.IsNullOrEmpty(text)) return new List<string>();

        var sentences = SentenceRegex.Split(text)
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .ToList();

        var chunks = new List<string>();
        var currentChunk = new List<string>();
        var currentLength = 0;

        foreach (var sentence in sentences)
        {
            if (currentLength + sentence.Length > ChunkSize && currentChunk.Count > 0)
            {
                chunks.Add(string.Join(" ", currentChunk).Trim());

                // Keep overlap portion
                var overlapLength = 0;
                var overlapChunk = new List<string>();
                for (var i = currentChunk.Count - 1; i >= 0 && overlapLength < ChunkOverlap; i--)
                {
                    overlapChunk.Insert(0, currentChunk[i]);
                    overlapLength += currentChunk[i].Length;
                }

                currentChunk = overlapChunk;
                currentLength = overlapLength;
            }

            currentChunk.Add(sentence);
            currentLength += sentence.Length;
        }

        if (currentChunk.Count > 0)
        {
            chunks.Add(string.Join(" ", currentChunk).Trim());
        }

        return chunks;
    }

    /// <summary>
    /// Splits text by paragraph boundaries.
    /// </summary>
    /// <param name="text">The text to split.</param>
    /// <returns>A list of text chunks.</returns>
    public List<string> SplitByParagraphs(string text)
    {
        if (string.IsNullOrEmpty(text)) return new List<string>();

        return text.Split(new[] { "\n\n", "\r\n\r\n" }, StringSplitOptions.RemoveEmptyEntries)
            .Select(p => p.Trim())
            .Where(p => !string.IsNullOrWhiteSpace(p))
            .ToList();
    }
    /// <summary>
    /// Splits text by token count approximation (1 token ≈ 4 chars).
    /// </summary>
    /// <param name="text">The text to split.</param>
    /// <returns>A list of text chunks.</returns>
    public List<string> SplitByTokens(string text)
    {
        // Save original chunk size
        var originalSize = ChunkSize;
        var originalOverlap = ChunkOverlap;

        try
        {
            // Adjust chunk size to characters (approx 4 chars per token)
            ChunkSize = originalSize * 4;
            ChunkOverlap = originalOverlap * 4;

            return SplitByCharacters(text);
        }
        finally
        {
            // Restore original settings
            ChunkSize = originalSize;
            ChunkOverlap = originalOverlap;
        }
    }
}

