using SharpAIKit.RAG;
using Xunit;

namespace SharpAIKit.Tests.RAG;

/// <summary>
/// Unit tests for the TextSplitter class.
/// </summary>
public class TextSplitterTests
{
    [Fact]
    public void SplitByCharacters_ShortText_ReturnsSingleChunk()
    {
        // Arrange
        var splitter = new TextSplitter(chunkSize: 100);
        var text = "This is a short text.";

        // Act
        var chunks = splitter.SplitByCharacters(text);

        // Assert
        Assert.Single(chunks);
        Assert.Equal(text, chunks[0]);
    }

    [Fact]
    public void SplitByCharacters_LongText_ReturnsMultipleChunks()
    {
        // Arrange
        var splitter = new TextSplitter(chunkSize: 20, chunkOverlap: 5);
        var text = "This is a longer text that should be split into multiple chunks.";

        // Act
        var chunks = splitter.SplitByCharacters(text);

        // Assert
        Assert.True(chunks.Count > 1);
    }

    [Fact]
    public void SplitByCharacters_EmptyText_ReturnsEmptyList()
    {
        // Arrange
        var splitter = new TextSplitter();
        var text = "";

        // Act
        var chunks = splitter.SplitByCharacters(text);

        // Assert
        Assert.Empty(chunks);
    }

    [Fact]
    public void SplitBySentences_MultipleSentences_SplitsCorrectly()
    {
        // Arrange
        var splitter = new TextSplitter(chunkSize: 50, chunkOverlap: 10);
        var text = "This is sentence one. This is sentence two. This is sentence three.";

        // Act
        var chunks = splitter.SplitBySentences(text);

        // Assert
        Assert.True(chunks.Count >= 1);
    }

    [Fact]
    public void SplitBySentences_EnglishText_SplitsCorrectly_Variant()
    {
        // Arrange
        var splitter = new TextSplitter(chunkSize: 20, chunkOverlap: 5);
        var text = "This is the first sentence. This is the second sentence. This is the third sentence.";

        // Act
        var chunks = splitter.SplitBySentences(text);

        // Assert
        Assert.True(chunks.Count >= 1);
    }

    [Fact]
    public void SplitByParagraphs_MultipleParagraphs_SplitsCorrectly()
    {
        // Arrange
        var splitter = new TextSplitter();
        var text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.";

        // Act
        var chunks = splitter.SplitByParagraphs(text);

        // Assert
        Assert.Equal(3, chunks.Count);
        Assert.Equal("Paragraph one.", chunks[0]);
        Assert.Equal("Paragraph two.", chunks[1]);
        Assert.Equal("Paragraph three.", chunks[2]);
    }

    [Fact]
    public void SplitByParagraphs_EmptyText_ReturnsEmptyList()
    {
        // Arrange
        var splitter = new TextSplitter();
        var text = "";

        // Act
        var chunks = splitter.SplitByParagraphs(text);

        // Assert
        Assert.Empty(chunks);
    }
}

