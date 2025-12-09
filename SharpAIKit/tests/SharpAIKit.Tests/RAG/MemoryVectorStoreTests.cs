using SharpAIKit.RAG;
using Xunit;

namespace SharpAIKit.Tests.RAG;

/// <summary>
/// Unit tests for the MemoryVectorStore class.
/// </summary>
public class MemoryVectorStoreTests
{
    [Fact]
    public async Task AddAsync_AddsDocument()
    {
        // Arrange
        var store = new MemoryVectorStore();
        var doc = new VectorDocument
        {
            Content = "Test content",
            Embedding = new float[] { 1, 2, 3 }
        };

        // Act
        await store.AddAsync(doc);
        var count = await store.CountAsync();

        // Assert
        Assert.Equal(1, count);
    }

    [Fact]
    public async Task AddRangeAsync_AddsMultipleDocuments()
    {
        // Arrange
        var store = new MemoryVectorStore();
        var docs = new[]
        {
            new VectorDocument { Content = "Doc 1", Embedding = new float[] { 1, 0, 0 } },
            new VectorDocument { Content = "Doc 2", Embedding = new float[] { 0, 1, 0 } },
            new VectorDocument { Content = "Doc 3", Embedding = new float[] { 0, 0, 1 } }
        };

        // Act
        await store.AddRangeAsync(docs);
        var count = await store.CountAsync();

        // Assert
        Assert.Equal(3, count);
    }

    [Fact]
    public async Task SearchAsync_ReturnsMostSimilarDocuments()
    {
        // Arrange
        var store = new MemoryVectorStore();
        await store.AddRangeAsync(new[]
        {
            new VectorDocument { Content = "Similar", Embedding = new float[] { 1, 0, 0 } },
            new VectorDocument { Content = "Different", Embedding = new float[] { 0, 1, 0 } }
        });

        // Act
        var results = await store.SearchAsync(new float[] { 1, 0, 0 }, topK: 1);

        // Assert
        Assert.Single(results);
        Assert.Equal("Similar", results[0].Document.Content);
    }

    [Fact]
    public async Task SearchAsync_RespectsTopK()
    {
        // Arrange
        var store = new MemoryVectorStore();
        await store.AddRangeAsync(new[]
        {
            new VectorDocument { Content = "Doc 1", Embedding = new float[] { 1, 0, 0 } },
            new VectorDocument { Content = "Doc 2", Embedding = new float[] { 0.9f, 0.1f, 0 } },
            new VectorDocument { Content = "Doc 3", Embedding = new float[] { 0, 1, 0 } }
        });

        // Act
        var results = await store.SearchAsync(new float[] { 1, 0, 0 }, topK: 2);

        // Assert
        Assert.Equal(2, results.Count);
    }

    [Fact]
    public async Task DeleteAsync_RemovesDocument()
    {
        // Arrange
        var store = new MemoryVectorStore();
        var doc = new VectorDocument
        {
            Id = "test-id",
            Content = "Test content",
            Embedding = new float[] { 1, 2, 3 }
        };
        await store.AddAsync(doc);

        // Act
        await store.DeleteAsync("test-id");
        var count = await store.CountAsync();

        // Assert
        Assert.Equal(0, count);
    }

    [Fact]
    public async Task ClearAsync_RemovesAllDocuments()
    {
        // Arrange
        var store = new MemoryVectorStore();
        await store.AddRangeAsync(new[]
        {
            new VectorDocument { Content = "Doc 1", Embedding = new float[] { 1, 0, 0 } },
            new VectorDocument { Content = "Doc 2", Embedding = new float[] { 0, 1, 0 } }
        });

        // Act
        await store.ClearAsync();
        var count = await store.CountAsync();

        // Assert
        Assert.Equal(0, count);
    }

    [Fact]
    public async Task SearchAsync_OrdersByScoreDescending()
    {
        // Arrange
        var store = new MemoryVectorStore();
        await store.AddRangeAsync(new[]
        {
            new VectorDocument { Content = "Low", Embedding = new float[] { 0, 1, 0 } },
            new VectorDocument { Content = "High", Embedding = new float[] { 1, 0, 0 } },
            new VectorDocument { Content = "Medium", Embedding = new float[] { 0.7f, 0.3f, 0 } }
        });

        // Act
        var results = await store.SearchAsync(new float[] { 1, 0, 0 }, topK: 3);

        // Assert
        Assert.Equal("High", results[0].Document.Content);
        Assert.True(results[0].Score >= results[1].Score);
        Assert.True(results[1].Score >= results[2].Score);
    }
}

