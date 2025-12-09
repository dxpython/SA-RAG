using SharpAIKit.RAG;
using Xunit;

namespace SharpAIKit.Tests.RAG;

/// <summary>
/// Unit tests for the Similarity class.
/// </summary>
public class SimilarityTests
{
    [Fact]
    public void Cosine_IdenticalVectors_ReturnsOne()
    {
        // Arrange
        var vec1 = new float[] { 1, 2, 3 };
        var vec2 = new float[] { 1, 2, 3 };

        // Act
        var result = Similarity.Cosine(vec1, vec2);

        // Assert
        Assert.Equal(1.0f, result, 0.0001f);
    }

    [Fact]
    public void Cosine_OrthogonalVectors_ReturnsZero()
    {
        // Arrange
        var vec1 = new float[] { 1, 0, 0 };
        var vec2 = new float[] { 0, 1, 0 };

        // Act
        var result = Similarity.Cosine(vec1, vec2);

        // Assert
        Assert.Equal(0.0f, result, 0.0001f);
    }

    [Fact]
    public void Cosine_OppositeVectors_ReturnsNegativeOne()
    {
        // Arrange
        var vec1 = new float[] { 1, 2, 3 };
        var vec2 = new float[] { -1, -2, -3 };

        // Act
        var result = Similarity.Cosine(vec1, vec2);

        // Assert
        Assert.Equal(-1.0f, result, 0.0001f);
    }

    [Fact]
    public void Cosine_DifferentDimensions_ThrowsException()
    {
        // Arrange
        var vec1 = new float[] { 1, 2, 3 };
        var vec2 = new float[] { 1, 2 };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => Similarity.Cosine(vec1, vec2));
    }

    [Fact]
    public void DotProduct_CalculatesCorrectly()
    {
        // Arrange
        var vec1 = new float[] { 1, 2, 3 };
        var vec2 = new float[] { 4, 5, 6 };

        // Act
        var result = Similarity.DotProduct(vec1, vec2);

        // Assert
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        Assert.Equal(32.0f, result, 0.0001f);
    }

    [Fact]
    public void Euclidean_IdenticalVectors_ReturnsZero()
    {
        // Arrange
        var vec1 = new float[] { 1, 2, 3 };
        var vec2 = new float[] { 1, 2, 3 };

        // Act
        var result = Similarity.Euclidean(vec1, vec2);

        // Assert
        Assert.Equal(0.0f, result, 0.0001f);
    }

    [Fact]
    public void Euclidean_CalculatesCorrectly()
    {
        // Arrange
        var vec1 = new float[] { 0, 0 };
        var vec2 = new float[] { 3, 4 };

        // Act
        var result = Similarity.Euclidean(vec1, vec2);

        // Assert
        // sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
        Assert.Equal(5.0f, result, 0.0001f);
    }

    [Fact]
    public void EuclideanSimilarity_IdenticalVectors_ReturnsOne()
    {
        // Arrange
        var vec1 = new float[] { 1, 2, 3 };
        var vec2 = new float[] { 1, 2, 3 };

        // Act
        var result = Similarity.EuclideanSimilarity(vec1, vec2);

        // Assert
        Assert.Equal(1.0f, result, 0.0001f);
    }

    [Fact]
    public void EuclideanSimilarity_DifferentVectors_ReturnsLessThanOne()
    {
        // Arrange
        var vec1 = new float[] { 0, 0 };
        var vec2 = new float[] { 3, 4 };

        // Act
        var result = Similarity.EuclideanSimilarity(vec1, vec2);

        // Assert
        // 1 / (1 + 5) = 1/6 ≈ 0.1667
        Assert.True(result < 1.0f);
        Assert.True(result > 0.0f);
    }
}

