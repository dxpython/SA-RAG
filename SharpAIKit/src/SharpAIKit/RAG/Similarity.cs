namespace SharpAIKit.RAG;

/// <summary>
/// Utility class for calculating vector similarity metrics.
/// </summary>
public static class Similarity
{
    /// <summary>
    /// Calculates the cosine similarity between two vectors.
    /// </summary>
    /// <param name="vec1">First vector.</param>
    /// <param name="vec2">Second vector.</param>
    /// <returns>Similarity value between -1 and 1.</returns>
    public static float Cosine(float[] vec1, float[] vec2)
    {
        if (vec1.Length != vec2.Length)
            throw new ArgumentException("Vector dimensions must match");

        float dotProduct = 0;
        float norm1 = 0;
        float norm2 = 0;

        for (var i = 0; i < vec1.Length; i++)
        {
            dotProduct += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }

        var denominator = MathF.Sqrt(norm1) * MathF.Sqrt(norm2);
        return denominator == 0 ? 0 : dotProduct / denominator;
    }

    /// <summary>
    /// Calculates the dot product of two vectors.
    /// </summary>
    /// <param name="vec1">First vector.</param>
    /// <param name="vec2">Second vector.</param>
    /// <returns>The dot product value.</returns>
    public static float DotProduct(float[] vec1, float[] vec2)
    {
        if (vec1.Length != vec2.Length)
            throw new ArgumentException("Vector dimensions must match");

        float result = 0;
        for (var i = 0; i < vec1.Length; i++)
        {
            result += vec1[i] * vec2[i];
        }

        return result;
    }

    /// <summary>
    /// Calculates the Euclidean distance between two vectors.
    /// </summary>
    /// <param name="vec1">First vector.</param>
    /// <param name="vec2">Second vector.</param>
    /// <returns>The Euclidean distance.</returns>
    public static float Euclidean(float[] vec1, float[] vec2)
    {
        if (vec1.Length != vec2.Length)
            throw new ArgumentException("Vector dimensions must match");

        float sum = 0;
        for (var i = 0; i < vec1.Length; i++)
        {
            var diff = vec1[i] - vec2[i];
            sum += diff * diff;
        }

        return MathF.Sqrt(sum);
    }

    /// <summary>
    /// Calculates Euclidean similarity (normalized inverse of distance).
    /// </summary>
    /// <param name="vec1">First vector.</param>
    /// <param name="vec2">Second vector.</param>
    /// <returns>Similarity value between 0 and 1.</returns>
    public static float EuclideanSimilarity(float[] vec1, float[] vec2)
    {
        var distance = Euclidean(vec1, vec2);
        return 1 / (1 + distance);
    }
}

