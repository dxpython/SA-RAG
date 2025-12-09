using SharpAIKit.LLM;
using SharpAIKit.Common;
using System.Text;

namespace SharpAIKit.Optimizer;

/// <summary>
/// Represents a metric for evaluating prompt performance.
/// </summary>
public delegate Task<double> MetricFunction(string input, string output, string expectedOutput);

/// <summary>
/// Represents a training example.
/// </summary>
public class TrainingExample
{
    /// <summary>
    /// Gets or sets the input.
    /// </summary>
    public string Input { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the expected output.
    /// </summary>
    public string ExpectedOutput { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets optional metadata.
    /// </summary>
    public Dictionary<string, object?> Metadata { get; set; } = new();
}

/// <summary>
/// Result of prompt optimization.
/// </summary>
public class OptimizationResult
{
    /// <summary>
    /// Gets or sets the optimized prompt.
    /// </summary>
    public string OptimizedPrompt { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the best score achieved.
    /// </summary>
    public double BestScore { get; set; }

    /// <summary>
    /// Gets or sets the number of iterations.
    /// </summary>
    public int Iterations { get; set; }

    /// <summary>
    /// Gets or sets the optimization history.
    /// </summary>
    public List<OptimizationStep> History { get; set; } = new();
}

/// <summary>
/// Represents a single optimization step.
/// </summary>
public class OptimizationStep
{
    /// <summary>
    /// Gets or sets the iteration number.
    /// </summary>
    public int Iteration { get; set; }

    /// <summary>
    /// Gets or sets the prompt used.
    /// </summary>
    public string Prompt { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the average score.
    /// </summary>
    public double Score { get; set; }

    /// <summary>
    /// Gets or sets the individual scores for each example.
    /// </summary>
    public List<double> ExampleScores { get; set; } = new();
}

/// <summary>
/// DSPy-style automatic prompt optimizer.
/// Automatically improves prompts through iterative optimization based on metrics.
/// </summary>
public class DSPyOptimizer
{
    private readonly ILLMClient _llmClient;
    private readonly List<TrainingExample> _examples = new();
    private MetricFunction? _metric;

    /// <summary>
    /// Gets or sets the maximum number of optimization iterations.
    /// </summary>
    public int MaxIterations { get; set; } = 10;

    /// <summary>
    /// Gets or sets the target score threshold (stops when reached).
    /// </summary>
    public double TargetScore { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the number of few-shot examples to include.
    /// </summary>
    public int FewShotExamples { get; set; } = 3;

    /// <summary>
    /// Gets or sets whether to use few-shot learning.
    /// </summary>
    public bool UseFewShot { get; set; } = true;

    /// <summary>
    /// Creates a new DSPy optimizer.
    /// </summary>
    public DSPyOptimizer(ILLMClient llmClient)
    {
        _llmClient = llmClient ?? throw new ArgumentNullException(nameof(llmClient));
    }

    /// <summary>
    /// Adds a training example.
    /// </summary>
    public DSPyOptimizer AddExample(string input, string expectedOutput, Dictionary<string, object?>? metadata = null)
    {
        _examples.Add(new TrainingExample
        {
            Input = input,
            ExpectedOutput = expectedOutput,
            Metadata = metadata ?? new Dictionary<string, object?>()
        });
        return this;
    }

    /// <summary>
    /// Sets the metric function for evaluation.
    /// </summary>
    public DSPyOptimizer SetMetric(MetricFunction metric)
    {
        _metric = metric;
        return this;
    }

    /// <summary>
    /// Optimizes a prompt template.
    /// </summary>
    public async Task<OptimizationResult> OptimizeAsync(string initialPrompt, CancellationToken cancellationToken = default)
    {
        if (_examples.Count == 0)
        {
            throw new InvalidOperationException("No training examples provided. Use AddExample() to add examples.");
        }

        if (_metric == null)
        {
            throw new InvalidOperationException("No metric function provided. Use SetMetric() to set a metric.");
        }

        var result = new OptimizationResult
        {
            OptimizedPrompt = initialPrompt,
            BestScore = 0.0
        };

        var currentPrompt = initialPrompt;
        var bestPrompt = initialPrompt;
        var bestScore = 0.0;

        for (var iteration = 0; iteration < MaxIterations; iteration++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Evaluate current prompt
            var scores = new List<double>();
            foreach (var example in _examples)
            {
                var output = await ExecutePromptAsync(currentPrompt, example.Input, cancellationToken);
                var score = await _metric(example.Input, output, example.ExpectedOutput);
                scores.Add(score);
            }

            var avgScore = scores.Average();
            var step = new OptimizationStep
            {
                Iteration = iteration + 1,
                Prompt = currentPrompt,
                Score = avgScore,
                ExampleScores = scores
            };
            result.History.Add(step);

            // Update best if improved
            if (avgScore > bestScore)
            {
                bestScore = avgScore;
                bestPrompt = currentPrompt;
                result.BestScore = bestScore;
                result.OptimizedPrompt = bestPrompt;
            }

            // Check if target reached
            if (avgScore >= TargetScore)
            {
                result.Iterations = iteration + 1;
                return result;
            }

            // Generate improved prompt
            currentPrompt = await GenerateImprovedPromptAsync(
                currentPrompt, 
                _examples, 
                scores, 
                cancellationToken);
        }

        result.Iterations = MaxIterations;
        return result;
    }

    private async Task<string> ExecutePromptAsync(string prompt, string input, CancellationToken cancellationToken)
    {
        var fullPrompt = prompt.Replace("{input}", input);
        var messages = new List<ChatMessage>
        {
            ChatMessage.User(fullPrompt)
        };
        return await _llmClient.ChatAsync(messages, cancellationToken);
    }

    private async Task<string> GenerateImprovedPromptAsync(
        string currentPrompt,
        List<TrainingExample> examples,
        List<double> scores,
        CancellationToken cancellationToken)
    {
        // Find best and worst examples
        var indexedScores = scores.Select((s, i) => new { Score = s, Index = i }).ToList();
        var bestExamples = indexedScores.OrderByDescending(x => x.Score).Take(FewShotExamples).ToList();
        var worstExamples = indexedScores.OrderBy(x => x.Score).Take(FewShotExamples).ToList();

        var sb = new StringBuilder();
        sb.AppendLine("You are optimizing a prompt template for an LLM. The current prompt is:");
        sb.AppendLine($"```");
        sb.AppendLine(currentPrompt);
        sb.AppendLine("```");
        sb.AppendLine();
        sb.AppendLine("Here are some examples with their scores:");
        sb.AppendLine();

        // Add best examples
        sb.AppendLine("Best performing examples:");
        foreach (var best in bestExamples)
        {
            var example = examples[best.Index];
            sb.AppendLine($"- Input: {example.Input}");
            sb.AppendLine($"  Expected: {example.ExpectedOutput}");
            sb.AppendLine($"  Score: {best.Score:F2}");
            sb.AppendLine();
        }

        // Add worst examples
        sb.AppendLine("Worst performing examples:");
        foreach (var worst in worstExamples)
        {
            var example = examples[worst.Index];
            sb.AppendLine($"- Input: {example.Input}");
            sb.AppendLine($"  Expected: {example.ExpectedOutput}");
            sb.AppendLine($"  Score: {worst.Score:F2}");
            sb.AppendLine();
        }

        sb.AppendLine("Generate an improved prompt template that:");
        sb.AppendLine("1. Maintains the same structure (use {input} as placeholder)");
        sb.AppendLine("2. Incorporates insights from the best examples");
        sb.AppendLine("3. Addresses issues in the worst examples");
        sb.AppendLine("4. Includes few-shot examples if helpful");
        sb.AppendLine();
        sb.AppendLine("Return ONLY the improved prompt template, nothing else:");

        var optimizationPrompt = sb.ToString();
        var messages = new List<ChatMessage>
        {
            ChatMessage.System("You are an expert at optimizing prompts for LLMs."),
            ChatMessage.User(optimizationPrompt)
        };

        var improved = await _llmClient.ChatAsync(messages, cancellationToken);
        
        // Clean up the response (remove markdown code blocks if present)
        improved = improved.Trim();
        if (improved.StartsWith("```"))
        {
            var lines = improved.Split('\n');
            improved = string.Join("\n", lines.Skip(1).TakeWhile(l => !l.Trim().StartsWith("```")));
        }

        return improved.Trim();
    }
}

/// <summary>
/// Predefined metric functions.
/// </summary>
public static class Metrics
{
    /// <summary>
    /// Exact match metric (1.0 if exact match, 0.0 otherwise).
    /// </summary>
    public static MetricFunction ExactMatch => async (input, output, expected) =>
    {
        return output.Trim().Equals(expected.Trim(), StringComparison.OrdinalIgnoreCase) ? 1.0 : 0.0;
    };

    /// <summary>
    /// Contains metric (1.0 if output contains expected, 0.0 otherwise).
    /// </summary>
    public static MetricFunction Contains => async (input, output, expected) =>
    {
        return output.Contains(expected, StringComparison.OrdinalIgnoreCase) ? 1.0 : 0.0;
    };

    /// <summary>
    /// Semantic similarity metric using embedding cosine similarity.
    /// </summary>
    public static MetricFunction SemanticSimilarity(ILLMClient llmClient)
    {
        return async (input, output, expected) =>
        {
            try
            {
                var outputEmbedding = await llmClient.EmbeddingAsync(output);
                var expectedEmbedding = await llmClient.EmbeddingAsync(expected);

                // Cosine similarity
                var dotProduct = outputEmbedding.Zip(expectedEmbedding, (a, b) => a * b).Sum();
                var magnitudeA = Math.Sqrt(outputEmbedding.Sum(x => x * x));
                var magnitudeB = Math.Sqrt(expectedEmbedding.Sum(x => x * x));

                if (magnitudeA == 0 || magnitudeB == 0) return 0.0;

                return dotProduct / (magnitudeA * magnitudeB);
            }
            catch
            {
                return 0.0;
            }
        };
    }

    /// <summary>
    /// Custom metric builder.
    /// </summary>
    public static MetricFunction Custom(Func<string, string, string, Task<double>> func)
    {
        return async (input, output, expected) => await func(input, output, expected);
    }
}
