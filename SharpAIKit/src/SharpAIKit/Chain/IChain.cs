namespace SharpAIKit.Chain;

/// <summary>
/// Represents the execution context passed through a chain.
/// This is the core data structure that flows through chain components.
/// </summary>
public class ChainContext
{
    private readonly Dictionary<string, object?> _data = new();

    /// <summary>
    /// Gets or sets a value by key.
    /// </summary>
    public object? this[string key]
    {
        get => _data.TryGetValue(key, out var value) ? value : null;
        set => _data[key] = value;
    }

    /// <summary>
    /// Gets a typed value from the context.
    /// </summary>
    public T? Get<T>(string key) => _data.TryGetValue(key, out var value) && value is T typed ? typed : default;

    /// <summary>
    /// Sets a value in the context.
    /// </summary>
    public ChainContext Set(string key, object? value)
    {
        _data[key] = value;
        return this;
    }

    /// <summary>
    /// Gets or sets the primary input text.
    /// </summary>
    public string Input
    {
        get => Get<string>("input") ?? string.Empty;
        set => Set("input", value);
    }

    /// <summary>
    /// Gets or sets the primary output text.
    /// </summary>
    public string Output
    {
        get => Get<string>("output") ?? string.Empty;
        set => Set("output", value);
    }

    /// <summary>
    /// Gets all keys in the context.
    /// </summary>
    public IEnumerable<string> Keys => _data.Keys;

    /// <summary>
    /// Creates a copy of this context.
    /// </summary>
    public ChainContext Clone()
    {
        var clone = new ChainContext();
        foreach (var kvp in _data)
        {
            clone._data[kvp.Key] = kvp.Value;
        }
        return clone;
    }

    /// <summary>
    /// Creates a new context from a dictionary.
    /// </summary>
    public static ChainContext FromDictionary(Dictionary<string, object?> data)
    {
        var context = new ChainContext();
        foreach (var kvp in data)
        {
            context._data[kvp.Key] = kvp.Value;
        }
        return context;
    }

    /// <summary>
    /// Creates a new context with just an input string.
    /// </summary>
    public static ChainContext FromInput(string input) => new ChainContext { Input = input };
}

/// <summary>
/// Base interface for all chain components.
/// Chains can be composed using the | operator (like LCEL in LangChain).
/// </summary>
public interface IChain
{
    /// <summary>
    /// Executes the chain with the given context.
    /// </summary>
    Task<ChainContext> InvokeAsync(ChainContext context, CancellationToken cancellationToken = default);

    /// <summary>
    /// Executes the chain with a simple string input.
    /// </summary>
    Task<string> InvokeAsync(string input, CancellationToken cancellationToken = default);

    /// <summary>
    /// Streams the chain execution, yielding partial results.
    /// </summary>
    IAsyncEnumerable<ChainContext> StreamAsync(ChainContext context, CancellationToken cancellationToken = default);
}

/// <summary>
/// Extension methods for chain composition.
/// </summary>
public static class ChainExtensions
{
    /// <summary>
    /// Pipes the output of this chain to another chain.
    /// This enables LCEL-style composition: chain1 | chain2 | chain3
    /// </summary>
    public static IChain Pipe(this IChain first, IChain second) => new PipelineChain(first, second);

    /// <summary>
    /// Maps the output using a transformation function.
    /// </summary>
    public static IChain Map(this IChain chain, Func<ChainContext, ChainContext> transform) 
        => new MapChain(chain, transform);

    /// <summary>
    /// Filters the context based on a predicate.
    /// </summary>
    public static IChain Where(this IChain chain, Func<ChainContext, bool> predicate)
        => new FilterChain(chain, predicate);

    /// <summary>
    /// Runs multiple chains in parallel and aggregates results.
    /// </summary>
    public static IChain Parallel(params IChain[] chains) => new ParallelChain(chains);

    /// <summary>
    /// Creates a branch based on a condition.
    /// </summary>
    public static IChain Branch(this IChain chain, Func<ChainContext, bool> condition, IChain trueBranch, IChain? falseBranch = null)
        => new BranchChain(chain, condition, trueBranch, falseBranch);
}

