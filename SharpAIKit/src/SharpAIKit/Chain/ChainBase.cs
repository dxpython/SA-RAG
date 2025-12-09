using System.Runtime.CompilerServices;

namespace SharpAIKit.Chain;

/// <summary>
/// Base class for all chain implementations.
/// Provides common functionality and operator overloading for composition.
/// </summary>
public abstract class ChainBase : IChain
{
    /// <summary>
    /// Override this to implement the chain logic.
    /// </summary>
    protected abstract Task<ChainContext> ExecuteAsync(ChainContext context, CancellationToken cancellationToken);

    /// <inheritdoc/>
    public async Task<ChainContext> InvokeAsync(ChainContext context, CancellationToken cancellationToken = default)
    {
        return await ExecuteAsync(context, cancellationToken);
    }

    /// <inheritdoc/>
    public async Task<string> InvokeAsync(string input, CancellationToken cancellationToken = default)
    {
        var context = ChainContext.FromInput(input);
        var result = await InvokeAsync(context, cancellationToken);
        return result.Output;
    }

    /// <inheritdoc/>
    public virtual async IAsyncEnumerable<ChainContext> StreamAsync(
        ChainContext context, 
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        // Default implementation: single yield of final result
        yield return await InvokeAsync(context, cancellationToken);
    }

    /// <summary>
    /// Enables LCEL-style composition: chain1 | chain2
    /// </summary>
    public static IChain operator |(ChainBase first, IChain second) => first.Pipe(second);
}

/// <summary>
/// A chain that pipes output from one chain to another.
/// </summary>
internal class PipelineChain : ChainBase
{
    private readonly IChain _first;
    private readonly IChain _second;

    public PipelineChain(IChain first, IChain second)
    {
        _first = first;
        _second = second;
    }

    protected override async Task<ChainContext> ExecuteAsync(ChainContext context, CancellationToken cancellationToken)
    {
        var intermediate = await _first.InvokeAsync(context, cancellationToken);
        // Pass the output of first chain as input to second
        intermediate.Input = intermediate.Output;
        return await _second.InvokeAsync(intermediate, cancellationToken);
    }

    public override async IAsyncEnumerable<ChainContext> StreamAsync(
        ChainContext context, 
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        // Stream through first chain
        await foreach (var intermediate in _first.StreamAsync(context, cancellationToken))
        {
            // Stream through second chain for each intermediate result
            intermediate.Input = intermediate.Output;
            await foreach (var final in _second.StreamAsync(intermediate, cancellationToken))
            {
                yield return final;
            }
        }
    }
}

/// <summary>
/// A chain that transforms the context.
/// </summary>
internal class MapChain : ChainBase
{
    private readonly IChain _inner;
    private readonly Func<ChainContext, ChainContext> _transform;

    public MapChain(IChain inner, Func<ChainContext, ChainContext> transform)
    {
        _inner = inner;
        _transform = transform;
    }

    protected override async Task<ChainContext> ExecuteAsync(ChainContext context, CancellationToken cancellationToken)
    {
        var result = await _inner.InvokeAsync(context, cancellationToken);
        return _transform(result);
    }
}

/// <summary>
/// A chain that filters based on a condition.
/// </summary>
internal class FilterChain : ChainBase
{
    private readonly IChain _inner;
    private readonly Func<ChainContext, bool> _predicate;

    public FilterChain(IChain inner, Func<ChainContext, bool> predicate)
    {
        _inner = inner;
        _predicate = predicate;
    }

    protected override async Task<ChainContext> ExecuteAsync(ChainContext context, CancellationToken cancellationToken)
    {
        var result = await _inner.InvokeAsync(context, cancellationToken);
        return _predicate(result) ? result : context;
    }
}

/// <summary>
/// A chain that runs multiple chains in parallel.
/// </summary>
internal class ParallelChain : ChainBase
{
    private readonly IChain[] _chains;

    public ParallelChain(IChain[] chains)
    {
        _chains = chains;
    }

    protected override async Task<ChainContext> ExecuteAsync(ChainContext context, CancellationToken cancellationToken)
    {
        var tasks = _chains.Select(c => c.InvokeAsync(context.Clone(), cancellationToken));
        var results = await Task.WhenAll(tasks);
        
        // Merge results
        var merged = context.Clone();
        for (var i = 0; i < results.Length; i++)
        {
            merged.Set($"parallel_{i}", results[i].Output);
        }
        merged.Output = string.Join("\n\n", results.Select(r => r.Output));
        return merged;
    }
}

/// <summary>
/// A chain that branches based on a condition.
/// </summary>
internal class BranchChain : ChainBase
{
    private readonly IChain _source;
    private readonly Func<ChainContext, bool> _condition;
    private readonly IChain _trueBranch;
    private readonly IChain? _falseBranch;

    public BranchChain(IChain source, Func<ChainContext, bool> condition, IChain trueBranch, IChain? falseBranch)
    {
        _source = source;
        _condition = condition;
        _trueBranch = trueBranch;
        _falseBranch = falseBranch;
    }

    protected override async Task<ChainContext> ExecuteAsync(ChainContext context, CancellationToken cancellationToken)
    {
        var intermediate = await _source.InvokeAsync(context, cancellationToken);
        intermediate.Input = intermediate.Output;
        
        if (_condition(intermediate))
        {
            return await _trueBranch.InvokeAsync(intermediate, cancellationToken);
        }
        else if (_falseBranch != null)
        {
            return await _falseBranch.InvokeAsync(intermediate, cancellationToken);
        }
        return intermediate;
    }
}

/// <summary>
/// A simple pass-through chain that applies a function.
/// </summary>
public class LambdaChain : ChainBase
{
    private readonly Func<ChainContext, Task<ChainContext>> _func;

    public LambdaChain(Func<ChainContext, ChainContext> func)
    {
        _func = ctx => Task.FromResult(func(ctx));
    }

    public LambdaChain(Func<ChainContext, Task<ChainContext>> func)
    {
        _func = func;
    }

    public LambdaChain(Func<string, string> func)
    {
        _func = ctx =>
        {
            ctx.Output = func(ctx.Input);
            return Task.FromResult(ctx);
        };
    }

    public LambdaChain(Func<string, Task<string>> func)
    {
        _func = async ctx =>
        {
            ctx.Output = await func(ctx.Input);
            return ctx;
        };
    }

    protected override Task<ChainContext> ExecuteAsync(ChainContext context, CancellationToken cancellationToken)
    {
        return _func(context);
    }

    /// <summary>
    /// Creates a lambda chain from a simple string transformation.
    /// </summary>
    public static LambdaChain From(Func<string, string> func) => new(func);

    /// <summary>
    /// Creates a lambda chain from an async string transformation.
    /// </summary>
    public static LambdaChain From(Func<string, Task<string>> func) => new(func);
}

