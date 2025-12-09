using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace SharpAIKit.Callback;

/// <summary>
/// Console callback handler that prints events to console.
/// </summary>
public class ConsoleCallbackHandler : CallbackHandlerBase
{
    /// <summary>
    /// Gets or sets whether to use colors.
    /// </summary>
    public bool UseColors { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to show timestamps.
    /// </summary>
    public bool ShowTimestamps { get; set; } = true;

    /// <inheritdoc/>
    public override Task OnLLMStartAsync(LLMEvent evt)
    {
        WriteColored($"[LLM] Starting call to {evt.Model}...", ConsoleColor.Cyan);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override Task OnLLMEndAsync(LLMEvent evt)
    {
        var sb = new StringBuilder();
        sb.Append($"[LLM] Completed in {evt.LatencyMs}ms");
        
        if (evt.Usage != null)
        {
            sb.Append($" | Tokens: {evt.Usage.TotalTokens} (prompt: {evt.Usage.PromptTokens}, completion: {evt.Usage.CompletionTokens})");
        }
        
        WriteColored(sb.ToString(), ConsoleColor.Green);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override Task OnLLMErrorAsync(LLMEvent evt)
    {
        WriteColored($"[LLM] Error: {evt.Error?.Message}", ConsoleColor.Red);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override Task OnChainStartAsync(ChainEvent evt)
    {
        WriteColored($"[Chain] Starting: {evt.ChainName ?? "unnamed"}", ConsoleColor.Blue);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override Task OnChainEndAsync(ChainEvent evt)
    {
        WriteColored($"[Chain] Completed: {evt.ChainName ?? "unnamed"} in {evt.LatencyMs}ms", ConsoleColor.Blue);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override Task OnAgentEventAsync(AgentEvent evt)
    {
        var message = evt.Type switch
        {
            AgentEventType.Thinking => $"[Agent] Thinking: {evt.Thought}",
            AgentEventType.ToolStart => $"[Agent] Calling tool: {evt.ToolName}",
            AgentEventType.ToolEnd => $"[Agent] Tool result: {Truncate(evt.ToolOutput, 200)}",
            AgentEventType.FinalAnswer => $"[Agent] Final answer: {Truncate(evt.FinalAnswer, 200)}",
            AgentEventType.Error => $"[Agent] Error occurred",
            _ => "[Agent] Unknown event"
        };

        var color = evt.Type switch
        {
            AgentEventType.Error => ConsoleColor.Red,
            AgentEventType.FinalAnswer => ConsoleColor.Green,
            _ => ConsoleColor.Yellow
        };

        WriteColored(message, color);
        return Task.CompletedTask;
    }

    private void WriteColored(string message, ConsoleColor color)
    {
        var timestamp = ShowTimestamps ? $"[{DateTime.Now:HH:mm:ss}] " : "";
        
        if (UseColors)
        {
            var originalColor = Console.ForegroundColor;
            Console.ForegroundColor = color;
            Console.WriteLine(timestamp + message);
            Console.ForegroundColor = originalColor;
        }
        else
        {
            Console.WriteLine(timestamp + message);
        }
    }

    private static string? Truncate(string? text, int maxLength)
    {
        if (text == null || text.Length <= maxLength) return text;
        return text.Substring(0, maxLength) + "...";
    }
}

/// <summary>
/// Logging callback handler using Microsoft.Extensions.Logging.
/// </summary>
public class LoggingCallbackHandler : CallbackHandlerBase
{
    private readonly ILogger _logger;

    /// <summary>
    /// Creates a new logging callback handler.
    /// </summary>
    public LoggingCallbackHandler(ILogger logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <inheritdoc/>
    public override Task OnLLMStartAsync(LLMEvent evt)
    {
        _logger.LogInformation("LLM call started. Model: {Model}, RunId: {RunId}", evt.Model, evt.RunId);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override Task OnLLMEndAsync(LLMEvent evt)
    {
        _logger.LogInformation(
            "LLM call completed. RunId: {RunId}, Latency: {LatencyMs}ms, Tokens: {TotalTokens}",
            evt.RunId, evt.LatencyMs, evt.Usage?.TotalTokens ?? 0);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override Task OnLLMErrorAsync(LLMEvent evt)
    {
        _logger.LogError(evt.Error, "LLM call failed. RunId: {RunId}", evt.RunId);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override Task OnChainStartAsync(ChainEvent evt)
    {
        _logger.LogDebug("Chain started. Name: {ChainName}, RunId: {RunId}", evt.ChainName, evt.RunId);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override Task OnChainEndAsync(ChainEvent evt)
    {
        _logger.LogDebug("Chain completed. Name: {ChainName}, RunId: {RunId}, Latency: {LatencyMs}ms",
            evt.ChainName, evt.RunId, evt.LatencyMs);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override Task OnAgentEventAsync(AgentEvent evt)
    {
        _logger.LogInformation("Agent event. Type: {Type}, Tool: {ToolName}, RunId: {RunId}",
            evt.Type, evt.ToolName, evt.RunId);
        return Task.CompletedTask;
    }
}

/// <summary>
/// Metrics callback handler for collecting performance metrics.
/// </summary>
public class MetricsCallbackHandler : CallbackHandlerBase
{
    private readonly object _lock = new();
    private int _llmCalls;
    private int _llmErrors;
    private long _totalLatencyMs;
    private int _totalTokens;
    private readonly Dictionary<string, int> _modelCalls = new();
    private readonly Dictionary<string, int> _toolCalls = new();

    /// <summary>
    /// Gets the total number of LLM calls.
    /// </summary>
    public int LLMCalls
    {
        get { lock (_lock) return _llmCalls; }
    }

    /// <summary>
    /// Gets the total number of LLM errors.
    /// </summary>
    public int LLMErrors
    {
        get { lock (_lock) return _llmErrors; }
    }

    /// <summary>
    /// Gets the average latency in milliseconds.
    /// </summary>
    public double AverageLatencyMs
    {
        get { lock (_lock) return _llmCalls > 0 ? (double)_totalLatencyMs / _llmCalls : 0; }
    }

    /// <summary>
    /// Gets the total tokens used.
    /// </summary>
    public int TotalTokens
    {
        get { lock (_lock) return _totalTokens; }
    }

    /// <summary>
    /// Gets a snapshot of model usage counts.
    /// </summary>
    public Dictionary<string, int> GetModelCounts()
    {
        lock (_lock) return new Dictionary<string, int>(_modelCalls);
    }

    /// <summary>
    /// Gets a snapshot of tool usage counts.
    /// </summary>
    public Dictionary<string, int> GetToolCounts()
    {
        lock (_lock) return new Dictionary<string, int>(_toolCalls);
    }

    /// <inheritdoc/>
    public override Task OnLLMEndAsync(LLMEvent evt)
    {
        lock (_lock)
        {
            _llmCalls++;
            _totalLatencyMs += evt.LatencyMs ?? 0;
            _totalTokens += evt.Usage?.TotalTokens ?? 0;
            
            if (!string.IsNullOrEmpty(evt.Model))
            {
                _modelCalls.TryGetValue(evt.Model, out var count);
                _modelCalls[evt.Model] = count + 1;
            }
        }
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override Task OnLLMErrorAsync(LLMEvent evt)
    {
        lock (_lock)
        {
            _llmErrors++;
        }
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override Task OnAgentEventAsync(AgentEvent evt)
    {
        if (evt.Type == AgentEventType.ToolEnd && !string.IsNullOrEmpty(evt.ToolName))
        {
            lock (_lock)
            {
                _toolCalls.TryGetValue(evt.ToolName, out var count);
                _toolCalls[evt.ToolName] = count + 1;
            }
        }
        return Task.CompletedTask;
    }

    /// <summary>
    /// Resets all metrics.
    /// </summary>
    public void Reset()
    {
        lock (_lock)
        {
            _llmCalls = 0;
            _llmErrors = 0;
            _totalLatencyMs = 0;
            _totalTokens = 0;
            _modelCalls.Clear();
            _toolCalls.Clear();
        }
    }

    /// <summary>
    /// Gets a summary of all metrics.
    /// </summary>
    public MetricsSummary GetSummary()
    {
        lock (_lock)
        {
            return new MetricsSummary
            {
                LLMCalls = _llmCalls,
                LLMErrors = _llmErrors,
                AverageLatencyMs = _llmCalls > 0 ? (double)_totalLatencyMs / _llmCalls : 0,
                TotalTokens = _totalTokens,
                ModelCounts = new Dictionary<string, int>(_modelCalls),
                ToolCounts = new Dictionary<string, int>(_toolCalls)
            };
        }
    }
}

/// <summary>
/// Summary of collected metrics.
/// </summary>
public class MetricsSummary
{
    /// <summary>Total LLM calls.</summary>
    public int LLMCalls { get; set; }
    /// <summary>Total LLM errors.</summary>
    public int LLMErrors { get; set; }
    /// <summary>Average latency in ms.</summary>
    public double AverageLatencyMs { get; set; }
    /// <summary>Total tokens used.</summary>
    public int TotalTokens { get; set; }
    /// <summary>Calls per model.</summary>
    public Dictionary<string, int> ModelCounts { get; set; } = new();
    /// <summary>Calls per tool.</summary>
    public Dictionary<string, int> ToolCounts { get; set; } = new();
}

/// <summary>
/// File-based callback handler for persistent logging.
/// </summary>
public class FileCallbackHandler : CallbackHandlerBase, IDisposable
{
    private readonly StreamWriter _writer;
    private readonly JsonSerializerOptions _jsonOptions = new() { WriteIndented = false };

    /// <summary>
    /// Creates a new file callback handler.
    /// </summary>
    public FileCallbackHandler(string filePath, bool append = true)
    {
        _writer = new StreamWriter(filePath, append) { AutoFlush = true };
    }

    /// <inheritdoc/>
    public override async Task OnLLMStartAsync(LLMEvent evt)
    {
        await WriteEventAsync("llm_start", evt);
    }

    /// <inheritdoc/>
    public override async Task OnLLMEndAsync(LLMEvent evt)
    {
        await WriteEventAsync("llm_end", evt);
    }

    /// <inheritdoc/>
    public override async Task OnLLMErrorAsync(LLMEvent evt)
    {
        await WriteEventAsync("llm_error", evt);
    }

    /// <inheritdoc/>
    public override async Task OnChainStartAsync(ChainEvent evt)
    {
        await WriteEventAsync("chain_start", evt);
    }

    /// <inheritdoc/>
    public override async Task OnChainEndAsync(ChainEvent evt)
    {
        await WriteEventAsync("chain_end", evt);
    }

    /// <inheritdoc/>
    public override async Task OnAgentEventAsync(AgentEvent evt)
    {
        await WriteEventAsync($"agent_{evt.Type.ToString().ToLower()}", evt);
    }

    private async Task WriteEventAsync(string eventType, object data)
    {
        var logEntry = new
        {
            timestamp = DateTime.UtcNow.ToString("O"),
            @event = eventType,
            data
        };
        
        var json = JsonSerializer.Serialize(logEntry, _jsonOptions);
        await _writer.WriteLineAsync(json);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _writer.Dispose();
    }
}

/// <summary>
/// Tracing context for distributed tracing support.
/// </summary>
public class TracingContext
{
    private static readonly AsyncLocal<TracingContext?> _current = new();

    /// <summary>
    /// Gets or sets the current tracing context.
    /// </summary>
    public static TracingContext? Current
    {
        get => _current.Value;
        set => _current.Value = value;
    }

    /// <summary>
    /// Gets the trace ID.
    /// </summary>
    public string TraceId { get; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the parent span ID.
    /// </summary>
    public string? ParentSpanId { get; set; }

    /// <summary>
    /// Gets the current span ID.
    /// </summary>
    public string SpanId { get; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets the tags.
    /// </summary>
    public Dictionary<string, string> Tags { get; } = new();

    /// <summary>
    /// Creates a child context.
    /// </summary>
    public TracingContext CreateChild()
    {
        return new TracingContext { ParentSpanId = SpanId };
    }

    /// <summary>
    /// Starts a new trace.
    /// </summary>
    public static TracingContext StartTrace()
    {
        var context = new TracingContext();
        Current = context;
        return context;
    }
}

