using SharpAIKit.Common;

namespace SharpAIKit.Callback;

/// <summary>
/// Event data for LLM operations.
/// </summary>
public class LLMEvent
{
    /// <summary>
    /// Gets or sets the event timestamp.
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the run ID.
    /// </summary>
    public string RunId { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the parent run ID.
    /// </summary>
    public string? ParentRunId { get; set; }

    /// <summary>
    /// Gets or sets the model name.
    /// </summary>
    public string? Model { get; set; }

    /// <summary>
    /// Gets or sets the input messages.
    /// </summary>
    public List<ChatMessage>? Messages { get; set; }

    /// <summary>
    /// Gets or sets the output text.
    /// </summary>
    public string? Output { get; set; }

    /// <summary>
    /// Gets or sets token usage information.
    /// </summary>
    public TokenUsage? Usage { get; set; }

    /// <summary>
    /// Gets or sets the latency in milliseconds.
    /// </summary>
    public long? LatencyMs { get; set; }

    /// <summary>
    /// Gets or sets any error that occurred.
    /// </summary>
    public Exception? Error { get; set; }

    /// <summary>
    /// Gets or sets additional metadata.
    /// </summary>
    public Dictionary<string, object?> Metadata { get; set; } = new();
}

/// <summary>
/// Token usage information.
/// </summary>
public class TokenUsage
{
    /// <summary>
    /// Gets or sets the prompt token count.
    /// </summary>
    public int PromptTokens { get; set; }

    /// <summary>
    /// Gets or sets the completion token count.
    /// </summary>
    public int CompletionTokens { get; set; }

    /// <summary>
    /// Gets the total token count.
    /// </summary>
    public int TotalTokens => PromptTokens + CompletionTokens;
}

/// <summary>
/// Event data for chain operations.
/// </summary>
public class ChainEvent
{
    /// <summary>
    /// Gets or sets the event timestamp.
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the run ID.
    /// </summary>
    public string RunId { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the chain name.
    /// </summary>
    public string? ChainName { get; set; }

    /// <summary>
    /// Gets or sets the input.
    /// </summary>
    public object? Input { get; set; }

    /// <summary>
    /// Gets or sets the output.
    /// </summary>
    public object? Output { get; set; }

    /// <summary>
    /// Gets or sets the latency in milliseconds.
    /// </summary>
    public long? LatencyMs { get; set; }

    /// <summary>
    /// Gets or sets any error.
    /// </summary>
    public Exception? Error { get; set; }
}

/// <summary>
/// Event data for agent operations.
/// </summary>
public class AgentEvent
{
    /// <summary>
    /// Gets or sets the event timestamp.
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the run ID.
    /// </summary>
    public string RunId { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the event type.
    /// </summary>
    public AgentEventType Type { get; set; }

    /// <summary>
    /// Gets or sets the thought/reasoning.
    /// </summary>
    public string? Thought { get; set; }

    /// <summary>
    /// Gets or sets the tool name.
    /// </summary>
    public string? ToolName { get; set; }

    /// <summary>
    /// Gets or sets the tool input.
    /// </summary>
    public object? ToolInput { get; set; }

    /// <summary>
    /// Gets or sets the tool output.
    /// </summary>
    public string? ToolOutput { get; set; }

    /// <summary>
    /// Gets or sets the final answer.
    /// </summary>
    public string? FinalAnswer { get; set; }
}

/// <summary>
/// Types of agent events.
/// </summary>
public enum AgentEventType
{
    /// <summary>Agent started thinking.</summary>
    Thinking,
    /// <summary>Agent is calling a tool.</summary>
    ToolStart,
    /// <summary>Tool execution completed.</summary>
    ToolEnd,
    /// <summary>Agent produced final answer.</summary>
    FinalAnswer,
    /// <summary>An error occurred.</summary>
    Error
}

/// <summary>
/// Interface for callback handlers.
/// </summary>
public interface ICallbackHandler
{
    /// <summary>
    /// Called when an LLM call starts.
    /// </summary>
    Task OnLLMStartAsync(LLMEvent evt);

    /// <summary>
    /// Called when an LLM call ends.
    /// </summary>
    Task OnLLMEndAsync(LLMEvent evt);

    /// <summary>
    /// Called when an LLM call errors.
    /// </summary>
    Task OnLLMErrorAsync(LLMEvent evt);

    /// <summary>
    /// Called when a chain starts.
    /// </summary>
    Task OnChainStartAsync(ChainEvent evt);

    /// <summary>
    /// Called when a chain ends.
    /// </summary>
    Task OnChainEndAsync(ChainEvent evt);

    /// <summary>
    /// Called for agent events.
    /// </summary>
    Task OnAgentEventAsync(AgentEvent evt);
}

/// <summary>
/// Base implementation of callback handler with empty methods.
/// </summary>
public abstract class CallbackHandlerBase : ICallbackHandler
{
    /// <inheritdoc/>
    public virtual Task OnLLMStartAsync(LLMEvent evt) => Task.CompletedTask;

    /// <inheritdoc/>
    public virtual Task OnLLMEndAsync(LLMEvent evt) => Task.CompletedTask;

    /// <inheritdoc/>
    public virtual Task OnLLMErrorAsync(LLMEvent evt) => Task.CompletedTask;

    /// <inheritdoc/>
    public virtual Task OnChainStartAsync(ChainEvent evt) => Task.CompletedTask;

    /// <inheritdoc/>
    public virtual Task OnChainEndAsync(ChainEvent evt) => Task.CompletedTask;

    /// <inheritdoc/>
    public virtual Task OnAgentEventAsync(AgentEvent evt) => Task.CompletedTask;
}

/// <summary>
/// Manages multiple callback handlers.
/// </summary>
public class CallbackManager
{
    private readonly List<ICallbackHandler> _handlers = new();

    /// <summary>
    /// Adds a callback handler.
    /// </summary>
    public CallbackManager AddHandler(ICallbackHandler handler)
    {
        _handlers.Add(handler);
        return this;
    }

    /// <summary>
    /// Removes a callback handler.
    /// </summary>
    public CallbackManager RemoveHandler(ICallbackHandler handler)
    {
        _handlers.Remove(handler);
        return this;
    }

    /// <summary>
    /// Notifies all handlers of an LLM start event.
    /// </summary>
    public async Task OnLLMStartAsync(LLMEvent evt)
    {
        foreach (var handler in _handlers)
        {
            await handler.OnLLMStartAsync(evt);
        }
    }

    /// <summary>
    /// Notifies all handlers of an LLM end event.
    /// </summary>
    public async Task OnLLMEndAsync(LLMEvent evt)
    {
        foreach (var handler in _handlers)
        {
            await handler.OnLLMEndAsync(evt);
        }
    }

    /// <summary>
    /// Notifies all handlers of an LLM error event.
    /// </summary>
    public async Task OnLLMErrorAsync(LLMEvent evt)
    {
        foreach (var handler in _handlers)
        {
            await handler.OnLLMErrorAsync(evt);
        }
    }

    /// <summary>
    /// Notifies all handlers of a chain start event.
    /// </summary>
    public async Task OnChainStartAsync(ChainEvent evt)
    {
        foreach (var handler in _handlers)
        {
            await handler.OnChainStartAsync(evt);
        }
    }

    /// <summary>
    /// Notifies all handlers of a chain end event.
    /// </summary>
    public async Task OnChainEndAsync(ChainEvent evt)
    {
        foreach (var handler in _handlers)
        {
            await handler.OnChainEndAsync(evt);
        }
    }

    /// <summary>
    /// Notifies all handlers of an agent event.
    /// </summary>
    public async Task OnAgentEventAsync(AgentEvent evt)
    {
        foreach (var handler in _handlers)
        {
            await handler.OnAgentEventAsync(evt);
        }
    }
}

