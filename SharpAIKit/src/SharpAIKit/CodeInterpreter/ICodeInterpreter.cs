namespace SharpAIKit.CodeInterpreter;

/// <summary>
/// Result of code execution.
/// </summary>
public class CodeExecutionResult
{
    /// <summary>
    /// Gets or sets whether execution was successful.
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Gets or sets the output text.
    /// </summary>
    public string Output { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the error message if execution failed.
    /// </summary>
    public string? Error { get; set; }

    /// <summary>
    /// Gets or sets the execution time in milliseconds.
    /// </summary>
    public long ExecutionTimeMs { get; set; }

    /// <summary>
    /// Gets or sets any variables that were set during execution.
    /// </summary>
    public Dictionary<string, object?> Variables { get; set; } = new();
}

/// <summary>
/// Interface for code interpreters that can execute C# code.
/// </summary>
public interface ICodeInterpreter
{
    /// <summary>
    /// Executes C# code and returns the result.
    /// </summary>
    Task<CodeExecutionResult> ExecuteAsync(string code, CancellationToken cancellationToken = default);

    /// <summary>
    /// Executes code with a specific return type.
    /// </summary>
    Task<T?> ExecuteAsync<T>(string code, CancellationToken cancellationToken = default);

    /// <summary>
    /// Resets the execution context (clears variables).
    /// </summary>
    void Reset();

    /// <summary>
    /// Gets a variable from the execution context.
    /// </summary>
    T? GetVariable<T>(string name);

    /// <summary>
    /// Sets a variable in the execution context.
    /// </summary>
    void SetVariable(string name, object? value);
}
