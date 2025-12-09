using Microsoft.CodeAnalysis.CSharp.Scripting;
using Microsoft.CodeAnalysis.Scripting;
using System.Diagnostics;
using System.Text;

namespace SharpAIKit.CodeInterpreter;

/// <summary>
/// Native C# code interpreter using Roslyn scripting engine.
/// This allows agents to write and execute C# code directly, without needing Python or external interpreters.
/// </summary>
public class RoslynCodeInterpreter : ICodeInterpreter
{
    private ScriptState<object>? _scriptState;
    private readonly ScriptOptions _scriptOptions;
    private readonly object _lock = new();

    /// <summary>
    /// Gets or sets the maximum execution time in milliseconds.
    /// </summary>
    public int MaxExecutionTimeMs { get; set; } = 30000; // 30 seconds

    /// <summary>
    /// Gets or sets whether to allow unsafe code.
    /// </summary>
    public bool AllowUnsafe { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to allow file system access.
    /// </summary>
    public bool AllowFileSystemAccess { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to allow network access.
    /// </summary>
    public bool AllowNetworkAccess { get; set; } = false;

    /// <summary>
    /// Creates a new Roslyn code interpreter.
    /// </summary>
    public RoslynCodeInterpreter()
    {
        _scriptOptions = ScriptOptions.Default
            .WithImports(
                "System",
                "System.Collections.Generic",
                "System.Linq",
                "System.Text",
                "System.IO",
                "System.Threading.Tasks",
                "System.Math"
            )
            .WithReferences(
                typeof(object).Assembly,
                typeof(Console).Assembly,
                typeof(System.Linq.Enumerable).Assembly,
                typeof(System.Collections.Generic.List<>).Assembly,
                typeof(System.Text.StringBuilder).Assembly,
                typeof(System.IO.File).Assembly,
                typeof(System.Math).Assembly
            );
    }

    /// <inheritdoc/>
    public async Task<CodeExecutionResult> ExecuteAsync(string code, CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();
        var result = new CodeExecutionResult();

        try
        {
            lock (_lock)
            {
                // Capture current state
                var currentState = _scriptState;
                
                // Execute code
                var task = currentState == null
                    ? CSharpScript.RunAsync(code, _scriptOptions, cancellationToken: cancellationToken)
                    : currentState.ContinueWithAsync(code, cancellationToken: cancellationToken);

                // Wait with timeout
                if (!task.Wait(MaxExecutionTimeMs, cancellationToken))
                {
                    throw new TimeoutException($"Code execution exceeded {MaxExecutionTimeMs}ms");
                }

                _scriptState = task.Result;
                result.Output = _scriptState.ReturnValue?.ToString() ?? string.Empty;
                result.Success = true;

                // Capture variables
                foreach (var variable in _scriptState.Variables)
                {
                    result.Variables[variable.Name] = variable.Value;
                }
            }
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.Error = ex.Message;
            result.Output = string.Empty;
        }
        finally
        {
            stopwatch.Stop();
            result.ExecutionTimeMs = stopwatch.ElapsedMilliseconds;
        }

        return result;
    }

    /// <inheritdoc/>
    public async Task<T?> ExecuteAsync<T>(string code, CancellationToken cancellationToken = default)
    {
        var result = await ExecuteAsync(code, cancellationToken);
        
        if (!result.Success)
        {
            throw new InvalidOperationException($"Code execution failed: {result.Error}");
        }

        // Try to get from variables first (if code sets a variable)
        if (result.Variables.TryGetValue("result", out var varValue) && varValue is T typedVarValue)
        {
            return typedVarValue;
        }

        // Try to get from return value
        lock (_lock)
        {
            if (_scriptState != null && _scriptState.ReturnValue is T typedReturn)
            {
                return typedReturn;
            }
        }

        // Try to parse from output string
        if (!string.IsNullOrEmpty(result.Output))
        {
            try
            {
                return (T)Convert.ChangeType(result.Output, typeof(T));
            }
            catch
            {
                // If conversion fails, try JSON deserialization
                try
                {
                    return System.Text.Json.JsonSerializer.Deserialize<T>(result.Output);
                }
                catch
                {
                    // Last resort: return default
                }
            }
        }
        
        return default(T);
    }

    /// <inheritdoc/>
    public void Reset()
    {
        lock (_lock)
        {
            _scriptState = null;
        }
    }

    /// <inheritdoc/>
    public T? GetVariable<T>(string name)
    {
        lock (_lock)
        {
            if (_scriptState == null) return default(T);
            
            var variable = _scriptState.Variables.FirstOrDefault(v => v.Name == name);
            if (variable == null) return default(T);
            
            if (variable.Value is T typedValue)
            {
                return typedValue;
            }
            
            try
            {
                return (T)Convert.ChangeType(variable.Value, typeof(T));
            }
            catch
            {
                return default(T);
            }
        }
    }

    /// <inheritdoc/>
    public void SetVariable(string name, object? value)
    {
        lock (_lock)
        {
            if (_scriptState == null)
            {
                // Create initial state with this variable
                var code = $"{GetTypeName(value)} {name} = {GetValueCode(value)};";
                var task = CSharpScript.RunAsync(code, _scriptOptions);
                task.Wait();
                _scriptState = task.Result;
            }
            else
            {
                var code = $"{name} = {GetValueCode(value)};";
                var task = _scriptState.ContinueWithAsync(code);
                task.Wait();
                _scriptState = task.Result;
            }
        }
    }

    private static string GetTypeName(object? value)
    {
        if (value == null) return "object";
        return value.GetType().FullName ?? value.GetType().Name;
    }

    private static string GetValueCode(object? value)
    {
        if (value == null) return "null";
        if (value is string str) return $"\"{str.Replace("\"", "\\\"")}\"";
        if (value is bool b) return b ? "true" : "false";
        if (value is int || value is long || value is float || value is double || value is decimal)
            return value.ToString() ?? "0";
        if (value is char c) return $"'{c}'";
        
        // For complex types, use JSON serialization
        return System.Text.Json.JsonSerializer.Serialize(value);
    }
}

/// <summary>
/// Helper class for common code interpreter operations.
/// </summary>
public static class CodeInterpreterHelpers
{
    /// <summary>
    /// Creates a code snippet to calculate a mathematical expression.
    /// </summary>
    public static string CreateMathExpression(string expression)
    {
        return $"var result = {expression}; result";
    }

    /// <summary>
    /// Creates a code snippet to process a list.
    /// </summary>
    public static string CreateListProcessing(string listVar, string operation)
    {
        return $"{listVar}.{operation}";
    }

    /// <summary>
    /// Creates a code snippet to read and process a CSV file.
    /// </summary>
    public static string CreateCsvProcessing(string filePath, string processingCode)
    {
        return $"""
            var lines = System.IO.File.ReadAllLines("{filePath}");
            var data = lines.Skip(1).Select(l => l.Split(',')).ToList();
            {processingCode}
            """;
    }

    /// <summary>
    /// Creates a code snippet for data analysis.
    /// </summary>
    public static string CreateDataAnalysis(string dataVar, string analysis)
    {
        return $"""
            var data = {dataVar};
            {analysis}
            """;
    }
}
