using System.Reflection;
using System.Text.Json;

namespace SharpAIKit.Agent;

/// <summary>
/// Attribute to mark a method as a tool.
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public class ToolAttribute : Attribute
{
    /// <summary>
    /// Gets or sets the tool name.
    /// </summary>
    public string Name { get; set; }

    /// <summary>
    /// Gets or sets the tool description.
    /// </summary>
    public string Description { get; set; }

    /// <summary>
    /// Creates a new tool attribute.
    /// </summary>
    /// <param name="name">The tool name.</param>
    /// <param name="description">The tool description.</param>
    public ToolAttribute(string name, string description)
    {
        Name = name;
        Description = description;
    }
}

/// <summary>
/// Attribute to describe a parameter.
/// </summary>
[AttributeUsage(AttributeTargets.Parameter)]
public class ParameterAttribute : Attribute
{
    /// <summary>
    /// Gets or sets the parameter description.
    /// </summary>
    public string Description { get; set; }

    /// <summary>
    /// Gets or sets whether the parameter is required.
    /// </summary>
    public bool Required { get; set; } = true;

    /// <summary>
    /// Creates a new parameter attribute.
    /// </summary>
    /// <param name="description">The parameter description.</param>
    public ParameterAttribute(string description)
    {
        Description = description;
    }
}

/// <summary>
/// Represents a tool definition.
/// </summary>
public class ToolDefinition
{
    /// <summary>
    /// Gets or sets the tool name.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the tool description.
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the parameter definitions.
    /// </summary>
    public List<ToolParameter> Parameters { get; set; } = new();

    /// <summary>
    /// Gets or sets the execution function.
    /// </summary>
    public Func<Dictionary<string, object?>, Task<string>>? Execute { get; set; }
}

/// <summary>
/// Represents a tool parameter definition.
/// </summary>
public class ToolParameter
{
    /// <summary>
    /// Gets or sets the parameter name.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the parameter type.
    /// </summary>
    public string Type { get; set; } = "string";

    /// <summary>
    /// Gets or sets the parameter description.
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets whether the parameter is required.
    /// </summary>
    public bool Required { get; set; } = true;
}

/// <summary>
/// Base class for tools.
/// All custom tools should inherit from this class.
/// </summary>
public abstract class ToolBase
{
    /// <summary>
    /// Gets the tool definitions by scanning methods with ToolAttribute.
    /// </summary>
    /// <returns>A list of tool definitions.</returns>
    public virtual List<ToolDefinition> GetToolDefinitions()
    {
        var definitions = new List<ToolDefinition>();
        var methods = GetType().GetMethods(BindingFlags.Public | BindingFlags.Instance);

        foreach (var method in methods)
        {
            var toolAttr = method.GetCustomAttribute<ToolAttribute>();
            if (toolAttr == null) continue;

            var parameters = method.GetParameters();
            var toolParams = new List<ToolParameter>();

            foreach (var param in parameters)
            {
                var paramAttr = param.GetCustomAttribute<ParameterAttribute>();
                toolParams.Add(new ToolParameter
                {
                    Name = param.Name ?? string.Empty,
                    Type = GetJsonType(param.ParameterType),
                    Description = paramAttr?.Description ?? string.Empty,
                    Required = paramAttr?.Required ?? true
                });
            }

            var definition = new ToolDefinition
            {
                Name = toolAttr.Name,
                Description = toolAttr.Description,
                Parameters = toolParams,
                Execute = async args => await InvokeMethod(method, args)
            };

            definitions.Add(definition);
        }

        return definitions;
    }

    /// <summary>
    /// Invokes a method with the given arguments.
    /// </summary>
    private async Task<string> InvokeMethod(MethodInfo method, Dictionary<string, object?> args)
    {
        var parameters = method.GetParameters();
        var invokeArgs = new object?[parameters.Length];

        for (var i = 0; i < parameters.Length; i++)
        {
            var param = parameters[i];
            if (args.TryGetValue(param.Name ?? string.Empty, out var value))
            {
                invokeArgs[i] = ConvertParameter(value, param.ParameterType);
            }
            else if (param.HasDefaultValue)
            {
                invokeArgs[i] = param.DefaultValue;
            }
            else
            {
                throw new ArgumentException($"Missing required parameter: {param.Name}");
            }
        }

        var result = method.Invoke(this, invokeArgs);

        if (result is Task<string> taskString)
        {
            return await taskString;
        }
        else if (result is Task task)
        {
            await task;
            return "Execution successful";
        }
        else
        {
            return result?.ToString() ?? string.Empty;
        }
    }

    /// <summary>
    /// Converts a parameter to the target type.
    /// </summary>
    private static object? ConvertParameter(object? value, Type targetType)
    {
        if (value == null) return null;

        if (value is JsonElement jsonElement)
        {
            return jsonElement.ValueKind switch
            {
                JsonValueKind.String => jsonElement.GetString(),
                JsonValueKind.Number when targetType == typeof(int) => jsonElement.GetInt32(),
                JsonValueKind.Number when targetType == typeof(long) => jsonElement.GetInt64(),
                JsonValueKind.Number when targetType == typeof(float) => jsonElement.GetSingle(),
                JsonValueKind.Number when targetType == typeof(double) => jsonElement.GetDouble(),
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                _ => value
            };
        }

        return Convert.ChangeType(value, targetType);
    }

    /// <summary>
    /// Gets the JSON type name for a .NET type.
    /// </summary>
    private static string GetJsonType(Type type)
    {
        if (type == typeof(int) || type == typeof(long) ||
            type == typeof(float) || type == typeof(double))
            return "number";
        if (type == typeof(bool))
            return "boolean";
        if (type == typeof(string))
            return "string";
        if (type.IsArray || typeof(System.Collections.IEnumerable).IsAssignableFrom(type))
            return "array";
        return "object";
    }
}

