using System.Text.Json;
using System.Text.RegularExpressions;
using SharpAIKit.Chain;

namespace SharpAIKit.Output;

/// <summary>
/// Interface for parsing LLM output into structured formats.
/// </summary>
/// <typeparam name="T">The output type.</typeparam>
public interface IOutputParser<T>
{
    /// <summary>
    /// Parses the LLM output string into the target type.
    /// </summary>
    T Parse(string output);

    /// <summary>
    /// Gets format instructions to include in the prompt.
    /// </summary>
    string GetFormatInstructions();
}

/// <summary>
/// Base class for output parsers that can be used as chain components.
/// </summary>
/// <typeparam name="T">The output type.</typeparam>
public abstract class OutputParserBase<T> : ChainBase, IOutputParser<T>
{
    /// <summary>
    /// Gets or sets the input key to read from context.
    /// </summary>
    public string InputKey { get; set; } = "output";

    /// <summary>
    /// Gets or sets the output key to write parsed result.
    /// </summary>
    public string OutputKey { get; set; } = "parsed";

    /// <inheritdoc/>
    public abstract T Parse(string output);

    /// <inheritdoc/>
    public abstract string GetFormatInstructions();

    /// <inheritdoc/>
    protected override Task<ChainContext> ExecuteAsync(ChainContext context, CancellationToken cancellationToken)
    {
        var input = context.Get<string>(InputKey) ?? context.Output;
        var parsed = Parse(input);
        context.Set(OutputKey, parsed);
        return Task.FromResult(context);
    }
}

/// <summary>
/// Parses JSON output into a strongly-typed object.
/// </summary>
/// <typeparam name="T">The target type.</typeparam>
public partial class JsonOutputParser<T> : OutputParserBase<T>
{
    private static readonly Regex JsonBlockPattern = MyRegex();

    /// <inheritdoc/>
    public override T Parse(string output)
    {
        // Try to extract JSON from markdown code blocks
        var match = JsonBlockPattern.Match(output);
        var jsonStr = match.Success ? match.Groups[1].Value : output.Trim();

        // Handle raw JSON
        var jsonStart = jsonStr.IndexOf('{');
        var jsonEnd = jsonStr.LastIndexOf('}');
        
        if (jsonStart >= 0 && jsonEnd > jsonStart)
        {
            jsonStr = jsonStr.Substring(jsonStart, jsonEnd - jsonStart + 1);
        }

        return JsonSerializer.Deserialize<T>(jsonStr, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        }) ?? throw new InvalidOperationException("Failed to parse JSON output");
    }

    /// <inheritdoc/>
    public override string GetFormatInstructions()
    {
        var properties = typeof(T).GetProperties();
        var schema = new Dictionary<string, object>();
        
        foreach (var prop in properties)
        {
            schema[prop.Name] = GetJsonTypeName(prop.PropertyType);
        }

        var schemaJson = JsonSerializer.Serialize(schema, new JsonSerializerOptions { WriteIndented = true });
        
        return $"""
            Please respond with a JSON object in the following format:
            ```json
            {schemaJson}
            ```
            """;
    }

    private static string GetJsonTypeName(Type type)
    {
        if (type == typeof(string)) return "string";
        if (type == typeof(int) || type == typeof(long) || type == typeof(float) || type == typeof(double)) return "number";
        if (type == typeof(bool)) return "boolean";
        if (type.IsArray || (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(List<>))) return "array";
        return "object";
    }

    [GeneratedRegex(@"```(?:json)?\s*([\s\S]*?)```", RegexOptions.IgnoreCase, "en-US")]
    private static partial Regex MyRegex();
}

/// <summary>
/// Parses comma-separated values into a list.
/// </summary>
public class CommaSeparatedListParser : OutputParserBase<List<string>>
{
    /// <inheritdoc/>
    public override List<string> Parse(string output)
    {
        return output
            .Split(',')
            .Select(s => s.Trim())
            .Where(s => !string.IsNullOrEmpty(s))
            .ToList();
    }

    /// <inheritdoc/>
    public override string GetFormatInstructions()
    {
        return "Please respond with a comma-separated list of items.";
    }
}

/// <summary>
/// Parses numbered list output.
/// </summary>
public partial class NumberedListParser : OutputParserBase<List<string>>
{
    private static readonly Regex ListItemPattern = MyRegex1();

    /// <inheritdoc/>
    public override List<string> Parse(string output)
    {
        var matches = ListItemPattern.Matches(output);
        return matches.Select(m => m.Groups[1].Value.Trim()).ToList();
    }

    /// <inheritdoc/>
    public override string GetFormatInstructions()
    {
        return """
            Please respond with a numbered list:
            1. First item
            2. Second item
            3. Third item
            """;
    }

    [GeneratedRegex(@"^\d+[.)] (.+)$", RegexOptions.Multiline)]
    private static partial Regex MyRegex1();
}

/// <summary>
/// Parses boolean yes/no responses.
/// </summary>
public partial class BooleanParser : OutputParserBase<bool>
{
    private static readonly Regex YesPattern = MyRegex2();
    private static readonly Regex NoPattern = MyRegex3();

    /// <inheritdoc/>
    public override bool Parse(string output)
    {
        var trimmed = output.Trim().ToLowerInvariant();
        
        if (YesPattern.IsMatch(trimmed)) return true;
        if (NoPattern.IsMatch(trimmed)) return false;
        
        // Try to find yes/no anywhere in the response
        if (trimmed.Contains("yes") || trimmed.Contains("true") || trimmed.Contains("correct"))
            return true;
        if (trimmed.Contains("no") || trimmed.Contains("false") || trimmed.Contains("incorrect"))
            return false;
            
        throw new InvalidOperationException($"Could not parse boolean from: {output}");
    }

    /// <inheritdoc/>
    public override string GetFormatInstructions()
    {
        return "Please respond with either 'Yes' or 'No'.";
    }

    [GeneratedRegex(@"^(yes|true|correct|affirmative|1)\.?$")]
    private static partial Regex MyRegex2();
    [GeneratedRegex(@"^(no|false|incorrect|negative|0)\.?$")]
    private static partial Regex MyRegex3();
}

/// <summary>
/// Parses structured output with specific fields using regex.
/// </summary>
public class RegexParser : OutputParserBase<Dictionary<string, string>>
{
    private readonly Dictionary<string, Regex> _patterns = new();
    private readonly string _formatInstructions;

    /// <summary>
    /// Creates a new regex parser.
    /// </summary>
    public RegexParser(string formatInstructions = "")
    {
        _formatInstructions = formatInstructions;
    }

    /// <summary>
    /// Adds a field pattern.
    /// </summary>
    public RegexParser AddField(string fieldName, string pattern)
    {
        _patterns[fieldName] = new Regex(pattern, RegexOptions.Multiline);
        return this;
    }

    /// <inheritdoc/>
    public override Dictionary<string, string> Parse(string output)
    {
        var result = new Dictionary<string, string>();
        
        foreach (var (fieldName, pattern) in _patterns)
        {
            var match = pattern.Match(output);
            if (match.Success)
            {
                result[fieldName] = match.Groups.Count > 1 
                    ? match.Groups[1].Value.Trim() 
                    : match.Value.Trim();
            }
        }
        
        return result;
    }

    /// <inheritdoc/>
    public override string GetFormatInstructions()
    {
        return _formatInstructions;
    }
}

/// <summary>
/// Extracts specific XML-like tags from output.
/// </summary>
public partial class XMLTagParser : OutputParserBase<Dictionary<string, string>>
{
    private readonly HashSet<string> _tags;

    /// <summary>
    /// Creates a new XML tag parser.
    /// </summary>
    public XMLTagParser(params string[] tags)
    {
        _tags = new HashSet<string>(tags);
    }

    /// <inheritdoc/>
    public override Dictionary<string, string> Parse(string output)
    {
        var result = new Dictionary<string, string>();
        
        foreach (var tag in _tags)
        {
            var pattern = new Regex($@"<{tag}>([\s\S]*?)</{tag}>", RegexOptions.IgnoreCase);
            var match = pattern.Match(output);
            if (match.Success)
            {
                result[tag] = match.Groups[1].Value.Trim();
            }
        }
        
        return result;
    }

    /// <inheritdoc/>
    public override string GetFormatInstructions()
    {
        var tagList = string.Join(", ", _tags.Select(t => $"<{t}>...</{t}>"));
        return $"Please structure your response using these XML tags: {tagList}";
    }
}

/// <summary>
/// Parses key-value pairs from output.
/// </summary>
public partial class KeyValueParser : OutputParserBase<Dictionary<string, string>>
{
    private static readonly Regex KvPattern = MyRegex4();

    /// <inheritdoc/>
    public override Dictionary<string, string> Parse(string output)
    {
        var result = new Dictionary<string, string>();
        var matches = KvPattern.Matches(output);
        
        foreach (Match match in matches)
        {
            result[match.Groups[1].Value.Trim()] = match.Groups[2].Value.Trim();
        }
        
        return result;
    }

    /// <inheritdoc/>
    public override string GetFormatInstructions()
    {
        return """
            Please respond with key-value pairs, one per line:
            Key1: Value1
            Key2: Value2
            """;
    }

    [GeneratedRegex(@"^(.+?):\s*(.+)$", RegexOptions.Multiline)]
    private static partial Regex MyRegex4();
}

