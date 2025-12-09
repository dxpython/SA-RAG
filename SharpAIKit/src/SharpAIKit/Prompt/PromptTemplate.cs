using System.Text;
using System.Text.RegularExpressions;
using SharpAIKit.Chain;
using SharpAIKit.Memory;

namespace SharpAIKit.Prompt;

/// <summary>
/// A powerful prompt template system with variable substitution and composition.
/// Supports {variable} syntax and advanced features like conditional sections.
/// </summary>
public partial class PromptTemplate : ChainBase
{
    private readonly string _template;
    private readonly Dictionary<string, object?> _partialVariables = new();
    private static readonly Regex VariablePattern = MyRegex();

    /// <summary>
    /// Gets the input variables expected by this template.
    /// </summary>
    public HashSet<string> InputVariables { get; }

    /// <summary>
    /// Gets or sets the output key for the formatted prompt.
    /// </summary>
    public string OutputKey { get; set; } = "prompt";

    /// <summary>
    /// Creates a new prompt template.
    /// </summary>
    /// <param name="template">The template string with {variable} placeholders.</param>
    public PromptTemplate(string template)
    {
        _template = template ?? throw new ArgumentNullException(nameof(template));
        InputVariables = ExtractVariables(template);
    }

    /// <summary>
    /// Creates a prompt template from a template string.
    /// </summary>
    public static PromptTemplate FromTemplate(string template) => new(template);

    /// <summary>
    /// Sets a partial variable that will always be substituted.
    /// </summary>
    public PromptTemplate WithPartial(string name, object? value)
    {
        _partialVariables[name] = value;
        InputVariables.Remove(name);
        return this;
    }

    /// <summary>
    /// Sets a partial variable from a function.
    /// </summary>
    public PromptTemplate WithPartial(string name, Func<object?> valueFactory)
    {
        _partialVariables[name] = valueFactory;
        InputVariables.Remove(name);
        return this;
    }

    /// <summary>
    /// Formats the template with the given variables.
    /// </summary>
    public string Format(Dictionary<string, object?>? variables = null)
    {
        var allVars = new Dictionary<string, object?>(_partialVariables);
        
        if (variables != null)
        {
            foreach (var kvp in variables)
            {
                allVars[kvp.Key] = kvp.Value;
            }
        }

        return VariablePattern.Replace(_template, match =>
        {
            var varName = match.Groups[1].Value;
            if (allVars.TryGetValue(varName, out var value))
            {
                // Handle function partial variables
                if (value is Func<object?> func)
                {
                    value = func();
                }
                return value?.ToString() ?? string.Empty;
            }
            return match.Value; // Keep unresolved variables as-is
        });
    }

    /// <summary>
    /// Formats the template with named arguments.
    /// </summary>
    public string Format(params (string Name, object? Value)[] variables)
    {
        var dict = new Dictionary<string, object?>();
        foreach (var (name, value) in variables)
        {
            dict[name] = value;
        }
        return Format(dict);
    }

    /// <inheritdoc/>
    protected override Task<ChainContext> ExecuteAsync(ChainContext context, CancellationToken cancellationToken)
    {
        var variables = new Dictionary<string, object?>();
        foreach (var key in context.Keys)
        {
            variables[key] = context[key];
        }
        
        // Also include input as a variable
        if (!variables.ContainsKey("input"))
        {
            variables["input"] = context.Input;
        }

        var formatted = Format(variables);
        context.Set(OutputKey, formatted);
        context.Output = formatted;
        return Task.FromResult(context);
    }

    private static HashSet<string> ExtractVariables(string template)
    {
        var matches = VariablePattern.Matches(template);
        return new HashSet<string>(matches.Select(m => m.Groups[1].Value));
    }

    [GeneratedRegex(@"\{(\w+)\}")]
    private static partial Regex MyRegex();
}

/// <summary>
/// A chat prompt template that generates structured messages.
/// </summary>
public class ChatPromptTemplate : ChainBase
{
    private readonly List<MessageTemplate> _messageTemplates = new();

    /// <summary>
    /// Adds a system message template.
    /// </summary>
    public ChatPromptTemplate AddSystemMessage(string template)
    {
        _messageTemplates.Add(new MessageTemplate("system", template));
        return this;
    }

    /// <summary>
    /// Adds a user message template.
    /// </summary>
    public ChatPromptTemplate AddUserMessage(string template)
    {
        _messageTemplates.Add(new MessageTemplate("user", template));
        return this;
    }

    /// <summary>
    /// Adds an assistant message template.
    /// </summary>
    public ChatPromptTemplate AddAssistantMessage(string template)
    {
        _messageTemplates.Add(new MessageTemplate("assistant", template));
        return this;
    }

    /// <summary>
    /// Adds a placeholder for conversation history.
    /// </summary>
    public ChatPromptTemplate AddHistoryPlaceholder(string variableName = "history")
    {
        _messageTemplates.Add(new MessageTemplate("placeholder", variableName));
        return this;
    }

    /// <summary>
    /// Formats the template to produce chat messages.
    /// </summary>
    public List<Common.ChatMessage> FormatMessages(Dictionary<string, object?>? variables = null)
    {
        var result = new List<Common.ChatMessage>();

        foreach (var msgTemplate in _messageTemplates)
        {
            if (msgTemplate.Role == "placeholder")
            {
                // Handle history placeholder
                if (variables != null && 
                    variables.TryGetValue(msgTemplate.Template, out var historyObj) &&
                    historyObj is IEnumerable<Common.ChatMessage> history)
                {
                    result.AddRange(history);
                }
            }
            else
            {
                var content = new PromptTemplate(msgTemplate.Template).Format(variables);
                result.Add(new Common.ChatMessage { Role = msgTemplate.Role, Content = content });
            }
        }

        return result;
    }

    /// <inheritdoc/>
    protected override Task<ChainContext> ExecuteAsync(ChainContext context, CancellationToken cancellationToken)
    {
        var variables = new Dictionary<string, object?>();
        foreach (var key in context.Keys)
        {
            variables[key] = context[key];
        }
        variables["input"] = context.Input;

        var messages = FormatMessages(variables);
        context.Set("messages", messages);
        context.Output = string.Join("\n", messages.Select(m => $"{m.Role}: {m.Content}"));
        return Task.FromResult(context);
    }

    /// <summary>
    /// Creates a simple chat prompt template.
    /// </summary>
    public static ChatPromptTemplate FromMessages(params (string Role, string Content)[] messages)
    {
        var template = new ChatPromptTemplate();
        foreach (var (role, content) in messages)
        {
            template._messageTemplates.Add(new MessageTemplate(role, content));
        }
        return template;
    }

    private class MessageTemplate
    {
        public string Role { get; }
        public string Template { get; }

        public MessageTemplate(string role, string template)
        {
            Role = role;
            Template = template;
        }
    }
}

/// <summary>
/// Few-shot prompt template that includes examples.
/// </summary>
public class FewShotPromptTemplate : ChainBase
{
    private readonly List<Example> _examples = new();
    private readonly string _prefix;
    private readonly string _suffix;
    private readonly string _exampleTemplate;
    private readonly string _exampleSeparator;

    /// <summary>
    /// Creates a new few-shot prompt template.
    /// </summary>
    /// <param name="prefix">Text before the examples.</param>
    /// <param name="suffix">Text after the examples (usually contains the input placeholder).</param>
    /// <param name="exampleTemplate">Template for each example.</param>
    /// <param name="exampleSeparator">Separator between examples.</param>
    public FewShotPromptTemplate(
        string prefix,
        string suffix,
        string exampleTemplate,
        string exampleSeparator = "\n\n")
    {
        _prefix = prefix;
        _suffix = suffix;
        _exampleTemplate = exampleTemplate;
        _exampleSeparator = exampleSeparator;
    }

    /// <summary>
    /// Adds an example.
    /// </summary>
    public FewShotPromptTemplate AddExample(Dictionary<string, object?> example)
    {
        _examples.Add(new Example(example));
        return this;
    }

    /// <summary>
    /// Adds an example with input/output.
    /// </summary>
    public FewShotPromptTemplate AddExample(string input, string output)
    {
        return AddExample(new Dictionary<string, object?>
        {
            ["input"] = input,
            ["output"] = output
        });
    }

    /// <summary>
    /// Formats the few-shot prompt.
    /// </summary>
    public string Format(Dictionary<string, object?>? variables = null)
    {
        var sb = new StringBuilder();
        
        // Prefix
        sb.Append(new PromptTemplate(_prefix).Format(variables));
        sb.AppendLine();

        // Examples
        var formattedExamples = new List<string>();
        foreach (var example in _examples)
        {
            var examplePrompt = new PromptTemplate(_exampleTemplate);
            formattedExamples.Add(examplePrompt.Format(example.Variables));
        }
        sb.Append(string.Join(_exampleSeparator, formattedExamples));
        sb.AppendLine();

        // Suffix
        sb.Append(new PromptTemplate(_suffix).Format(variables));

        return sb.ToString();
    }

    /// <inheritdoc/>
    protected override Task<ChainContext> ExecuteAsync(ChainContext context, CancellationToken cancellationToken)
    {
        var variables = new Dictionary<string, object?>();
        foreach (var key in context.Keys)
        {
            variables[key] = context[key];
        }
        variables["input"] = context.Input;

        var formatted = Format(variables);
        context.Output = formatted;
        return Task.FromResult(context);
    }

    private class Example
    {
        public Dictionary<string, object?> Variables { get; }

        public Example(Dictionary<string, object?> variables)
        {
            Variables = variables;
        }
    }
}

