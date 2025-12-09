using System.Text.Json;
using SaAgentOS.Models;

namespace SaAgentOS.Execution;

/// <summary>
/// Exports execution graphs for visualization and analysis
/// </summary>
public class ExecutionGraphExporter
{
    /// <summary>
    /// Export reasoning trace to JSON
    /// </summary>
    public static string ExportToJson(ReasoningTrace trace)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        return JsonSerializer.Serialize(trace, options);
    }

    /// <summary>
    /// Export execution graph to Graphviz DOT format
    /// </summary>
    public static string ExportToDot(ExecutionGraph? graph)
    {
        if (graph == null)
        {
            return "digraph EmptyGraph { }";
        }

        var dot = new System.Text.StringBuilder();
        dot.AppendLine("digraph ExecutionGraph {");
        dot.AppendLine("  rankdir=LR;");
        dot.AppendLine("  node [shape=box];");
        dot.AppendLine();

        // Add nodes
        foreach (var node in graph.Nodes)
        {
            var label = $"{node.NodeType}\\n{node.Description}";
            dot.AppendLine($"  \"{node.NodeId}\" [label=\"{label}\"];");
        }

        dot.AppendLine();

        // Add edges
        foreach (var edge in graph.Edges)
        {
            dot.AppendLine($"  \"{edge.FromNode}\" -> \"{edge.ToNode}\" [label=\"{edge.EdgeType}\"];");
        }

        dot.AppendLine("}");
        return dot.ToString();
    }

    /// <summary>
    /// Export reasoning trace to DOT format (combining agent steps and execution graph)
    /// </summary>
    public static string ExportTraceToDot(ReasoningTrace trace)
    {
        var dot = new System.Text.StringBuilder();
        dot.AppendLine("digraph ReasoningTrace {");
        dot.AppendLine("  rankdir=TB;");
        dot.AppendLine("  node [shape=box];");
        dot.AppendLine();

        // Add query node
        dot.AppendLine($"  \"query\" [label=\"Query:\\n{trace.Query}\"];");

        // Add step nodes
        foreach (var step in trace.Steps)
        {
            var stepId = $"step_{step.StepId}";
            var label = $"{step.StepType}\\n{step.ToolName ?? "N/A"}";
            dot.AppendLine($"  \"{stepId}\" [label=\"{label}\"];");
        }

        // Add answer node
        dot.AppendLine($"  \"answer\" [label=\"Answer:\\n{trace.FinalAnswer.Substring(0, Math.Min(50, trace.FinalAnswer.Length))}...\"];");

        dot.AppendLine();

        // Add edges between steps
        dot.AppendLine("  \"query\" -> \"step_1\";");
        for (int i = 1; i < trace.Steps.Count; i++)
        {
            dot.AppendLine($"  \"step_{i}\" -> \"step_{i + 1}\";");
        }
        if (trace.Steps.Any())
        {
            dot.AppendLine($"  \"step_{trace.Steps.Count}\" -> \"answer\";");
        }

        dot.AppendLine("}");
        return dot.ToString();
    }

    /// <summary>
    /// Save execution graph to file
    /// </summary>
    public static async Task SaveToFileAsync(
        ReasoningTrace trace,
        string filePath,
        string format = "json")
    {
        string content;
        string extension;

        if (format.ToLowerInvariant() == "dot")
        {
            content = ExportTraceToDot(trace);
            extension = ".dot";
        }
        else
        {
            content = ExportToJson(trace);
            extension = ".json";
        }

        var fullPath = filePath.EndsWith(extension) ? filePath : filePath + extension;
        await File.WriteAllTextAsync(fullPath, content);
    }
}

