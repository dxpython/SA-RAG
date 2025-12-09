using System.Collections.Concurrent;
using SharpAIKit.Common;

namespace SharpAIKit.Graph;

/// <summary>
/// Represents a state in the graph.
/// </summary>
public class GraphState
{
    /// <summary>
    /// Gets or sets the state data.
    /// </summary>
    public Dictionary<string, object?> Data { get; set; } = new();

    /// <summary>
    /// Gets or sets the current node name.
    /// </summary>
    public string CurrentNode { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the next node to transition to.
    /// </summary>
    public string? NextNode { get; set; }

    /// <summary>
    /// Gets or sets whether execution should end.
    /// </summary>
    public bool ShouldEnd { get; set; }

    /// <summary>
    /// Gets or sets the final output.
    /// </summary>
    public string? Output { get; set; }

    /// <summary>
    /// Gets a typed value from the state.
    /// </summary>
    public T? Get<T>(string key) => Data.TryGetValue(key, out var value) && value is T typed ? typed : default;

    /// <summary>
    /// Sets a value in the state.
    /// </summary>
    public GraphState Set(string key, object? value)
    {
        Data[key] = value;
        return this;
    }
}

/// <summary>
/// Represents a node in the graph.
/// </summary>
public class GraphNode
{
    /// <summary>
    /// Gets or sets the node name.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the node description.
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the action to execute when entering this node.
    /// </summary>
    public Func<GraphState, Task<GraphState>>? Action { get; set; }

    /// <summary>
    /// Gets or sets the condition function to determine the next node.
    /// </summary>
    public Func<GraphState, string?>? Condition { get; set; }

    /// <summary>
    /// Gets or sets the edges (transitions) from this node.
    /// </summary>
    public List<GraphEdge> Edges { get; set; } = new();
}

/// <summary>
/// Represents an edge (transition) between nodes.
/// </summary>
public class GraphEdge
{
    /// <summary>
    /// Gets or sets the target node name.
    /// </summary>
    public string Target { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the condition function (returns true if this edge should be taken).
    /// </summary>
    public Func<GraphState, bool>? Condition { get; set; }

    /// <summary>
    /// Gets or sets the edge label/description.
    /// </summary>
    public string Label { get; set; } = string.Empty;
}

/// <summary>
/// SharpGraph - A graph-based agent orchestration engine using Finite State Machine (FSM).
/// Supports loops, conditional branches, and complex workflows that LangChain's linear chains cannot handle.
/// </summary>
public class SharpGraph
{
    private readonly Dictionary<string, GraphNode> _nodes = new();
    private readonly string _entryNode;
    private readonly int _maxIterations;

    /// <summary>
    /// Gets or sets the maximum number of iterations to prevent infinite loops.
    /// </summary>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Creates a new SharpGraph.
    /// </summary>
    /// <param name="entryNode">The name of the entry node.</param>
    /// <param name="maxIterations">Maximum iterations to prevent infinite loops.</param>
    public SharpGraph(string entryNode, int maxIterations = 100)
    {
        _entryNode = entryNode ?? throw new ArgumentNullException(nameof(entryNode));
        _maxIterations = maxIterations;
    }

    /// <summary>
    /// Adds a node to the graph.
    /// </summary>
    public SharpGraph AddNode(string name, Func<GraphState, Task<GraphState>> action, string? description = null)
    {
        _nodes[name] = new GraphNode
        {
            Name = name,
            Description = description ?? name,
            Action = action
        };
        return this;
    }

    /// <summary>
    /// Adds a node with a condition function.
    /// </summary>
    public SharpGraph AddNode(string name, Func<GraphState, Task<GraphState>> action, Func<GraphState, string?> condition, string? description = null)
    {
        _nodes[name] = new GraphNode
        {
            Name = name,
            Description = description ?? name,
            Action = action,
            Condition = condition
        };
        return this;
    }

    /// <summary>
    /// Adds an edge (transition) between nodes.
    /// </summary>
    public SharpGraph AddEdge(string from, string to, Func<GraphState, bool>? condition = null, string? label = null)
    {
        if (!_nodes.TryGetValue(from, out var fromNode))
        {
            throw new ArgumentException($"Node '{from}' does not exist");
        }

        fromNode.Edges.Add(new GraphEdge
        {
            Target = to,
            Condition = condition,
            Label = label ?? $"{from} -> {to}"
        });

        return this;
    }

    /// <summary>
    /// Adds a default edge (taken when no other conditions match).
    /// </summary>
    public SharpGraph AddDefaultEdge(string from, string to, string? label = null)
    {
        return AddEdge(from, to, condition: null, label: label ?? $"default: {from} -> {to}");
    }

    /// <summary>
    /// Executes the graph starting from the entry node.
    /// </summary>
    public async Task<GraphState> ExecuteAsync(GraphState? initialState = null, CancellationToken cancellationToken = default)
    {
        var state = initialState ?? new GraphState();
        state.CurrentNode = _entryNode;

        var iterations = 0;
        var visitedNodes = new HashSet<string>();

        while (iterations < MaxIterations && !state.ShouldEnd)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (!_nodes.TryGetValue(state.CurrentNode, out var currentNode))
            {
                throw new InvalidOperationException($"Node '{state.CurrentNode}' does not exist");
            }

            visitedNodes.Add(state.CurrentNode);

            // Execute the node action
            if (currentNode.Action != null)
            {
                state = await currentNode.Action(state);
            }

            // Check if we should end
            if (state.ShouldEnd)
            {
                break;
            }

            // Determine next node
            string? nextNode = null;

            // First, check if the node has a condition function
            if (currentNode.Condition != null)
            {
                nextNode = currentNode.Condition(state);
            }

            // If no condition result, check edges
            if (nextNode == null && currentNode.Edges.Count > 0)
            {
                foreach (var edge in currentNode.Edges)
                {
                    if (edge.Condition == null || edge.Condition(state))
                    {
                        nextNode = edge.Target;
                        break;
                    }
                }
            }

            // If still no next node, check state's NextNode
            if (nextNode == null)
            {
                nextNode = state.NextNode;
            }

            if (nextNode == null)
            {
                // No transition found, end execution
                state.ShouldEnd = true;
                break;
            }

            state.CurrentNode = nextNode;
            state.NextNode = null;
            iterations++;
        }

        if (iterations >= MaxIterations)
        {
            throw new InvalidOperationException($"Graph execution exceeded maximum iterations ({MaxIterations})");
        }

        return state;
    }

    /// <summary>
    /// Gets a visual representation of the graph (for debugging).
    /// </summary>
    public string GetGraphViz()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("digraph SharpGraph {");
        sb.AppendLine("  rankdir=LR;");

        foreach (var node in _nodes.Values)
        {
            foreach (var edge in node.Edges)
            {
                var label = string.IsNullOrEmpty(edge.Label) ? "" : $" [label=\"{edge.Label}\"]";
                sb.AppendLine($"  \"{node.Name}\" -> \"{edge.Target}\"{label};");
            }
        }

        sb.AppendLine($"  \"ENTRY\" -> \"{_entryNode}\" [style=dashed];");
        sb.AppendLine("}");
        return sb.ToString();
    }
}

/// <summary>
/// Builder for creating SharpGraph instances fluently.
/// </summary>
public class SharpGraphBuilder
{
    private readonly SharpGraph _graph;

    /// <summary>
    /// Creates a new graph builder.
    /// </summary>
    public SharpGraphBuilder(string entryNode, int maxIterations = 100)
    {
        _graph = new SharpGraph(entryNode, maxIterations);
    }

    /// <summary>
    /// Adds a node.
    /// </summary>
    public SharpGraphBuilder Node(string name, Func<GraphState, Task<GraphState>> action, string? description = null)
    {
        _graph.AddNode(name, action, description);
        return this;
    }

    /// <summary>
    /// Adds a node with condition.
    /// </summary>
    public SharpGraphBuilder Node(string name, Func<GraphState, Task<GraphState>> action, Func<GraphState, string?> condition, string? description = null)
    {
        _graph.AddNode(name, action, condition, description);
        return this;
    }

    /// <summary>
    /// Adds an edge.
    /// </summary>
    public SharpGraphBuilder Edge(string from, string to, Func<GraphState, bool>? condition = null, string? label = null)
    {
        _graph.AddEdge(from, to, condition, label);
        return this;
    }

    /// <summary>
    /// Adds a default edge.
    /// </summary>
    public SharpGraphBuilder DefaultEdge(string from, string to, string? label = null)
    {
        _graph.AddDefaultEdge(from, to, label);
        return this;
    }

    /// <summary>
    /// Builds the graph.
    /// </summary>
    public SharpGraph Build() => _graph;
}
