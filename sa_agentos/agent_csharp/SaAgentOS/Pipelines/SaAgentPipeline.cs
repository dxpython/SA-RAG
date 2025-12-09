using Microsoft.Extensions.Logging;
using SaAgentOS.Agents;
using SaAgentOS.Models;

namespace SaAgentOS.Pipelines;

/// <summary>
/// SA-AgentOS Pipeline: Main orchestration loop for agent execution
/// </summary>
public class SaAgentPipeline
{
    private readonly SaRagAgent _agent;
    private readonly ILogger<SaAgentPipeline>? _logger;

    public SaAgentPipeline(
        SaRagAgent agent,
        ILogger<SaAgentPipeline>? logger = null)
    {
        _agent = agent;
        _logger = logger;
    }

    /// <summary>
    /// Get the agent instance (for accessing agent methods)
    /// </summary>
    public SaRagAgent Agent => _agent;

    /// <summary>
    /// Run the agent pipeline
    /// </summary>
    public async Task<ReasoningTrace> RunAsync(
        string userQuery,
        CancellationToken cancellationToken = default)
    {
        var startTime = DateTimeOffset.UtcNow;
        _logger?.LogInformation("Starting agent pipeline for query: '{Query}'", userQuery);

        try
        {
            // Process query through agent
            var response = await _agent.ProcessQueryAsync(userQuery, cancellationToken);

            // Build reasoning trace
            var trace = new ReasoningTrace
            {
                Query = userQuery,
                Steps = response.Steps,
                FinalAnswer = response.FinalAnswer,
                ExecutionTimeMs = (DateTimeOffset.UtcNow - startTime).TotalMilliseconds,
                ExecutionGraph = response.ExecutionGraph
            };

            _logger?.LogInformation(
                "Agent pipeline completed: Steps={StepCount}, ExecutionTime={ExecutionTime}ms",
                trace.Steps.Count, trace.ExecutionTimeMs);

            return trace;
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Error in agent pipeline");
            throw;
        }
    }

    /// <summary>
    /// Run multiple queries in sequence
    /// </summary>
    public async Task<List<ReasoningTrace>> RunBatchAsync(
        IEnumerable<string> queries,
        CancellationToken cancellationToken = default)
    {
        var traces = new List<ReasoningTrace>();

        foreach (var query in queries)
        {
            var trace = await RunAsync(query, cancellationToken);
            traces.Add(trace);
        }

        return traces;
    }
}

