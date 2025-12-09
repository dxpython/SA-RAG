using System.Net.Http.Json;
using System.Text.Json;
using System.Linq;
using Microsoft.Extensions.Logging;
using SaAgentOS.Models;
using SharpAIKit.Agent;

namespace SaAgentOS.Tools;

/// <summary>
/// Tool for retrieving knowledge from SA-RAG
/// Inherits from SharpAIKit's ToolBase for seamless integration
/// </summary>
public class KnowledgeRetrievalTool : ToolBase
{
    private readonly HttpClient _httpClient;
    private readonly SaRagConfig _config;
    private readonly ILogger<KnowledgeRetrievalTool>? _logger;

    public KnowledgeRetrievalTool(
        HttpClient httpClient,
        SaRagConfig config,
        ILogger<KnowledgeRetrievalTool>? logger = null)
    {
        _httpClient = httpClient;
        _config = config;
        _logger = logger;
    }

    /// <summary>
    /// Execute knowledge retrieval from SA-RAG
    /// </summary>
    [Tool("knowledge_retrieval", "Retrieves relevant knowledge from SA-RAG knowledge base. Use this when you need to search for information in the knowledge base.")]
    public async Task<string> RetrieveKnowledge(
        [Parameter("The query to search for in the knowledge base")] string query,
        [Parameter("Whether to use graph expansion (default: true)")] bool useGraph = true,
        [Parameter("Whether to use memory retrieval (default: true)")] bool useMemory = true,
        [Parameter("Number of results to return (default: 6)")] int topK = 6)
    {
        try
        {
            var request = new RAGQueryRequest
            {
                Query = query,
                UseGraph = useGraph,
                UseMemory = useMemory,
                TopK = topK
            };

            _logger?.LogInformation(
                "Executing knowledge retrieval: Query='{Query}', UseGraph={UseGraph}, UseMemory={UseMemory}, TopK={TopK}",
                query, request.UseGraph, request.UseMemory, request.TopK);

            var response = await _httpClient.PostAsJsonAsync(
                _config.Endpoint,
                request);

            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<RAGQueryResponse>(
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

            if (result == null)
            {
                return "Error: Failed to retrieve knowledge from SA-RAG";
            }

            _logger?.LogInformation(
                "Knowledge retrieval completed: Found {NodeCount} nodes, Answer length={AnswerLength}",
                result.Nodes.Count, result.Answer.Length);

            // Format result as a readable string for the agent
            var formattedResult = $"Retrieved {result.Nodes.Count} relevant nodes:\n\n";
            formattedResult += $"Answer: {result.Answer}\n\n";
            formattedResult += "Top results:\n";
            foreach (var node in result.Nodes.Take(3))
            {
                formattedResult += $"- [{node.Score:F2}] {node.Text.Substring(0, Math.Min(100, node.Text.Length))}...\n";
            }

            return formattedResult;
        }
        catch (HttpRequestException ex)
        {
            _logger?.LogError(ex, "HTTP error during knowledge retrieval");
            return $"Error: Failed to retrieve knowledge from SA-RAG: {ex.Message}";
        }
        catch (TaskCanceledException ex)
        {
            _logger?.LogError(ex, "Timeout during knowledge retrieval");
            return "Error: Knowledge retrieval request timed out";
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Unexpected error during knowledge retrieval");
            return $"Error: {ex.Message}";
        }
    }

    /// <summary>
    /// Execute knowledge retrieval (legacy method for backward compatibility)
    /// </summary>
    public async Task<RAGQueryResponse> ExecuteAsync(
        string query,
        bool? useGraph = null,
        bool? useMemory = null,
        int? topK = null,
        CancellationToken cancellationToken = default)
    {
        var request = new RAGQueryRequest
        {
            Query = query,
            UseGraph = useGraph ?? _config.UseGraph,
            UseMemory = useMemory ?? _config.UseMemory,
            TopK = topK ?? _config.TopK
        };

        var response = await _httpClient.PostAsJsonAsync(
            _config.Endpoint,
            request,
            cancellationToken);

        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<RAGQueryResponse>(
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true },
            cancellationToken);

        if (result == null)
        {
            throw new InvalidOperationException("Failed to deserialize RAG query response");
        }

        return result;
    }

    /// <summary>
    /// Check if SA-RAG service is available
    /// </summary>
    public async Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var healthUrl = _config.Endpoint.Replace("/rag/query", "/health");
            var response = await _httpClient.GetAsync(healthUrl, cancellationToken);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }
}

