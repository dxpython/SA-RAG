using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using SaAgentOS.Models;
using SharpAIKit.Agent;
using SharpAIKit.LLM;

namespace SaAgentOS.Tools;

/// <summary>
/// Tool for calling DeepSeek API for reasoning
/// Uses SharpAIKit's LLM client for better integration
/// </summary>
public class DeepSeekReasoningTool : ToolBase
{
    private readonly ILLMClient _llmClient;
    private readonly DeepSeekConfig _config;
    private readonly ILogger<DeepSeekReasoningTool>? _logger;

    public DeepSeekReasoningTool(
        ILLMClient llmClient,
        DeepSeekConfig config,
        ILogger<DeepSeekReasoningTool>? logger = null)
    {
        _llmClient = llmClient ?? throw new ArgumentNullException(nameof(llmClient));
        _config = config;
        _logger = logger;
    }

    /// <summary>
    /// Legacy constructor for backward compatibility
    /// </summary>
    public DeepSeekReasoningTool(
        HttpClient httpClient,
        DeepSeekConfig config,
        ILogger<DeepSeekReasoningTool>? logger = null)
    {
        // Create LLM client from config
        var apiKey = ResolveApiKey(config.ApiKey);
        _llmClient = LLMClientFactory.Create(
            apiKey: apiKey,
            baseUrl: config.BaseUrl,
            model: config.Model
        );
        _config = config;
        _logger = logger;
    }

    /// <summary>
    /// Resolve API key from config (supports ENV: prefix)
    /// </summary>
    private static string ResolveApiKey(string configValue)
    {
        if (string.IsNullOrEmpty(configValue))
        {
            return string.Empty;
        }

        if (configValue.StartsWith("ENV:", StringComparison.OrdinalIgnoreCase))
        {
            var envVarName = configValue.Substring(4);
            return Environment.GetEnvironmentVariable(envVarName) ?? string.Empty;
        }

        return configValue;
    }

    /// <summary>
    /// Run reasoning with DeepSeek (SharpAIKit tool method)
    /// </summary>
    [Tool("deepseek_reasoning", "Performs reasoning using DeepSeek Chat/Reasoner API. Use this for complex reasoning tasks or when you need to generate detailed answers.")]
    public async Task<string> Reason(
        [Parameter("The prompt or question to reason about")] string prompt,
        [Parameter("Additional context to include (optional)")] string? extraContext = null)
    {
        try
        {
            var fullPrompt = prompt;
            if (!string.IsNullOrEmpty(extraContext))
            {
                fullPrompt = $"Context: {extraContext}\n\nQuestion: {prompt}";
            }

            _logger?.LogInformation("Calling DeepSeek API: Model={Model}, Prompt length={PromptLength}",
                _config.Model, fullPrompt.Length);

            var response = await _llmClient.ChatAsync(fullPrompt);

            _logger?.LogInformation("DeepSeek reasoning completed: Response length={ResponseLength}",
                response.Length);

            return response;
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Error during DeepSeek reasoning");
            return $"Error: Failed to perform reasoning: {ex.Message}";
        }
    }

    /// <summary>
    /// Run reasoning with DeepSeek (legacy method for backward compatibility)
    /// </summary>
    public async Task<string> RunReasoningAsync(
        string prompt,
        object? extraContext = null,
        CancellationToken cancellationToken = default)
    {
        var fullPrompt = prompt;
        if (extraContext != null)
        {
            var contextStr = JsonSerializer.Serialize(extraContext);
            fullPrompt = $"Context: {contextStr}\n\nQuestion: {prompt}";
        }

        return await _llmClient.ChatAsync(fullPrompt);
    }

    /// <summary>
    /// Check if DeepSeek API is configured
    /// </summary>
    public bool IsConfigured()
    {
        var apiKey = ResolveApiKey(_config.ApiKey);
        return !string.IsNullOrEmpty(apiKey);
    }
}

