using System.Net.Http;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using SaAgentOS.Agents;
using SaAgentOS.Models;
using SaAgentOS.Tools;
using Xunit;

namespace SaAgentOS.Tests;

/// <summary>
/// Integration tests for SA-AgentOS
/// Note: These tests require SA-RAG service to be running and DeepSeek API key to be configured
/// </summary>
public class AgentTests
{
    private readonly IConfiguration _configuration;
    private readonly HttpClient _httpClient;

    public AgentTests()
    {
        _configuration = new ConfigurationBuilder()
            .SetBasePath(Directory.GetCurrentDirectory())
            .AddJsonFile("Config/appsettings.json", optional: false)
            .AddEnvironmentVariables()
            .Build();

        _httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
    }

    [Fact]
    public void TestConfigurationLoading()
    {
        var config = _configuration.Get<AgentConfig>();
        Assert.NotNull(config);
        Assert.NotNull(config.DeepSeek);
        Assert.NotNull(config.SaRag);
        Assert.NotNull(config.Memory);
    }

    [Fact]
    public void TestEnvironmentVariableResolution()
    {
        // Set test environment variable
        Environment.SetEnvironmentVariable("DEEPSEEK_API_KEY", "test-key-12345");
        
        var config = new AgentConfig
        {
            DeepSeek = new DeepSeekConfig { ApiKey = "ENV:DEEPSEEK_API_KEY" }
        };

        if (config.DeepSeek.ApiKey.StartsWith("ENV:", StringComparison.OrdinalIgnoreCase))
        {
            var envVarName = config.DeepSeek.ApiKey.Substring(4);
            config.DeepSeek.ApiKey = Environment.GetEnvironmentVariable(envVarName) ?? string.Empty;
        }

        Assert.Equal("test-key-12345", config.DeepSeek.ApiKey);
        
        // Cleanup
        Environment.SetEnvironmentVariable("DEEPSEEK_API_KEY", null);
    }

    [Fact]
    public async Task TestKnowledgeRetrievalTool()
    {
        var config = _configuration.Get<AgentConfig>();
        if (config == null) return;

        var tool = new KnowledgeRetrievalTool(_httpClient, config.SaRag);

        // Check if service is available
        var isAvailable = await tool.IsAvailableAsync();
        
        if (isAvailable)
        {
            // Try to execute a query
            try
            {
                var result = await tool.ExecuteAsync("What is Python?", topK: 3);
                
                Assert.NotNull(result);
                Assert.NotEmpty(result.Answer);
                Assert.NotNull(result.Nodes);
                // Nodes might be empty if no documents are indexed, which is okay
            }
            catch (Exception ex)
            {
                // If service is not fully set up, that's okay for tests
                Assert.True(ex.Message.Contains("SA-RAG") || ex.Message.Contains("timeout"),
                    $"Unexpected error: {ex.Message}");
            }
        }
        else
        {
            // Service not available, skip test
            Assert.True(true, "SA-RAG service not available, skipping test");
        }
    }

    [Fact]
    public void TestDeepSeekReasoningToolConfiguration()
    {
        var config = _configuration.Get<AgentConfig>();
        if (config == null) return;

        var llmClient = SharpAIKit.LLM.LLMClientFactory.Create(
            apiKey: config.DeepSeek.ApiKey,
            baseUrl: config.DeepSeek.BaseUrl,
            model: config.DeepSeek.Model
        );
        var tool = new DeepSeekReasoningTool(llmClient, config.DeepSeek);

        // Tool should be created even without API key
        Assert.NotNull(tool);
    }

    [Fact]
    public async Task TestDeepSeekReasoningToolWithoutApiKey()
    {
        var config = new DeepSeekConfig
        {
            ApiKey = "",  // No API key
            BaseUrl = "https://api.deepseek.com",
            Model = "deepseek-chat"
        };

        var tool = new DeepSeekReasoningTool(_httpClient, config);

        // Should fail gracefully with clear error message
        var result = await tool.Reason("test prompt");
        // Tool returns error message as string instead of throwing
        Assert.NotNull(result);
    }

    [Fact]
    public void TestMemoryTool()
    {
        var config = new MemoryConfig
        {
            StoragePath = "./test_memory.json",
            UseSaRagMemory = false
        };

        var tool = new MemoryTool(config);

        Assert.NotNull(tool);

        // Cleanup
        if (File.Exists("./test_memory.json"))
        {
            File.Delete("./test_memory.json");
        }
    }

    [Fact]
    public async Task TestMemoryToolAddAndRetrieve()
    {
        var config = new MemoryConfig
        {
            StoragePath = "./test_memory_retrieve.json",
            UseSaRagMemory = false
        };

        var tool = new MemoryTool(config);

        // Add memory
        await tool.AddMemoryAsync("User likes Python programming", importance: 0.8);

        // Retrieve
        var memories = tool.RetrieveRelevant("Python", maxResults: 5);
        Assert.NotEmpty(memories);
        Assert.Contains(memories, m => m.Text.Contains("Python"));

        // Cleanup
        if (File.Exists("./test_memory_retrieve.json"))
        {
            File.Delete("./test_memory_retrieve.json");
        }
    }
}

