using System.Text.Json;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using SaAgentOS.Agents;
using SaAgentOS.Execution;
using SaAgentOS.Models;
using SaAgentOS.Pipelines;
using SaAgentOS.Tools;
using SharpAIKit.LLM;

namespace SaAgentOS;

class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("╔════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║          SA-AgentOS: Intelligent Agent System              ║");
        Console.WriteLine("║     Knowledge Brain powered by SA-RAG Core + DeepSeek      ║");
        Console.WriteLine("╚════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        // Load configuration
        var configuration = new ConfigurationBuilder()
            .SetBasePath(AppContext.BaseDirectory)
            .AddJsonFile("Config/appsettings.json", optional: false, reloadOnChange: true)
            .AddEnvironmentVariables()
            .Build();

        var config = configuration.Get<AgentConfig>() ?? new AgentConfig();
        
        // Resolve environment variables in config
        ResolveEnvironmentVariables(config);

        // Setup logging
        using var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder
                .AddConsole()
                .SetMinimumLevel(LogLevel.Information);
        });

        var logger = loggerFactory.CreateLogger<Program>();

        // Validate configuration
        if (!ValidateConfiguration(config, logger))
        {
            Console.WriteLine("Configuration validation failed. Please check your settings.");
            return;
        }

        // Initialize HTTP client
        var httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromSeconds(30)
        };

        // Initialize SharpAIKit LLM client
        // Support both direct API key and ENV: prefix for environment variables
        var apiKey = config.DeepSeek.ApiKey;
        if (apiKey.StartsWith("ENV:", StringComparison.OrdinalIgnoreCase))
        {
            var envVarName = apiKey.Substring(4);
            apiKey = Environment.GetEnvironmentVariable(envVarName) ?? "";
            if (string.IsNullOrEmpty(apiKey))
            {
                logger.LogWarning("Environment variable {EnvVar} is not set, using empty API key", envVarName);
            }
        }

        var llmClient = SharpAIKit.LLM.LLMClientFactory.Create(
            apiKey: apiKey,
            baseUrl: config.DeepSeek.BaseUrl,
            model: config.DeepSeek.Model
        );

        // Initialize tools
        var knowledgeTool = new KnowledgeRetrievalTool(
            httpClient,
            config.SaRag,
            loggerFactory.CreateLogger<KnowledgeRetrievalTool>());

        var reasoningTool = new DeepSeekReasoningTool(
            llmClient,
            config.DeepSeek,
            loggerFactory.CreateLogger<DeepSeekReasoningTool>());

        var memoryTool = new MemoryTool(
            config.Memory,
            httpClient,
            loggerFactory.CreateLogger<MemoryTool>());

        // Check SA-RAG availability
        logger.LogInformation("Checking SA-RAG service availability...");
        var isRagAvailable = await knowledgeTool.IsAvailableAsync();
        if (!isRagAvailable)
        {
            logger.LogWarning("SA-RAG service is not available at {Endpoint}. Some features may not work.",
                config.SaRag.Endpoint);
        }
        else
        {
            logger.LogInformation("✅ SA-RAG service is available");
        }

        // Initialize agent
        var agent = new SaRagAgent(
            llmClient,
            knowledgeTool,
            reasoningTool,
            memoryTool,
            loggerFactory.CreateLogger<SaRagAgent>());

        // Initialize pipeline
        var pipeline = new SaAgentPipeline(
            agent,
            loggerFactory.CreateLogger<SaAgentPipeline>());

        // Interactive mode
        if (args.Length == 0)
        {
            await RunInteractiveMode(pipeline, logger);
        }
        else
        {
            // Single query mode
            var query = string.Join(" ", args);
            await ProcessSingleQuery(pipeline, query, logger);
        }
    }

    static async Task RunInteractiveMode(SaAgentPipeline pipeline, ILogger logger)
    {
        Console.WriteLine("Interactive mode. Type 'exit' or 'quit' to exit.");
        Console.WriteLine("Type 'clear' or 'reset' to clear conversation history.");
        Console.WriteLine("Type 'export <filename>' to export the last execution graph.");
        Console.WriteLine();

        ReasoningTrace? lastTrace = null;

        while (true)
        {
            Console.Write("> ");
            var input = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(input))
                continue;

            if (input.ToLowerInvariant() is "exit" or "quit")
            {
                Console.WriteLine("Goodbye!");
                break;
            }

            if (input.ToLowerInvariant() is "clear" or "reset")
            {
                // Clear conversation history
                await pipeline.Agent.ClearHistoryAsync();
                Console.WriteLine("✅ Conversation history cleared.");
                continue;
            }

            if (input.ToLowerInvariant().StartsWith("export "))
            {
                if (lastTrace != null)
                {
                    var filename = input.Substring(7).Trim();
                    await ExecutionGraphExporter.SaveToFileAsync(lastTrace, filename);
                    Console.WriteLine($"Execution graph exported to {filename}");
                }
                else
                {
                    Console.WriteLine("No previous query to export.");
                }
                continue;
            }

            try
            {
                var trace = await pipeline.RunAsync(input);
                lastTrace = trace;

                Console.WriteLine();
                Console.WriteLine("Answer:");
                Console.WriteLine(trace.FinalAnswer);
                Console.WriteLine();

                if (trace.ExecutionGraph != null)
                {
                    Console.WriteLine($"Execution graph: {trace.ExecutionGraph.Nodes.Count} nodes, " +
                                    $"{trace.ExecutionGraph.Edges.Count} edges");
                }

                Console.WriteLine($"Execution time: {trace.ExecutionTimeMs:F2}ms");
                Console.WriteLine();
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Error processing query");
                Console.WriteLine($"Error: {ex.Message}");
            }
        }
    }

    static async Task ProcessSingleQuery(SaAgentPipeline pipeline, string query, ILogger logger)
    {
        try
        {
            var trace = await pipeline.RunAsync(query);
            
            Console.WriteLine("Query: " + query);
            Console.WriteLine();
            Console.WriteLine("Answer:");
            Console.WriteLine(trace.FinalAnswer);
            Console.WriteLine();
            Console.WriteLine($"Execution time: {trace.ExecutionTimeMs:F2}ms");
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error processing query");
            Console.WriteLine($"Error: {ex.Message}");
            Environment.Exit(1);
        }
    }

    static void ResolveEnvironmentVariables(AgentConfig config)
    {
        if (config.DeepSeek.ApiKey.StartsWith("ENV:", StringComparison.OrdinalIgnoreCase))
        {
            var envVarName = config.DeepSeek.ApiKey.Substring(4);
            config.DeepSeek.ApiKey = Environment.GetEnvironmentVariable(envVarName) ?? string.Empty;
        }
    }

    static bool ValidateConfiguration(AgentConfig config, ILogger logger)
    {
        var isValid = true;

        if (string.IsNullOrEmpty(config.DeepSeek.ApiKey))
        {
            logger.LogError("DeepSeek API key is not configured. Please set DEEPSEEK_API_KEY environment variable.");
            isValid = false;
        }

        if (string.IsNullOrEmpty(config.SaRag.Endpoint))
        {
            logger.LogError("SA-RAG endpoint is not configured.");
            isValid = false;
        }

        return isValid;
    }
}

