using SharpAIKit.Agent;
using SharpAIKit.LLM;

Console.WriteLine("=== SharpAIKit Agent Demo (DeepSeek) ===\n");

// Use DeepSeek API
var client = LLMClientFactory.CreateDeepSeek("YOUR-API-KEY");

// Create AI Agent
var agent = new AiAgent(client);

// Register Tools
agent.AddTool(new CalculatorTool());
agent.AddTool(new WebSearchTool());
agent.AddTool(new FileWriterTool { BaseDirectory = "." });

try
{
    // Example 1: Calculator Tool
    Console.WriteLine("[Example 1: Calculator Tool]");
    var calcResult = await agent.RunAsync("Calculate 3 to the power of 5", CancellationToken.None);
    Console.WriteLine($"Task: Calculate 3 to the power of 5");
    Console.WriteLine($"Result: {calcResult.Answer}");
    Console.WriteLine($"Steps: {calcResult.Steps.Count}\n");

    // Show execution steps
    foreach (var step in calcResult.Steps)
    {
        if (step.Type == "tool")
        {
            Console.WriteLine($"  [Tool] {step.ToolName}: {step.Result}");
        }
        else if (step.Type == "answer")
        {
            Console.WriteLine($"  [Answer] {step.Result}");
        }
    }
    Console.WriteLine();

    // Example 2: Complex Calculation
    Console.WriteLine("[Example 2: Complex Calculation]");
    var complexResult = await agent.RunAsync("What is the square root of 144 plus the square of 5?", CancellationToken.None);
    Console.WriteLine($"Task: What is the square root of 144 plus the square of 5?");
    Console.WriteLine($"Result: {complexResult.Answer}\n");

    // Example 3: Web Search (Simulated)
    Console.WriteLine("[Example 3: Web Search Tool]");
    var searchResult = await agent.RunAsync("Search for information about C# programming", CancellationToken.None);
    Console.WriteLine($"Task: Search for information about C# programming");
    Console.WriteLine($"Result: {searchResult.Answer}\n");

    // Example 4: File Operations
    Console.WriteLine("[Example 4: File Operations Tool]");

    // Write to file
    var writeResult = await agent.RunAsync("Write 'Hello from SharpAIKit Agent!' to a file named test_output.txt", CancellationToken.None);
    Console.WriteLine($"Task: Write to file");
    Console.WriteLine($"Result: {writeResult.Answer}\n");

    // Read file
    var readResult = await agent.RunAsync("Read the content of test_output.txt", CancellationToken.None);
    Console.WriteLine($"Task: Read file");
    Console.WriteLine($"Result: {readResult.Answer}\n");

    // Cleanup
    if (File.Exists("test_output.txt"))
    {
        File.Delete("test_output.txt");
    }

    // Example 5: Interactive Mode
    Console.WriteLine("[Example 5: Interactive Agent Mode] (Type 'exit' to quit)");

    while (true)
    {
        Console.Write("\nYou: ");
        var input = Console.ReadLine();

        if (string.IsNullOrEmpty(input) || input.ToLower() == "exit")
        {
            Console.WriteLine("Goodbye!");
            break;
        }

        Console.WriteLine("Agent is thinking...");
        var result = await agent.RunAsync(input, CancellationToken.None);

        Console.WriteLine($"\nAgent: {result.Answer}");

        if (result.Steps.Count > 1)
        {
            Console.WriteLine("\n  Execution Trace:");
            foreach (var step in result.Steps)
            {
                if (step.Type == "tool")
                {
                    Console.WriteLine($"    -> Call {step.ToolName}");
                    if (!string.IsNullOrEmpty(step.Thought))
                    {
                        Console.WriteLine($"       Thought: {step.Thought}");
                    }
                }
            }
        }
    }
}
catch (Exception ex)
{
    Console.WriteLine($"Error: {ex.Message}");
}
finally
{
    client.Dispose();
}

Console.WriteLine("\n=== Agent Demo Completed ===");
