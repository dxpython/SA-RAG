using SharpAIKit.LLM;

Console.WriteLine("=== SharpAIKit DeepSeek Demo ===\n");

// Use DeepSeek API
var client = LLMClientFactory.CreateDeepSeek("YOUR-API-KEY");

try
{
    // Test 1: Simple Chat
    Console.WriteLine("[Test 1: Simple Chat]");
    var response = await client.ChatAsync("Hello! Please introduce yourself in one sentence.");
    Console.WriteLine($"DeepSeek: {response}\n");

    // Test 2: Streaming Output
    Console.WriteLine("[Test 2: Streaming Output]");
    Console.Write("DeepSeek: ");
    await foreach (var chunk in client.ChatStreamAsync("Write a short poem about programming, no more than 4 lines."))
    {
        Console.Write(chunk);
    }
    Console.WriteLine("\n");

    // Test 3: Multi-turn Chat
    Console.WriteLine("[Test 3: Multi-turn Chat]");
    var messages = new List<SharpAIKit.Common.ChatMessage>
    {
        SharpAIKit.Common.ChatMessage.System("You are a friendly programming assistant, be concise."),
        SharpAIKit.Common.ChatMessage.User("What is C#?"),
    };

    var reply1 = await client.ChatAsync(messages);
    Console.WriteLine($"User: What is C#?");
    Console.WriteLine($"DeepSeek: {reply1}\n");

    messages.Add(SharpAIKit.Common.ChatMessage.Assistant(reply1));
    messages.Add(SharpAIKit.Common.ChatMessage.User("What is the difference between it and Python?"));

    var reply2 = await client.ChatAsync(messages);
    Console.WriteLine($"User: What is the difference between it and Python?");
    Console.WriteLine($"DeepSeek: {reply2}\n");

    Console.WriteLine("=== Test Completed! ===");
}
catch (Exception ex)
{
    Console.WriteLine($"Error: {ex.Message}");
}
finally
{
    client.Dispose();
}
