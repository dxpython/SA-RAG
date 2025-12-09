using SharpAIKit.LLM;
using SharpAIKit.Chain;
using SharpAIKit.Memory;
using SharpAIKit.Prompt;
using SharpAIKit.Output;
using SharpAIKit.DocumentLoader;
using SharpAIKit.Callback;
using SharpAIKit.Agent;
using SharpAIKit.MultiModal;

Console.WriteLine("🚀 SharpAIKit Advanced Features Demo (DeepSeek)\n");
Console.WriteLine("=".PadRight(50, '='));

// Using DeepSeek API
var client = LLMClientFactory.CreateDeepSeek("sk-e164311ef7914e46a5d760c505714b94");

// ============================================================
// 1. Chain Module - LCEL Style Composition
// ============================================================
Console.WriteLine("\n📦 1. Chain Module Demo\n");

// Simple LLM Chain
var llmChain = new LLMChain(client, "You are a friendly assistant.");

// Prompt Template Chain
var promptChain = PromptTemplate.FromTemplate(
    "Translate the following text to {language}: {input}"
);

// Using | operator for pipeline composition
var translationPipeline = promptChain.Pipe(llmChain);

// Execute Chain
var context = new ChainContext()
    .Set("language", "French")
    .Set("input", "Hello, how are you?");

var result = await translationPipeline.InvokeAsync(context);
Console.WriteLine($"Translation Result: {result.Output}");

// Parallel Chain
var parallelChain = ChainExtensions.Parallel(
    new LLMChain(client, "You are an optimist. Give a positive perspective."),
    new LLMChain(client, "You are a realist. Give a balanced perspective."),
    new LLMChain(client, "You are a critic. Give a critical perspective.")
);

Console.WriteLine("✅ Chain Module Ready - Supports LCEL Style Composition!");

// ============================================================
// 2. Memory Module - Multiple Memory Strategies
// ============================================================
Console.WriteLine("\n📦 2. Memory Module Demo\n");

// Buffer Memory - Simply keeps last N messages
var bufferMemory = new BufferMemory { MaxMessages = 10 };
await bufferMemory.AddExchangeAsync("What is Python?", "Python is a programming language.");
await bufferMemory.AddExchangeAsync("Who created it?", "Guido van Rossum created Python.");

// Window Buffer Memory - Keeps last N conversation turns
var windowMemory = new WindowBufferMemory { WindowSize = 5 };

// Summary Memory - Summarizes old conversations
var summaryMemory = new SummaryMemory(client) { RecentMessagesCount = 6 };

// Vector Memory - Uses semantic search to retrieve relevant conversation history
var vectorMemory = new VectorMemory(client) { TopK = 5 };

// Entity Memory - Extracts and tracks entities
var entityMemory = new EntityMemory(client);

Console.WriteLine($"Buffer Memory has {await bufferMemory.GetCountAsync()} messages");
Console.WriteLine("✅ Memory Module Ready - Supports 5 different memory strategies!");

// ============================================================
// 3. Prompt Module - Advanced Prompt Templates
// ============================================================
Console.WriteLine("\n📦 3. Prompt Module Demo\n");

// Simple Template
var simpleTemplate = PromptTemplate.FromTemplate(
    "You are {role}. Answer the question: {input}"
);

// With Partial Variables
var templateWithPartials = PromptTemplate.FromTemplate(
    "Current time: {time}\nUser: {input}"
).WithPartial("time", () => DateTime.Now.ToString());

// Chat Prompt Template
var chatTemplate = new ChatPromptTemplate()
    .AddSystemMessage("You are a {role} assistant.")
    .AddHistoryPlaceholder("history")
    .AddUserMessage("{input}");

// Few-shot Prompt Template
var fewShotTemplate = new FewShotPromptTemplate(
    prefix: "Classify the sentiment of the following texts:",
    suffix: "Text: {input}\nSentiment:",
    exampleTemplate: "Text: {input}\nSentiment: {output}"
)
.AddExample("I love this product!", "Positive")
.AddExample("This is terrible.", "Negative")
.AddExample("It's okay, nothing special.", "Neutral");

Console.WriteLine("✅ Prompt Module Ready - Supports Templates, Chat Prompts, Few-shot!");

// ============================================================
// 4. Output Parser Module - Structured Output Parsing
// ============================================================
Console.WriteLine("\n📦 4. Output Parser Module Demo\n");

// JSON Output Parser
var jsonParser = new JsonOutputParser<ProductReview>();
Console.WriteLine("Format Instructions:");
Console.WriteLine(jsonParser.GetFormatInstructions());

// Boolean Parser
var boolParser = new BooleanParser();
Console.WriteLine($"Parsed 'yes' = {boolParser.Parse("yes")}");
Console.WriteLine($"Parsed 'no' = {boolParser.Parse("no")}");

// List Parser
var listParser = new CommaSeparatedListParser();
Console.WriteLine($"Parsed List: [{string.Join(", ", listParser.Parse("Apple, Banana, Orange"))}]");

// XML Tag Parser
var xmlParser = new XMLTagParser("answer", "reasoning");
var xmlResult = xmlParser.Parse("<answer>42</answer><reasoning>Because...</reasoning>");
Console.WriteLine($"XML Parsing: answer={xmlResult["answer"]}, reasoning={xmlResult["reasoning"]}");

Console.WriteLine("✅ Output Parser Module Ready - JSON, Boolean, List, XML Parsing!");

// ============================================================
// 5. Document Loader Module - Multi-format Support
// ============================================================
Console.WriteLine("\n📦 5. Document Loader Module Demo\n");

// Text File Loader
// var textLoader = new TextFileLoader("sample.txt");

// CSV Loader (Column Aware)
// var csvLoader = new CsvLoader("data.csv", hasHeader: true)
// {
//     OneDocumentPerRow = true,
//     ContentColumns = new[] { "title", "content" },
//     MetadataColumns = new[] { "id", "category" }
// };

// Markdown Loader (Split by Headers)
// var mdLoader = new MarkdownLoader("readme.md")
// {
//     SplitByHeaders = true,
//     SplitHeaderLevel = 2
// };

Console.WriteLine("✅ Document Loader Module Ready - Text, CSV, JSON, Markdown, Web!");

// ============================================================
// 6. Callback Module - Observability
// ============================================================
Console.WriteLine("\n📦 6. Callback Module Demo\n");

// Console Callback (For Debugging)
var consoleCallback = new ConsoleCallbackHandler { UseColors = true };

// Metrics Callback (Performance Monitoring)
var metricsCallback = new MetricsCallbackHandler();

// Callback Manager
var callbackManager = new CallbackManager()
    .AddHandler(consoleCallback)
    .AddHandler(metricsCallback);

// Simulate Events
await callbackManager.OnLLMStartAsync(new LLMEvent { Model = "deepseek-chat" });
await callbackManager.OnLLMEndAsync(new LLMEvent 
{ 
    Model = "deepseek-chat", 
    LatencyMs = 150,
    Usage = new TokenUsage { PromptTokens = 100, CompletionTokens = 50 }
});

var metrics = metricsCallback.GetSummary();
Console.WriteLine($"Metrics: {metrics.LLMCalls} calls, {metrics.AverageLatencyMs}ms avg latency, {metrics.TotalTokens} total tokens");

Console.WriteLine("✅ Callback Module Ready - Console, Logging, Metrics, File Tracing!");

// ============================================================
// 7. MultiModal Module - Vision Support
// ============================================================
Console.WriteLine("\n📦 7. MultiModal Module Demo\n");

// Create vision message from URL
var visionMessage = MultiModalMessage.User(
    "What is in this image?",
    "https://example.com/image.jpg"
);

// Fluent Builder
var multiModalMessage = new MultiModalMessageBuilder()
    .WithRole("user")
    .AddText("Compare these two images:")
    .AddImage("https://example.com/image1.jpg")
    .AddImage("https://example.com/image2.jpg")
    .Build();

Console.WriteLine("✅ MultiModal Module Ready - Supports Images, URLs, Base64!");

// ============================================================
// 8. Agents - ReAct, Plan-and-Execute, Multi-Agent
// ============================================================
Console.WriteLine("\n📦 8. Agents Demo\n");

// ReAct Agent - Reasoning + Acting
var reactAgent = new ReActAgent(client)
    .AddTool(new CalculatorTool())
    .AddTool(new WebSearchTool());

Console.WriteLine("ReAct Agent: Uses Thought -> Action -> Observation Loop");

// Plan-and-Execute Agent
var planAgent = new PlanAndExecuteAgent(client)
    .AddTool(new CalculatorTool());

Console.WriteLine("Plan-Execute Agent: Creates plan first, then executes step by step");

// Multi-Agent System
var multiAgentSystem = new MultiAgentSystem(client)
    .AddAgent("researcher", "Research Expert", "You are a professional researcher...")
    .AddAgent("writer", "Content Writer", "You are an excellent technical writer...")
    .AddAgent("reviewer", "Quality Reviewer", "You are a rigorous reviewer...");

Console.WriteLine("Multi-Agent System: Coordinates multiple specialized agents");

Console.WriteLine("✅ Advanced Agents Ready - ReAct, Plan-Execute, Multi-Agent!");

// ============================================================
// 9. Real Demo - Chat with DeepSeek
// ============================================================
Console.WriteLine("\n📦 9. Real Chat Demo\n");

try
{
    // Simple Chat
    Console.WriteLine("[Simple Chat Test]");
    var chatResponse = await client.ChatAsync("Introduce SharpAIKit framework in one sentence");
    Console.WriteLine($"DeepSeek: {chatResponse}\n");

    // Use ConversationChain for chat with memory
    Console.WriteLine("[Chat with Memory Test]");
    var conversation = new ConversationChain(client, "You are a friendly programming assistant.");
    
    var resp1 = await conversation.InvokeAsync("What is C#?");
    Console.WriteLine($"Q: What is C#?");
    Console.WriteLine($"A: {resp1}\n");

    var resp2 = await conversation.InvokeAsync("What is the difference between it and Java?");
    Console.WriteLine($"Q: What is the difference between it and Java?");
    Console.WriteLine($"A: {resp2}\n");

    Console.WriteLine($"Conversation History Length: {conversation.History.Count} messages");
}
catch (Exception ex)
{
    Console.WriteLine($"Error: {ex.Message}");
}

// ============================================================
// Summary
// ============================================================
Console.WriteLine("\n" + "=".PadRight(50, '='));
Console.WriteLine("🎉 SharpAIKit Feature Summary:");
Console.WriteLine("=".PadRight(50, '='));
Console.WriteLine(@"
✅ Chain Module      - LCEL style composition, parallel execution
✅ Memory Module     - Buffer, Window, Summary, Vector, Entity memory
✅ Prompt Module     - Templates, Chat prompts, Few-shot learning
✅ Output Parser     - JSON, Boolean, List, XML, Regex parsing
✅ Document Loader   - Text, CSV, JSON, Markdown, Web loading
✅ Callback Module   - Console, Logging, Metrics, File tracing
✅ MultiModal        - Vision image support
✅ Advanced Agents   - ReAct, Plan-and-Execute, Multi-Agent

🚀 All core features of LangChain are now available in C#!
🎯 Plus .NET native advantages: Strong typing, async/await, LINQ
");

client.Dispose();

// Product Review Class (For JSON Parser)
public class ProductReview
{
    public string Product { get; set; } = "";
    public int Rating { get; set; }
    public string Summary { get; set; } = "";
    public List<string> Pros { get; set; } = new();
    public List<string> Cons { get; set; } = new();
}
