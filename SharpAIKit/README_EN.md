<div align="center">

## SharpAIKit: A Unified AI/LLM Toolkit for .NET

### 🎯 More Powerful Than LangChain, Simpler Than LangChain

<img src="imgs/logo.jpg" alt="SharpAIKit Logo" width="900">

[![.NET Version](https://img.shields.io/badge/.NET-8.0-512BD4?style=for-the-badge&logo=dotnet&logoColor=white)](https://dotnet.microsoft.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![NuGet](https://img.shields.io/badge/NuGet-v2.0.0-004880?style=for-the-badge&logo=nuget&logoColor=white)](https://www.nuget.org/)

[中文文档](README_CN.md) | [🏠 Home](README.md)

</div>

---

## 🆚 SharpAIKit vs LangChain

| Feature | SharpAIKit | LangChain |
|:--------|:----------:|:---------:|
| **Chain (LCEL)** | ✅ Pipe operator | ✅ |
| **Memory** | ✅ 5 strategies | ✅ |
| **Prompt Templates** | ✅ Type-safe | ✅ |
| **Output Parsers** | ✅ Strongly-typed | ✅ |
| **Document Loaders** | ✅ Multi-format | ✅ |
| **Observability** | ✅ Full tracing | ✅ |
| **MultiModal** | ✅ Image support | ✅ |
| **Advanced Agents** | ✅ ReAct/Multi-Agent | ✅ |
| **Code Interpreter** | ✅ **Native C# (Roslyn)** | ❌ Python dependency |
| **Graph Orchestration** | ✅ **SharpGraph (FSM)** | ⚠️ LangGraph (new) |
| **Auto Optimization** | ✅ **DSPy-style** | ❌ None |
| **Type Safety** | ✅ **C# Strong typing** | ❌ Python weak typing |
| **Performance** | ✅ **Native compilation** | ❌ Interpreted |
| **Code Simplicity** | ✅ **Minimal API** | ❌ Heavy abstractions |
| **Dependencies** | ✅ **Minimal** | ❌ Many dependencies |

---

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Chain (LCEL)](#-chain-lcel)
- [Memory](#-memory)
- [Prompt Templates](#-prompt-templates)
- [Output Parsers](#-output-parsers)
- [Document Loaders](#-document-loaders)
- [Observability (Callback)](#-observability-callback)
- [MultiModal](#-multimodal)
- [Advanced Agents](#-advanced-agents)
- [RAG Engine](#-rag-engine)
- [🔮 Native C# Code Interpreter](#-native-c-code-interpreter) ⭐ **Killer Feature**
- [🕸️ SharpGraph](#️-sharpgraph) ⭐ **Killer Feature**
- [🧬 DSPy-style Optimizer](#-dspy-style-optimizer) ⭐ **Killer Feature**
- [Supported Providers](#-supported-providers)

---

## ✨ Features

| Module | Feature | Description |
|:-------|:--------|:------------|
| 🔗 **Chain** | Chain Composition | LCEL-style pipe composition with `\|` operator |
| 🧠 **Memory** | Conversation Memory | Buffer, Window, Summary, Vector, Entity strategies |
| 📝 **Prompt** | Prompt Templates | Variable substitution, Chat templates, Few-shot learning |
| 📤 **Output** | Output Parsers | JSON, Boolean, List, XML, Regex parsers |
| 📄 **Loader** | Document Loading | Text, CSV, JSON, Markdown, Web multi-format |
| 📊 **Callback** | Observability | Console, Logging, Metrics, File full-trace |
| 🖼️ **MultiModal** | MultiModal | Image URL, local files, Base64 support |
| 🤖 **Agent** | Intelligent Agents | ReAct, Plan-Execute, Multi-Agent systems |
| 📚 **RAG** | Retrieval-Augmented | Document indexing, vector search, Q&A |
| 🔮 **Code Interpreter** | Code Execution | **Native C# code execution**, no Python, based on Roslyn |
| 🕸️ **SharpGraph** | Graph Orchestration | **Finite State Machine**, supports loops and complex branches |
| 🧬 **DSPy Optimizer** | Auto Optimization | **Automatic prompt optimization**, gets smarter over time |

---

## 📦 Installation

```bash
dotnet add package SharpAIKit
```

---

## 🚀 Quick Start

```csharp
using SharpAIKit.LLM;

// Three lines of code, works with any OpenAI-compatible API
var client = LLMClientFactory.Create("your-api-key", "https://api.deepseek.com/v1", "deepseek-chat");
var response = await client.ChatAsync("Hello!");
Console.WriteLine(response);

// Streaming output
await foreach (var chunk in client.ChatStreamAsync("Tell me a story"))
{
    Console.Write(chunk);
}
```

---

## 🔗 Chain (LCEL)

**LCEL-style pipe composition, elegant like LangChain!**

```csharp
using SharpAIKit.Chain;
using SharpAIKit.Prompt;

// Create chain components
var prompt = PromptTemplate.FromTemplate("Translate the following to {language}: {input}");
var llm = new LLMChain(client);

// Pipe composition (using | operator, like LangChain LCEL!)
var chain = prompt.Pipe(llm);

// Execute
var result = await chain.InvokeAsync(new ChainContext()
    .Set("language", "French")
    .Set("input", "Hello, world"));

Console.WriteLine(result.Output);

// Parallel execution
var parallel = ChainExtensions.Parallel(
    new LLMChain(client, "You are an optimist"),
    new LLMChain(client, "You are a realist"),
    new LLMChain(client, "You are a critic")
);

// Conditional branching
var branched = chain.Branch(
    ctx => ctx.Output.Contains("error"),
    trueBranch: errorHandlerChain,
    falseBranch: successChain
);
```

---

## 🧠 Memory

**5 memory strategies, richer than LangChain!**

```csharp
using SharpAIKit.Memory;

// 1. Buffer Memory - Keep last N messages
var buffer = new BufferMemory { MaxMessages = 20 };

// 2. Window Buffer - Keep last N conversation turns
var window = new WindowBufferMemory { WindowSize = 5 };

// 3. Summary Memory - Auto-summarize old conversations
var summary = new SummaryMemory(client) { RecentMessagesCount = 6 };

// 4. Vector Memory - Semantic search over conversation history
var vector = new VectorMemory(client) { TopK = 5 };

// 5. Entity Memory - Extract and track entities
var entity = new EntityMemory(client);

// Usage
await buffer.AddExchangeAsync("What is Python?", "Python is a programming language...");
var context = await buffer.GetContextStringAsync();
```

---

## 📝 Prompt Templates

**Type-safe template system!**

```csharp
using SharpAIKit.Prompt;

// Simple template
var template = PromptTemplate.FromTemplate("You are {role}, answer: {input}");
var prompt = template.Format(("role", "AI Assistant"), ("input", "What is C#?"));

// With partial variables
var withPartials = PromptTemplate.FromTemplate("Current time: {time}\nUser: {input}")
    .WithPartial("time", () => DateTime.Now.ToString());

// Chat template
var chatTemplate = new ChatPromptTemplate()
    .AddSystemMessage("You are a {role} assistant")
    .AddHistoryPlaceholder("history")
    .AddUserMessage("{input}");

// Few-shot template
var fewShot = new FewShotPromptTemplate(
    prefix: "Classify sentiment:",
    suffix: "Text: {input}\nSentiment:",
    exampleTemplate: "Text: {input}\nSentiment: {output}"
)
.AddExample("I love this product!", "Positive")
.AddExample("This is terrible", "Negative");
```

---

## 📤 Output Parsers

**Strongly-typed generic parsing, safer than LangChain!**

```csharp
using SharpAIKit.Output;

// JSON parsing to strongly-typed object
var jsonParser = new JsonOutputParser<ProductReview>();
var review = jsonParser.Parse(llmOutput);
Console.WriteLine(review.Rating);  // Strongly-typed access!

// Boolean parsing
var boolParser = new BooleanParser();
bool isTrue = boolParser.Parse("yes");  // true

// List parsing
var listParser = new CommaSeparatedListParser();
List<string> items = listParser.Parse("apple, banana, orange");

// XML tag parsing
var xmlParser = new XMLTagParser("answer", "reasoning");
var result = xmlParser.Parse("<answer>42</answer><reasoning>...</reasoning>");

// Get format instructions (for Prompt)
string instructions = jsonParser.GetFormatInstructions();
```

---

## 📄 Document Loaders

**Multi-format support, ready to use!**

```csharp
using SharpAIKit.DocumentLoader;

// Text file
var textLoader = new TextFileLoader("document.txt");

// CSV file (column-aware)
var csvLoader = new CsvLoader("data.csv")
{
    OneDocumentPerRow = true,
    ContentColumns = new[] { "title", "content" },
    MetadataColumns = new[] { "id", "category" }
};

// Markdown (supports header splitting)
var mdLoader = new MarkdownLoader("readme.md")
{
    SplitByHeaders = true,
    SplitHeaderLevel = 2
};

// Directory batch loading
var dirLoader = new TextDirectoryLoader("./docs", "*.txt", recursive: true);

// Web loading
var webLoader = new WebLoader("https://example.com/api/data");

// Load documents
var documents = await csvLoader.LoadAsync();
foreach (var doc in documents)
{
    Console.WriteLine($"Content: {doc.Content}");
    Console.WriteLine($"Source: {doc.Metadata["source"]}");
}
```

---

## 📊 Observability (Callback)

**Full-trace observability, essential for production!**

```csharp
using SharpAIKit.Callback;

// Console output (for debugging)
var consoleCallback = new ConsoleCallbackHandler { UseColors = true };

// Logging
var loggingCallback = new LoggingCallbackHandler(logger);

// Performance metrics
var metricsCallback = new MetricsCallbackHandler();

// File persistence
var fileCallback = new FileCallbackHandler("llm_logs.jsonl");

// Callback manager
var manager = new CallbackManager()
    .AddHandler(consoleCallback)
    .AddHandler(metricsCallback);

// View metrics
var metrics = metricsCallback.GetSummary();
Console.WriteLine($"Calls: {metrics.LLMCalls}");
Console.WriteLine($"Avg Latency: {metrics.AverageLatencyMs}ms");
Console.WriteLine($"Total Tokens: {metrics.TotalTokens}");
```

---

## 🖼️ MultiModal

**Image understanding, Vision support!**

```csharp
using SharpAIKit.MultiModal;

// Create image message from URL
var message = MultiModalMessage.User(
    "What's in this image?",
    "https://example.com/image.jpg"
);

// Fluent builder
var multiModal = new MultiModalMessageBuilder()
    .WithRole("user")
    .AddText("Compare these two images:")
    .AddImage("https://example.com/img1.jpg")
    .AddImage("https://example.com/img2.jpg")
    .Build();

// From local file (auto Base64 encoding)
var localImage = MultiModalExtensions.CreateVisionMessageFromFile(
    "Describe this image in detail",
    "./my-image.png"
);
```

---

## 🤖 Advanced Agents

**ReAct, Plan-Execute, Multi-Agent three modes!**

```csharp
using SharpAIKit.Agent;

// 1. ReAct Agent - Reasoning + Acting loop
var reactAgent = new ReActAgent(client)
    .AddTool(new CalculatorTool())
    .AddTool(new WebSearchTool());

var result = await reactAgent.RunAsync("Search for latest AI news and summarize");
// Output includes: Thought -> Action -> Observation loop

// 2. Plan-and-Execute Agent - Plan first, then execute
var planAgent = new PlanAndExecuteAgent(client)
    .AddTool(new CalculatorTool());

var planResult = await planAgent.RunAsync("Analyze this data and generate a report");
// Output includes: Plan steps + execution results for each step

// 3. Multi-Agent System - Multi-agent collaboration
var multiAgent = new MultiAgentSystem(client)
    .AddAgent("researcher", "Research Expert", "You are a professional researcher...")
    .AddAgent("writer", "Content Writer", "You are an excellent technical writer...")
    .AddAgent("reviewer", "Quality Reviewer", "You are a meticulous reviewer...");

var teamResult = await multiAgent.RunAsync("Write a technical blog about AI");
// Output includes: Task delegation + agent responses + synthesized answer
```

---

## 📚 RAG Engine

```csharp
using SharpAIKit.RAG;

var rag = new RagEngine(client);

// Index documents
await rag.IndexDocumentAsync("docs/guide.txt");
await rag.IndexContentAsync("Your document content...");
await rag.IndexDirectoryAsync("./knowledge", "*.md");

// Intelligent Q&A
var answer = await rag.AskAsync("How to use this feature?");

// Streaming answer
await foreach (var chunk in rag.AskStreamAsync("What is RAG?"))
{
    Console.Write(chunk);
}

// Retrieve only (no generation)
var docs = await rag.RetrieveAsync("relevant query", topK: 5);
```

---

## 🔮 Native C# Code Interpreter

**🎯 Killer Feature: Let agents write and execute C# code directly, no Python needed!**

### Why This Is a Killer Feature?

- **Pain Point**: LangChain's Code Interpreter depends on Python, deployment is troublesome and slow
- **Advantage**: Uses .NET's Roslyn compiler technology, executes in-memory, extremely fast
- **Effect**: Agents are no longer "language giants, math dwarfs", can write code to solve math and data processing problems

### Basic Usage

```csharp
using SharpAIKit.CodeInterpreter;

var interpreter = new RoslynCodeInterpreter();

// Math calculation
var mathCode = """
    var a = 3;
    var b = 5;
    var result = Math.Pow(a, b);
    result
    """;
var result = await interpreter.ExecuteAsync<double>(mathCode);
Console.WriteLine($"3^5 = {result}");  // Output: 243

// Fibonacci sequence
var fibCode = """
    var n = 10;
    var fib = new List<int> { 0, 1 };
    for (int i = 2; i < n; i++)
    {
        fib.Add(fib[i-1] + fib[i-2]);
    }
    string.Join(", ", fib)
    """;
var fibResult = await interpreter.ExecuteAsync(fibCode);
Console.WriteLine(fibResult.Output);  // Output: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
```

### Data Processing

```csharp
// List processing
var dataCode = """
    var numbers = new List<int> { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var evens = numbers.Where(n => n % 2 == 0).ToList();
    var sum = evens.Sum();
    var avg = evens.Average();
    $"Evens: {string.Join(", ", evens)}, Sum: {sum}, Avg: {avg}"
    """;
var dataResult = await interpreter.ExecuteAsync(dataCode);
Console.WriteLine(dataResult.Output);
```

### Integration with Agent

```csharp
using SharpAIKit.Agent;

// Create tool using Code Interpreter
public class CodeInterpreterTool : ToolBase
{
    private readonly ICodeInterpreter _interpreter;

    public CodeInterpreterTool(ICodeInterpreter interpreter)
    {
        _interpreter = interpreter;
    }

    [Tool("execute_code", "Execute C# code and return result")]
    public async Task<string> ExecuteCode(
        [Parameter("C# code to execute")] string code)
    {
        var result = await _interpreter.ExecuteAsync(code);
        return result.Success ? result.Output : $"Error: {result.Error}";
    }
}

// Use in Agent
var agent = new AiAgent(client);
agent.AddTool(new CodeInterpreterTool(interpreter));

var answer = await agent.RunAsync("Calculate the sum of squares of even numbers from 1 to 100");
// Agent will automatically write code and execute!
```

---

## 🕸️ SharpGraph

**🎯 Killer Feature: Graph-based orchestration engine using Finite State Machine (FSM), supports loops and complex branches!**

### Why This Is a Killer Feature?

- **Pain Point**: LangChain's Chain is linear (DAG), hard to handle loops (e.g., write code → run → error → fix → run again)
- **Advantage**: Implements graph orchestration based on Finite State Machine, can define States and Edges, let agents "walk" in the graph
- **Effect**: Easily build complex agents with "self-correcting" capabilities

### Basic Usage

```csharp
using SharpAIKit.Graph;

// Create a simple graph
var graph = new SharpGraphBuilder("start", maxIterations: 20)
    .Node("start", async state =>
    {
        Console.WriteLine("Starting task");
        state.Set("task", "Calculate Fibonacci");
        state.NextNode = "process";
        return state;
    })
    .Node("process", async state =>
    {
        Console.WriteLine("Processing task");
        state.Set("result", "Done");
        state.NextNode = "end";
        return state;
    })
    .Node("end", async state =>
    {
        Console.WriteLine("Task completed");
        state.Output = state.Get<string>("result");
        state.ShouldEnd = true;
        return state;
    })
    .Build();

var result = await graph.ExecuteAsync();
Console.WriteLine($"Result: {result.Output}");
```

### Loops and Conditional Branches

```csharp
// Create a "write code → run → check error → fix" loop graph
var graph = new SharpGraphBuilder("start", maxIterations: 20)
    .Node("start", async state =>
    {
        state.Set("attempts", 0);
        state.NextNode = "write_code";
        return state;
    })
    .Node("write_code", async state =>
    {
        var attempts = state.Get<int>("attempts");
        Console.WriteLine($"Attempt #{attempts + 1}: Writing code");
        state.Set("code", "var x = 10; x * 2");
        state.NextNode = "execute_code";
        return state;
    })
    .Node("execute_code", async state =>
    {
        var code = state.Get<string>("code");
        try
        {
            var result = await interpreter.ExecuteAsync(code);
            state.Set("result", result.Output);
            state.Set("error", (string?)null);
            state.NextNode = "check_result";
        }
        catch (Exception ex)
        {
            state.Set("error", ex.Message);
            state.NextNode = "fix_code";
        }
        return state;
    })
    .Node("check_result", async state =>
    {
        var result = state.Get<string>("result");
        if (!string.IsNullOrEmpty(result))
        {
            state.Output = result;
            state.ShouldEnd = true;
        }
        else
        {
            state.NextNode = "fix_code";
        }
        return state;
    })
    .Node("fix_code", async state =>
    {
        var attempts = state.Get<int>("attempts") + 1;
        state.Set("attempts", attempts);
        
        if (attempts >= 3)
        {
            state.ShouldEnd = true;
            state.Output = "Max attempts reached";
        }
        else
        {
            state.NextNode = "write_code";  // Loop back
        }
        return state;
    })
    .Build();

var finalState = await graph.ExecuteAsync();
```

---

## 🧬 DSPy-style Optimizer

**🎯 Killer Feature: Automatically optimize prompts, let the framework write prompts itself, gets smarter over time!**

### Why This Is a Killer Feature?

- **Pain Point**: LangChain's Prompts are hardcoded strings, poor effects require manual tweaking, like "alchemy"
- **Advantage**: Implements DSPy-like mechanism, define tasks and evaluation metrics, framework automatically optimizes through iterations
- **Effect**: Automatically optimize from "Help me" to "You are an expert... [Few-Shot Examples]..."

### Basic Usage

```csharp
using SharpAIKit.Optimizer;

var optimizer = new DSPyOptimizer(client)
{
    MaxIterations = 10,
    TargetScore = 0.9,
    FewShotExamples = 3
};

// Add training examples
optimizer
    .AddExample("What is C#?", "C# is an object-oriented programming language developed by Microsoft")
    .AddExample("What is Python?", "Python is an interpreted, object-oriented high-level programming language")
    .AddExample("What is Java?", "Java is a cross-platform object-oriented programming language");

// Set evaluation metric
optimizer.SetMetric(Metrics.Contains);

// Optimize prompt
var initialPrompt = "Answer questions about programming languages: {input}";
var result = await optimizer.OptimizeAsync(initialPrompt);

Console.WriteLine($"Optimized prompt:\n{result.OptimizedPrompt}");
Console.WriteLine($"Best score: {result.BestScore:F2}");
Console.WriteLine($"Iterations: {result.Iterations}");
```

### Evaluation Metrics

```csharp
// 1. Exact match
optimizer.SetMetric(Metrics.ExactMatch);

// 2. Contains match
optimizer.SetMetric(Metrics.Contains);

// 3. Semantic similarity (using embeddings)
optimizer.SetMetric(Metrics.SemanticSimilarity(client));

// 4. Custom metric
optimizer.SetMetric(Metrics.Custom(async (input, output, expected) =>
{
    var score = 0.0;
    if (output.Contains(expected)) score += 0.5;
    if (output.Length > 50) score += 0.3;
    if (output.Contains("programming language")) score += 0.2;
    return score;
}));
```

---

## 🌐 Supported Providers

| Provider | Base URL | Preset Method |
|:---------|:---------|:--------------|
| OpenAI | `https://api.openai.com/v1` | `CreateOpenAI()` |
| DeepSeek | `https://api.deepseek.com/v1` | `CreateDeepSeek()` |
| Qwen (Alibaba) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `CreateQwen()` |
| Mistral | `https://api.mistral.ai/v1` | `CreateMistral()` |
| Yi (01.AI) | `https://api.lingyiwanwu.com/v1` | `CreateYi()` |
| Groq | `https://api.groq.com/openai/v1` | `CreateGroq()` |
| Moonshot (Kimi) | `https://api.moonshot.cn/v1` | `CreateMoonshot()` |
| Zhipu GLM | `https://open.bigmodel.cn/api/paas/v4` | `CreateZhipu()` |
| Ollama (Local) | `http://localhost:11434` | `CreateOllama()` |
| **Any OpenAI-compatible** | Custom | `Create(key, url, model)` |

---

## 📁 Project Structure

```
SharpAIKit/
├── 📂 src/SharpAIKit/
│   ├── 📂 LLM/              # LLM clients
│   ├── 📂 Chain/            # Chain composition ⭐ NEW
│   ├── 📂 Memory/           # Conversation memory ⭐ NEW
│   ├── 📂 Prompt/           # Prompt templates ⭐ NEW
│   ├── 📂 Output/           # Output parsers ⭐ NEW
│   ├── 📂 DocumentLoader/   # Document loading ⭐ NEW
│   ├── 📂 Callback/         # Observability ⭐ NEW
│   ├── 📂 MultiModal/       # MultiModal ⭐ NEW
│   ├── 📂 Agent/            # Intelligent agents (ReAct/MultiAgent)
│   ├── 📂 RAG/              # RAG engine
│   ├── 📂 CodeInterpreter/  # Code interpreter 🔮 Killer Feature
│   ├── 📂 Graph/            # Graph orchestration 🕸️ Killer Feature
│   └── 📂 Optimizer/        # Auto optimization 🧬 Killer Feature
├── 📂 samples/              # Example projects
└── 📂 tests/                # Unit tests
```

---

## 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">

### 🚀 SharpAIKit - LangChain for .NET, But Better!

**🎯 Three Killer Features LangChain Doesn't Have:**
- 🔮 **Native C# Code Interpreter** - No Python needed, native C# code execution
- 🕸️ **SharpGraph** - Graph orchestration with loops and complex branches
- 🧬 **DSPy Optimizer** - Auto-optimize prompts, gets smarter over time

**Made with ❤️ for the .NET community**

If this project helps you, please give it a ⭐ **Star**!

</div>
