# SharpAIKit

A unified .NET large-model application and agentic AI development framework.

## Overview

SharpAIKit is a comprehensive AI/LLM toolkit for .NET that provides a unified interface for working with large language models, building AI agents, and implementing RAG (Retrieval-Augmented Generation) applications. It's designed to be more powerful and simpler than LangChain, with killer features that leverage the .NET ecosystem.

## Key Features

### Core Capabilities
- **Unified LLM Interface** - One API for all OpenAI-compatible models
- **LCEL-style Chains** - Elegant chain composition with pipe operators
- **Multiple Memory Strategies** - Buffer, Window, Summary, Vector, and Entity memory
- **Advanced Agents** - ReAct, Plan-and-Execute, and Multi-Agent systems
- **RAG Engine** - Built-in document indexing and intelligent Q&A
- **Full Observability** - Console, logging, metrics, and file tracing

### Killer Features Beyond LangChain
- **🔮 Native C# Code Interpreter** - Execute C# code directly using Roslyn, no Python needed
- **🕸️ SharpGraph** - Graph-based orchestration with FSM, supports loops and complex branches
- **🧬 DSPy-style Optimizer** - Automatically optimize prompts through iterative improvement

## Installation

```bash
dotnet add package SharpAIKit
```

## Quick Start

```csharp
using SharpAIKit.LLM;

// Create a client for any OpenAI-compatible API
var client = LLMClientFactory.Create(
    apiKey: "your-api-key",
    baseUrl: "https://api.deepseek.com/v1",
    model: "deepseek-chat"
);

// Simple chat
var response = await client.ChatAsync("Hello!");
Console.WriteLine(response);

// Streaming output
await foreach (var chunk in client.ChatStreamAsync("Tell me a story"))
{
    Console.Write(chunk);
}
```

## Supported LLM Providers

SharpAIKit supports all OpenAI-compatible APIs, including:

- OpenAI
- DeepSeek
- Qwen (Alibaba)
- Mistral
- Yi (01.AI)
- Groq
- Moonshot (Kimi)
- Zhipu GLM
- Ollama (Local)
- Any custom OpenAI-compatible endpoint

## Example: Code Interpreter

```csharp
using SharpAIKit.CodeInterpreter;

var interpreter = new RoslynCodeInterpreter();
var result = await interpreter.ExecuteAsync<double>("Math.Pow(3, 5)");
Console.WriteLine($"3^5 = {result}");  // Output: 243
```

## Example: Graph Orchestration

```csharp
using SharpAIKit.Graph;

var graph = new SharpGraphBuilder("start")
    .Node("start", async state => {
        state.NextNode = "process";
        return state;
    })
    .Node("process", async state => {
        state.Output = "Done";
        state.ShouldEnd = true;
        return state;
    })
    .Build();

var result = await graph.ExecuteAsync();
```

## Example: Prompt Optimization

```csharp
using SharpAIKit.Optimizer;

var optimizer = new DSPyOptimizer(client)
    .AddExample("What is C#?", "C# is an object-oriented programming language...")
    .SetMetric(Metrics.Contains);

var result = await optimizer.OptimizeAsync("Answer: {input}");
Console.WriteLine(result.OptimizedPrompt);
```

## Documentation

For detailed documentation and examples, visit:
- [GitHub Repository](https://github.com/dxpython/SharpAIKit)
- [English Documentation](https://github.com/dxpython/SharpAIKit/blob/main/README_EN.md)
- [中文文档](https://github.com/dxpython/SharpAIKit/blob/main/README_CN.md)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/dxpython/SharpAIKit/blob/main/LICENSE) file for details.


