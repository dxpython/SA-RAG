# Release Notes

## Version 0.1.0 - Initial Release

### Overview

This is the initial release of SharpAIKit, a unified .NET large-model application and agentic AI development framework. SharpAIKit provides a comprehensive toolkit for building AI applications with large language models, featuring killer capabilities that go beyond what LangChain offers.

### Major Features

#### Core Framework
- **Unified LLM Interface** - Single API supporting all OpenAI-compatible models
- **Streaming Support** - Real-time token-by-token output streaming
- **Async/Await** - Full async support for .NET 8
- **Retry Policy** - Built-in Polly retry mechanism for resilient API calls

#### Chain Composition (LCEL-style)
- **Pipe Operators** - Elegant chain composition using `|` operator
- **Parallel Execution** - Run multiple chains in parallel
- **Conditional Branching** - Dynamic execution paths based on conditions
- **Lambda Chains** - Custom chain logic with lambda functions

#### Memory Management
- **Buffer Memory** - Simple last N messages storage
- **Window Buffer Memory** - Keep last N conversation turns
- **Summary Memory** - Automatically summarize old conversations
- **Vector Memory** - Semantic search over conversation history
- **Entity Memory** - Extract and track entities from conversations

#### Prompt Engineering
- **Prompt Templates** - Variable substitution with type safety
- **Chat Prompt Templates** - Structured message templates
- **Few-shot Templates** - Built-in few-shot learning support
- **Partial Variables** - Pre-filled template variables

#### Output Parsing
- **JSON Parser** - Strongly-typed JSON parsing with generics
- **Boolean Parser** - Smart boolean value extraction
- **List Parsers** - Comma-separated and numbered list parsing
- **XML Parser** - Extract structured data from XML tags
- **Regex Parser** - Custom pattern-based parsing

#### Document Loading
- **Text Loader** - Plain text file loading
- **CSV Loader** - Column-aware CSV processing
- **JSON Loader** - JSON file parsing
- **Markdown Loader** - Markdown with header splitting
- **Web Loader** - Load content from URLs
- **Directory Loader** - Batch loading from directories

#### Observability
- **Console Callback** - Colored console output for debugging
- **Logging Callback** - Integration with Microsoft.Extensions.Logging
- **Metrics Callback** - Performance metrics collection
- **File Callback** - Persistent logging to files
- **Callback Manager** - Multi-handler callback system

#### MultiModal Support
- **Image Content** - Support for image URLs and local files
- **Base64 Encoding** - Automatic image encoding
- **MultiModal Messages** - Combine text and images in messages
- **Fluent Builder** - Elegant API for building multimodal messages

#### Advanced Agents
- **ReAct Agent** - Reasoning + Acting loop for complex problem solving
- **Plan-and-Execute Agent** - Create plans before execution
- **Multi-Agent System** - Coordinate multiple specialized agents
- **Tool Integration** - Easy tool registration and execution

#### RAG Engine
- **Document Indexing** - Index documents with embeddings
- **Vector Search** - Semantic similarity search
- **Intelligent Q&A** - Context-aware question answering
- **Streaming Answers** - Real-time answer generation
- **Multiple Vector Stores** - In-memory and file-based storage

### Killer Features

#### 🔮 Native C# Code Interpreter
- **Roslyn Integration** - Direct C# code execution using Roslyn compiler
- **No Python Dependency** - Pure .NET solution, no external processes
- **High Performance** - Native compilation, 10-100x faster than Python
- **Type Safety** - Full C# type system support
- **Sandbox Execution** - Configurable execution limits and security

#### 🕸️ SharpGraph
- **Finite State Machine** - Graph-based orchestration engine
- **Loop Support** - Handle cycles and retry logic
- **Conditional Branches** - Dynamic execution paths
- **State Management** - Complete state passing and management
- **Visualization** - GraphViz format export

#### 🧬 DSPy-style Optimizer
- **Automatic Optimization** - Iterative prompt improvement
- **Few-shot Learning** - Auto-generate few-shot examples
- **Multiple Metrics** - Exact match, contains, semantic similarity
- **Custom Metrics** - Define your own evaluation functions
- **Optimization History** - Track improvement over iterations

### Supported LLM Providers

- OpenAI (GPT-4, GPT-3.5, o1, o3)
- DeepSeek (DeepSeek-V3, R1, Coder)
- Qwen (Tongyi Qianwen)
- Mistral AI
- Yi (01.AI)
- Groq
- Moonshot (Kimi)
- Zhipu GLM
- Baichuan
- Together AI
- Ollama (Local models)
- Any OpenAI-compatible API

### Technical Details

- **Target Framework**: .NET 8.0
- **Dependencies**: Minimal external dependencies
- **License**: MIT
- **Documentation**: Comprehensive XML documentation included

### Getting Started

```bash
dotnet add package SharpAIKit
```

```csharp
using SharpAIKit.LLM;

var client = LLMClientFactory.Create("api-key", "https://api.deepseek.com/v1", "deepseek-chat");
var response = await client.ChatAsync("Hello!");
```

### Documentation

- [GitHub Repository](https://github.com/dxpython/SharpAIKit)
- [English Documentation](https://github.com/dxpython/SharpAIKit/blob/main/README_EN.md)
- [中文文档](https://github.com/dxpython/SharpAIKit/blob/main/README_CN.md)

### Breaking Changes

None - This is the initial release.

### Known Issues

None at this time.

### Future Roadmap

- Additional vector store backends (Pinecone, Weaviate, etc.)
- More document loaders (PDF, Word, etc.)
- Enhanced multi-agent coordination
- Additional optimization strategies
- Performance improvements

---

**Thank you for using SharpAIKit!**


