# SA-AgentOS: Intelligent Agent System

**SA-AgentOS** is an intelligent agent system that serves as the "knowledge brain" for AI agents, built on top of **SA-RAG Core**, **DeepSeek API**, and **SharpAIKit** (C# Agent framework).

## 🎯 Overview

SA-AgentOS provides a complete agent system with:

- **Planning**: Intelligent decision-making about when to retrieve knowledge
- **World Model (Memory)**: Long-term memory management using SA-RAG's cognitive memory system
- **Knowledge Retrieval**: Seamless integration with SA-RAG Core for multi-stage, graph-enhanced retrieval
- **Execution Graph**: Visual representation of agent reasoning and knowledge retrieval flow
- **Self-Correction**: Built-in debugging and reflection capabilities

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    C# Agent Layer                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Agent   │  │ Planner  │  │  Memory  │  │ Debugger │   │
│  │ Pipeline │  │          │  │  Tool    │  │          │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │              │              │          │
│       └─────────────┴──────────────┴──────────────┘          │
│                         │                                     │
│              ┌──────────┴──────────┐                         │
│              │      Tools          │                         │
│              │  ┌──────────────┐  │                         │
│              │  │ Knowledge    │  │                         │
│              │  │ Retrieval    │  │                         │
│              │  └──────┬───────┘  │                         │
│              │  ┌──────┴───────┐  │                         │
│              │  │ DeepSeek     │  │                         │
│              │  │ Reasoning    │  │                         │
│              │  └──────────────┘  │                         │
└──────────────┼────────────────────┼─────────────────────────┘
               │ HTTP/gRPC          │
               ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│              Python HTTP Service                             │
│              (FastAPI)                                       │
│              POST /rag/query                                 │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              SA-RAG Core                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Multi-   │  │ Graph-   │  │ Memory   │  │ Execution│   │
│  │ Stage    │  │ RAG      │  │ Store    │  │ Graph    │   │
│  │ Retrieval│  │          │  │          │  │          │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
sa_agentos/
├── README.md                    # This file
├── agent_csharp/                # C# Agent project
│   └── SaAgentOS/
│       ├── SaAgentOS.csproj
│       ├── Program.cs           # Main entry point
│       ├── Config/
│       │   └── appsettings.json # Configuration
│       ├── Agents/
│       │   └── SaRagAgent.cs    # Core agent class
│       ├── Tools/
│       │   ├── KnowledgeRetrievalTool.cs
│       │   ├── DeepSeekReasoningTool.cs
│       │   └── MemoryTool.cs
│       ├── Pipelines/
│       │   └── SaAgentPipeline.cs
│       ├── Models/
│       │   ├── AgentConfig.cs
│       │   ├── RetrievalResult.cs
│       │   └── ReasoningTrace.cs
│       ├── Logging/
│       │   └── AgentLogger.cs
│       ├── Execution/
│       │   └── ExecutionGraphExporter.cs
│       └── Tests/
│           └── AgentTests.cs
└── python_server/               # Python HTTP service
    ├── server.py                # FastAPI server
    ├── requirements.txt
    └── tests/
        └── test_rag_endpoint.py
```

## 🚀 Quick Start

### Prerequisites

- .NET 8.0 SDK
- Python 3.10+ with `uv` package manager
- DeepSeek API key (set as `DEEPSEEK_API_KEY` environment variable)

### Step 1: Start SA-RAG Python Service

```bash
cd sa_agentos/python_server

# Install dependencies (using uv)
uv sync

# Or using pip
pip install -r requirements.txt

# Start the server
uv run python server.py
# Or: python server.py
```

The server will start on `http://localhost:8000` by default.

### Step 2: Configure DeepSeek API Key

Set the environment variable:

```bash
export DEEPSEEK_API_KEY="sk-e164311ef7914e46a5d760c505714b94"
```

Or on Windows:

```cmd
set DEEPSEEK_API_KEY=sk-e164311ef7914e46a5d760c505714b94
```

**Note**: The API key is configured in the example above. Replace it with your own key if needed.

### Step 3: Build and Run C# Agent

```bash
cd sa_agentos/agent_csharp/SaAgentOS

# Restore dependencies
dotnet restore

# Build
dotnet build

# Run
dotnet run
```

### Step 4: Use the Agent

In interactive mode, you can ask questions:

```
> What is Python?
> How does machine learning work?
> export trace.json
> exit
```

## 🔧 Configuration

### Python Server Configuration

The Python server uses environment variables:

- `PORT`: Server port (default: 8000)

### C# Agent Configuration

Edit `Config/appsettings.json`:

```json
{
  "DeepSeek": {
    "ApiKey": "ENV:DEEPSEEK_API_KEY",
    "BaseUrl": "https://api.deepseek.com",
    "Model": "deepseek-chat"
  },
  "SaRag": {
    "Endpoint": "http://localhost:8000/rag/query",
    "UseGraph": true,
    "UseMemory": true,
    "TopK": 6
  },
  "Memory": {
    "StoragePath": "./agent_memory.json",
    "UseSaRagMemory": true
  }
}
```

**Note**: Values starting with `ENV:` will be resolved from environment variables.

## 🧪 Testing

### Python Server Tests

```bash
cd sa_agentos/python_server
uv run pytest tests/test_rag_endpoint.py -v
```

### C# Agent Tests

```bash
cd sa_agentos/agent_csharp/SaAgentOS
dotnet test
```

## 📊 Agent Pipeline Flow

1. **Planning**: Agent analyzes the query to determine if knowledge retrieval is needed
2. **Knowledge Retrieval**: If needed, queries SA-RAG with graph expansion and memory
3. **Memory Retrieval**: Retrieves relevant memories from long-term storage
4. **Reasoning**: Uses DeepSeek API to generate answer based on retrieved context
5. **Memory Storage**: Stores important information for future use
6. **Execution Graph**: Records the complete execution trace for visualization

## 🔌 API Endpoints

### POST /rag/query

Query the SA-RAG knowledge base.

**Request:**
```json
{
  "query": "What is Python?",
  "use_graph": true,
  "use_memory": true,
  "top_k": 6
}
```

**Response:**
```json
{
  "answer": "Python is a high-level programming language...",
  "nodes": [
    {
      "id": 1,
      "text": "Python is a programming language...",
      "score": 0.95,
      "source": "search",
      "node_type": "text"
    }
  ],
  "execution_graph": {
    "query": "What is Python?",
    "nodes": [...],
    "edges": [...],
    "execution_trace": [...],
    "total_time_ms": 45.2
  },
  "query": "What is Python?",
  "top_k": 6
}
```

### POST /rag/memory

Add memory to the knowledge base.

**Request:**
```
POST /rag/memory?text=User prefers Python&importance=0.8
```

## 🛠️ Integration with SharpAIKit

When SharpAIKit is available, `SaRagAgent` can be easily integrated:

```csharp
// Example integration (when SharpAIKit is available)
public class SaRagAgent : AgentBase  // or : IAgent
{
    // Implementation would inherit from SharpAIKit's base classes
    // Tools would implement ITool interface
}
```

The current implementation is designed to be compatible with SharpAIKit's architecture while remaining functional as a standalone system.

## 📝 Execution Graph Export

Export execution graphs for visualization:

```bash
# In interactive mode
> What is machine learning?
> export trace.json

# Or programmatically
var exporter = new ExecutionGraphExporter();
await exporter.SaveToFileAsync(trace, "trace.json", "json");
await exporter.SaveToFileAsync(trace, "trace.dot", "dot");
```

## 🔍 Troubleshooting

### SA-RAG Service Not Available

- Ensure the Python server is running: `uv run python server.py`
- Check the endpoint in `appsettings.json` matches the server URL
- Verify SA-RAG Core is properly installed

### DeepSeek API Errors

- Verify `DEEPSEEK_API_KEY` environment variable is set
- Check API key is valid and has sufficient credits
- Ensure network connectivity to `api.deepseek.com`

### Memory Not Persisting

- Check file permissions for `agent_memory.json`
- Verify `StoragePath` in configuration is writable
- If using SA-RAG memory, ensure the Python service is running

## 📚 Dependencies

### Python Server
- FastAPI
- Uvicorn
- SA-RAG Core (from parent directory)

### C# Agent
- .NET 8.0
- Microsoft.Extensions.* (Configuration, Logging)
- System.Text.Json
- Newtonsoft.Json
- xUnit (for tests)

## 🤝 Contributing

This is part of the SA-RAG ecosystem. Contributions should maintain compatibility with SA-RAG Core and follow the existing code style.

## 📄 License

Same as SA-RAG Core project.

