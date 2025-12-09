# SA-RAG Python Package

Python orchestration layer for SA-RAG, providing high-level APIs for RAG operations.

## Installation

```bash
# From project root
cd python
pip install -e .

# Or using maturin to build Rust core
maturin develop --release
```

## Quick Start

```python
from sa_rag import RAG

# Initialize
rag = RAG(
    llm_provider="mock",
    embedding_provider="mock"
)

# Index documents
doc_ids = rag.index_documents([
    "Document 1 text...",
    "Document 2 text...",
], generate_embeddings=True)

# Query
result = rag.ask("Your question here", top_k=5)
print(result['answer'])
```

## API Overview

### RAG Class

Main entry point for SA-RAG functionality.

#### Methods

- `index_documents(texts, generate_embeddings=True)`: Index multiple documents
- `ask(query, top_k=5, use_graph=True, use_memory=False)`: Ask a question
- `search(query, top_k=5, use_graph=True, use_memory=False)`: Search only
- `add_memory(text, importance=0.5)`: Add long-term memory
- `update_document(doc_id, new_text)`: Update a document

### Client Class

Lower-level client API for more control.

### RAGPipeline Class

Core RAG pipeline implementation.

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: For OpenAI LLM/embeddings
- `DEEPSEEK_API_KEY`: For DeepSeek LLM/embeddings

### Provider Options

- **LLM**: `"openai"`, `"deepseek"`, `"local"`, `"mock"`
- **Embedding**: `"openai"`, `"deepseek"`, `"local"`, `"mock"`

## Examples

See `examples/` directory for complete examples.
