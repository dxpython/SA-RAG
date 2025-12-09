# API Reference

Complete API documentation for SA-RAG.

## Python API

### `sa_rag.RAG`

Main entry point for SA-RAG functionality.

#### `__init__(llm_provider="mock", embedding_provider="mock", **kwargs)`

Initialize the RAG system.

**Parameters:**
- `llm_provider` (str): LLM provider ("openai", "deepseek", "local", "mock")
- `embedding_provider` (str): Embedding provider ("openai", "deepseek", "local", "mock")
- `**kwargs`: Additional configuration parameters

**Example:**
```python
rag = RAG(
    llm_provider="openai",
    embedding_provider="openai"
)
```

#### `index_documents(texts: List[str], generate_embeddings: bool = True) -> List[int]`

Index multiple documents.

**Parameters:**
- `texts` (List[str]): List of document texts
- `generate_embeddings` (bool): Whether to generate embeddings

**Returns:**
- `List[int]`: List of document IDs

**Example:**
```python
doc_ids = rag.index_documents([
    "Document 1 text...",
    "Document 2 text...",
], generate_embeddings=True)
```

#### `ask(query: str, top_k: int = 5, use_graph: bool = True, use_memory: bool = False) -> dict`

Ask a question and get an answer.

**Parameters:**
- `query` (str): User query
- `top_k` (int): Number of top results to retrieve
- `use_graph` (bool): Whether to use graph expansion
- `use_memory` (bool): Whether to use long-term memory

**Returns:**
- `dict`: Dictionary containing:
  - `answer` (str): Generated answer
  - `used_semantic_nodes` (List[dict]): Semantic nodes used
  - `used_graph_nodes` (List[dict]): Graph nodes used
  - `used_memory_nodes` (List[dict]): Memory nodes used
  - `scoring_details` (dict): Scoring information
  - `raw_results` (List[dict]): Raw retrieval results

**Example:**
```python
result = rag.ask(
    query="What is the main topic?",
    top_k=5,
    use_graph=True
)
print(result['answer'])
```

#### `search(query: str, top_k: int = 5, use_graph: bool = True, use_memory: bool = False) -> List[dict]`

Search for relevant documents without generating an answer.

**Parameters:**
- `query` (str): Query text
- `top_k` (int): Number of results to return
- `use_graph` (bool): Whether to use graph expansion
- `use_memory` (bool): Whether to use long-term memory

**Returns:**
- `List[dict]`: List of search results

#### `add_memory(text: str, importance: float = 0.5)`

Add a long-term memory.

**Parameters:**
- `text` (str): Memory content
- `importance` (float): Importance score (0.0-1.0)

**Example:**
```python
rag.add_memory("User prefers detailed explanations", importance=0.8)
```

#### `update_document(doc_id: int, new_text: str)`

Update a document using differential indexing.

**Parameters:**
- `doc_id` (int): Document ID
- `new_text` (str): New document text

### `sa_rag.Client`

Lower-level client API for more control.

#### `index(texts: Union[str, List[str]], generate_embeddings: bool = True) -> List[int]`

Index documents.

#### `search(query: str, top_k: int = 5, use_graph: bool = True, use_memory: bool = False) -> List[dict]`

Search for documents.

#### `ask(query: str, top_k: int = 5, use_graph: bool = True, use_memory: bool = False) -> dict`

Ask a question.

#### `get_stats() -> dict`

Get system statistics.

### `sa_rag.RAGPipeline`

Core RAG pipeline implementation.

#### `index_documents(texts: List[str], generate_embeddings: bool = True, batch_size: int = 32) -> List[int]`

Index documents with embedding generation.

#### `search(query: str, top_k: int = 3, use_graph_expansion: bool = False, graph_hops: int = 1, use_memory: bool = False) -> List[dict]`

Search for relevant documents.

#### `ask(query: str, top_k: int = 5, use_graph: bool = True, use_memory: bool = False) -> dict`

Complete Q&A workflow.

#### `generate_answer(query: str, context_nodes: List[dict], include_sources: bool = False) -> str`

Generate answer from context.

## Rust API (via PyO3)

### `rust_core.RustCoreEngine`

Rust core engine exposed via PyO3.

#### `index_documents(texts: List[str]) -> List[int]`

Index documents and return document IDs.

#### `search(query: str, top_k: int, query_vector: Optional[List[float]] = None) -> List[Tuple[str, float]]`

Search for relevant documents.

**Returns:**
- `List[Tuple[str, float]]`: List of (text, score) tuples

#### `search_full(query: str, top_k: int, query_vector: Optional[List[float]] = None) -> List[Tuple[int, str, float]]`

Full search returning node IDs.

**Returns:**
- `List[Tuple[int, str, float]]`: List of (node_id, text, score) tuples

#### `update_embeddings(node_ids: List[int], embeddings: List[List[float]])`

Update embeddings for nodes.

#### `get_node_ids_for_doc(doc_id: int) -> List[int]`

Get all node IDs for a document.

#### `get_node_info(node_id: int) -> Optional[dict]`

Get node information.

**Returns:**
- `dict`: Node metadata including text, level, node_type, etc.

#### `get_node_text(node_id: int) -> Optional[str]`

Get node text content.

#### `get_node_metadata(node_id: int) -> Optional[dict]`

Get node metadata.

#### `expand_nodes(node_ids: List[int], hops: int) -> List[int]`

Expand nodes in the graph.

**Parameters:**
- `node_ids` (List[int]): Seed node IDs
- `hops` (int): Number of hops for expansion

**Returns:**
- `List[int]`: Expanded node IDs

#### `expand_nodes_smart(node_ids: List[int], hops: int, min_weight: float, max_nodes: int) -> List[int]`

Smart graph expansion with filtering.

**Parameters:**
- `node_ids` (List[int]): Seed node IDs
- `hops` (int): Number of hops
- `min_weight` (float): Minimum edge weight threshold
- `max_nodes` (int): Maximum number of nodes

**Returns:**
- `List[int]`: Expanded node IDs

#### `add_memory(text: str, importance: float)`

Add a long-term memory.

#### `search_with_memory(query: str, top_k: int = 3) -> List[str]`

Search long-term memory.

#### `update_document(doc_id: int, new_text: str)`

Update a document using differential indexing.

#### `get_graph_stats() -> dict`

Get graph statistics.

**Returns:**
- `dict`: Dictionary with `num_nodes` and `num_edges`

## Embedding Service

### `sa_rag.EmbeddingService`

Embedding vector generation service.

#### `__init__(provider="mock", model_name=None, api_key=None, dimension=128)`

Initialize embedding service.

#### `get_embedding(text: str) -> List[float]`

Get embedding for a single text.

#### `get_embeddings_batch(texts: List[str]) -> List[List[float]]`

Get embeddings for multiple texts.

## LLM Service

### `sa_rag.LLMService`

Large Language Model service.

#### `__init__(provider="mock", model_name=None, api_key=None, base_url=None, temperature=0.7)`

Initialize LLM service.

#### `chat_completion(prompt: str, system_prompt: str = "...", context: Optional[List[dict]] = None, max_tokens: Optional[int] = None) -> str`

Generate chat completion.

#### `generate_with_rag(query: str, retrieved_context: List[dict], system_prompt: Optional[str] = None) -> str`

Generate answer with RAG context.

## Orchestrator

### `sa_rag.Orchestrator`

Query orchestration and result fusion.

#### `rewrite_query(query: str, use_llm: bool = False) -> str`

Rewrite query for better retrieval.

#### `plan_retrieval(query: str, available_methods: Optional[List[str]] = None) -> dict`

Plan retrieval strategy.

#### `fuse_results(results_list: List[List[dict]], method: str = "rrf", top_k: int = 5) -> List[dict]`

Fuse results from multiple retrieval methods.

**Methods:**
- `"rrf"`: Reciprocal Rank Fusion
- `"weighted"`: Weighted fusion
- `"max"`: Maximum score fusion
