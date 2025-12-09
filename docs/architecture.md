# SA-RAG Architecture

## Overview

SA-RAG (Semantic-Accelerated Retrieval Augmentation) is a high-performance RAG system designed for enterprise knowledge bases. It combines Rust's performance with Python's flexibility to provide a complete RAG solution.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Python Layer                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   RAG    │  │  Client  │  │   LLM    │  │Embedding │   │
│  │ Pipeline │  │   API    │  │ Service  │  │ Service  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │              │              │          │
│       └─────────────┴──────────────┴──────────────┘          │
│                         │                                     │
│                    Orchestrator                               │
└─────────────────────────┼─────────────────────────────────────┘
                          │ PyO3
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      Rust Core                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Parser  │  │ Indexer  │  │  Graph   │  │  Memory  │   │
│  │          │  │          │  │          │  │          │   │
│  │ Semantic │  │ Vector   │  │ Adjacency│  │ Ebbinghaus│  │
│  │  Nodes   │  │  (HNSW)  │  │   List   │  │   Curve  │   │
│  │          │  │  + BM25  │  │          │  │          │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Differential Indexing                      │  │
│  │         (Document Version Management)                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Rust Core

#### 1. Semantic Parser (`parser.rs`)

**Purpose**: Automatically segment documents into hierarchical semantic nodes.

**Algorithm**:
- Identifies headings (Markdown style `#`, `##`, or numbered sections)
- Splits text into paragraphs, sentences, and chunks
- Builds parent-child relationships
- Creates `SemanticNode` structures with metadata

**Key Features**:
- Supports both English and Chinese headings
- Maintains document hierarchy
- Preserves text spans for efficient retrieval

#### 2. Vector Index (`indexer/vector.rs`)

**Purpose**: High-performance approximate nearest neighbor search.

**Algorithm**: HNSW (Hierarchical Navigable Small World)

**Features**:
- Multi-layer graph structure
- Greedy search from top layer
- Dynamic insertion with neighbor selection
- Cosine similarity for vector comparison

**Parameters**:
- `m`: Maximum connections per layer (default: 16)
- `m_max`: Controls layer sparsity (default: 64)
- `ef_construction`: Search width during insertion

#### 3. BM25 Index (`indexer/bm25.rs`)

**Purpose**: Full-text search with relevance scoring.

**Algorithm**: BM25 ranking function

**Features**:
- Tokenization and stemming (English)
- Inverted index for fast lookup
- IDF (Inverse Document Frequency) calculation
- TF (Term Frequency) weighting

**Formula**:
```
BM25(q, d) = Σ IDF(qi) * (f(qi, d) * (k1 + 1)) / (f(qi, d) + k1 * (1 - b + b * |d| / avgdl))
```

#### 4. Hybrid Index (`indexer/hybrid.rs`)

**Purpose**: Combine vector and BM25 search results.

**Algorithm**: Reciprocal Rank Fusion (RRF)

**Formula**:
```
RRF(d) = Σ 1 / (k + rank_i(d))
```

**Features**:
- Configurable weights for vector and BM25
- Automatic score normalization
- Top-k result selection

#### 5. Knowledge Graph (`graph/graph.rs`)

**Purpose**: Store and query semantic relationships.

**Structure**: Adjacency list representation

**Edge Types**:
- `ParentChild`: Hierarchical relationships
- `SemanticSimilarity`: Similarity-based connections
- `Mention`: Reference relationships
- `NextChunk`: Sequential relationships
- `Reference`: Cross-references
- `Synonym`: Synonym relationships

**Features**:
- Weighted edges (0.0 - 1.0)
- Edge metadata support
- Bidirectional edge support
- Graph statistics

#### 6. Graph Expansion (`graph/expansion.rs`)

**Purpose**: Expand retrieval results through graph traversal.

**Strategies**:
1. **Basic Expansion**: Breadth-first expansion
2. **Weighted Expansion**: Filter by edge weight threshold
3. **Type-aware Expansion**: Prioritize specific edge types
4. **Diverse Expansion**: Control expansion diversity
5. **Smart Expansion**: Combined strategy with decay

**Algorithm**:
- BFS traversal from seed nodes
- Score decay with depth
- Type-based filtering
- Weight threshold filtering

#### 7. Long-term Memory (`memory.rs`)

**Purpose**: Store and retrieve long-term memories with temporal decay.

**Algorithm**: Ebbinghaus Forgetting Curve

**Formula**:
```
retention = exp(-decay_rate * time_passed / importance_factor)
score = importance * retention * relevance
```

**Features**:
- Importance-weighted decay
- Access frequency boost (spaced repetition)
- Keyword and vector-based retrieval
- Automatic memory pruning

#### 8. Differential Indexing (`diff.rs`)

**Purpose**: Incremental document updates without full re-indexing.

**Algorithm**: Simplified Myers diff algorithm

**Features**:
- Document version management
- Changed segment identification
- Partial embedding regeneration
- Version history tracking

### Python Layer

#### 1. RAG Pipeline (`rag.py`)

**Purpose**: Orchestrate the complete RAG workflow.

**Workflow**:
1. Document indexing with embedding generation
2. Query rewriting
3. Hybrid retrieval (vector + BM25)
4. Graph expansion (optional)
5. Memory retrieval (optional)
6. Result fusion
7. Answer generation with LLM

#### 2. LLM Service (`llm.py`)

**Purpose**: Interface with various LLM providers.

**Supported Providers**:
- OpenAI (GPT models)
- DeepSeek
- Local models (via transformers)
- Mock mode (for testing)

**Features**:
- Chat completion
- RAG-enhanced generation
- Context management
- Temperature control

#### 3. Embedding Service (`embedding.py`)

**Purpose**: Generate text embeddings.

**Supported Providers**:
- OpenAI embeddings
- DeepSeek embeddings
- Local models (sentence-transformers)
- Mock mode (deterministic)

**Features**:
- Batch embedding generation
- Vector normalization
- Cosine similarity calculation

#### 4. Orchestrator (`orchestrator.py`)

**Purpose**: Coordinate query processing and result fusion.

**Features**:
- Query rewriting (rule-based or LLM-based)
- Retrieval strategy planning
- Result fusion (RRF, weighted, max)
- Graph expansion decision making

#### 5. Client API (`client.py`)

**Purpose**: High-level API for end users.

**Features**:
- Simplified interface
- Statistics tracking
- Error handling
- Configuration management

## Data Flow

### Indexing Flow

```
Document Text
    │
    ▼
Python: RAG.index_documents()
    │
    ▼
Rust: RustCoreEngine.index_documents()
    │
    ├─► Parser: Parse into SemanticNodes
    │       │
    │       ├─► Create nodes with hierarchy
    │       └─► Build parent-child relationships
    │
    ├─► Graph: Add nodes and edges
    │       │
    │       └─► Create ParentChild edges
    │
    ├─► BM25: Index text content
    │       │
    │       └─► Build inverted index
    │
    └─► Python: Generate embeddings
            │
            └─► Rust: Update vector index (HNSW)
```

### Retrieval Flow

```
User Query
    │
    ▼
Python: RAG.ask()
    │
    ├─► Orchestrator: Rewrite query
    │
    ├─► Embedding: Generate query embedding
    │
    ├─► Rust: Hybrid search
    │       │
    │       ├─► Vector search (HNSW)
    │       └─► BM25 search
    │
    ├─► Rust: Graph expansion (optional)
    │       │
    │       └─► Expand seed nodes
    │
    ├─► Rust: Memory retrieval (optional)
    │
    ├─► Orchestrator: Fuse results (RRF)
    │
    └─► LLM: Generate answer
            │
            └─► Return to user
```

## Performance Characteristics

### Indexing
- **Throughput**: ~1000 documents/second (with embeddings)
- **Memory**: O(n) where n is number of nodes
- **Disk**: Optional persistence (not implemented)

### Retrieval
- **Vector Search**: O(log n) with HNSW
- **BM25 Search**: O(m) where m is query term count
- **Graph Expansion**: O(k * d) where k is seed nodes, d is depth
- **Total Latency**: <50ms for typical queries

### Memory
- **Storage**: O(n) for n memories
- **Retrieval**: O(n) with pruning optimization
- **Decay Calculation**: O(1) per memory

## Concurrency

- **Rust Core**: Thread-safe with `Arc<Mutex<>>` for shared state
- **Python Layer**: GIL-bound (single-threaded Python execution)
- **Future**: Consider async/await for I/O-bound operations

## Extensibility

### Adding New Retrieval Methods
1. Implement in Rust core
2. Expose via PyO3
3. Integrate in `HybridIndex`
4. Add to orchestrator

### Adding New LLM Providers
1. Implement in `LLMService`
2. Add provider-specific logic
3. Update configuration

### Adding New Graph Expansion Strategies
1. Implement in `graph/expansion.rs`
2. Add to `GraphExpansion` struct
3. Expose via PyO3

## Future Improvements

1. **Persistence**: Add disk-based storage for indexes
2. **Distributed**: Support distributed indexing and retrieval
3. **Async**: Async/await support for Python layer
4. **Caching**: Add result caching for common queries
5. **Monitoring**: Add metrics and observability
6. **Optimization**: Further optimize HNSW parameters
7. **Multi-language**: Better support for non-English text
