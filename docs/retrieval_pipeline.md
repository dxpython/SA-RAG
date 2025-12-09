# Retrieval Pipeline

Detailed documentation of the SA-RAG retrieval pipeline.

## Overview

The retrieval pipeline combines multiple retrieval methods to find the most relevant information for a user query. It supports:

- Vector similarity search (HNSW)
- Full-text search (BM25)
- Graph-based expansion (Graph-RAG)
- Long-term memory retrieval

## Pipeline Stages

### 1. Query Preprocessing

**Input**: Raw user query

**Process**:
- Query rewriting (optional, via orchestrator)
- Embedding generation
- Keyword extraction

**Output**: 
- Rewritten query
- Query embedding vector
- Query keywords

### 2. Hybrid Retrieval

#### 2.1 Vector Search (HNSW)

**Algorithm**: Hierarchical Navigable Small World

**Process**:
1. Start from top layer entry point
2. Greedy search for nearest neighbor
3. Move to lower layer
4. Repeat until bottom layer
5. Return top-k results

**Parameters**:
- `ef_search`: Search width (default: 50)
- `top_k`: Number of results

**Output**: List of (node_id, score) tuples

#### 2.2 BM25 Search

**Algorithm**: BM25 ranking function

**Process**:
1. Tokenize query
2. Stem tokens (English)
3. Lookup in inverted index
4. Calculate BM25 scores
5. Return top-k results

**Parameters**:
- `k1`: Term frequency saturation (default: 1.2)
- `b`: Length normalization (default: 0.75)

**Output**: List of (node_id, score) tuples

#### 2.3 Score Fusion

**Method**: Reciprocal Rank Fusion (RRF)

**Formula**:
```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

Where:
- `k`: RRF constant (default: 60)
- `rank_i(d)`: Rank of document d in result set i

**Process**:
1. Collect results from all methods
2. Calculate RRF scores
3. Sort by RRF score
4. Return top-k

**Output**: Fused result list

### 3. Graph Expansion (Optional)

**Trigger**: When `use_graph=True`

**Process**:
1. Take top-k results as seed nodes
2. Expand graph from seed nodes
3. Filter by edge weight and type
4. Re-rank expanded nodes

**Expansion Strategies**:

#### Basic Expansion
- Breadth-first traversal
- Fixed depth (hops)
- No filtering

#### Weighted Expansion
- Filter edges by weight threshold
- Only expand high-weight edges
- Better precision

#### Type-aware Expansion
- Prioritize specific edge types
- Boost scores for preferred types
- Better semantic relevance

#### Smart Expansion
- Combined strategy
- Depth decay
- Weight and type filtering
- Diversity control

**Output**: Expanded node list

### 4. Memory Retrieval (Optional)

**Trigger**: When `use_memory=True`

**Process**:
1. Extract keywords from query
2. Calculate memory scores:
   - Retention (Ebbinghaus curve)
   - Relevance (keyword + vector)
   - Importance
3. Sort by score
4. Return top-k memories

**Scoring Formula**:
```
score = importance * retention * (0.3 + relevance * 0.7)
retention = exp(-decay_rate * hours_passed / importance_factor)
```

**Output**: Relevant memory list

### 5. Result Fusion

**Input**: 
- Hybrid retrieval results
- Graph expansion results
- Memory retrieval results

**Methods**:

#### RRF (Reciprocal Rank Fusion)
- Default method
- Combines rankings from all sources
- Good for diverse result sets

#### Weighted Fusion
- Assign weights to each source
- Weighted sum of scores
- Good when source quality varies

#### Max Fusion
- Take maximum score across sources
- Good for high-precision scenarios

**Output**: Final ranked result list

### 6. Answer Generation

**Input**: 
- User query
- Top-k retrieved contexts

**Process**:
1. Build context string from results
2. Format with source information
3. Call LLM with query + context
4. Generate answer

**Prompt Template**:
```
Context:
[Document 1, Score: 0.85]
{context_text_1}

[Document 2, Score: 0.72]
{context_text_2}

Question: {query}

Please provide a comprehensive answer based on the context above.
```

**Output**: Generated answer text

## Configuration

### Retrieval Parameters

```python
result = rag.ask(
    query="Your question",
    top_k=5,              # Number of results
    use_graph=True,       # Enable graph expansion
    use_memory=False,     # Enable memory retrieval
)
```

### Advanced Configuration

```python
# Via RAGPipeline
pipeline = RAGPipeline()

# Custom search
results = pipeline.search(
    query="Your question",
    top_k=10,
    use_graph_expansion=True,
    graph_hops=2,         # Graph expansion depth
    use_memory=True,
)
```

## Performance Optimization

### Indexing Optimization
- Batch embedding generation
- Parallel document processing
- Incremental indexing

### Retrieval Optimization
- Early termination in HNSW search
- Result caching
- Query result limit

### Graph Optimization
- Limit expansion depth
- Filter by edge weight
- Control expansion diversity

## Example Workflow

```python
from sa_rag import RAG

# Initialize
rag = RAG()

# Index documents
doc_ids = rag.index_documents([
    "Document 1: Machine learning basics...",
    "Document 2: Deep learning architectures...",
])

# Query with full pipeline
result = rag.ask(
    query="What is the relationship between ML and DL?",
    top_k=5,
    use_graph=True,      # Use graph expansion
    use_memory=False,    # Don't use memory
)

# Result contains:
# - answer: Generated answer
# - used_semantic_nodes: Retrieved semantic nodes
# - used_graph_nodes: Graph-expanded nodes
# - scoring_details: Score breakdown
```

## Troubleshooting

### Low Retrieval Quality
- Increase `top_k` for more results
- Enable graph expansion
- Check embedding quality
- Verify document indexing

### Slow Retrieval
- Reduce `top_k`
- Disable graph expansion
- Reduce graph expansion depth
- Use smaller embedding dimensions

### Memory Issues
- Reduce batch size
- Limit graph expansion
- Prune old memories
- Use smaller indexes

