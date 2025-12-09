# Semantic Node Design

Documentation of the semantic node structure and parsing algorithm.

## Overview

Semantic nodes represent hierarchical chunks of text with relationships. They form the foundation of SA-RAG's retrieval system, enabling both fine-grained and hierarchical retrieval.

## Node Structure

### SemanticNode

```rust
pub struct SemanticNode {
    pub node_id: u64,           // Unique node identifier
    pub doc_id: u64,            // Parent document ID
    pub span_start: usize,      // Start position in document
    pub span_end: usize,        // End position in document
    pub level: u8,              // Hierarchy level (0=root, 1=heading, etc.)
    pub node_type: NodeType,    // Node type
    pub text: String,           // Node text content
    pub embedding: Option<Vec<f32>>,  // Vector embedding
    pub parents: Vec<u64>,      // Parent node IDs
    pub children: Vec<u64>,     // Child node IDs
}
```

### Node Types

```rust
pub enum NodeType {
    Root,        // Document root
    Heading,     // Section heading
    Paragraph,   // Paragraph
    Sentence,    // Sentence
    Chunk,       // Text chunk
}
```

## Parsing Algorithm

### Step 1: Heading Detection

**Patterns**:
- Markdown headings: `# Title`, `## Subtitle`, etc.
- Chinese numbered: `第一章`, `第一节`, etc.
- Numbered sections: `1.`, `1.1`, etc.

**Regex**:
```rust
r"(?m)^(#{1,6}\s+.+?)$|^(第[一二三四五六七八九十\d]+[章节部分]\s+.+?)$|^(\d+\.\d*\s+.+?)$"
```

**Output**: List of (start, end, level) tuples

### Step 2: Hierarchy Construction

**Process**:
1. Create root node for document
2. For each heading:
   - Create heading node
   - Set parent to previous heading at lower level
   - Add to parent's children
3. For text between headings:
   - Create paragraph/sentence nodes
   - Set parent to nearest heading
   - Build sentence-level chunks

**Rules**:
- Heading level determines hierarchy
- Lower level headings are children of higher level
- Text belongs to nearest preceding heading

### Step 3: Text Segmentation

**Paragraph Splitting**:
- Double newlines (`\n\n`)
- Heading boundaries
- Section markers

**Sentence Splitting**:
- Periods (`.`)
- Question marks (`?`)
- Exclamation marks (`!`)
- Chinese punctuation (。！？)

**Chunk Creation**:
- Fixed-size chunks (optional)
- Sentence boundaries
- Semantic boundaries

## Hierarchy Example

```
Document Root (level 0)
├── # Main Title (level 1)
│   ├── Paragraph 1 (level 2)
│   │   ├── Sentence 1 (level 3)
│   │   └── Sentence 2 (level 3)
│   └── ## Subsection (level 2)
│       ├── Paragraph 2 (level 3)
│       └── Paragraph 3 (level 3)
└── # Another Section (level 1)
    └── ...
```

## Relationships

### Parent-Child Relationships

**Bidirectional**:
- Parent nodes maintain `children` list
- Child nodes maintain `parents` list
- Graph edges created for both directions

**Edge Types**:
- `ParentChild`: Direct parent-child (weight: 1.0)
- Reverse edge: Child-parent (weight: 0.8)

### Semantic Relationships

**Similarity Edges**:
- Created based on embedding similarity
- Weight: cosine similarity
- Type: `SemanticSimilarity`

**Reference Edges**:
- Created when nodes reference each other
- Detected via keyword matching
- Type: `Reference` or `Mention`

## Usage in Retrieval

### Hierarchical Retrieval

When a heading node is retrieved:
- Optionally include child nodes
- Maintain context hierarchy
- Preserve document structure

### Fine-grained Retrieval

Individual nodes can be retrieved:
- Sentence-level precision
- Paragraph-level context
- Chunk-level granularity

### Graph Expansion

Parent-child relationships enable:
- Upward expansion (to parent context)
- Downward expansion (to child details)
- Sibling expansion (related sections)

## Implementation Details

### Node ID Generation

```rust
static NODE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_node_id() -> u64 {
    NODE_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}
```

**Thread-safe**: Uses atomic counter for concurrent access

### Span Management

**Text Spans**:
- `span_start`: Byte offset in original document
- `span_end`: Byte offset in original document
- Used for efficient text extraction

**Benefits**:
- No text duplication
- Efficient memory usage
- Easy text updates

### Embedding Storage

**Optional Embeddings**:
- Nodes can have embeddings
- Generated on-demand or during indexing
- Stored in vector index (HNSW)

**Update Process**:
1. Generate embedding for node text
2. Store in node structure
3. Update vector index
4. Maintain consistency

## Best Practices

### Document Structure

**Recommended**:
- Use clear headings
- Maintain consistent hierarchy
- Separate sections clearly

**Example**:
```markdown
# Main Topic

## Subtopic 1

Content for subtopic 1.

## Subtopic 2

Content for subtopic 2.
```

### Node Granularity

**Balance**:
- Too fine: Many small nodes, high overhead
- Too coarse: Less precise retrieval

**Guidelines**:
- Paragraph-level for general content
- Sentence-level for precise queries
- Heading-level for structure

### Relationship Management

**Parent-Child**:
- Always maintain bidirectional links
- Update when structure changes
- Use for graph expansion

**Semantic**:
- Create based on similarity threshold
- Update periodically
- Use for related content discovery

## Performance Considerations

### Parsing Performance
- **Time**: O(n) where n is document length
- **Space**: O(m) where m is number of nodes
- **Optimization**: Lazy parsing, incremental updates

### Storage
- **Memory**: ~200 bytes per node (without embedding)
- **Embedding**: +512 bytes per node (for 128-dim embedding)
- **Total**: ~700 bytes per node average

### Retrieval
- **Node Lookup**: O(1) with hash map
- **Hierarchy Traversal**: O(d) where d is depth
- **Graph Expansion**: O(k * d) where k is neighbors, d is depth

## Future Enhancements

1. **Multi-language Support**: Better parsing for non-English text
2. **Custom Parsers**: Plugin system for domain-specific parsing
3. **Lazy Loading**: Load node content on-demand
4. **Compression**: Compress node text for storage
5. **Caching**: Cache parsed structures
6. **Incremental Parsing**: Parse only changed sections

