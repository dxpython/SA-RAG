use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Node type: defines semantic types of nodes for better retrieval and ranking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NodeType {
    /// Document root node
    Root,
    /// Section heading (h1-h6)
    Heading,
    /// Regular paragraph
    Paragraph,
    /// Sentence-level chunk
    Sentence,
    /// Text chunk (smallest unit)
    Chunk,
    /// Definition node (defines a concept)
    Definition,
    /// Example node (provides examples)
    Example,
    /// Conclusion node (conclusions or summaries)
    Conclusion,
    /// Procedure node (step-by-step instructions)
    Procedure,
    /// Code block
    Code,
    /// Table (row-level or cell-level)
    Table,
    /// List item (ordered or unordered)
    List,
    /// Quote or citation
    Quote,
    /// Reference or citation link
    Reference,
    /// Theorem or formula
    Theorem,
}

/// Semantic node: represents a semantic unit in a document with rich metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticNode {
    pub node_id: u64,
    pub doc_id: u64,
    /// Start offset in the original text (characters/bytes)
    pub span_start: usize,
    /// End offset
    pub span_end: usize,
    /// Hierarchy level (0 = root, 1 = heading, etc.)
    pub level: u8,
    pub node_type: NodeType,
    pub text: String,
    /// Embedding vector (optional because not all nodes might have embeddings immediately)
    pub embedding: Option<Vec<f32>>,
    /// Parent node IDs
    pub parents: Vec<u64>,
    /// Child node IDs
    pub children: Vec<u64>,
    /// Additional metadata (e.g., language, format, importance)
    pub metadata: HashMap<String, String>,
    /// Priority score for retrieval (higher = more important)
    pub priority: f32,
}

impl SemanticNode {
    pub fn new(
        node_id: u64,
        doc_id: u64,
        span_start: usize,
        span_end: usize,
        level: u8,
        node_type: NodeType,
        text: String,
    ) -> Self {
        // Calculate default priority based on node type
        let priority = Self::default_priority(&node_type);
        
        Self {
            node_id,
            doc_id,
            span_start,
            span_end,
            level,
            node_type,
            text,
            embedding: None,
            parents: Vec::new(),
            children: Vec::new(),
            metadata: HashMap::new(),
            priority,
        }
    }
    
    /// Get default priority score for node type (used in retrieval ranking)
    pub fn default_priority(node_type: &NodeType) -> f32 {
        match node_type {
            NodeType::Root => 0.1,
            NodeType::Heading => 0.9,
            NodeType::Definition => 0.95,
            NodeType::Theorem => 0.9,
            NodeType::Conclusion => 0.85,
            NodeType::Procedure => 0.8,
            NodeType::Example => 0.75,
            NodeType::Code => 0.7,
            NodeType::Table => 0.7,
            NodeType::Quote => 0.65,
            NodeType::Reference => 0.6,
            NodeType::Paragraph => 0.5,
            NodeType::List => 0.5,
            NodeType::Sentence => 0.4,
            NodeType::Chunk => 0.3,
        }
    }
    
    /// Check if node type is a concept node (Definition, Theorem, etc.)
    #[allow(dead_code)]
    pub fn is_concept_node(&self) -> bool {
        matches!(
            self.node_type,
            NodeType::Definition | NodeType::Theorem | NodeType::Heading
        )
    }

    #[allow(dead_code)]
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        self.embedding = Some(embedding);
    }

    pub fn add_child(&mut self, child_id: u64) {
        if !self.children.contains(&child_id) {
            self.children.push(child_id);
        }
    }

    pub fn add_parent(&mut self, parent_id: u64) {
        if !self.parents.contains(&parent_id) {
            self.parents.push(parent_id);
        }
    }
}
