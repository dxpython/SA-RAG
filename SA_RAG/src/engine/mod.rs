// Engine module for retrieval results
// Provides common types used across the engine

use serde::{Deserialize, Serialize};

/// Retrieval result with scores from different stages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    pub node_id: u64,
    pub score: f32,
    pub vector_score: f32,
    pub bm25_score: f32,
    pub graph_score: f32,
    pub keyword_score: f32,
    pub memory_score: f32,
}

impl RetrievalResult {
    pub fn new(node_id: u64) -> Self {
        Self {
            node_id,
            score: 0.0,
            vector_score: 0.0,
            bm25_score: 0.0,
            graph_score: 0.0,
            keyword_score: 0.0,
            memory_score: 0.0,
        }
    }
}

