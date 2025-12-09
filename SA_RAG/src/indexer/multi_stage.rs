/// Multi-Stage Retrieval Engine: Implements three-stage retrieval pipeline
/// 
/// Stage 1: Vector Coarse Retrieval (HNSW, top_k=200)
/// Stage 2: BM25 + Keyword Fine Ranking
/// Stage 3: Score Fusion with configurable weights
/// 
/// Final score formula:
/// final_score = α * vector_score + β * bm25_score + γ * keyword_score + δ * node_type_prior
/// 
/// All weights (α, β, γ, δ) are configurable.

use crate::indexer::vector::VectorIndex;
use crate::indexer::bm25::BM25Index;
use crate::semantic_node::SemanticNode;
use std::collections::{HashMap, HashSet};

/// Multi-stage retrieval configuration
#[derive(Debug, Clone)]
pub struct MultiStageConfig {
    /// Stage 1: Coarse retrieval top_k
    pub coarse_top_k: usize,
    /// Stage 2: Fine ranking top_k (from coarse results)
    pub fine_top_k: usize,
    /// Final output top_k
    pub final_top_k: usize,
    /// Weight for vector score (α)
    pub alpha: f32,
    /// Weight for BM25 score (β)
    pub beta: f32,
    /// Weight for keyword score (γ)
    pub gamma: f32,
    /// Weight for node type prior (δ)
    pub delta: f32,
}

impl Default for MultiStageConfig {
    fn default() -> Self {
        Self {
            coarse_top_k: 200,
            fine_top_k: 50,
            final_top_k: 10,
            alpha: 0.4,
            beta: 0.3,
            gamma: 0.2,
            delta: 0.1,
        }
    }
}

/// Multi-stage retrieval engine
pub struct MultiStageRetrieval {
    vector_index: VectorIndex,
    bm25_index: BM25Index,
    config: MultiStageConfig,
    /// Node ID -> SemanticNode (for accessing node metadata)
    node_store: HashMap<u64, SemanticNode>,
}

impl MultiStageRetrieval {
    pub fn new(max_elements: usize, config: MultiStageConfig) -> Self {
        Self {
            vector_index: VectorIndex::new(max_elements),
            bm25_index: BM25Index::new(),
            config,
            node_store: HashMap::new(),
        }
    }

    /// Insert a node into all indexes
    pub fn insert(&mut self, node: SemanticNode) {
        let node_id = node.node_id;
        
        // Store node for metadata access
        self.node_store.insert(node_id, node.clone());
        
        // Insert into BM25 index
        self.bm25_index.insert(node_id, &node.text);
        
        // Insert into vector index if embedding exists
        if let Some(embedding) = &node.embedding {
            let _ = self.vector_index.insert(node_id, embedding.clone());
        }
    }

    /// Stage 1: Vector coarse retrieval
    fn stage1_coarse_retrieval(
        &self,
        query_vector: &[f32],
    ) -> Result<Vec<(u64, f32)>, String> {
        // Use HNSW to get top-k candidates
        self.vector_index
            .search(query_vector, 64, self.config.coarse_top_k)
            .map_err(|e| format!("Vector search error: {}", e))
            .map(|results| {
                // Convert distance to similarity score (1.0 - normalized_distance)
                results
                    .into_iter()
                    .map(|(node_id, dist)| {
                        // Normalize distance to [0, 1] and convert to similarity
                        let similarity = 1.0 / (1.0 + dist);
                        (node_id, similarity)
                    })
                    .collect()
            })
    }

    /// Stage 2: BM25 + Keyword fine ranking
    fn stage2_fine_ranking(
        &self,
        query_text: &str,
        candidate_node_ids: &[u64],
    ) -> Vec<(u64, f32, f32)> {
        // Extract keywords from query
        let keywords = self.extract_keywords(query_text);
        
        let mut results = Vec::new();
        
        for &node_id in candidate_node_ids {
            // Get BM25 score
            let bm25_score = self.bm25_index.score_document(query_text, node_id);
            
            // Get keyword match score
            let keyword_score = if let Some(node) = self.node_store.get(&node_id) {
                self.calculate_keyword_score(&keywords, &node.text)
            } else {
                0.0
            };
            
            results.push((node_id, bm25_score, keyword_score));
        }
        
        // Sort by combined score (BM25 + keyword)
        results.sort_by(|a, b| {
            let score_a = a.1 + a.2;
            let score_b = b.1 + b.2;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Return top-k
        results.into_iter().take(self.config.fine_top_k).collect()
    }

    /// Stage 3: Score fusion with node type prior
    fn stage3_fusion(
        &self,
        vector_scores: &HashMap<u64, f32>,
        fine_results: &[(u64, f32, f32)],
    ) -> Vec<(u64, f32)> {
        let mut final_scores: HashMap<u64, f32> = HashMap::new();
        
        for (node_id, bm25_score, keyword_score) in fine_results {
            // Get vector score (default to 0.0 if not in coarse results)
            let vector_score = vector_scores.get(node_id).copied().unwrap_or(0.0);
            
            // Get node type prior
            let node_type_prior = if let Some(node) = self.node_store.get(node_id) {
                node.priority
            } else {
                0.5
            };
            
            // Calculate final score
            let final_score = self.config.alpha * vector_score
                + self.config.beta * bm25_score
                + self.config.gamma * keyword_score
                + self.config.delta * node_type_prior;
            
            final_scores.insert(*node_id, final_score);
        }
        
        // Sort by final score
        let mut sorted_results: Vec<(u64, f32)> = final_scores.into_iter().collect();
        sorted_results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Return top-k
        sorted_results.into_iter().take(self.config.final_top_k).collect()
    }

    /// Main retrieval method: executes all three stages
    pub fn search(
        &self,
        query_text: &str,
        query_vector: Option<&[f32]>,
    ) -> Result<Vec<(u64, f32)>, String> {
        // Stage 1: Coarse retrieval
        let coarse_results = if let Some(vec) = query_vector {
            self.stage1_coarse_retrieval(vec)?
        } else {
            // If no vector, skip to stage 2 with all nodes
            self.node_store.keys().copied().map(|id| (id, 0.0)).collect()
        };
        
        let candidate_ids: Vec<u64> = coarse_results.iter().map(|(id, _)| *id).collect();
        let vector_scores: HashMap<u64, f32> = coarse_results.into_iter().collect();
        
        // Stage 2: Fine ranking
        let fine_results = self.stage2_fine_ranking(query_text, &candidate_ids);
        
        // Stage 3: Fusion
        let final_results = self.stage3_fusion(&vector_scores, &fine_results);
        
        Ok(final_results)
    }

    /// Extract keywords from query text
    fn extract_keywords(&self, text: &str) -> HashSet<String> {
        // Simple keyword extraction: split by whitespace and punctuation
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() > 2)
            .map(|s| s.to_string())
            .collect()
    }

    /// Calculate keyword match score
    fn calculate_keyword_score(&self, keywords: &HashSet<String>, text: &str) -> f32 {
        let text_lower = text.to_lowercase();
        let text_keywords: HashSet<String> = text_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() > 2)
            .map(|s| s.to_string())
            .collect();
        
        // Calculate Jaccard similarity
        let intersection = keywords.intersection(&text_keywords).count();
        let union = keywords.union(&text_keywords).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Update configuration
    pub fn update_config(&mut self, config: MultiStageConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn get_config(&self) -> &MultiStageConfig {
        &self.config
    }
}

