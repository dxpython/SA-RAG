use crate::indexer::vector::VectorIndex;
use crate::indexer::bm25::BM25Index;
use std::collections::HashMap;

pub struct HybridIndex {
    pub vector_index: VectorIndex,
    pub bm25_index: BM25Index,
    pub alpha: f32, // Weight for vector search (0.0 - 1.0). 1.0 = Pure Vector, 0.0 = Pure BM25
}

impl HybridIndex {
    pub fn new(max_elements: usize, alpha: f32) -> Self {
        Self {
            vector_index: VectorIndex::new(max_elements),
            bm25_index: BM25Index::new(),
            alpha,
        }
    }

    pub fn insert(&mut self, node_id: u64, text: &str, vector: Option<Vec<f32>>) {
        self.bm25_index.insert(node_id, text);
        if let Some(vec) = vector {
            let _ = self.vector_index.insert(node_id, vec);
        }
    }

    /// Reciprocal Rank Fusion (RRF) .
    /// Here we use weighted sum with min-max normalization simulated by purely weighting scores?
    /// Actually, BM25 and Cosine/L2 are in different ranges. 
    /// Standard approach is RRF. Let's do RRF for stability.
    pub fn search(&self, query_text: &str, query_vector: Option<&[f32]>, k: usize) -> Vec<(u64, f32)> {
        let mut rrf_scores: HashMap<u64, f32> = HashMap::new();
        let k_rrf = 60.0; // RRF constant

        // BM25 Search
        let bm25_results = self.bm25_index.search(query_text, k * 2);
        for (rank, (node_id, _score)) in bm25_results.iter().enumerate() {
            let contribution = (1.0 - self.alpha) / (k_rrf + rank as f32 + 1.0);
            *rrf_scores.entry(*node_id).or_insert(0.0) += contribution;
        }

        // Vector Search
        if let Some(vec) = query_vector {
            if let Ok(vec_results) = self.vector_index.search(vec, 64, k * 2) {
                // HNSW returns distance. Map to "similarity" rank or just rank.
                // results are sorted by distance (ascending).
                for (rank, (node_id, _dist)) in vec_results.iter().enumerate() {
                    let contribution = self.alpha / (k_rrf + rank as f32 + 1.0);
                    *rrf_scores.entry(*node_id).or_insert(0.0) += contribution;
                }
            }
        }

        let mut final_results: Vec<(u64, f32)> = rrf_scores.into_iter().collect();
        final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        final_results.into_iter().take(k).collect()
    }
}
