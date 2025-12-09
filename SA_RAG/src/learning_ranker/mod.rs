// Self-Evolving Ranker Module
// Implements RL-based fusion and learning-to-rank capabilities

pub mod rl_ranker;
pub mod contrastive_learner;

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use rl_ranker::RLRanker;
pub use contrastive_learner::ContrastiveLearner;

/// Ranker weights learned from data
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct LearnedWeights {
    #[pyo3(get, set)]
    pub vector_weight: f32,
    #[pyo3(get, set)]
    pub bm25_weight: f32,
    #[pyo3(get, set)]
    pub graph_weight: f32,
    #[pyo3(get, set)]
    pub keyword_weight: f32,
    #[pyo3(get, set)]
    pub memory_weight: f32,
    #[pyo3(get, set)]
    pub confidence: f32, // Confidence in learned weights (0.0-1.0)
}

#[pymethods]
impl LearnedWeights {
    #[new]
    pub fn new() -> Self {
        Self {
            vector_weight: 0.4,
            bm25_weight: 0.3,
            graph_weight: 0.2,
            keyword_weight: 0.1,
            memory_weight: 0.0,
            confidence: 0.0,
        }
    }
}

/// Training sample for learning ranker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub query: String,
    pub query_embedding: Vec<f32>,
    pub results: Vec<RetrievalResult>,
    pub relevance_labels: Vec<f32>, // Ground truth relevance scores
    pub context: HashMap<String, f32>, // Additional context features
}

/// Retrieval result with scores from different stages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct RetrievalResult {
    #[pyo3(get, set)]
    pub node_id: u64,
    #[pyo3(get, set)]
    pub vector_score: f32,
    #[pyo3(get, set)]
    pub bm25_score: f32,
    #[pyo3(get, set)]
    pub graph_score: f32,
    #[pyo3(get, set)]
    pub keyword_score: f32,
    #[pyo3(get, set)]
    pub memory_score: f32,
}

#[pymethods]
impl RetrievalResult {
    #[new]
    fn new(
        node_id: u64,
        vector_score: f32,
        bm25_score: f32,
        graph_score: f32,
        keyword_score: f32,
        memory_score: f32,
    ) -> Self {
        Self {
            node_id,
            vector_score,
            bm25_score,
            graph_score,
            keyword_score,
            memory_score,
        }
    }
}

/// Self-evolving ranker that learns from data
#[pyclass]
pub struct SelfEvolvingRanker {
    rl_ranker: RLRanker,
    contrastive_learner: ContrastiveLearner,
    current_weights: LearnedWeights,
    training_samples: Vec<TrainingSample>,
}

#[pymethods]
impl SelfEvolvingRanker {
    #[new]
    pub fn new() -> Self {
        Self {
            rl_ranker: RLRanker::new(),
            contrastive_learner: ContrastiveLearner::new(),
            current_weights: LearnedWeights::new(),
            training_samples: Vec::new(),
        }
    }

    /// Rank results using learned weights
    fn rank(
        &self,
        results: Vec<RetrievalResult>,
    ) -> PyResult<Vec<(u64, f32)>> {
        let mut scored: Vec<(u64, f32)> = results
            .iter()
            .map(|r| {
                let score = 
                    r.vector_score * self.current_weights.vector_weight +
                    r.bm25_score * self.current_weights.bm25_weight +
                    r.graph_score * self.current_weights.graph_weight +
                    r.keyword_score * self.current_weights.keyword_weight +
                    r.memory_score * self.current_weights.memory_weight;
                (r.node_id, score)
            })
            .collect();
        
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(scored)
    }

    /// Add training sample
    fn add_training_sample(
        &mut self,
        query: String,
        query_embedding: Vec<f32>,
        results: Vec<RetrievalResult>,
        relevance_labels: Vec<f32>,
    ) -> PyResult<()> {
        if results.len() != relevance_labels.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "results and relevance_labels must have same length"
            ));
        }

        let sample = TrainingSample {
            query,
            query_embedding,
            results,
            relevance_labels,
            context: HashMap::new(),
        };
        self.training_samples.push(sample);
        Ok(())
    }

    /// Train ranker using collected samples
    pub fn train(&mut self, epochs: usize) -> PyResult<LearnedWeights> {
        if self.training_samples.is_empty() {
            return Ok(self.current_weights.clone());
        }

        // Use RL-based training
        let new_weights = self.rl_ranker.train(&self.training_samples, epochs);
        self.current_weights = new_weights.clone();
        Ok(new_weights)
    }

    /// Get current learned weights
    pub fn get_weights(&self) -> LearnedWeights {
        self.current_weights.clone()
    }

    /// Update weights (for manual tuning or transfer learning)
    fn update_weights(&mut self, weights: LearnedWeights) {
        self.current_weights = weights;
    }
}

