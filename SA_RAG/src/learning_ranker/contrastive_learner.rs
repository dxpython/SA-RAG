// Contrastive Learning for Ranker
// Learns from positive/negative pairs

use super::{LearnedWeights, TrainingSample};

pub struct ContrastiveLearner {
    margin: f32,
    learning_rate: f32,
}

impl ContrastiveLearner {
    pub fn new() -> Self {
        Self {
            margin: 0.1,
            learning_rate: 0.001,
        }
    }

    /// Train using contrastive learning
    pub fn train(&self, samples: &[TrainingSample], epochs: usize) -> LearnedWeights {
        let mut weights = LearnedWeights::new();
        
        for _epoch in 0..epochs {
            for sample in samples {
                // Find positive and negative pairs
                let pairs = self.find_pairs(sample);
                
                for (pos_idx, neg_idx) in pairs {
                    let pos_score = self.score_result(&sample.results[pos_idx], &weights);
                    let neg_score = self.score_result(&sample.results[neg_idx], &weights);
                    
                    // Contrastive loss: maximize margin between positive and negative
                    let loss = (self.margin - (pos_score - neg_score)).max(0.0);
                    
                    if loss > 0.0 {
                        // Update weights to increase positive score and decrease negative score
                        self.update_weights_contrastive(
                            &mut weights,
                            &sample.results[pos_idx],
                            &sample.results[neg_idx],
                            loss,
                        );
                    }
                }
            }
        }
        
        weights.confidence = 0.8; // Contrastive learning typically has good confidence
        weights
    }

    fn find_pairs(&self, sample: &TrainingSample) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        
        for (i, label_i) in sample.relevance_labels.iter().enumerate() {
            for (j, label_j) in sample.relevance_labels.iter().enumerate() {
                if i != j && *label_i > *label_j {
                    pairs.push((i, j)); // i is positive, j is negative
                }
            }
        }
        
        pairs
    }

    fn score_result(&self, result: &super::RetrievalResult, weights: &LearnedWeights) -> f32 {
        result.vector_score * weights.vector_weight +
        result.bm25_score * weights.bm25_weight +
        result.graph_score * weights.graph_weight +
        result.keyword_score * weights.keyword_weight +
        result.memory_score * weights.memory_weight
    }

    fn update_weights_contrastive(
        &self,
        weights: &mut LearnedWeights,
        positive: &super::RetrievalResult,
        negative: &super::RetrievalResult,
        loss: f32,
    ) {
        // Update weights to increase positive score and decrease negative score
        let gradient = self.learning_rate * loss;
        
        weights.vector_weight += gradient * (positive.vector_score - negative.vector_score);
        weights.bm25_weight += gradient * (positive.bm25_score - negative.bm25_score);
        weights.graph_weight += gradient * (positive.graph_score - negative.graph_score);
        weights.keyword_weight += gradient * (positive.keyword_score - negative.keyword_score);
        weights.memory_weight += gradient * (positive.memory_score - negative.memory_score);
        
        // Normalize
        let total = weights.vector_weight + weights.bm25_weight + weights.graph_weight +
                   weights.keyword_weight + weights.memory_weight;
        if total > 0.0 {
            weights.vector_weight /= total;
            weights.bm25_weight /= total;
            weights.graph_weight /= total;
            weights.keyword_weight /= total;
            weights.memory_weight /= total;
        }
    }
}

impl Default for ContrastiveLearner {
    fn default() -> Self {
        Self::new()
    }
}

