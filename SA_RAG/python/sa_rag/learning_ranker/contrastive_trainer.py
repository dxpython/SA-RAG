"""Contrastive Learning Trainer
Contrastive learning for ranking optimization
"""

from typing import List, Dict
import numpy as np


class ContrastiveTrainer:
    """Contrastive learning trainer"""
    
    def __init__(self, margin: float = 0.1, learning_rate: float = 0.001):
        """Initialize contrastive trainer
        
        Args:
            margin: Margin for contrastive loss
            learning_rate: Learning rate
        """
        self.margin = margin
        self.learning_rate = learning_rate
    
    def train(
        self,
        samples: List[Dict],
        epochs: int = 10,
    ) -> Dict:
        """Train using contrastive learning
        
        Args:
            samples: Training samples
            epochs: Number of epochs
            
        Returns:
            Learned weights
        """
        # Initialize weights
        weights = {
            "vector_weight": 0.4,
            "bm25_weight": 0.3,
            "graph_weight": 0.2,
            "keyword_weight": 0.1,
            "memory_weight": 0.0,
        }
        
        for epoch in range(epochs):
            for sample in samples:
                # Find positive/negative pairs
                pairs = self._find_pairs(sample["results"], sample["relevance_labels"])
                
                for pos_idx, neg_idx in pairs:
                    pos_score = self._score_result(sample["results"][pos_idx], weights)
                    neg_score = self._score_result(sample["results"][neg_idx], weights)
                    
                    # Contrastive loss
                    loss = max(0.0, self.margin - (pos_score - neg_score))
                    
                    if loss > 0.0:
                        # Update weights
                        self._update_weights_contrastive(
                            weights,
                            sample["results"][pos_idx],
                            sample["results"][neg_idx],
                            loss,
                        )
        
        weights["confidence"] = 0.8
        return weights
    
    def _find_pairs(
        self,
        results: List[Dict],
        labels: List[float],
    ) -> List[tuple]:
        """Find positive/negative pairs"""
        pairs = []
        for i, label_i in enumerate(labels):
            for j, label_j in enumerate(labels):
                if i != j and label_i > label_j:
                    pairs.append((i, j))
        return pairs
    
    def _score_result(self, result: Dict, weights: Dict) -> float:
        """Score a result with given weights"""
        return (
            result.get("vector_score", 0.0) * weights["vector_weight"] +
            result.get("bm25_score", 0.0) * weights["bm25_weight"] +
            result.get("graph_score", 0.0) * weights["graph_weight"] +
            result.get("keyword_score", 0.0) * weights["keyword_weight"] +
            result.get("memory_score", 0.0) * weights["memory_weight"]
        )
    
    def _update_weights_contrastive(
        self,
        weights: Dict,
        positive: Dict,
        negative: Dict,
        loss: float,
    ):
        """Update weights using contrastive learning"""
        gradient = self.learning_rate * loss
        
        weights["vector_weight"] += gradient * (
            positive.get("vector_score", 0.0) - negative.get("vector_score", 0.0)
        )
        weights["bm25_weight"] += gradient * (
            positive.get("bm25_score", 0.0) - negative.get("bm25_score", 0.0)
        )
        weights["graph_weight"] += gradient * (
            positive.get("graph_score", 0.0) - negative.get("graph_score", 0.0)
        )
        weights["keyword_weight"] += gradient * (
            positive.get("keyword_score", 0.0) - negative.get("keyword_score", 0.0)
        )
        weights["memory_weight"] += gradient * (
            positive.get("memory_score", 0.0) - negative.get("memory_score", 0.0)
        )
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            for key in weights:
                weights[key] /= total

