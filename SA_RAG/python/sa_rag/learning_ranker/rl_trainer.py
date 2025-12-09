"""RL-based Trainer
Reinforcement learning for ranking weight optimization
"""

from typing import List, Dict
import numpy as np


class RLTrainer:
    """RL-based ranking trainer"""
    
    def __init__(self, learning_rate: float = 0.01):
        """Initialize RL trainer
        
        Args:
            learning_rate: Learning rate for weight updates
        """
        self.learning_rate = learning_rate
    
    def train(
        self,
        samples: List[Dict],
        epochs: int = 10,
    ) -> Dict:
        """Train using policy gradient
        
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
            total_reward = 0.0
            gradients = {
                "vector_weight": 0.0,
                "bm25_weight": 0.0,
                "graph_weight": 0.0,
                "keyword_weight": 0.0,
                "memory_weight": 0.0,
            }
            
            for sample in samples:
                # Calculate scores with current weights
                ranked = self._rank_results(sample["results"], weights)
                
                # Calculate reward (NDCG-like)
                reward = self._calculate_reward(ranked, sample["relevance_labels"])
                total_reward += reward
                
                # Calculate gradients
                sample_gradients = self._calculate_gradients(
                    sample["results"],
                    weights,
                    reward,
                )
                
                for key in gradients:
                    gradients[key] += sample_gradients[key]
            
            # Update weights
            for key in weights:
                weights[key] += self.learning_rate * (gradients[key] / len(samples))
            
            # Normalize weights
            total = sum(weights.values())
            if total > 0:
                for key in weights:
                    weights[key] /= total
        
        weights["confidence"] = min(total_reward / (len(samples) * epochs), 1.0)
        return weights
    
    def _rank_results(self, results: List[Dict], weights: Dict) -> List[tuple]:
        """Rank results with given weights"""
        scored = []
        for r in results:
            score = (
                r.get("vector_score", 0.0) * weights["vector_weight"] +
                r.get("bm25_score", 0.0) * weights["bm25_weight"] +
                r.get("graph_score", 0.0) * weights["graph_weight"] +
                r.get("keyword_score", 0.0) * weights["keyword_weight"] +
                r.get("memory_score", 0.0) * weights["memory_weight"]
            )
            scored.append((r.get("node_id", 0), score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def _calculate_reward(self, ranked: List[tuple], labels: List[float]) -> float:
        """Calculate NDCG-like reward"""
        reward = 0.0
        for rank, (idx, _) in enumerate(ranked):
            if idx < len(labels):
                relevance = labels[idx]
                position_discount = 1.0 / np.log1p(rank + 1)
                reward += relevance * position_discount
        return reward
    
    def _calculate_gradients(
        self,
        results: List[Dict],
        weights: Dict,
        reward: float,
    ) -> Dict:
        """Calculate gradients for weight updates"""
        gradients = {
            "vector_weight": 0.0,
            "bm25_weight": 0.0,
            "graph_weight": 0.0,
            "keyword_weight": 0.0,
            "memory_weight": 0.0,
        }
        
        for r in results:
            gradients["vector_weight"] += r.get("vector_score", 0.0) * reward
            gradients["bm25_weight"] += r.get("bm25_score", 0.0) * reward
            gradients["graph_weight"] += r.get("graph_score", 0.0) * reward
            gradients["keyword_weight"] += r.get("keyword_score", 0.0) * reward
            gradients["memory_weight"] += r.get("memory_score", 0.0) * reward
        
        return gradients

