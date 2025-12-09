"""Learning Ranker Trainer
Python wrapper for self-evolving ranker
"""

from typing import List, Dict, Optional
import rust_core


class LearningRankerTrainer:
    """Trainer for self-evolving ranker"""
    
    def __init__(self):
        """Initialize learning ranker trainer"""
        self.ranker = rust_core.SelfEvolvingRanker()
    
    def add_training_sample(
        self,
        query: str,
        query_embedding: List[float],
        results: List[Dict],
        relevance_labels: List[float],
    ):
        """Add a training sample
        
        Args:
            query: Query text
            query_embedding: Query embedding vector
            results: List of retrieval results with scores
            relevance_labels: Ground truth relevance scores
        """
        # Convert results to RetrievalResult format
        retrieval_results = []
        for r in results:
            result = rust_core.RetrievalResult(
                node_id=r.get("node_id", 0),
                vector_score=r.get("vector_score", 0.0),
                bm25_score=r.get("bm25_score", 0.0),
                graph_score=r.get("graph_score", 0.0),
                keyword_score=r.get("keyword_score", 0.0),
                memory_score=r.get("memory_score", 0.0),
            )
            retrieval_results.append(result)
        
        self.ranker.add_training_sample(
            query,
            query_embedding,
            retrieval_results,
            relevance_labels,
        )
    
    def train(self, epochs: int = 10) -> Dict:
        """Train the ranker
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            Learned weights dictionary
        """
        weights = self.ranker.train(epochs)
        return {
            "vector_weight": weights.vector_weight,
            "bm25_weight": weights.bm25_weight,
            "graph_weight": weights.graph_weight,
            "keyword_weight": weights.keyword_weight,
            "memory_weight": weights.memory_weight,
            "confidence": weights.confidence,
        }
    
    def get_weights(self) -> Dict:
        """Get current learned weights"""
        weights = self.ranker.get_weights()
        return {
            "vector_weight": weights.vector_weight,
            "bm25_weight": weights.bm25_weight,
            "graph_weight": weights.graph_weight,
            "keyword_weight": weights.keyword_weight,
            "memory_weight": weights.memory_weight,
            "confidence": weights.confidence,
        }
    
    def rank(self, results: List[Dict]) -> List[tuple]:
        """Rank results using learned weights
        
        Args:
            results: List of retrieval results
            
        Returns:
            List of (node_id, score) tuples, sorted by score
        """
        retrieval_results = []
        for r in results:
            result = rust_core.RetrievalResult(
                node_id=r.get("node_id", 0),
                vector_score=r.get("vector_score", 0.0),
                bm25_score=r.get("bm25_score", 0.0),
                graph_score=r.get("graph_score", 0.0),
                keyword_score=r.get("keyword_score", 0.0),
                memory_score=r.get("memory_score", 0.0),
            )
            retrieval_results.append(result)
        
        return self.ranker.rank(retrieval_results)

