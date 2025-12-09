"""Consistency Checker
Checks retrieval consistency and detects drift
"""

from typing import List, Dict, Optional
import rust_core


class ConsistencyChecker:
    """Checker for retrieval consistency"""
    
    def __init__(self, engine: Optional[rust_core.RustCoreEngine] = None):
        """Initialize consistency checker
        
        Args:
            engine: Rust core engine instance
        """
        self.engine = engine
        self.checker = rust_core.ConsistencyChecker() if engine else None
    
    def check(
        self,
        query: str,
        results: List[int],
    ) -> Dict:
        """Check consistency for a query
        
        Args:
            query: Query text
            results: List of result node IDs
            
        Returns:
            Consistency report dictionary
        """
        if self.checker:
            report = self.checker.check_consistency(query, results)
            return {
                "query_hash": report.query_hash,
                "previous_results": report.previous_results,
                "current_results": report.current_results,
                "similarity_score": report.similarity_score,
                "drift_detected": report.drift_detected,
                "drift_reason": report.drift_reason,
                "timestamp": report.timestamp,
            }
        else:
            # Fallback implementation
            return {
                "query_hash": str(hash(query)),
                "previous_results": [],
                "current_results": results,
                "similarity_score": 1.0,
                "drift_detected": False,
                "drift_reason": None,
                "timestamp": 0,
            }
    
    def set_drift_threshold(self, threshold: float):
        """Set drift detection threshold
        
        Args:
            threshold: Threshold value (0.0-1.0)
        """
        if self.checker:
            self.checker.set_drift_threshold(threshold)
    
    def get_history(self, query: str) -> List[tuple]:
        """Get query history
        
        Args:
            query: Query text
            
        Returns:
            List of (timestamp, results) tuples
        """
        if self.checker:
            return self.checker.get_history(query)
        else:
            return []

