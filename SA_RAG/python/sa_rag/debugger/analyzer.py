"""Autonomous Debugger
Analyzes retrieval failures and suggests optimizations
"""

from typing import List, Dict, Optional, Tuple
import rust_core


class AutonomousDebugger:
    """Autonomous debugger for retrieval analysis"""
    
    def __init__(self, engine: Optional[rust_core.RustCoreEngine] = None):
        """Initialize debugger
        
        Args:
            engine: Rust core engine instance
        """
        self.engine = engine
        self.debugger = rust_core.AutonomousDebugger() if engine else None
    
    def analyze(
        self,
        query: str,
        vector_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        graph_results: List[Tuple[int, float]],
        final_results: List[Tuple[int, float]],
        answer_quality: Optional[float] = None,
    ) -> Dict:
        """Analyze retrieval performance
        
        Args:
            query: Query text
            vector_results: Vector search results
            bm25_results: BM25 search results
            graph_results: Graph expansion results
            final_results: Final ranked results
            answer_quality: Answer quality score (optional)
            
        Returns:
            Debug analysis dictionary
        """
        if self.debugger:
            analysis = self.debugger.analyze_retrieval(
                query,
                None,
                vector_results,
                bm25_results,
                graph_results,
                final_results,
                answer_quality,
            )
            
            return {
                "query": analysis.query,
                "success": analysis.success,
                "failure_stage": analysis.failure_stage,
                "issues": [
                    {
                        "stage": issue.stage,
                        "severity": issue.severity,
                        "description": issue.description,
                        "details": dict(issue.details),
                    }
                    for issue in analysis.issues
                ],
                "suggestions": [
                    {
                        "category": sug.category,
                        "action": sug.action,
                        "description": sug.description,
                        "expected_improvement": sug.expected_improvement,
                        "parameters": dict(sug.parameters),
                    }
                    for sug in analysis.suggestions
                ],
                "metrics": dict(analysis.metrics),
            }
        else:
            # Fallback implementation
            return self._analyze_fallback(
                query,
                vector_results,
                bm25_results,
                graph_results,
                final_results,
                answer_quality,
            )
    
    def _analyze_fallback(
        self,
        query: str,
        vector_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        graph_results: List[Tuple[int, float]],
        final_results: List[Tuple[int, float]],
        answer_quality: Optional[float],
    ) -> Dict:
        """Fallback analysis without Rust engine"""
        issues = []
        suggestions = []
        metrics = {}
        
        if not vector_results:
            issues.append({
                "stage": "vector_search",
                "severity": "warning",
                "description": "Vector search returned no results",
                "details": {},
            })
        
        if not bm25_results:
            issues.append({
                "stage": "bm25_search",
                "severity": "warning",
                "description": "BM25 search returned no results",
                "details": {},
            })
        
        if len(final_results) < 3:
            issues.append({
                "stage": "ranking",
                "severity": "warning",
                "description": f"Too few results: {len(final_results)}",
                "details": {},
            })
            suggestions.append({
                "category": "ranking",
                "action": "increase_top_k",
                "description": "Increase top_k parameter",
                "expected_improvement": 0.1,
                "parameters": {},
            })
        
        metrics["result_count"] = len(final_results)
        if answer_quality is not None:
            metrics["answer_quality"] = answer_quality
        
        return {
            "query": query,
            "success": len(issues) == 0,
            "failure_stage": None,
            "issues": issues,
            "suggestions": suggestions,
            "metrics": metrics,
        }
    
    def get_optimization_suggestions(self) -> List[Dict]:
        """Get optimization suggestions based on history
        
        Returns:
            List of optimization suggestions
        """
        if self.debugger:
            suggestions = self.debugger.get_optimization_suggestions()
            return [
                {
                    "category": sug.category,
                    "action": sug.action,
                    "description": sug.description,
                    "expected_improvement": sug.expected_improvement,
                    "parameters": dict(sug.parameters),
                }
                for sug in suggestions
            ]
        else:
            return []

