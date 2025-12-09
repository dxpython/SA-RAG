"""Orchestrator Module: Responsible for query rewriting, retrieval strategy planning, and execution scheduling

Features:
- Query rewriting: Convert user queries to forms more suitable for retrieval
- Query type inference: Classify queries (Definition, Comparison, Procedure, etc.)
- Structured query generation: Convert queries to structured format
- Retrieval strategy planning: Decide which retrieval methods to use based on query type
- Result fusion: Merge results from multiple retrieval methods
"""

from typing import List, Optional, Dict, Any
from .llm import LLMService
from .query_analysis import QueryAnalyzer, StructuredQuery, QueryIntent


class Orchestrator:
    """RAG Orchestrator: Coordinates query processing and retrieval strategies"""
    
    def __init__(self, llm_service: LLMService, embedding_service=None):
        """
        Initialize orchestrator
        
        Args:
            llm_service: LLM service instance (for query rewriting)
            embedding_service: Embedding service (optional, for query analysis)
        """
        self.llm = llm_service
        self.embedding = embedding_service
        self.query_analyzer = QueryAnalyzer(llm_service)
    
    def rewrite_query(self, query: str, use_llm: bool = False) -> str:
        """
        Rewrite query to be more suitable for retrieval
        
        Args:
            query: Original query
            use_llm: Whether to use LLM for intelligent rewriting
            
        Returns:
            Rewritten query
        """
        if not use_llm:
            # Simple rewrite: remove stop words, normalize
            return self._simple_rewrite(query)
        
        # Use LLM for intelligent rewriting
        prompt = f"""Please rewrite the following user query to be more suitable for information retrieval.
Requirements:
1. Preserve core intent and keywords
2. Expand synonyms and related concepts
3. Clarify entities and relationships in the query
4. If the query is ambiguous, provide multiple possible interpretations

Original query: {query}

Rewritten query:"""
        
        try:
            rewritten = self.llm.chat_completion(
                prompt,
                system_prompt="You are a query rewriting expert for information retrieval systems.",
                max_tokens=200,
            )
            return rewritten.strip()
        except Exception as e:
            print(f"Error in LLM query rewriting: {e}, using simple rewrite")
            return self._simple_rewrite(query)
    
    def _simple_rewrite(self, query: str) -> str:
        """Simple query rewriting (rule-based)"""
        # Remove extra spaces
        query = " ".join(query.split())
        
        # Can add more rules: synonym expansion, entity recognition, etc.
        return query
    
    def analyze_query(self, query: str, use_llm: bool = False) -> StructuredQuery:
        """
        Analyze query and generate structured query
        
        Args:
            query: User query
            use_llm: Whether to use LLM for analysis
            
        Returns:
            StructuredQuery object
        """
        return self.query_analyzer.analyze(query, use_llm=use_llm)
    
    def plan_retrieval(
        self,
        query: str,
        available_methods: Optional[List[str]] = None,
        use_llm_analysis: bool = False,
    ) -> Dict[str, Any]:
        """
        Plan retrieval strategy based on query analysis
        
        Args:
            query: User query
            available_methods: List of available retrieval methods
            use_llm_analysis: Whether to use LLM for query analysis
            
        Returns:
            Retrieval strategy configuration
        """
        if available_methods is None:
            available_methods = ["hybrid", "graph", "memory"]
        
        # Analyze query to get structured query
        structured_query = self.analyze_query(query, use_llm=use_llm_analysis)
        
        # Determine retrieval strategy based on intent
        strategy = self._get_strategy_for_intent(structured_query.intent)
        
        # Override with structured query requirements
        use_graph = structured_query.requires_graph and "graph" in available_methods
        use_memory = structured_query.requires_memory and "memory" in available_methods
        top_k = structured_query.top_k
        
        return {
            "use_graph": use_graph,
            "use_memory": use_memory,
            "use_hybrid": True,
            "top_k": top_k,
            "graph_hops": strategy.get("graph_hops", 2),
            "intent": structured_query.intent.value,
            "entities": structured_query.entities,
            "keywords": structured_query.keywords,
            "rewritten_query": structured_query.rewritten_query,
        }
    
    def _get_strategy_for_intent(self, intent: QueryIntent) -> Dict[str, Any]:
        """Get retrieval strategy configuration for query intent"""
        strategies = {
            QueryIntent.DEFINITION: {
                "graph_hops": 1,  # Definitions usually don't need deep expansion
                "weight_vector": 0.5,
                "weight_bm25": 0.3,
                "weight_graph": 0.2,
            },
            QueryIntent.COMPARISON: {
                "graph_hops": 3,  # Comparisons benefit from graph expansion
                "weight_vector": 0.3,
                "weight_bm25": 0.2,
                "weight_graph": 0.5,  # Emphasize graph for relationships
            },
            QueryIntent.PROCEDURE: {
                "graph_hops": 2,
                "weight_vector": 0.4,
                "weight_bm25": 0.4,  # Procedures benefit from keyword matching
                "weight_graph": 0.2,
            },
            QueryIntent.FACTUAL: {
                "graph_hops": 1,
                "weight_vector": 0.3,
                "weight_bm25": 0.5,  # Facts benefit from exact keyword matching
                "weight_graph": 0.2,
            },
            QueryIntent.CONCEPTUAL: {
                "graph_hops": 2,
                "weight_vector": 0.5,  # Concepts benefit from semantic search
                "weight_bm25": 0.2,
                "weight_graph": 0.3,
            },
            QueryIntent.OTHER: {
                "graph_hops": 2,
                "weight_vector": 0.4,
                "weight_bm25": 0.3,
                "weight_graph": 0.3,
            },
        }
        return strategies.get(intent, strategies[QueryIntent.OTHER])
        
        return {
            "use_hybrid": use_hybrid,
            "use_graph": use_graph,
            "use_memory": use_memory,
            "top_k": top_k,
            "graph_hops": 2 if use_graph else 0,
            "graph_min_weight": 0.3,
            "fusion_method": "rrf",  # Reciprocal Rank Fusion
        }
    
    def fuse_results(
        self,
        results_list: List[List[Dict[str, Any]]],
        method: str = "rrf",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Fuse multiple retrieval results
        
        Args:
            results_list: List of results from multiple retrieval methods
            method: Fusion method ("rrf", "weighted", "max")
            top_k: Return top-k results
            
        Returns:
            Fused result list
        """
        if method == "rrf":
            return self._reciprocal_rank_fusion(results_list, top_k)
        elif method == "weighted":
            return self._weighted_fusion(results_list, top_k)
        elif method == "max":
            return self._max_fusion(results_list, top_k)
        else:
            return self._reciprocal_rank_fusion(results_list, top_k)
    
    def _reciprocal_rank_fusion(
        self,
        results_list: List[List[Dict[str, Any]]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion (RRF)"""
        k = 60  # RRF constant
        scores: Dict[str, Dict[str, Any]] = {}
        
        for results in results_list:
            for rank, result in enumerate(results):
                # Use node_id or text as unique identifier
                key = str(result.get("node_id", result.get("text", "")))
                
                if key not in scores:
                    scores[key] = result.copy()
                    scores[key]["rrf_score"] = 0.0
                
                # RRF score: 1 / (k + rank)
                scores[key]["rrf_score"] += 1.0 / (k + rank + 1)
        
        # Sort by RRF score
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x.get("rrf_score", 0.0),
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def _weighted_fusion(
        self,
        results_list: List[List[Dict[str, Any]]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Weighted fusion"""
        # Assign weights to each retrieval method
        weights = [1.0 / len(results_list)] * len(results_list)
        
        scores: Dict[str, Dict[str, Any]] = {}
        
        for weight, results in zip(weights, results_list):
            for result in results:
                key = str(result.get("node_id", result.get("text", "")))
                
                if key not in scores:
                    scores[key] = result.copy()
                    scores[key]["weighted_score"] = 0.0
                
                original_score = result.get("score", 0.0)
                scores[key]["weighted_score"] += weight * original_score
        
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x.get("weighted_score", 0.0),
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def _max_fusion(
        self,
        results_list: List[List[Dict[str, Any]]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Take maximum score fusion"""
        scores: Dict[str, Dict[str, Any]] = {}
        
        for results in results_list:
            for result in results:
                key = str(result.get("node_id", result.get("text", "")))
                
                if key not in scores:
                    scores[key] = result.copy()
                    scores[key]["max_score"] = 0.0
                
                original_score = result.get("score", 0.0)
                scores[key]["max_score"] = max(scores[key]["max_score"], original_score)
        
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x.get("max_score", 0.0),
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def should_use_graph_expansion(
        self,
        query: str,
        initial_results: List[Dict[str, Any]],
    ) -> bool:
        """
        Determine whether to use graph expansion
        
        Args:
            query: User query
            initial_results: Initial retrieval results
            
        Returns:
            Whether to use graph expansion
        """
        # If initial results are few, use graph expansion
        if len(initial_results) < 3:
            return True
        
        # If result scores are low, use graph expansion
        if initial_results and initial_results[0].get("score", 0.0) < 0.5:
            return True
        
        return False
