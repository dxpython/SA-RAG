"""Execution Graph Builder
Builds DAG representation of query execution
"""

from typing import List, Dict, Optional
import rust_core
import json


class ExecutionGraphBuilder:
    """Builder for execution graphs"""
    
    def __init__(self, engine: Optional[rust_core.RustCoreEngine] = None):
        """Initialize builder
        
        Args:
            engine: Rust core engine instance
        """
        self.engine = engine
    
    def build_graph(
        self,
        query: str,
        intent: str,
        knowledge_types: List[str],
    ) -> Dict:
        """Build execution graph for a query
        
        Args:
            query: Query text
            intent: Detected intent
            knowledge_types: List of required knowledge types
            
        Returns:
            Execution graph as dictionary
        """
        if self.engine:
            graph = self.engine.build_execution_graph(query, intent, knowledge_types)
            return json.loads(graph.to_json())
        else:
            # Fallback implementation
            return self._build_graph_fallback(query, intent, knowledge_types)
    
    def _build_graph_fallback(
        self,
        query: str,
        intent: str,
        knowledge_types: List[str],
    ) -> Dict:
        """Fallback graph building without Rust engine"""
        nodes = [
            {
                "node_id": "query",
                "node_type": "QUERY",
                "description": f"Query: {query}",
            },
            {
                "node_id": "intent",
                "node_type": "INTENT",
                "description": f"Intent: {intent}",
            },
            {
                "node_id": "retrieval_plan",
                "node_type": "RETRIEVAL_PLAN",
                "description": "Retrieval Plan",
            },
        ]
        
        edges = [
            {"from_node": "query", "to_node": "intent", "edge_type": "data_flow"},
            {"from_node": "intent", "to_node": "retrieval_plan", "edge_type": "data_flow"},
        ]
        
        # Add knowledge type nodes
        for i, kt in enumerate(knowledge_types):
            node_id = f"knowledge_type_{i}"
            nodes.append({
                "node_id": node_id,
                "node_type": "KNOWLEDGE_TYPE",
                "description": f"Knowledge Type: {kt}",
            })
            edges.append({
                "from_node": "intent",
                "to_node": node_id,
                "edge_type": "data_flow",
            })
        
        return {
            "query": query,
            "nodes": nodes,
            "edges": edges,
            "execution_trace": [n["node_id"] for n in nodes],
            "total_time_ms": 0.0,
        }

