"""Execution Graph Visualizer
Visualizes execution graphs for debugging and explainability
"""

from typing import Dict, Optional
import json


class ExecutionGraphVisualizer:
    """Visualizer for execution graphs"""
    
    def __init__(self):
        """Initialize visualizer"""
        pass
    
    def to_dot(self, graph: Dict) -> str:
        """Convert graph to Graphviz DOT format
        
        Args:
            graph: Execution graph dictionary
            
        Returns:
            DOT format string
        """
        lines = ["digraph ExecutionGraph {"]
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=box];')
        
        # Add nodes
        for node in graph.get("nodes", []):
            node_id = node["node_id"]
            node_type = node.get("node_type", "UNKNOWN")
            description = node.get("description", "")
            lines.append(f'  "{node_id}" [label="{node_type}\\n{description}"];')
        
        # Add edges
        for edge in graph.get("edges", []):
            from_node = edge["from_node"]
            to_node = edge["to_node"]
            edge_type = edge.get("edge_type", "data_flow")
            lines.append(f'  "{from_node}" -> "{to_node}" [label="{edge_type}"];')
        
        lines.append("}")
        return "\n".join(lines)
    
    def to_json(self, graph: Dict, indent: int = 2) -> str:
        """Convert graph to formatted JSON
        
        Args:
            graph: Execution graph dictionary
            indent: JSON indentation
            
        Returns:
            Formatted JSON string
        """
        return json.dumps(graph, indent=indent)
    
    def get_execution_trace(self, graph: Dict) -> list:
        """Get execution trace from graph
        
        Args:
            graph: Execution graph dictionary
            
        Returns:
            List of node IDs in execution order
        """
        return graph.get("execution_trace", [])

