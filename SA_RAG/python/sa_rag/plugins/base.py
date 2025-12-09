"""Base Plugin Classes
Base classes for implementing custom plugins
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple


class BaseRankerPlugin(ABC):
    """Base class for custom ranker plugins"""
    
    @abstractmethod
    def rank(self, results: List[Dict]) -> List[Tuple[int, float]]:
        """Rank retrieval results
        
        Args:
            results: List of retrieval results with scores
            
        Returns:
            List of (node_id, score) tuples, sorted by score
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass


class BaseNodeParserPlugin(ABC):
    """Base class for custom node parser plugins"""
    
    @abstractmethod
    def parse(self, doc_id: int, text: str) -> List[Dict]:
        """Parse document into semantic nodes
        
        Args:
            doc_id: Document ID
            text: Document text
            
        Returns:
            List of semantic node dictionaries
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass


class BaseGraphPolicyPlugin(ABC):
    """Base class for custom graph policy plugins"""
    
    @abstractmethod
    def should_expand(self, node_id: int, current_hops: int) -> bool:
        """Determine if a node should be expanded
        
        Args:
            node_id: Node ID
            current_hops: Current expansion depth
            
        Returns:
            True if node should be expanded
        """
        pass
    
    @abstractmethod
    def expansion_priority(self, node_id: int) -> float:
        """Get expansion priority for a node
        
        Args:
            node_id: Node ID
            
        Returns:
            Priority score (0.0-1.0)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass

