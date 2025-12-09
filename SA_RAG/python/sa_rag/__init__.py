"""SA-RAG: Semantic-Accelerated Retrieval Augmentation

A high-performance RAG system including:
- Rust high-performance core
- Python LLM/RAG orchestration layer
- Semantic node retrieval
- Graph structure retrieval
- Hybrid retrieval (vector + BM25)
- Long-term memory management
- Differential indexing
"""

from typing import List
# Lazy import to avoid rust_core dependency at module level
# from .rag import RAGPipeline
from .client import Client
from .llm import LLMService
from .embedding import EmbeddingService
from .orchestrator import Orchestrator

__version__ = "0.1.0"
__all__ = ["RAG", "Client", "RAGPipeline", "LLMService", "EmbeddingService", "Orchestrator"]


class RAG:
    """SA-RAG Main Class: Provides simplest usage interface"""
    
    def __init__(
        self,
        llm_provider: str = "mock",
        embedding_provider: str = "mock",
        **kwargs
    ):
        """
        Initialize RAG system
        
        Args:
            llm_provider: LLM service provider ("openai", "deepseek", "mock")
            embedding_provider: Embedding service provider ("openai", "deepseek", "mock")
            **kwargs: Other configuration parameters
        """
        self.client = Client(
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            **kwargs
        )
        self.pipeline = self.client.pipeline
    
    def index_text(self, doc_id_str: str, text: str, generate_embeddings: bool = True):
        """
        Index a single document
        
        Args:
            doc_id_str: Document identifier (string, for display)
            text: Document text
            generate_embeddings: Whether to generate embeddings
        """
        self.client.index([text], generate_embeddings=generate_embeddings)
    
    def index_documents(self, texts: List[str], generate_embeddings: bool = True):
        """
        Index multiple documents
        
        Args:
            texts: List of document texts
            generate_embeddings: Whether to generate embeddings
            
        Returns:
            List of document IDs
        """
        return self.client.index(texts, generate_embeddings=generate_embeddings)
    
    def ask(
        self,
        query: str,
        top_k: int = 5,
        use_graph: bool = True,
        use_memory: bool = False,
    ) -> dict:
        """
        Q&A interface
        
        Args:
            query: User query
            top_k: Retrieve top-k results
            use_graph: Whether to use graph expansion
            use_memory: Whether to use long-term memory
            
        Returns:
            Dictionary containing answer, sources, scores, etc.
        """
        return self.client.ask(
            query=query,
            top_k=top_k,
            use_graph=use_graph,
            use_memory=use_memory,
        )
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        use_graph: bool = True,
        use_memory: bool = False,
    ) -> List[dict]:
        """
        Search interface
        
        Args:
            query: Query text
            top_k: Return top-k results
            use_graph: Whether to use graph expansion
            use_memory: Whether to use long-term memory
            
        Returns:
            List of search results
        """
        return self.client.search(
            query=query,
            top_k=top_k,
            use_graph=use_graph,
            use_memory=use_memory,
        )
    
    def add_memory(self, text: str, importance: float = 0.5):
        """
        Add long-term memory
        
        Args:
            text: Memory content
            importance: Importance score (0.0-1.0)
        """
        self.client.add_memory(text, importance)
    
    def update_document(self, doc_id: int, new_text: str):
        """
        Update document
        
        Args:
            doc_id: Document ID
            new_text: New document text
        """
        self.client.update_document(doc_id, new_text)
