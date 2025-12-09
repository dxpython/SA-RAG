"""Client API Module: Provides concise high-level interface

Wraps RAGPipeline to provide easier-to-use API
"""

from typing import List, Dict, Any, Optional, Union
from .rag import RAGPipeline
from .llm import LLMService
from .embedding import EmbeddingService
from .orchestrator import Orchestrator


class Client:
    """SA-RAG Client: Provides concise high-level API"""
    
    def __init__(
        self,
        llm_provider: str = "mock",
        embedding_provider: str = "mock",
        **kwargs
    ):
        """
        Initialize client
        
        Args:
            llm_provider: LLM service provider
            embedding_provider: Embedding service provider
            **kwargs: Other configuration parameters
        """
        llm_service = LLMService(provider=llm_provider, **kwargs)
        embedding_service = EmbeddingService(provider=embedding_provider, **kwargs)
        orchestrator = Orchestrator(llm_service)
        
        self.pipeline = RAGPipeline(
            llm_service=llm_service,
            embedding_service=embedding_service,
            orchestrator=orchestrator,
        )
    
    def index(
        self,
        texts: Union[str, List[str]],
        generate_embeddings: bool = True,
    ) -> List[int]:
        """
        Index documents
        
        Args:
            texts: Document text (string or list)
            generate_embeddings: Whether to generate embeddings
            
        Returns:
            List of document IDs
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.pipeline.index_documents(texts, generate_embeddings=generate_embeddings)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        use_graph: bool = True,
        use_memory: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents
        
        Args:
            query: Query text
            top_k: Return top-k results
            use_graph: Whether to use graph expansion
            use_memory: Whether to use long-term memory
            
        Returns:
            List of search results
        """
        return self.pipeline.search(
            query=query,
            top_k=top_k,
            use_graph_expansion=use_graph,
            use_memory=use_memory,
        )
    
    def ask(
        self,
        query: str,
        top_k: int = 5,
        use_graph: bool = True,
        use_memory: bool = False,
    ) -> Dict[str, Any]:
        """
        Q&A: retrieval + generation
        
        Args:
            query: User query
            top_k: Retrieve top-k results
            use_graph: Whether to use graph expansion
            use_memory: Whether to use long-term memory
            
        Returns:
            Dictionary containing answer and source information
        """
        return self.pipeline.ask(
            query=query,
            top_k=top_k,
            use_graph=use_graph,
            use_memory=use_memory,
        )
    
    def update_document(self, doc_id: int, new_text: str) -> bool:
        """
        Update document
        
        Args:
            doc_id: Document ID
            new_text: New document text
            
        Returns:
            Whether successful
        """
        return self.pipeline.update_document(doc_id, new_text)
    
    def add_memory(self, text: str, importance: float = 0.5) -> bool:
        """
        Add long-term memory
        
        Args:
            text: Memory content
            importance: Importance score (0.0-1.0)
            
        Returns:
            Whether successful
        """
        return self.pipeline.add_memory(text, importance)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics
        
        Returns:
            Statistics dictionary
        """
        try:
            graph_stats = self.pipeline.engine.get_graph_stats()
            return {
                "graph": graph_stats,
            }
        except Exception as e:
            return {"error": str(e)}
