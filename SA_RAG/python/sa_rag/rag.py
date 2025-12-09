"""RAG Pipeline Module: Implements complete retrieval-augmented generation workflow

Features:
- Document indexing and embedding
- Hybrid retrieval (vector + BM25 + graph)
- Result fusion and ranking
- Context-based answer generation
"""

try:
    import sa_rag_core as rust_core
except ImportError:
    rust_core = None  # Will use mock implementations
from typing import List, Dict, Any, Optional
from .llm import LLMService
from .embedding import EmbeddingService
from .orchestrator import Orchestrator


class MockRustCoreEngine:
    def index_documents(self, texts):
        return [i + 1 for i in range(len(texts))]
    
    def get_node_ids_for_doc(self, doc_id):
        # Return dummy node IDs
        return [doc_id * 10 + i for i in range(3)]
        
    def update_embeddings(self, node_ids, embeddings):
        pass
        
    def search_full(self, query, k, vec):
        # Return dummy results: (node_id, text, score)
        return [(i, f"Mock result for {query} node {i}", 0.8 - (i*0.1)) for i in range(1, min(k, 5) + 1)]
        
    def expand_nodes(self, seed_ids, hops):
        return seed_ids + [999]
    
    def search_with_memory(self, query, top_k=5):
        return ["Mock memory 1", "Mock memory 2"][:top_k]

    def update_document(self, doc_id, new_text):
        return True
        
    def add_memory(self, text, importance):
        return True
    
    def get_graph_stats(self):
        return {"numNodes": 100, "numEdges": 200}

class RAGPipeline:
    """RAG Pipeline: Complete retrieval-augmented generation workflow"""
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        orchestrator: Optional[Orchestrator] = None,
    ):
        """
        Initialize RAG pipeline
        
        Args:
            llm_service: LLM service instance
            embedding_service: Embedding service instance
            orchestrator: Orchestrator instance
        """
        if rust_core:
            self.engine = rust_core.RustCoreEngine()
        else:
            print("Warning: Using MockRustCoreEngine")
            self.engine = MockRustCoreEngine()
        self.llm = llm_service or LLMService()
        self.embedding = embedding_service or EmbeddingService()
        self.orchestrator = orchestrator or Orchestrator(self.llm)
        self._indexed_texts = []  # Store indexed texts for embedding generation
    
    def index_documents(
        self,
        texts: List[str],
        generate_embeddings: bool = True,
        batch_size: int = 32,
    ) -> List[int]:
        """
        Index documents
        
        Args:
            texts: List of document texts
            generate_embeddings: Whether to generate embeddings
            batch_size: Batch processing size
            
        Returns:
            List of document IDs
        """
        print(f"Indexing {len(texts)} documents...")
        
        # 1. Index documents to Rust engine (parse semantic nodes, build graph, BM25 index)
        # Store texts for later use in embedding generation
        self._indexed_texts = texts.copy() if hasattr(texts, 'copy') else list(texts)
        try:
            doc_ids = self.engine.index_documents(texts)
            if doc_ids is None:
                doc_ids = []
        except Exception as e:
            print(f"Error indexing documents: {e}")
            doc_ids = []
        
        # 2. Generate embeddings (if needed)
        if generate_embeddings:
            print("Generating embeddings...")
            for i, doc_id in enumerate(doc_ids):
                # Get all nodes for document
                try:
                    node_ids = self.engine.get_node_ids_for_doc(doc_id)
                except Exception as e:
                    print(f"Warning: Could not get node IDs for doc {doc_id}: {e}")
                    continue
                
                if not node_ids or len(node_ids) == 0:
                    continue
                
                # Get node texts (simplified: use document text for all nodes)
                # In actual implementation, should get actual node text from engine
                # For now, we'll generate embeddings based on document text
                # This is a simplified approach - in production, nodes should have their own text
                doc_text = self._indexed_texts[i] if i < len(self._indexed_texts) else ""
                
                # Generate one embedding per node (simplified: use document text)
                # In production, each node should have its own text extracted
                node_texts = [doc_text] * len(node_ids)  # Simplified: use doc text for all nodes
                
                # Generate embeddings in batch
                embeddings = []
                for j in range(0, len(node_texts), batch_size):
                    batch_texts = node_texts[j:j+batch_size]
                    batch_embeddings = self.embedding.get_embeddings_batch(batch_texts)
                    embeddings.extend(batch_embeddings)
                
                # Update embeddings to index
                if embeddings and len(embeddings) == len(node_ids):
                    self.engine.update_embeddings(node_ids, embeddings)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(doc_ids)} documents")
        
        print("Indexing complete.")
        return doc_ids
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        use_graph_expansion: bool = False,
        graph_hops: int = 2,
        use_memory: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents
        
        Args:
            query: Query text
            top_k: Return top-k results
            use_graph_expansion: Whether to use graph expansion
            graph_hops: Number of graph expansion hops
            use_memory: Whether to use long-term memory
            
        Returns:
            List of search results (containing node_id, text, score, etc.)
        """
        # 1. Query rewriting
        rewritten_query = self.orchestrator.rewrite_query(query, use_llm=False)
        
        # 2. Generate query embedding
        query_vec = self.embedding.get_embedding(rewritten_query)
        
        # 3. Hybrid retrieval (vector + BM25)
        results = self.engine.search_full(rewritten_query, top_k * 2, query_vec)
        
        formatted_results = []
        for (node_id, text, score) in results:
            formatted_results.append({
                "node_id": node_id,
                "text": text,
                "score": score,
                "source": "hybrid",
            })
        
        # 4. Graph expansion (if needed)
        graph_nodes = []
        if use_graph_expansion and formatted_results:
            seed_node_ids = [r["node_id"] for r in formatted_results[:top_k]]
            expanded_node_ids = self.engine.expand_nodes(seed_node_ids, graph_hops)
            
            # Get expanded node information
            # Simplified: we can't get node text directly, so we'll skip graph nodes for now
            # In production, we should have a method to get node text by ID
            for node_id in expanded_node_ids:
                if node_id not in seed_node_ids:
                    # Try to get text via search (simplified approach)
                    # In production, should have get_node_info or similar method
                    graph_nodes.append({
                        "node_id": node_id,
                        "text": f"[Graph-expanded node {node_id}]",  # Placeholder
                        "score": 0.5,  # Default score for graph-expanded nodes
                        "source": "graph",
                    })
        
        # 5. Long-term memory retrieval (if needed)
        memory_results = []
        if use_memory:
            # Old version doesn't support top_k parameter
            try:
                memory_texts = self.engine.search_with_memory(query, top_k)
            except TypeError:
                # Fallback for old version
                memory_texts = self.engine.search_with_memory(query)
                memory_texts = memory_texts[:top_k] if len(memory_texts) > top_k else memory_texts
            for i, mem_text in enumerate(memory_texts):
                memory_results.append({
                    "node_id": 0,  # Memory has no node ID
                    "text": mem_text,
                    "score": 0.8 - i * 0.1,  # Memory score
                    "source": "memory",
                })
        
        # 6. Fuse results
        all_results = [formatted_results, graph_nodes, memory_results]
        fused_results = self.orchestrator.fuse_results(
            all_results,
            method="rrf",
            top_k=top_k,
        )
        
        return fused_results
    
    def generate_answer(
        self,
        query: str,
        context_nodes: List[Dict[str, Any]],
        include_sources: bool = True,
    ) -> str:
        """
        Generate answer based on retrieval results
        
        Args:
            query: User query
            context_nodes: Retrieved context nodes
            include_sources: Whether to include source information in answer
            
        Returns:
            Generated answer
        """
        # Build context
        context_parts = []
        for i, node in enumerate(context_nodes):
            source_info = f"[Source {i+1}: {node.get('source', 'unknown')}]"
            if include_sources:
                context_parts.append(f"{source_info}\n{node.get('text', '')}")
            else:
                context_parts.append(node.get('text', ''))
        
        context_str = "\n\n".join(context_parts)
        
        # Use LLM to generate answer
        answer = self.llm.generate_with_rag(
            query=query,
            retrieved_context=context_nodes,
        )
        
        return answer
    
    def ask(
        self,
        query: str,
        top_k: int = 5,
        use_graph: bool = True,
        use_memory: bool = False,
    ) -> Dict[str, Any]:
        """
        Complete Q&A workflow: retrieval + generation
        
        Args:
            query: User query
            top_k: Retrieve top-k results
            use_graph: Whether to use graph expansion
            use_memory: Whether to use long-term memory
            
        Returns:
            Dictionary containing answer, sources, scores, etc.
        """
        # 1. Plan retrieval strategy
        strategy = self.orchestrator.plan_retrieval(query)
        
        # 2. Execute retrieval
        results = self.search(
            query=query,
            top_k=top_k,
            use_graph_expansion=use_graph and strategy.get("use_graph", False),
            graph_hops=strategy.get("graph_hops", 2),
            use_memory=use_memory and strategy.get("use_memory", False),
        )
        
        # 3. Generate answer
        answer = self.generate_answer(query, results, include_sources=True)
        
        # 4. Extract used node information
        used_semantic_nodes = [
            r for r in results if r.get("source") in ["hybrid", "graph"]
        ]
        used_graph_nodes = [
            r for r in results if r.get("source") == "graph"
        ]
        
        # 5. Build return result
        return {
            "answer": answer,
            "used_semantic_nodes": used_semantic_nodes,
            "used_graph_nodes": used_graph_nodes,
            "scoring_details": {
                r["node_id"]: {
                    "score": r.get("score", 0.0),
                    "source": r.get("source", "unknown"),
                }
                for r in results
            },
            "total_results": len(results),
        }
    
    def update_document(self, doc_id: int, new_text: str) -> bool:
        """
        Update document (differential indexing)
        
        Args:
            doc_id: Document ID
            new_text: New document text
            
        Returns:
            Whether successful
        """
        try:
            self.engine.update_document(doc_id, new_text)
            return True
        except Exception as e:
            print(f"Failed to update document: {e}")
            return False
    
    def add_memory(self, text: str, importance: float = 0.5) -> bool:
        """
        Add long-term memory
        
        Args:
            text: Memory content
            importance: Importance score (0.0-1.0)
            
        Returns:
            Whether successful
        """
        try:
            self.engine.add_memory(text, importance)
            return True
        except Exception as e:
            print(f"Failed to add memory: {e}")
            return False
    
    def debug_retrieval(
        self,
        query: str,
        top_k: int = 5,
        use_graph: bool = False,
        use_memory: bool = False,
    ) -> Dict[str, Any]:
        """
        Debug retrieval process: returns detailed information about retrieval
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            use_graph: Whether to use graph expansion
            use_memory: Whether to use long-term memory
            
        Returns:
            Dictionary containing:
            - query_analysis: Structured query analysis
            - retrieval_strategy: Planned retrieval strategy
            - results: Retrieved results with scores
            - score_breakdown: Score breakdown by source
            - statistics: Retrieval statistics
        """
        # Analyze query
        structured_query = self.orchestrator.analyze_query(query, use_llm=False)
        
        # Plan retrieval strategy
        strategy = self.orchestrator.plan_retrieval(query, use_llm_analysis=False)
        
        # Generate query embedding
        query_vec = self.embedding.get_embedding(structured_query.rewritten_query)
        
        # Execute retrieval
        results = self.search(
            query=structured_query.rewritten_query,
            top_k=top_k,
            use_graph_expansion=use_graph or strategy.get("use_graph", False),
            graph_hops=strategy.get("graph_hops", 1),
            use_memory=use_memory or strategy.get("use_memory", False),
        )
        
        # Collect statistics
        vector_count = sum(1 for r in results if "vector" in r.get("source", ""))
        bm25_count = sum(1 for r in results if "bm25" in r.get("source", ""))
        graph_count = sum(1 for r in results if r.get("source") == "graph")
        memory_count = sum(1 for r in results if r.get("source") == "memory")
        
        # Score statistics
        scores = [r.get("score", 0.0) for r in results]
        score_stats = {
            "max": max(scores) if scores else 0.0,
            "min": min(scores) if scores else 0.0,
            "avg": sum(scores) / len(scores) if scores else 0.0,
            "count": len(scores),
        }
        
        # Score breakdown by source
        score_breakdown = {}
        for r in results:
            source = r.get("source", "unknown")
            if source not in score_breakdown:
                score_breakdown[source] = []
            score_breakdown[source].append(r.get("score", 0.0))
        
        return {
            "query_analysis": {
                "original_query": structured_query.original_query,
                "rewritten_query": structured_query.rewritten_query,
                "intent": structured_query.intent.value,
                "entities": structured_query.entities,
                "keywords": structured_query.keywords,
                "requires_graph": structured_query.requires_graph,
                "requires_memory": structured_query.requires_memory,
            },
            "retrieval_strategy": strategy,
            "results": results,
            "score_breakdown": {
                source: {
                    "count": len(scores),
                    "avg": sum(scores) / len(scores) if scores else 0.0,
                    "max": max(scores) if scores else 0.0,
                    "min": min(scores) if scores else 0.0,
                }
                for source, scores in score_breakdown.items()
            },
            "statistics": {
                "total_results": len(results),
                "vector_count": vector_count,
                "bm25_count": bm25_count,
                "graph_count": graph_count,
                "memory_count": memory_count,
                "score_stats": score_stats,
            },
        }
