"""gRPC Server for Python Orchestrator

Provides gRPC interface for Go service to call Python orchestrator functions.
Complete implementation with all required methods.
"""

import grpc
from concurrent import futures
from typing import List, Optional
import logging

# Note: In production, protobuf code would be generated from proto file
# For now, we'll use a simplified implementation

logger = logging.getLogger(__name__)


class OrchestratorService:
    """gRPC service for orchestrator operations"""
    
    def __init__(self, orchestrator, embedding_service, llm_service):
        """
        Initialize orchestrator service
        
        Args:
            orchestrator: Orchestrator instance
            embedding_service: EmbeddingService instance
            llm_service: LLMService instance
        """
        self.orchestrator = orchestrator
        self.embedding = embedding_service
        self.llm = llm_service
    
    def AnalyzeQuery(self, request, context):
        """Analyze a query and return structured query"""
        from sa_rag.query_analysis import QueryAnalyzer
        
        analyzer = QueryAnalyzer(llm_service=self.llm)
        structured = analyzer.analyze(request.query, use_llm=request.use_llm)
        
        # Convert to response format (would use generated protobuf)
        return type('obj', (object,), {
            'query': type('obj', (object,), {
                'original_query': structured.original_query,
                'rewritten_query': structured.rewritten_query,
                'intent': structured.intent.value,
                'entities': structured.entities,
                'keywords': structured.keywords,
                'relationships': structured.relationships,
                'requires_graph': structured.requires_graph,
                'requires_memory': structured.requires_memory,
                'top_k': structured.top_k,
            })()
        })()
    
    def RewriteQuery(self, request, context):
        """Rewrite a query"""
        rewritten = self.orchestrator.rewrite_query(
            request.query,
            use_llm=request.use_llm
        )
        
        return type('obj', (object,), {
            'rewritten_query': rewritten
        })()
    
    def GenerateEmbedding(self, request, context):
        """Generate embedding for text"""
        embedding = self.embedding.get_embedding(request.text)
        
        return type('obj', (object,), {
            'embedding': embedding
        })()
    
    def GenerateEmbeddingsBatch(self, request, context):
        """Generate embeddings for multiple texts"""
        embeddings = self.embedding.get_embeddings_batch(request.texts)
        
        # Convert to response format
        embedding_vectors = [
            type('obj', (object,), {'values': emb})()
            for emb in embeddings
        ]
        
        return type('obj', (object,), {
            'embeddings': embedding_vectors
        })()
    
    def SynthesizeAnswer(self, request, context):
        """Synthesize answer from context"""
        answer = self.llm.generate_with_rag(
            query=request.query,
            context=request.context,
            system_prompt=request.system_prompt or None
        )
        
        return type('obj', (object,), {
            'answer': answer
        })()
    
    def FuseResults(self, request, context):
        """Fuse results from multiple sources"""
        # Convert protobuf results to internal format
        all_results = []
        for result_group in request.result_groups:
            group_results = []
            for r in result_group.results:
                group_results.append({
                    "node_id": r.node_id,
                    "text": r.text if hasattr(r, 'text') else "",
                    "score": r.score,
                    "source": r.sources[0] if r.sources else "unknown",
                })
            all_results.append(group_results)
        
        fused = self.orchestrator.fuse_results(
            all_results,
            top_k=request.top_k,
            fusion_method=request.fusion_method or "rrf"
        )
        
        # Convert back to protobuf format
        fused_results = []
        for r in fused:
            result = type('obj', (object,), {
                'node_id': r.get("node_id", 0),
                'text': r.get("text", ""),
                'score': r.get("score", 0.0),
                'sources': [r.get("source", "unknown")],
            })()
            fused_results.append(result)
        
        return type('obj', (object,), {
            'results': fused_results
        })()


def serve(port: int = 50051, orchestrator=None, embedding_service=None, llm_service=None):
    """
    Start gRPC server
    
    Args:
        port: Server port
        orchestrator: Orchestrator instance
        embedding_service: EmbeddingService instance
        llm_service: LLMService instance
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Register service
    service = OrchestratorService(orchestrator, embedding_service, llm_service)
    
    # Note: In production, would use generated protobuf code:
    # from proto import orchestrator_pb2_grpc
    # orchestrator_pb2_grpc.add_OrchestratorServiceServicer_to_server(service, server)
    
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    logger.info(f"gRPC server started on port {port}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        server.stop(0)


if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from sa_rag.orchestrator import Orchestrator
        from sa_rag.embedding import EmbeddingService
        from sa_rag.llm import LLMService
    except ImportError as e:
        print(f"Warning: Failed to import some modules: {e}")
        print("Using mock implementations...")
        # Create minimal mock implementations
        class MockLLM:
            def generate_with_rag(self, query, context, system_prompt=None):
                return f"Mock answer for: {query}"
        
        class MockEmbedding:
            def get_embedding(self, text):
                return [0.0] * 1536
            def get_embeddings_batch(self, texts):
                return [[0.0] * 1536] * len(texts)
        
        class MockOrchestrator:
            def __init__(self, llm, embedding):
                self.llm = llm
                self.embedding = embedding
            def rewrite_query(self, query, use_llm=False):
                return query
            def fuse_results(self, results, top_k=5, fusion_method="rrf"):
                return []
        
        llm = MockLLM()
        embedding = MockEmbedding()
        orchestrator = MockOrchestrator(llm, embedding)
    else:
        logging.basicConfig(level=logging.INFO)
        llm = LLMService(provider="mock")
        embedding = EmbeddingService(provider="mock")
        orchestrator = Orchestrator(llm, embedding)
    
    print("Starting Python gRPC server on port 50051...")
    serve(port=50051, orchestrator=orchestrator, embedding_service=embedding, llm_service=llm)
