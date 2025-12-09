"""Tests for Phase 2 Enhanced Features"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', '.venv', 'lib', 'python3.10', 'site-packages'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from sa_rag import RAG
from sa_rag.query_analysis import QueryAnalyzer, QueryIntent
from sa_rag.llm import LLMService
from sa_rag.embedding import EmbeddingService


def test_query_analyzer():
    """Test QueryAnalyzer functionality"""
    print("\n[Test] Query Analyzer")
    print("-" * 70)
    
    analyzer = QueryAnalyzer()
    
    # Test definition query
    query1 = "What is machine learning?"
    result1 = analyzer.analyze(query1, use_llm=False)
    assert result1.intent == QueryIntent.DEFINITION
    assert len(result1.keywords) > 0
    print(f"✓ Definition query: {result1.intent.value}")
    
    # Test comparison query
    query2 = "Compare Python and Rust"
    result2 = analyzer.analyze(query2, use_llm=False)
    assert result2.intent == QueryIntent.COMPARISON
    assert result2.requires_graph
    print(f"✓ Comparison query: {result2.intent.value}, requires_graph={result2.requires_graph}")
    
    # Test procedure query
    query3 = "How to implement a neural network?"
    result3 = analyzer.analyze(query3, use_llm=False)
    assert result3.intent == QueryIntent.PROCEDURE
    print(f"✓ Procedure query: {result3.intent.value}")
    
    print("✓ Query Analyzer tests passed")


def test_structured_query():
    """Test StructuredQuery generation"""
    print("\n[Test] Structured Query")
    print("-" * 70)
    
    analyzer = QueryAnalyzer()
    query = "What is the relationship between AI and machine learning?"
    
    structured = analyzer.analyze(query, use_llm=False)
    
    assert structured.original_query == query
    assert structured.rewritten_query is not None
    assert structured.intent in QueryIntent
    assert isinstance(structured.entities, list)
    assert isinstance(structured.keywords, list)
    assert isinstance(structured.requires_graph, bool)
    assert isinstance(structured.requires_memory, bool)
    assert structured.top_k > 0
    
    print(f"✓ Original query: {structured.original_query}")
    print(f"✓ Rewritten query: {structured.rewritten_query}")
    print(f"✓ Intent: {structured.intent.value}")
    print(f"✓ Keywords: {structured.keywords}")
    print(f"✓ Requires graph: {structured.requires_graph}")
    print(f"✓ Top-k: {structured.top_k}")
    print("✓ Structured Query tests passed")


def test_rag_debug_retrieval():
    """Test debug_retrieval functionality"""
    print("\n[Test] RAG Debug Retrieval")
    print("-" * 70)
    
    rag = RAG()
    
    # Index some documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Python is a popular programming language for data science."
    ]
    
    doc_ids = rag.index_documents(documents, generate_embeddings=True)
    assert len(doc_ids) == 3
    print(f"✓ Indexed {len(doc_ids)} documents")
    
    # Test debug retrieval
    debug_info = rag.pipeline.debug_retrieval(
        "What is machine learning?",
        top_k=3,
        use_graph=False,
        use_memory=False
    )
    
    assert "query_analysis" in debug_info
    assert "retrieval_strategy" in debug_info
    assert "results" in debug_info
    assert "score_breakdown" in debug_info
    assert "statistics" in debug_info
    
    print(f"✓ Query analysis: {debug_info['query_analysis']['intent']}")
    print(f"✓ Results count: {debug_info['statistics']['total_results']}")
    print(f"✓ Score breakdown: {list(debug_info['score_breakdown'].keys())}")
    print("✓ Debug Retrieval tests passed")


def test_embedding_cache():
    """Test embedding service cache"""
    print("\n[Test] Embedding Cache")
    print("-" * 70)
    
    embedding = EmbeddingService(provider="mock", enable_cache=True)
    
    text = "Test text for caching"
    
    # First call
    emb1 = embedding.get_embedding(text)
    assert len(emb1) > 0
    
    # Second call (should use cache)
    emb2 = embedding.get_embedding(text)
    assert emb1 == emb2
    
    # Check cache stats
    stats = embedding.get_cache_stats()
    assert stats["size"] > 0
    
    print(f"✓ Cache size: {stats['size']}")
    print(f"✓ Cache hit verified")
    print("✓ Embedding Cache tests passed")


def test_orchestrator_strategy():
    """Test orchestrator strategy planning"""
    print("\n[Test] Orchestrator Strategy")
    print("-" * 70)
    
    llm = LLMService(provider="mock")
    from sa_rag.orchestrator import Orchestrator
    orchestrator = Orchestrator(llm)
    
    # Test different query types
    queries = [
        ("What is AI?", QueryIntent.DEFINITION),
        ("Compare Python and Rust", QueryIntent.COMPARISON),
        ("How to train a model?", QueryIntent.PROCEDURE),
    ]
    
    for query, expected_intent in queries:
        strategy = orchestrator.plan_retrieval(query, use_llm_analysis=False)
        
        assert "use_graph" in strategy
        assert "use_memory" in strategy
        assert "top_k" in strategy
        assert "intent" in strategy
        
        print(f"✓ Query: {query}")
        print(f"  Intent: {strategy['intent']}")
        print(f"  Use graph: {strategy['use_graph']}")
        print(f"  Top-k: {strategy['top_k']}")
    
    print("✓ Orchestrator Strategy tests passed")


def test_multi_stage_retrieval():
    """Test multi-stage retrieval workflow"""
    print("\n[Test] Multi-Stage Retrieval")
    print("-" * 70)
    
    rag = RAG()
    
    documents = [
        """# Artificial Intelligence
Artificial Intelligence (AI) is a branch of computer science.
## Machine Learning
Machine Learning is a subset of AI.
### Deep Learning
Deep Learning uses neural networks."""
    ]
    
    rag.index_documents(documents, generate_embeddings=True)
    print("✓ Documents indexed")
    
    # Test search with different strategies
    result1 = rag.ask("What is AI?", use_graph=False, use_memory=False)
    assert "answer" in result1
    assert len(result1.get("used_semantic_nodes", [])) > 0
    print(f"✓ Basic search: {len(result1.get('used_semantic_nodes', []))} nodes")
    
    result2 = rag.ask("What is the relationship between AI and ML?", use_graph=True, use_memory=False)
    assert "used_graph_nodes" in result2
    print(f"✓ Graph search: {len(result2.get('used_graph_nodes', []))} graph nodes")
    
    print("✓ Multi-Stage Retrieval tests passed")


def run_all_tests():
    """Run all Phase 2 feature tests"""
    print("=" * 70)
    print("Phase 2 Enhanced Features Tests")
    print("=" * 70)
    
    tests = [
        test_query_analyzer,
        test_structured_query,
        test_rag_debug_retrieval,
        test_embedding_cache,
        test_orchestrator_strategy,
        test_multi_stage_retrieval,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

