"""Tests for Python layer functionality"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', '.venv', 'lib', 'python3.10', 'site-packages'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from sa_rag import RAG
from sa_rag.embedding import EmbeddingService
from sa_rag.llm import LLMService
from sa_rag.orchestrator import Orchestrator


def test_rag_import():
    """Test RAG import"""
    assert RAG is not None
    print("✓ RAG imported successfully")


def test_rag_initialization():
    """Test RAG initialization"""
    rag = RAG()
    assert rag is not None
    print("✓ RAG initialized successfully")


def test_embedding_service():
    """Test embedding service"""
    embedding = EmbeddingService(provider="mock")
    emb = embedding.get_embedding("Test text")
    assert emb is not None
    assert len(emb) > 0
    print(f"✓ Embedding service: {len(emb)} dimensions")


def test_llm_service():
    """Test LLM service"""
    llm = LLMService(provider="mock")
    response = llm.chat_completion("Hello")
    assert response is not None
    assert len(response) > 0
    print(f"✓ LLM service: {len(response)} characters")


def test_orchestrator():
    """Test orchestrator"""
    llm = LLMService(provider="mock")
    orchestrator = Orchestrator(llm)
    rewritten = orchestrator.rewrite_query("Test query")
    assert rewritten is not None
    print(f"✓ Orchestrator: {rewritten}")


def test_document_indexing():
    """Test document indexing"""
    rag = RAG()
    doc_ids = rag.index_documents(["Test document 1", "Test document 2"], generate_embeddings=True)
    assert doc_ids is not None
    assert len(doc_ids) == 2
    print(f"✓ Document indexing: {len(doc_ids)} documents")


def test_basic_query():
    """Test basic query"""
    rag = RAG()
    rag.index_documents(["Python is a programming language"], generate_embeddings=True)
    result = rag.ask("What is Python?", top_k=3, use_graph=False, use_memory=False)
    assert result is not None
    assert 'answer' in result
    print(f"✓ Basic query: {len(result['answer'])} characters")


def test_graph_query():
    """Test graph expansion query"""
    rag = RAG()
    rag.index_documents(["Python is a programming language"], generate_embeddings=True)
    result = rag.ask("What is Python?", top_k=3, use_graph=True, use_memory=False)
    assert result is not None
    assert 'used_graph_nodes' in result
    print(f"✓ Graph query: {len(result.get('used_graph_nodes', []))} graph nodes")


def test_memory_query():
    """Test memory query"""
    rag = RAG()
    rag.add_memory("User likes Python", importance=0.8)
    result = rag.ask("What do I like?", use_memory=True)
    assert result is not None
    print(f"✓ Memory query: {len(result.get('used_memory_nodes', []))} memories")


def test_search():
    """Test search functionality"""
    rag = RAG()
    rag.index_documents(["Python programming", "Machine learning"], generate_embeddings=True)
    results = rag.search("Python", top_k=3)
    assert results is not None
    assert len(results) > 0
    print(f"✓ Search: {len(results)} results")


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("Python Layer Tests")
    print("=" * 70)
    
    tests = [
        test_rag_import,
        test_rag_initialization,
        test_embedding_service,
        test_llm_service,
        test_orchestrator,
        test_document_indexing,
        test_basic_query,
        test_graph_query,
        test_memory_query,
        test_search,
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

