"""Integration tests for complete SA-RAG workflow"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', '.venv', 'lib', 'python3.10', 'site-packages'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from sa_rag import RAG


def test_complete_workflow():
    """Test complete RAG workflow"""
    print("\n[Integration Test] Complete Workflow")
    print("-" * 70)
    
    rag = RAG()
    
    # Index documents
    documents = [
        """# Artificial Intelligence
Artificial Intelligence (AI) is a branch of computer science.
## Machine Learning
Machine Learning is a core technology of AI.
### Deep Learning
Deep Learning uses neural networks.""",
        """# Programming Languages
Python is a high-level language.
Rust focuses on safety."""
    ]
    
    doc_ids = rag.index_documents(documents, generate_embeddings=True)
    assert len(doc_ids) == 2
    print(f"✓ Indexed {len(doc_ids)} documents")
    
    # Add memory
    rag.add_memory("User is interested in AI", importance=0.9)
    print("✓ Added memory")
    
    # Basic query
    result1 = rag.ask("What is AI?", use_graph=False, use_memory=False)
    assert result1 is not None
    assert 'answer' in result1
    print(f"✓ Basic query: {len(result1['answer'])} characters")
    
    # Graph query
    result2 = rag.ask("What is the relationship between AI and ML?", use_graph=True, use_memory=False)
    assert result2 is not None
    assert len(result2.get('used_graph_nodes', [])) > 0
    print(f"✓ Graph query: {len(result2.get('used_graph_nodes', []))} graph nodes")
    
    # Memory query
    result3 = rag.ask("What am I interested in?", use_memory=True)
    assert result3 is not None
    print(f"✓ Memory query: {len(result3.get('used_memory_nodes', []))} memories")
    
    # Search
    results = rag.search("programming", top_k=3)
    assert len(results) > 0
    print(f"✓ Search: {len(results)} results")
    
    print("✓ Complete workflow test passed")


def test_hierarchical_documents():
    """Test hierarchical document parsing"""
    print("\n[Integration Test] Hierarchical Documents")
    print("-" * 70)
    
    rag = RAG()
    
    documents = [
        """# Main Topic
## Subtopic 1
Content for subtopic 1.
### Detail 1.1
More details.
## Subtopic 2
Content for subtopic 2."""
    ]
    
    doc_ids = rag.index_documents(documents, generate_embeddings=True)
    assert len(doc_ids) == 1
    print(f"✓ Indexed hierarchical document")
    
    # Query should retrieve relevant sections
    result = rag.ask("What is in subtopic 1?", use_graph=True)
    assert result is not None
    print(f"✓ Retrieved hierarchical content")


def test_mixed_retrieval():
    """Test mixed retrieval methods"""
    print("\n[Integration Test] Mixed Retrieval")
    print("-" * 70)
    
    rag = RAG()
    
    documents = [
        "Python is a programming language.",
        "Machine learning uses Python.",
        "Deep learning is a subset of machine learning."
    ]
    
    rag.index_documents(documents, generate_embeddings=True)
    
    # Query with all methods
    result = rag.ask(
        "What is the relationship between Python and machine learning?",
        top_k=5,
        use_graph=True,
        use_memory=False
    )
    
    assert result is not None
    assert len(result.get('used_semantic_nodes', [])) > 0
    print(f"✓ Mixed retrieval: {len(result.get('used_semantic_nodes', []))} semantic nodes")
    print(f"  Graph nodes: {len(result.get('used_graph_nodes', []))}")


def run_all_tests():
    """Run all integration tests"""
    print("=" * 70)
    print("Integration Tests")
    print("=" * 70)
    
    tests = [
        test_complete_workflow,
        test_hierarchical_documents,
        test_mixed_retrieval,
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

