"""Tests for Rust core functionality"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', '.venv', 'lib', 'python3.10', 'site-packages'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import rust_core


def test_rust_core_import():
    """Test that rust_core can be imported"""
    assert rust_core is not None
    print("✓ Rust core imported successfully")


def test_engine_creation():
    """Test RustCoreEngine creation"""
    engine = rust_core.RustCoreEngine()
    assert engine is not None
    print("✓ RustCoreEngine created successfully")


def test_document_indexing():
    """Test document indexing"""
    engine = rust_core.RustCoreEngine()
    doc_ids = engine.index_documents(["Test document 1", "Test document 2"])
    assert doc_ids is not None
    assert len(doc_ids) == 2
    print(f"✓ Document indexing: {doc_ids}")


def test_node_queries():
    """Test node queries"""
    engine = rust_core.RustCoreEngine()
    doc_ids = engine.index_documents(["Test document"])
    node_ids = engine.get_node_ids_for_doc(doc_ids[0])
    assert node_ids is not None
    assert len(node_ids) > 0
    print(f"✓ Node queries: {len(node_ids)} nodes")


def test_node_info():
    """Test node information retrieval"""
    engine = rust_core.RustCoreEngine()
    doc_ids = engine.index_documents(["Test document"])
    node_ids = engine.get_node_ids_for_doc(doc_ids[0])
    
    if node_ids:
        node_info = engine.get_node_info(node_ids[0])
        assert node_info is not None
        assert 'text' in node_info
        print(f"✓ Node info: {node_info.get('text', '')[:30]}...")


def test_graph_expansion():
    """Test graph expansion"""
    engine = rust_core.RustCoreEngine()
    doc_ids = engine.index_documents(["Test document 1", "Test document 2"])
    node_ids = engine.get_node_ids_for_doc(doc_ids[0])
    
    if node_ids:
        expanded = engine.expand_nodes(node_ids[:2], hops=1)
        assert expanded is not None
        assert len(expanded) >= len(node_ids[:2])
        print(f"✓ Graph expansion: {len(expanded)} nodes")


def test_smart_expansion():
    """Test smart graph expansion"""
    engine = rust_core.RustCoreEngine()
    doc_ids = engine.index_documents(["Test document"])
    node_ids = engine.get_node_ids_for_doc(doc_ids[0])
    
    if node_ids:
        expanded = engine.expand_nodes_smart(
            node_ids[:2], 
            hops=1, 
            min_weight=0.1, 
            max_nodes=10
        )
        assert expanded is not None
        print(f"✓ Smart expansion: {len(expanded)} nodes")


def test_graph_stats():
    """Test graph statistics"""
    engine = rust_core.RustCoreEngine()
    engine.index_documents(["Test document"])
    stats = engine.get_graph_stats()
    assert stats is not None
    assert 'num_nodes' in stats
    assert 'num_edges' in stats
    print(f"✓ Graph stats: {stats['num_nodes']} nodes, {stats['num_edges']} edges")


def test_search():
    """Test search functionality"""
    engine = rust_core.RustCoreEngine()
    engine.index_documents(["Test document with keywords"])
    results = engine.search("keywords", top_k=3)
    assert results is not None
    print(f"✓ Search: {len(results)} results")


def test_memory():
    """Test long-term memory"""
    engine = rust_core.RustCoreEngine()
    engine.add_memory("Test memory", importance=0.7)
    memories = engine.search_with_memory("Test", top_k=3)
    assert memories is not None
    print(f"✓ Memory: {len(memories)} memories")


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("Rust Core Tests")
    print("=" * 70)
    
    tests = [
        test_rust_core_import,
        test_engine_creation,
        test_document_indexing,
        test_node_queries,
        test_node_info,
        test_graph_expansion,
        test_smart_expansion,
        test_graph_stats,
        test_search,
        test_memory,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

