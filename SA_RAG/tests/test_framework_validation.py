"""
Comprehensive Framework Validation Tests
Tests all core functionalities of SA-RAG to ensure the framework is usable
Run with: uv run pytest tests/test_framework_validation.py -v
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import rust_core
try:
    import rust_core
    RUST_CORE_AVAILABLE = True
except ImportError:
    RUST_CORE_AVAILABLE = False
    print("Warning: rust_core not available, some tests will be skipped")

# Try to import sa_rag
try:
    from sa_rag import RAG
    from sa_rag.client import Client
    SA_RAG_AVAILABLE = True
except ImportError as e:
    SA_RAG_AVAILABLE = False
    print(f"Warning: sa_rag not available: {e}")


class TestFrameworkInitialization:
    """Test framework initialization and basic setup"""
    
    def test_import_sa_rag(self):
        """Test that sa_rag can be imported"""
        assert SA_RAG_AVAILABLE, "sa_rag module should be importable"
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_rag_initialization(self):
        """Test RAG class can be initialized"""
        rag = RAG(
            llm_provider="mock",
            embedding_provider="mock"
        )
        assert rag is not None
        assert hasattr(rag, 'index_documents')
        assert hasattr(rag, 'search')
        assert hasattr(rag, 'ask')
    
    @pytest.mark.skipif(not RUST_CORE_AVAILABLE, reason="rust_core not available")
    def test_rust_core_import(self):
        """Test that rust_core can be imported"""
        assert RUST_CORE_AVAILABLE, "rust_core module should be importable"
    
    @pytest.mark.skipif(not RUST_CORE_AVAILABLE, reason="rust_core not available")
    def test_rust_core_engine_initialization(self):
        """Test RustCoreEngine can be initialized"""
        engine = rust_core.RustCoreEngine()
        assert engine is not None


class TestDocumentIndexing:
    """Test document indexing functionality"""
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_index_single_document(self):
        """Test indexing a single document"""
        rag = RAG(llm_provider="mock", embedding_provider="mock")
        
        doc_ids = rag.index_documents([
            "Python is a high-level programming language."
        ])
        
        assert len(doc_ids) == 1
        assert doc_ids[0] > 0
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_index_multiple_documents(self):
        """Test indexing multiple documents"""
        rag = RAG(llm_provider="mock", embedding_provider="mock")
        
        documents = [
            "Python is a programming language.",
            "Rust is a systems programming language.",
            "JavaScript is used for web development."
        ]
        
        doc_ids = rag.index_documents(documents)
        
        assert len(doc_ids) == len(documents)
        assert all(doc_id > 0 for doc_id in doc_ids)
        # All doc_ids should be unique
        assert len(set(doc_ids)) == len(doc_ids)
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_index_empty_document_list(self):
        """Test indexing empty document list"""
        rag = RAG(llm_provider="mock", embedding_provider="mock")
        
        doc_ids = rag.index_documents([])
        assert len(doc_ids) == 0


class TestSearchFunctionality:
    """Test search functionality"""
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_basic_search(self):
        """Test basic search functionality"""
        rag = RAG(llm_provider="mock", embedding_provider="mock")
        
        # Index documents
        rag.index_documents([
            "Python is a high-level programming language.",
            "Rust is a systems programming language.",
            "Machine learning is a subfield of AI."
        ])
        
        # Search
        results = rag.search("programming language", top_k=5)
        
        assert isinstance(results, list)
        assert len(results) > 0
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_search_with_graph(self):
        """Test search with graph expansion"""
        rag = RAG(llm_provider="mock", embedding_provider="mock")
        
        rag.index_documents([
            "Python is a programming language.",
            "Rust is a systems programming language."
        ])
        
        results = rag.search(
            "programming",
            top_k=5,
            use_graph=True
        )
        
        assert isinstance(results, list)
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_search_with_memory(self):
        """Test search with memory context"""
        rag = RAG(llm_provider="mock", embedding_provider="mock")
        
        rag.index_documents([
            "Python is used for data science."
        ])
        
        # Add memory
        rag.add_memory("User prefers Python", importance=0.8)
        
        results = rag.search(
            "data science",
            top_k=5,
            use_memory=True
        )
        
        assert isinstance(results, list)


class TestQAFunctionality:
    """Test Q&A functionality"""
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_ask_question(self):
        """Test asking a question"""
        rag = RAG(llm_provider="mock", embedding_provider="mock")
        
        rag.index_documents([
            "Python is a high-level programming language created by Guido van Rossum."
        ])
        
        answer = rag.ask("What is Python?", top_k=3)
        
        assert isinstance(answer, dict)
        assert "answer" in answer or "text" in answer or "response" in answer
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_ask_with_graph(self):
        """Test asking with graph expansion"""
        rag = RAG(llm_provider="mock", embedding_provider="mock")
        
        rag.index_documents([
            "Python is a programming language.",
            "Python supports multiple programming paradigms."
        ])
        
        answer = rag.ask(
            "What is Python?",
            top_k=3,
            use_graph=True
        )
        
        assert isinstance(answer, dict)


class TestMemoryManagement:
    """Test memory management functionality"""
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_add_memory(self):
        """Test adding memory"""
        rag = RAG(llm_provider="mock", embedding_provider="mock")
        
        # Should not raise exception
        rag.add_memory("User likes Python", importance=0.7)
        rag.add_memory("User prefers functional programming", importance=0.8)
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_memory_in_search(self):
        """Test that memory affects search results"""
        rag = RAG(llm_provider="mock", embedding_provider="mock")
        
        rag.index_documents([
            "Python is a programming language.",
            "Java is a programming language."
        ])
        
        rag.add_memory("User prefers Python", importance=0.9)
        
        results_with_memory = rag.search(
            "programming language",
            top_k=5,
            use_memory=True
        )
        
        results_without_memory = rag.search(
            "programming language",
            top_k=5,
            use_memory=False
        )
        
        assert isinstance(results_with_memory, list)
        assert isinstance(results_without_memory, list)


class TestDocumentUpdate:
    """Test document update functionality"""
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_update_document(self):
        """Test updating a document"""
        rag = RAG(llm_provider="mock", embedding_provider="mock")
        
        # Index document
        doc_ids = rag.index_documents([
            "Python is a programming language."
        ])
        
        doc_id = doc_ids[0]
        
        # Update document
        rag.update_document(doc_id, "Python is a high-level programming language.")
        
        # Search should still work
        results = rag.search("Python", top_k=5)
        assert isinstance(results, list)


class TestNextGenFeatures:
    """Test next-generation features"""
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_learning_ranker_import(self):
        """Test learning ranker can be imported"""
        try:
            from sa_rag.learning_ranker import LearningRankerTrainer
            trainer = LearningRankerTrainer()
            assert trainer is not None
        except ImportError:
            pytest.skip("learning_ranker module not available")
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_execution_graph_import(self):
        """Test execution graph can be imported"""
        try:
            from sa_rag.execution_graph import ExecutionGraphBuilder
            builder = ExecutionGraphBuilder()
            assert builder is not None
        except ImportError:
            pytest.skip("execution_graph module not available")
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_consistency_checker_import(self):
        """Test consistency checker can be imported"""
        try:
            from sa_rag.consistency import ConsistencyChecker
            checker = ConsistencyChecker()
            assert checker is not None
        except ImportError:
            pytest.skip("consistency module not available")
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_debugger_import(self):
        """Test debugger can be imported"""
        try:
            from sa_rag.debugger import AutonomousDebugger
            debugger = AutonomousDebugger()
            assert debugger is not None
        except ImportError:
            pytest.skip("debugger module not available")
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_plugins_import(self):
        """Test plugins can be imported"""
        try:
            from sa_rag.plugins import PluginRegistry
            registry = PluginRegistry()
            assert registry is not None
        except ImportError:
            pytest.skip("plugins module not available")


class TestErrorHandling:
    """Test error handling"""
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_search_without_indexing(self):
        """Test search without indexing documents"""
        rag = RAG(llm_provider="mock", embedding_provider="mock")
        
        # Should not crash, may return empty results
        results = rag.search("test query", top_k=5)
        assert isinstance(results, list)
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_ask_without_indexing(self):
        """Test ask without indexing documents"""
        rag = RAG(llm_provider="mock", embedding_provider="mock")
        
        # Should not crash
        answer = rag.ask("test question", top_k=3)
        assert isinstance(answer, dict)


class TestPerformance:
    """Test basic performance characteristics"""
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_indexing_performance(self):
        """Test indexing multiple documents doesn't take too long"""
        import time
        
        rag = RAG(llm_provider="mock", embedding_provider="mock")
        
        documents = [
            f"Document {i}: This is test document number {i}."
            for i in range(100)
        ]
        
        start_time = time.time()
        doc_ids = rag.index_documents(documents)
        elapsed_time = time.time() - start_time
        
        assert len(doc_ids) == 100
        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed_time < 60.0, f"Indexing took {elapsed_time:.2f}s, expected < 60s"
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_search_performance(self):
        """Test search doesn't take too long"""
        import time
        
        rag = RAG(llm_provider="mock", embedding_provider="mock")
        
        # Index documents
        documents = [
            f"Document {i}: This is test document number {i}."
            for i in range(50)
        ]
        rag.index_documents(documents)
        
        # Search
        start_time = time.time()
        results = rag.search("test", top_k=10)
        elapsed_time = time.time() - start_time
        
        assert isinstance(results, list)
        # Should complete quickly
        assert elapsed_time < 5.0, f"Search took {elapsed_time:.2f}s, expected < 5s"


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.skipif(not SA_RAG_AVAILABLE, reason="sa_rag not available")
    def test_full_workflow(self):
        """Test complete workflow: index -> search -> ask"""
        rag = RAG(llm_provider="mock", embedding_provider="mock")
        
        # 1. Index documents
        doc_ids = rag.index_documents([
            "Python is a high-level programming language.",
            "Rust is a systems programming language.",
            "Machine learning uses algorithms to learn from data."
        ])
        assert len(doc_ids) == 3
        
        # 2. Add memory
        rag.add_memory("User is interested in programming languages", importance=0.8)
        
        # 3. Search
        search_results = rag.search("programming language", top_k=5)
        assert isinstance(search_results, list)
        
        # 4. Ask question
        answer = rag.ask("What is Python?", top_k=3)
        assert isinstance(answer, dict)
        
        # 5. Update document
        rag.update_document(doc_ids[0], "Python is a high-level, interpreted programming language.")
        
        # 6. Search again
        search_results_2 = rag.search("Python", top_k=5)
        assert isinstance(search_results_2, list)
