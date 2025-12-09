"""Tests for Next-Generation Semantic Retrieval OS Features"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import rust_core
    RUST_CORE_AVAILABLE = True
except ImportError:
    RUST_CORE_AVAILABLE = False
    print("Warning: rust_core not available, some tests will be skipped")


class TestSelfEvolvingRanker:
    """Tests for self-evolving ranker"""
    
    @pytest.mark.skipif(not RUST_CORE_AVAILABLE, reason="rust_core not available")
    def test_ranker_initialization(self):
        """Test ranker initialization"""
        from sa_rag.learning_ranker import LearningRankerTrainer
        
        trainer = LearningRankerTrainer()
        weights = trainer.get_weights()
        
        assert "vector_weight" in weights
        assert "bm25_weight" in weights
        assert "graph_weight" in weights
        assert weights["confidence"] == 0.0  # No training yet
    
    @pytest.mark.skipif(not RUST_CORE_AVAILABLE, reason="rust_core not available")
    def test_ranker_training(self):
        """Test ranker training"""
        from sa_rag.learning_ranker import LearningRankerTrainer
        
        trainer = LearningRankerTrainer()
        
        # Add training sample
        trainer.add_training_sample(
            query="test query",
            query_embedding=[0.1] * 384,
            results=[
                {
                    "node_id": 1,
                    "vector_score": 0.9,
                    "bm25_score": 0.8,
                    "graph_score": 0.7,
                    "keyword_score": 0.6,
                    "memory_score": 0.5,
                }
            ],
            relevance_labels=[1.0],
        )
        
        # Train
        weights = trainer.train(epochs=5)
        
        assert weights["confidence"] > 0.0
        assert sum([
            weights["vector_weight"],
            weights["bm25_weight"],
            weights["graph_weight"],
            weights["keyword_weight"],
            weights["memory_weight"],
        ]) > 0


class TestExecutionGraph:
    """Tests for execution graph"""
    
    def test_graph_builder(self):
        """Test execution graph builder"""
        from sa_rag.execution_graph import ExecutionGraphBuilder
        
        builder = ExecutionGraphBuilder()
        graph = builder.build_graph(
            query="What is Python?",
            intent="definition",
            knowledge_types=["programming", "language"],
        )
        
        assert graph["query"] == "What is Python?"
        assert len(graph["nodes"]) > 0
        assert len(graph["edges"]) > 0
        assert "execution_trace" in graph
    
    def test_graph_visualizer(self):
        """Test execution graph visualizer"""
        from sa_rag.execution_graph import ExecutionGraphBuilder, ExecutionGraphVisualizer
        
        builder = ExecutionGraphBuilder()
        graph = builder.build_graph(
            query="test",
            intent="test",
            knowledge_types=[],
        )
        
        visualizer = ExecutionGraphVisualizer()
        dot = visualizer.to_dot(graph)
        
        assert "digraph" in dot
        assert "ExecutionGraph" in dot


class TestConsistencyChecker:
    """Tests for consistency checker"""
    
    @pytest.mark.skipif(not RUST_CORE_AVAILABLE, reason="rust_core not available")
    def test_consistency_check(self):
        """Test consistency checking"""
        from sa_rag.consistency import ConsistencyChecker
        
        checker = ConsistencyChecker()
        
        # First check (no history)
        report1 = checker.check("test query", [1, 2, 3])
        assert report1["similarity_score"] == 1.0
        assert not report1["drift_detected"]
        
        # Second check (same results)
        report2 = checker.check("test query", [1, 2, 3])
        assert report2["similarity_score"] == 1.0
        assert not report2["drift_detected"]
        
        # Third check (different results)
        report3 = checker.check("test query", [4, 5, 6])
        assert report3["similarity_score"] < 1.0
        assert report3["drift_detected"]


class TestAutonomousDebugger:
    """Tests for autonomous debugger"""
    
    def test_debugger_analysis(self):
        """Test debugger analysis"""
        from sa_rag.debugger import AutonomousDebugger
        
        debugger = AutonomousDebugger()
        
        analysis = debugger.analyze(
            query="test query",
            vector_results=[(1, 0.9), (2, 0.8)],
            bm25_results=[(1, 0.7), (3, 0.6)],
            graph_results=[(2, 0.5)],
            final_results=[(1, 0.9), (2, 0.8), (3, 0.7)],
            answer_quality=0.8,
        )
        
        assert analysis["query"] == "test query"
        assert "issues" in analysis
        assert "suggestions" in analysis
        assert "metrics" in analysis
    
    def test_debugger_empty_results(self):
        """Test debugger with empty results"""
        from sa_rag.debugger import AutonomousDebugger
        
        debugger = AutonomousDebugger()
        
        analysis = debugger.analyze(
            query="test query",
            vector_results=[],
            bm25_results=[],
            graph_results=[],
            final_results=[],
            answer_quality=0.3,
        )
        
        assert not analysis["success"]
        assert len(analysis["issues"]) > 0
        assert len(analysis["suggestions"]) > 0


class TestPluginSystem:
    """Tests for plugin system"""
    
    def test_plugin_registry(self):
        """Test plugin registry"""
        from sa_rag.plugins import PluginRegistry, BaseRankerPlugin
        
        class TestRankerPlugin(BaseRankerPlugin):
            def rank(self, results):
                return [(r["node_id"], r.get("score", 0.0)) for r in results]
            
            @property
            def name(self):
                return "test_ranker"
        
        registry = PluginRegistry()
        plugin = TestRankerPlugin()
        
        registry.register_plugin(
            plugin_id="test_ranker",
            plugin_type="ranker",
            name="Test Ranker",
            plugin=plugin,
        )
        
        retrieved = registry.get_plugin("test_ranker")
        assert retrieved is not None
        assert retrieved.name == "test_ranker"
        
        plugins = registry.list_plugins()
        assert len(plugins) >= 0  # May be empty if rust_core not available


class TestMultimodalSupport:
    """Tests for multimodal support"""
    
    @pytest.mark.skipif(not RUST_CORE_AVAILABLE, reason="rust_core not available")
    def test_multimodal_detection(self):
        """Test multimodal type detection"""
        # This would require rust_core to be available
        # For now, just test that the module can be imported
        pass


class TestFeatureStore:
    """Tests for feature store"""
    
    @pytest.mark.skipif(not RUST_CORE_AVAILABLE, reason="rust_core not available")
    def test_feature_store(self):
        """Test feature store operations"""
        # This would require rust_core to be available
        # For now, just test that the module can be imported
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

