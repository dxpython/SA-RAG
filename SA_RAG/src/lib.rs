/// SA-RAG Rust Core Engine PyO3 Bindings
/// 
/// Next-Generation Semantic Retrieval OS
/// 
/// Provides complete RAG functionality interface:
/// - Document indexing and retrieval
/// - Semantic node parsing
/// - Hybrid retrieval (vector + BM25)
/// - Graph structure retrieval and expansion
/// - Long-term memory management
/// - Differential indexing and incremental updates
/// - Self-evolving ranker
/// - Execution graph for explainability
/// - Cognitive memory system
/// - Multimodal node support
/// - Feature store with versioning
/// - Consistency checking
/// - Plugin system

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

mod semantic_node;
mod parser;
mod indexer;
mod graph;
mod memory;
mod diff;
mod utils;
mod engine;
mod learning_ranker;
mod execution_graph;
mod feature_store;
mod multimodal;
mod consistency;
mod plugins;
mod debugger;

// Sub-modules (inline for now, can be refactored later)
#[path = "parser/enhanced.rs"]
mod parser_enhanced;
#[path = "diff/rolling_hash.rs"]
mod diff_rolling_hash;

use semantic_node::SemanticNode;
use parser::SemanticParser;
use indexer::hybrid::HybridIndex;
use graph::graph::{Graph, EdgeType};
use graph::expansion::GraphExpansion;
use memory::MemoryStore;
use diff::{DocumentVersionManager, calculate_text_diff};
use learning_ranker::{SelfEvolvingRanker, LearnedWeights};
use execution_graph::{GraphExecutor, QueryExecutionGraph, ExecutionNode, ExecutionEdge};
use feature_store::{FeatureStore, FeatureVector};
use consistency::ConsistencyChecker;
use plugins::PluginRegistry;
use debugger::AutonomousDebugger;

/// Rust Core Engine: Provides all RAG functionality
#[pyclass]
pub struct RustCoreEngine {
    /// Hybrid index (vector + BM25)
    index: Arc<Mutex<HybridIndex>>,
    /// Knowledge graph
    graph: Arc<Mutex<Graph>>,
    /// Long-term memory store
    memory: Arc<Mutex<MemoryStore>>,
    /// Document version manager (for differential indexing)
    version_manager: Arc<Mutex<DocumentVersionManager>>,
    /// Document ID counter
    doc_counter: Arc<Mutex<u64>>,
    /// Node storage: node_id -> SemanticNode
    nodes: Arc<Mutex<HashMap<u64, SemanticNode>>>,
    /// Document text storage: doc_id -> text (for diff calculation)
    doc_texts: Arc<Mutex<HashMap<u64, String>>>,
    /// Self-evolving ranker
    learning_ranker: Arc<Mutex<SelfEvolvingRanker>>,
    /// Execution graph executor
    graph_executor: Arc<GraphExecutor>,
    /// Feature store
    feature_store: Arc<FeatureStore>,
    /// Consistency checker
    consistency_checker: Arc<Mutex<ConsistencyChecker>>,
    /// Plugin registry
    plugin_registry: Arc<Mutex<PluginRegistry>>,
    /// Autonomous debugger
    debugger: Arc<Mutex<AutonomousDebugger>>,
}

#[pymethods]
impl RustCoreEngine {
    /// Create a new engine instance
    #[new]
    fn new() -> Self {
        Self {
            index: Arc::new(Mutex::new(HybridIndex::new(100000, 0.5))),
            graph: Arc::new(Mutex::new(Graph::new())),
            memory: Arc::new(Mutex::new(MemoryStore::new(0.1))),
            version_manager: Arc::new(Mutex::new(DocumentVersionManager::new())),
            doc_counter: Arc::new(Mutex::new(1)),
            nodes: Arc::new(Mutex::new(HashMap::new())),
            doc_texts: Arc::new(Mutex::new(HashMap::new())),
            learning_ranker: Arc::new(Mutex::new(SelfEvolvingRanker::new())),
            graph_executor: Arc::new(GraphExecutor::new()),
            feature_store: Arc::new(FeatureStore::new()),
            consistency_checker: Arc::new(Mutex::new(ConsistencyChecker::new())),
            plugin_registry: Arc::new(Mutex::new(PluginRegistry::new())),
            debugger: Arc::new(Mutex::new(AutonomousDebugger::new())),
        }
    }

    /// Index documents
    /// 
    /// # Arguments
    /// - `texts`: List of document texts
    /// 
    /// # Returns
    /// List of document IDs
    fn index_documents(&self, texts: Vec<String>) -> PyResult<Vec<u64>> {
        let mut doc_ids = Vec::new();
        let mut index = self.index.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock index: {}", e)))?;
        let mut graph = self.graph.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock graph: {}", e)))?;
        let mut nodes_map = self.nodes.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock nodes: {}", e)))?;
        let mut doc_texts = self.doc_texts.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock doc_texts: {}", e)))?;
        let mut counter = self.doc_counter.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock counter: {}", e)))?;
        let mut version_manager = self.version_manager.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock version_manager: {}", e)))?;

        for text in texts {
            let doc_id = *counter;
            *counter += 1;
            doc_ids.push(doc_id);

            // Save document text
            doc_texts.insert(doc_id, text.clone());

            // Add version
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            version_manager.add_version(doc_id, text.clone(), timestamp);

            // Parse semantic nodes
            let nodes = SemanticParser::parse(doc_id, &text);

            // Index nodes
            for node in &nodes {
                let embedding = node.embedding.clone();
                index.insert(node.node_id, &node.text, embedding);
                nodes_map.insert(node.node_id, node.clone());
            }

            // Build graph (add edges between sequential nodes)
            for i in 0..nodes.len().saturating_sub(1) {
                graph.add_edge(nodes[i].node_id, nodes[i + 1].node_id, EdgeType::Next, 0.8);
            }
        }

        Ok(doc_ids)
    }

    /// Get node IDs for a document
    fn get_node_ids_for_doc(&self, doc_id: u64) -> PyResult<Vec<u64>> {
        let nodes = self.nodes.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock nodes: {}", e)))?;
        let node_ids: Vec<u64> = nodes
            .values()
            .filter(|n| n.doc_id == doc_id)
            .map(|n| n.node_id)
            .collect();
        Ok(node_ids)
    }

    /// Update embeddings for nodes
    fn update_embeddings(&self, node_ids: Vec<u64>, embeddings: Vec<Vec<f32>>) -> PyResult<()> {
        if node_ids.len() != embeddings.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "node_ids and embeddings must have the same length"
            ));
        }

        let mut index = self.index.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock index: {}", e)))?;
        let mut nodes = self.nodes.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock nodes: {}", e)))?;

        for (node_id, embedding) in node_ids.iter().zip(embeddings.iter()) {
            if let Some(node) = nodes.get_mut(node_id) {
                node.embedding = Some(embedding.clone());
                // Update index with new embedding
                if let Some(text) = nodes.get(node_id).map(|n| n.text.clone()) {
                    index.insert(*node_id, &text, Some(embedding.clone()));
                }
            }
        }

        Ok(())
    }

    /// Full search: hybrid retrieval with graph expansion
    fn search_full(
        &self,
        query: String,
        k: usize,
        query_vec: Vec<f32>,
    ) -> PyResult<Vec<(u64, String, f32)>> {
        let index = self.index.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock index: {}", e)))?;
        let nodes = self.nodes.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock nodes: {}", e)))?;
        let graph = self.graph.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock graph: {}", e)))?;

        // Hybrid search
        let results = index.search(&query, Some(&query_vec), k * 2);

        // Graph expansion
        let seed_ids: Vec<u64> = results.iter().map(|r| r.0).collect();
        let expanded_ids = GraphExpansion::expand(&graph, &seed_ids, 1);

        // Combine and deduplicate
        let mut all_ids: Vec<u64> = seed_ids;
        all_ids.extend(expanded_ids);
        all_ids.sort();
        all_ids.dedup();

        // Get top-k results
        let mut final_results: Vec<(u64, String, f32)> = Vec::new();
        for id in all_ids.iter().take(k) {
            if let Some(node) = nodes.get(id) {
                let score = results
                    .iter()
                    .find(|(nid, _)| *nid == *id)
                    .map(|(_, s)| *s)
                    .unwrap_or(0.5);
                final_results.push((*id, node.text.clone(), score));
            }
        }

        Ok(final_results)
    }

    /// Expand nodes in graph
    fn expand_nodes(&self, seed_ids: Vec<u64>, hops: usize) -> PyResult<Vec<u64>> {
        let graph = self.graph.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock graph: {}", e)))?;
        let expanded = GraphExpansion::expand(&graph, &seed_ids, hops);
        Ok(expanded)
    }

    /// Search with memory context
    fn search_with_memory(&self, query: String, top_k: usize) -> PyResult<Vec<String>> {
        let mut memory = self.memory.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock memory: {}", e)))?;
        let memories = memory.retrieve_relevant(&query, top_k, None);
        Ok(memories.iter().map(|m| m.content.clone()).collect())
    }

    /// Update document (differential indexing)
    fn update_document(&self, doc_id: u64, new_text: String) -> PyResult<bool> {
        let mut doc_texts = self.doc_texts.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock doc_texts: {}", e)))?;
        let mut version_manager = self.version_manager.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock version_manager: {}", e)))?;
        let mut index = self.index.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock index: {}", e)))?;
        let mut nodes = self.nodes.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock nodes: {}", e)))?;
        let mut graph = self.graph.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock graph: {}", e)))?;

        let old_text = doc_texts.get(&doc_id).cloned().unwrap_or_default();
        
        // Calculate diff
        let diff = calculate_text_diff(&old_text, &new_text);
        
        // Update document text
        doc_texts.insert(doc_id, new_text.clone());
        
        // Add new version
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        version_manager.add_version(doc_id, new_text.clone(), timestamp);

        // Re-parse and update nodes for changed segments
        if !diff.changed_ranges.is_empty() || old_text != new_text {
            // Remove old nodes for this document
            let old_node_ids: Vec<u64> = nodes.values()
                .filter(|n| n.doc_id == doc_id)
                .map(|n| n.node_id)
                .collect();
            
            for node_id in &old_node_ids {
                nodes.remove(node_id);
                graph.remove_node(*node_id);
            }
            
            // Re-parse entire document (simplified implementation)
            let new_nodes = SemanticParser::parse(doc_id, &new_text);
            
            // Add new nodes
            for node in &new_nodes {
                let embedding = node.embedding.clone();
                index.insert(node.node_id, &node.text, embedding);
                nodes.insert(node.node_id, node.clone());
            }
            
            // Rebuild graph edges for sequential nodes
            for i in 0..new_nodes.len().saturating_sub(1) {
                graph.add_edge(new_nodes[i].node_id, new_nodes[i + 1].node_id, EdgeType::Next, 0.8);
            }
        }

        Ok(true)
    }

    /// Add memory
    fn add_memory(&self, text: String, importance: f32) -> PyResult<bool> {
        let mut memory = self.memory.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock memory: {}", e)))?;
        memory.add_memory(text, importance, None, None, None);
        Ok(true)
    }

    /// Get graph statistics
    fn get_graph_stats(&self) -> PyResult<HashMap<String, usize>> {
        let graph = self.graph.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock graph: {}", e)))?;
        let nodes = self.nodes.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock nodes: {}", e)))?;
        
        let graph_stats = graph.get_stats();
        let mut stats = HashMap::new();
        stats.insert("numNodes".to_string(), nodes.len());
        stats.insert("numEdges".to_string(), graph_stats.num_edges);
        Ok(stats)
    }

    /// Build execution graph for query
    fn build_execution_graph(
        &self,
        query: String,
        intent: String,
        knowledge_types: Vec<String>,
    ) -> PyResult<QueryExecutionGraph> {
        Ok(self.graph_executor.build_execution_graph(&query, &intent, &knowledge_types))
    }

    /// Train learning ranker
    fn train_ranker(&self, epochs: usize) -> PyResult<LearnedWeights> {
        let mut ranker = self.learning_ranker.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock ranker: {}", e)))?;
        ranker.train(epochs)
    }

    /// Get learned weights
    fn get_learned_weights(&self) -> PyResult<LearnedWeights> {
        let ranker = self.learning_ranker.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock ranker: {}", e)))?;
        Ok(ranker.get_weights())
    }

    /// Check retrieval consistency
    fn check_consistency(&self, query: String, results: Vec<u64>) -> PyResult<consistency::ConsistencyReport> {
        let mut checker = self.consistency_checker.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock checker: {}", e)))?;
        checker.check_consistency(&query, results)
    }

    /// Store feature vector
    fn store_feature(&self, feature_id: String, vector: Vec<f32>, model_version: Option<String>) -> PyResult<()> {
        self.feature_store.store(feature_id, vector, model_version, None)
    }

    /// Get feature vector
    fn get_feature(&self, feature_id: String) -> PyResult<Option<feature_store::FeatureVector>> {
        self.feature_store.get(&feature_id)
    }

    /// Analyze retrieval with debugger
    fn debug_retrieval(
        &self,
        query: String,
        vector_results: Vec<(u64, f32)>,
        bm25_results: Vec<(u64, f32)>,
        graph_results: Vec<(u64, f32)>,
        final_results: Vec<(u64, f32)>,
    ) -> PyResult<debugger::DebugAnalysis> {
        let mut debugger = self.debugger.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to lock debugger: {}", e)))?;
        debugger.analyze_retrieval(query, vector_results, bm25_results, graph_results, final_results, None, None)
    }
}

/// Python module initialization
#[pymodule]
fn sa_rag_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustCoreEngine>()?;
    m.add_class::<SelfEvolvingRanker>()?;
    m.add_class::<LearnedWeights>()?;
    m.add_class::<QueryExecutionGraph>()?;
    m.add_class::<ExecutionNode>()?;
    m.add_class::<ExecutionEdge>()?;
    m.add_class::<FeatureStore>()?;
    m.add_class::<FeatureVector>()?;
    m.add_class::<ConsistencyChecker>()?;
    m.add_class::<consistency::ConsistencyReport>()?;
    m.add_class::<PluginRegistry>()?;
    m.add_class::<plugins::PluginMetadata>()?;
    m.add_class::<AutonomousDebugger>()?;
    m.add_class::<debugger::DebugAnalysis>()?;
    m.add_class::<debugger::DebugIssue>()?;
    m.add_class::<debugger::OptimizationSuggestion>()?;
    m.add_class::<learning_ranker::RetrievalResult>()?;
    Ok(())
}
