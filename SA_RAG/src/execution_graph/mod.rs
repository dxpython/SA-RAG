// Semantic Execution Graph Module
// Represents query execution as a DAG for explainability

pub mod graph;
pub mod executor;

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use petgraph::{Graph, Directed, graph::NodeIndex};

pub use graph::ExecutionGraphBuilder;
pub use executor::GraphExecutor;

/// Execution node type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[pyclass]
pub enum ExecutionNodeType {
    #[pyo3(name = "QUERY")]
    Query,
    #[pyo3(name = "INTENT")]
    Intent,
    #[pyo3(name = "KNOWLEDGE_TYPE")]
    KnowledgeType,
    #[pyo3(name = "RETRIEVAL_PLAN")]
    RetrievalPlan,
    #[pyo3(name = "VECTOR_SEARCH")]
    VectorSearch,
    #[pyo3(name = "BM25_SEARCH")]
    BM25Search,
    #[pyo3(name = "GRAPH_EXPANSION")]
    GraphExpansion,
    #[pyo3(name = "MEMORY_RETRIEVAL")]
    MemoryRetrieval,
    #[pyo3(name = "FUSION")]
    Fusion,
    #[pyo3(name = "RANKING")]
    Ranking,
    #[pyo3(name = "ANSWER")]
    Answer,
}

/// Execution node
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct ExecutionNode {
    #[pyo3(get, set)]
    pub node_id: String,
    #[pyo3(get, set)]
    pub node_type: String, // ExecutionNodeType as string
    #[pyo3(get, set)]
    pub description: String,
    #[pyo3(get, set)]
    pub input_data: String, // JSON string
    #[pyo3(get, set)]
    pub output_data: String, // JSON string
    #[pyo3(get, set)]
    pub metadata: std::collections::HashMap<String, String>,
    #[pyo3(get, set)]
    pub execution_time_ms: f64,
    #[pyo3(get, set)]
    pub source_node_ids: Vec<u64>, // Original semantic node IDs
}

#[pymethods]
impl ExecutionNode {
    #[new]
    fn new(node_id: String, node_type: String, description: String) -> Self {
        Self {
            node_id,
            node_type,
            description,
            input_data: String::new(),
            output_data: String::new(),
            metadata: std::collections::HashMap::new(),
            execution_time_ms: 0.0,
            source_node_ids: Vec::new(),
        }
    }
}

/// Execution edge
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct ExecutionEdge {
    #[pyo3(get, set)]
    pub from_node: String,
    #[pyo3(get, set)]
    pub to_node: String,
    #[pyo3(get, set)]
    pub edge_type: String, // "data_flow", "control_flow", "dependency"
    #[pyo3(get, set)]
    pub weight: f32,
}

#[pymethods]
impl ExecutionEdge {
    #[new]
    fn new(from_node: String, to_node: String, edge_type: String) -> Self {
        Self {
            from_node,
            to_node,
            edge_type,
            weight: 1.0,
        }
    }
}

/// Complete execution graph
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct QueryExecutionGraph {
    #[pyo3(get, set)]
    pub query: String,
    #[pyo3(get, set)]
    pub nodes: Vec<ExecutionNode>,
    #[pyo3(get, set)]
    pub edges: Vec<ExecutionEdge>,
    #[pyo3(get, set)]
    pub execution_trace: Vec<String>, // Execution order
    #[pyo3(get, set)]
    pub total_time_ms: f64,
}

#[pymethods]
impl QueryExecutionGraph {
    #[new]
    fn new(query: String) -> Self {
        Self {
            query,
            nodes: Vec::new(),
            edges: Vec::new(),
            execution_trace: Vec::new(),
            total_time_ms: 0.0,
        }
    }

    /// Convert to JSON for visualization
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("JSON serialization failed: {}", e)))
    }
}

