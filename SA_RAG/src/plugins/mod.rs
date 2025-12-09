// Plugin System
// Allows users to extend SA-RAG with custom components

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Plugin trait (Rust side)
pub trait RankerPlugin: Send + Sync {
    fn rank(&self, results: &[crate::engine::RetrievalResult]) -> Vec<(u64, f32)>;
    fn name(&self) -> &str;
}

/// Retrieval result for plugins
pub use crate::engine::RetrievalResult;

pub trait NodeParserPlugin: Send + Sync {
    fn parse(&self, doc_id: u64, text: &str) -> Vec<crate::semantic_node::SemanticNode>;
    fn name(&self) -> &str;
}

pub trait GraphPolicyPlugin: Send + Sync {
    fn should_expand(&self, node_id: u64, current_hops: usize) -> bool;
    fn expansion_priority(&self, node_id: u64) -> f32;
    fn name(&self) -> &str;
}

/// Plugin metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct PluginMetadata {
    #[pyo3(get, set)]
    pub plugin_id: String,
    #[pyo3(get, set)]
    pub plugin_type: String, // "ranker", "parser", "graph_policy"
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub version: String,
    #[pyo3(get, set)]
    pub author: String,
    #[pyo3(get, set)]
    pub description: String,
    #[pyo3(get, set)]
    pub enabled: bool,
}

#[pymethods]
impl PluginMetadata {
    #[new]
    fn new(plugin_id: String, plugin_type: String, name: String) -> Self {
        Self {
            plugin_id,
            plugin_type,
            name,
            version: "1.0.0".to_string(),
            author: "Unknown".to_string(),
            description: String::new(),
            enabled: true,
        }
    }
}

/// Plugin registry
#[pyclass]
pub struct PluginRegistry {
    ranker_plugins: HashMap<String, Box<dyn RankerPlugin>>,
    parser_plugins: HashMap<String, Box<dyn NodeParserPlugin>>,
    graph_policy_plugins: HashMap<String, Box<dyn GraphPolicyPlugin>>,
    metadata: HashMap<String, PluginMetadata>,
}

#[pymethods]
impl PluginRegistry {
    #[new]
    pub fn new() -> Self {
        Self {
            ranker_plugins: HashMap::new(),
            parser_plugins: HashMap::new(),
            graph_policy_plugins: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Register plugin metadata
    fn register_metadata(&mut self, metadata: PluginMetadata) {
        self.metadata.insert(metadata.plugin_id.clone(), metadata);
    }

    /// Get plugin metadata
    fn get_metadata(&self, plugin_id: &str) -> Option<PluginMetadata> {
        self.metadata.get(plugin_id).cloned()
    }

    /// List all plugins
    fn list_plugins(&self) -> Vec<PluginMetadata> {
        self.metadata.values().cloned().collect()
    }
}

// Default plugin implementations

/// Default ranker plugin
pub struct DefaultRankerPlugin;

impl RankerPlugin for DefaultRankerPlugin {
    fn rank(&self, results: &[crate::engine::RetrievalResult]) -> Vec<(u64, f32)> {
        results.iter()
            .map(|r| (r.node_id, r.score))
            .collect()
    }

    fn name(&self) -> &str {
        "default_ranker"
    }
}

/// Default node parser plugin
pub struct DefaultNodeParserPlugin;

impl NodeParserPlugin for DefaultNodeParserPlugin {
    fn parse(&self, doc_id: u64, text: &str) -> Vec<crate::semantic_node::SemanticNode> {
        crate::parser::SemanticParser::parse(doc_id, text)
    }

    fn name(&self) -> &str {
        "default_parser"
    }
}

/// Default graph policy plugin
pub struct DefaultGraphPolicyPlugin {
    max_hops: usize,
}

impl DefaultGraphPolicyPlugin {
    pub fn new(max_hops: usize) -> Self {
        Self { max_hops }
    }
}

impl GraphPolicyPlugin for DefaultGraphPolicyPlugin {
    fn should_expand(&self, _node_id: u64, current_hops: usize) -> bool {
        current_hops < self.max_hops
    }

    fn expansion_priority(&self, _node_id: u64) -> f32 {
        1.0
    }

    fn name(&self) -> &str {
        "default_graph_policy"
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

