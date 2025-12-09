// Execution Graph Builder
// Builds DAG representation of query execution

use super::{ExecutionNode, ExecutionEdge, QueryExecutionGraph};
use petgraph::{Graph, Directed, graph::NodeIndex};
use std::collections::HashMap;

pub struct ExecutionGraphBuilder {
    graph: Graph<ExecutionNode, ExecutionEdge, Directed>,
    node_map: HashMap<String, NodeIndex>,
    query: String,
}

impl ExecutionGraphBuilder {
    pub fn new(query: String) -> Self {
        Self {
            graph: Graph::new(),
            node_map: HashMap::new(),
            query,
        }
    }

    /// Add execution node
    pub fn add_node(&mut self, node: ExecutionNode) -> String {
        let node_id = node.node_id.clone();
        let idx = self.graph.add_node(node);
        self.node_map.insert(node_id.clone(), idx);
        node_id
    }

    /// Add execution edge
    pub fn add_edge(&mut self, from: &str, to: &str, edge: ExecutionEdge) {
        if let (Some(&from_idx), Some(&to_idx)) = (self.node_map.get(from), self.node_map.get(to)) {
            self.graph.add_edge(from_idx, to_idx, edge);
        }
    }

    /// Build final execution graph
    pub fn build(self) -> QueryExecutionGraph {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut execution_trace = Vec::new();

        // Extract nodes
        for idx in self.graph.node_indices() {
            if let Some(node) = self.graph.node_weight(idx) {
                nodes.push(node.clone());
            }
        }

        // Extract edges
        for edge_idx in self.graph.edge_indices() {
            if let Some(edge) = self.graph.edge_weight(edge_idx) {
                edges.push(edge.clone());
            }
        }

        // Topological sort for execution trace
        if let Ok(topo) = petgraph::algo::toposort(&self.graph, None) {
            for idx in topo {
                if let Some(node) = self.graph.node_weight(idx) {
                    execution_trace.push(node.node_id.clone());
                }
            }
        }

        let total_time: f64 = nodes.iter().map(|n| n.execution_time_ms).sum();

        QueryExecutionGraph {
            query: self.query,
            nodes,
            edges,
            execution_trace,
            total_time_ms: total_time,
        }
    }

    /// Get node by ID
    pub fn get_node(&self, node_id: &str) -> Option<&ExecutionNode> {
        self.node_map.get(node_id)
            .and_then(|&idx| self.graph.node_weight(idx))
    }
}

