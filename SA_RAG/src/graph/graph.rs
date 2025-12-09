/// Graph Structure Module: Implements knowledge graph storage and querying
/// 
/// Algorithm:
/// 1. Adjacency list storage: each node maintains its outgoing edge list
/// 2. Edge types: support multiple semantic relationships (parent-child, similarity, mention, sequence, etc.)
/// 3. Edge weights: represent relationship strength
/// 4. Bidirectional edges: some relationships are bidirectional (e.g., similarity)
/// 
/// Graph structure is used for:
/// - Hierarchical relationships between semantic nodes
/// - Reference relationships between documents
/// - Association relationships between concepts
/// - Supporting graph expansion retrieval (Graph-RAG)

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Edge type: defines relationship types between nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    /// Parent-child relationship (hierarchical structure)
    ParentChild,
    /// Semantic similarity (based on embedding similarity)
    SemanticSimilarity,
    /// Mention relationship (one node mentions another)
    Mention,
    /// Sequential relationship (order in document)
    NextChunk,
    /// Reference relationship (one node references another)
    Reference,
    /// Synonym relationship (represents same or similar concepts)
    Synonym,
    /// NEXT: Next node in sequence (for sequential traversal)
    Next,
    /// PARENT_OF: Parent relationship (explicit parent-of)
    ParentOf,
    /// REFERS_TO: One node refers to another
    RefersTo,
    /// CITES: Citation relationship
    Cites,
}

/// Graph edge: represents relationship between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Target node ID
    pub target_id: u64,
    /// Edge type
    pub edge_type: EdgeType,
    /// Edge weight (0.0 - 1.0), represents relationship strength
    pub weight: f32,
    /// Edge metadata (optional)
    pub metadata: Option<HashMap<String, String>>,
}

/// Knowledge graph
pub struct Graph {
    /// Adjacency list: NodeID -> List of Edges
    adj: HashMap<u64, Vec<Edge>>,
    /// Reverse adjacency list: for fast lookup of incoming edges (optional, for query optimization)
    reverse_adj: HashMap<u64, Vec<u64>>,
    /// Node metadata (optional)
    node_metadata: HashMap<u64, HashMap<String, String>>,
}

impl Graph {
    /// Create a new empty graph
    pub fn new() -> Self {
        Self {
            adj: HashMap::new(),
            reverse_adj: HashMap::new(),
            node_metadata: HashMap::new(),
        }
    }

    /// Add edge
    /// 
    /// # Arguments
    /// - `source`: Source node ID
    /// - `target`: Target node ID
    /// - `edge_type`: Edge type
    /// - `weight`: Edge weight (0.0-1.0)
    /// - `bidirectional`: Whether to add reverse edge
    pub fn add_edge(
        &mut self,
        source: u64,
        target: u64,
        edge_type: EdgeType,
        weight: f32,
    ) {
        let weight = weight.max(0.0).min(1.0);
        
        let edge = Edge {
            target_id: target,
            edge_type,
            weight,
            metadata: None,
        };

        // Add to adjacency list
        let edges = self.adj.entry(source).or_insert_with(Vec::new);
        
        // Check if same edge already exists, update weight if exists
        if let Some(existing_edge) = edges.iter_mut().find(|e| e.target_id == target && e.edge_type == edge_type) {
            existing_edge.weight = weight;
        } else {
            edges.push(edge);
        }

        // Update reverse adjacency list
        self.reverse_adj.entry(target).or_insert_with(Vec::new).push(source);
    }

    /// Add edge with metadata
    pub fn add_edge_with_metadata(
        &mut self,
        source: u64,
        target: u64,
        edge_type: EdgeType,
        weight: f32,
        metadata: HashMap<String, String>,
    ) {
        let weight = weight.max(0.0).min(1.0);
        
        let edges = self.adj.entry(source).or_insert_with(Vec::new);
        
        if let Some(existing_edge) = edges.iter_mut().find(|e| e.target_id == target && e.edge_type == edge_type) {
            existing_edge.weight = weight;
            existing_edge.metadata = Some(metadata);
        } else {
            let edge = Edge {
                target_id: target,
                edge_type,
                weight,
                metadata: Some(metadata),
            };
            edges.push(edge);
        }

        self.reverse_adj.entry(target).or_insert_with(Vec::new).push(source);
    }

    /// Get all outgoing edges (neighbors) of a node
    pub fn get_neighbors(&self, node_id: u64) -> Option<&Vec<Edge>> {
        self.adj.get(&node_id)
    }

    /// Get all incoming nodes (nodes that point to it)
    pub fn get_incoming_nodes(&self, node_id: u64) -> Option<&Vec<u64>> {
        self.reverse_adj.get(&node_id)
    }

    /// Get edges of specific type
    pub fn get_edges_by_type(&self, node_id: u64, edge_type: EdgeType) -> Vec<&Edge> {
        self.adj.get(&node_id)
            .map(|edges| edges.iter().filter(|e| e.edge_type == edge_type).collect())
            .unwrap_or_default()
    }

    /// Get edge between two nodes
    pub fn get_edge(&self, source: u64, target: u64) -> Option<&Edge> {
        self.adj.get(&source)?
            .iter()
            .find(|e| e.target_id == target)
    }

    /// Check if two nodes are adjacent
    pub fn are_adjacent(&self, source: u64, target: u64) -> bool {
        self.get_edge(source, target).is_some()
    }

    /// Get node degree (out-degree)
    pub fn get_out_degree(&self, node_id: u64) -> usize {
        self.adj.get(&node_id).map(|edges| edges.len()).unwrap_or(0)
    }

    /// Get node in-degree
    pub fn get_in_degree(&self, node_id: u64) -> usize {
        self.reverse_adj.get(&node_id).map(|nodes| nodes.len()).unwrap_or(0)
    }

    /// Remove edge
    pub fn remove_edge(&mut self, source: u64, target: u64, edge_type: Option<EdgeType>) {
        if let Some(edges) = self.adj.get_mut(&source) {
            if let Some(et) = edge_type {
                edges.retain(|e| !(e.target_id == target && e.edge_type == et));
            } else {
                edges.retain(|e| e.target_id != target);
            }
        }

        if let Some(nodes) = self.reverse_adj.get_mut(&target) {
            nodes.retain(|&id| id != source);
        }
    }

    /// Remove node (and all its edges)
    pub fn remove_node(&mut self, node_id: u64) {
        // Remove all outgoing edges
        if let Some(edges) = self.adj.remove(&node_id) {
            // Remove from reverse adjacency list
            for edge in edges {
                if let Some(nodes) = self.reverse_adj.get_mut(&edge.target_id) {
                    nodes.retain(|&id| id != node_id);
                }
            }
        }

        // Remove all incoming edges
        if let Some(sources) = self.reverse_adj.remove(&node_id) {
            for source_id in sources {
                if let Some(edges) = self.adj.get_mut(&source_id) {
                    edges.retain(|e| e.target_id != node_id);
                }
            }
        }

        // Remove node metadata
        self.node_metadata.remove(&node_id);
    }

    /// Set node metadata
    pub fn set_node_metadata(&mut self, node_id: u64, metadata: HashMap<String, String>) {
        self.node_metadata.insert(node_id, metadata);
    }

    /// Get node metadata
    pub fn get_node_metadata(&self, node_id: u64) -> Option<&HashMap<String, String>> {
        self.node_metadata.get(&node_id)
    }

    /// Get all node IDs
    pub fn get_all_nodes(&self) -> HashSet<u64> {
        let mut nodes = HashSet::new();
        nodes.extend(self.adj.keys().copied());
        nodes.extend(self.reverse_adj.keys().copied());
        nodes
    }

    /// Get graph statistics
    pub fn get_stats(&self) -> GraphStats {
        let num_nodes = self.get_all_nodes().len();
        let num_edges: usize = self.adj.values().map(|edges| edges.len()).sum();
        
        let mut degree_distribution = HashMap::new();
        for node_id in self.get_all_nodes() {
            let degree = self.get_out_degree(node_id);
            *degree_distribution.entry(degree).or_insert(0) += 1;
        }

        GraphStats {
            num_nodes,
            num_edges,
            degree_distribution,
        }
    }

    /// Simple graph expansion (for Graph-RAG)
    /// Returns all nodes reachable from start nodes within specified hops
    pub fn expand(&self, start_nodes: &[u64], hops: usize) -> HashSet<u64> {
        let mut visited = HashSet::new();
        let mut current_level: Vec<u64> = start_nodes.to_vec();
        
        // Add start nodes
        for &start in start_nodes {
            visited.insert(start);
        }

        // Expand layer by layer
        for _ in 0..hops {
            let mut next_level = Vec::new();
            
            for node_id in &current_level {
                if let Some(edges) = self.get_neighbors(*node_id) {
                    for edge in edges {
                        if !visited.contains(&edge.target_id) {
                            visited.insert(edge.target_id);
                            next_level.push(edge.target_id);
                        }
                    }
                }
            }
            
            if next_level.is_empty() {
                break;
            }
            
            current_level = next_level;
        }
        
        visited
    }

    /// Weighted graph expansion: consider edge weights, only expand edges above threshold
    pub fn expand_weighted(&self, start_nodes: &[u64], hops: usize, min_weight: f32) -> HashSet<u64> {
        let mut visited = HashSet::new();
        let mut current_level: Vec<u64> = start_nodes.to_vec();
        
        for &start in start_nodes {
            visited.insert(start);
        }

        for _ in 0..hops {
            let mut next_level = Vec::new();
            
            for node_id in &current_level {
                if let Some(edges) = self.get_neighbors(*node_id) {
                    for edge in edges {
                        if edge.weight >= min_weight && !visited.contains(&edge.target_id) {
                            visited.insert(edge.target_id);
                            next_level.push(edge.target_id);
                        }
                    }
                }
            }
            
            if next_level.is_empty() {
                break;
            }
            
            current_level = next_level;
        }
        
        visited
    }
}

/// Graph statistics
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub degree_distribution: HashMap<usize, usize>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
