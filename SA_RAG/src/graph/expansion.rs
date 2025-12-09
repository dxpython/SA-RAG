/// Graph Expansion Module: Implements Graph-RAG graph expansion algorithms
/// 
/// Algorithm:
/// Graph-RAG enhances retrieval results through graph expansion:
/// 1. Start from initial retrieval results (seed nodes)
/// 2. Expand along graph edges with multiple hops
/// 3. Filter based on edge type and weight
/// 4. Re-rank expanded nodes
/// 
/// Expansion strategies:
/// - Breadth-first expansion: expand layer by layer, control expansion depth
/// - Weight filtering: only expand high-weight edges
/// - Type filtering: prioritize expanding specific edge types (e.g., semantic similarity)
/// - Diversity control: avoid over-concentration on certain nodes

use crate::graph::graph::{Graph, EdgeType};
use std::collections::{HashSet, HashMap, VecDeque};

/// Graph expander
pub struct GraphExpansion;

impl GraphExpansion {
    /// Basic graph expansion: expand from seed nodes by specified depth
    /// 
    /// # Arguments
    /// - `graph`: Knowledge graph
    /// - `seed_nodes`: Seed nodes (initial retrieval results)
    /// - `depth`: Expansion depth (number of hops)
    /// 
    /// # Returns
    /// List of expanded node IDs (including original seed nodes)
    pub fn expand(graph: &Graph, seed_nodes: &[u64], depth: usize) -> Vec<u64> {
        let expanded_set = graph.expand(seed_nodes, depth);
        expanded_set.into_iter().collect()
    }

    /// Weighted graph expansion: only expand edges above threshold
    /// 
    /// # Arguments
    /// - `graph`: Knowledge graph
    /// - `seed_nodes`: Seed nodes
    /// - `depth`: Expansion depth
    /// - `min_weight`: Minimum edge weight threshold
    pub fn expand_weighted(
        graph: &Graph,
        seed_nodes: &[u64],
        depth: usize,
        min_weight: f32,
    ) -> Vec<u64> {
        let expanded_set = graph.expand_weighted(seed_nodes, depth, min_weight);
        expanded_set.into_iter().collect()
    }

    /// Type-aware expansion: prioritize expanding specific edge types
    /// 
    /// # Arguments
    /// - `graph`: Knowledge graph
    /// - `seed_nodes`: Seed nodes
    /// - `depth`: Expansion depth
    /// - `preferred_types`: List of preferred edge types to expand
    /// - `type_weights`: Weight mapping for edge types
    pub fn expand_by_type(
        graph: &Graph,
        seed_nodes: &[u64],
        depth: usize,
        preferred_types: &[EdgeType],
        type_weights: &HashMap<EdgeType, f32>,
    ) -> Vec<u64> {
        let mut visited = HashSet::new();
        let mut current_level: Vec<(u64, f32)> = seed_nodes.iter()
            .map(|&id| (id, 1.0))
            .collect();
        
        for &(id, _) in &current_level {
            visited.insert(id);
        }

        for _ in 0..depth {
            let mut next_level = Vec::new();
            
            for (node_id, path_score) in &current_level {
                if let Some(edges) = graph.get_neighbors(*node_id) {
                    for edge in edges {
                        if visited.contains(&edge.target_id) {
                            continue;
                        }

                        // Calculate path score (consider edge weight and type weight)
                        let type_weight = type_weights.get(&edge.edge_type).copied().unwrap_or(1.0);
                        let edge_score = edge.weight * type_weight;
                        let new_path_score = path_score * edge_score;

                        // If edge type is in preferred list, boost score
                        let final_score = if preferred_types.contains(&edge.edge_type) {
                            new_path_score * 1.5
                        } else {
                            new_path_score
                        };

                        visited.insert(edge.target_id);
                        next_level.push((edge.target_id, final_score));
                    }
                }
            }

            if next_level.is_empty() {
                break;
            }

            // Sort by score, keep high-scoring nodes
            next_level.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            // Limit number of nodes expanded per level (avoid explosion)
            let max_per_level = 50;
            next_level.truncate(max_per_level);
            
            current_level = next_level;
        }

        visited.into_iter().collect()
    }

    /// Diversity expansion: control expansion diversity, avoid over-concentration in certain areas
    /// 
    /// # Arguments
    /// - `graph`: Knowledge graph
    /// - `seed_nodes`: Seed nodes
    /// - `depth`: Expansion depth
    /// - `max_nodes_per_seed`: Maximum number of nodes to expand per seed node
    pub fn expand_diverse(
        graph: &Graph,
        seed_nodes: &[u64],
        depth: usize,
        max_nodes_per_seed: usize,
    ) -> Vec<u64> {
        let mut all_expanded = HashSet::new();
        
        // Expand independently for each seed node
        for &seed in seed_nodes {
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            queue.push_back((seed, 0));
            visited.insert(seed);
            all_expanded.insert(seed);

            while let Some((node_id, current_depth)) = queue.pop_front() {
                if current_depth >= depth {
                    continue;
                }

                if let Some(edges) = graph.get_neighbors(node_id) {
                    // Sort edges by weight
                    let mut sorted_edges: Vec<_> = edges.iter().collect();
                    sorted_edges.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));

                    // Only expand top N high-weight edges
                    let mut added = 0;
                    for edge in sorted_edges {
                        if !visited.contains(&edge.target_id) && added < max_nodes_per_seed {
                            visited.insert(edge.target_id);
                            all_expanded.insert(edge.target_id);
                            queue.push_back((edge.target_id, current_depth + 1));
                            added += 1;
                        }
                    }
                }
            }
        }

        all_expanded.into_iter().collect()
    }

    /// Smart expansion: expansion method combining multiple strategies
    /// 
    /// # Arguments
    /// - `graph`: Knowledge graph
    /// - `seed_nodes`: Seed nodes
    /// - `depth`: Expansion depth
    /// - `min_weight`: Minimum edge weight
    /// - `preferred_types`: Preferred edge types
    /// - `max_total_nodes`: Maximum total number of nodes
    pub fn expand_smart(
        graph: &Graph,
        seed_nodes: &[u64],
        depth: usize,
        min_weight: f32,
        preferred_types: &[EdgeType],
        max_total_nodes: usize,
    ) -> Vec<u64> {
        let mut visited = HashSet::new();
        let mut current_level: Vec<(u64, f32)> = seed_nodes.iter()
            .map(|&id| (id, 1.0))
            .collect();
        
        for &(id, _) in &current_level {
            visited.insert(id);
        }

        for hop in 0..depth {
            let mut next_level = Vec::new();
            
            for (node_id, path_score) in &current_level {
                if visited.len() >= max_total_nodes {
                    break;
                }

                if let Some(edges) = graph.get_neighbors(*node_id) {
                    for edge in edges {
                        if visited.contains(&edge.target_id) {
                            continue;
                        }

                        // Filter low-weight edges
                        if edge.weight < min_weight {
                            continue;
                        }

                        // Calculate path score
                        let type_boost = if preferred_types.contains(&edge.edge_type) {
                            1.5
                        } else {
                            1.0
                        };
                        
                        let new_score = path_score * edge.weight * type_boost;
                        
                        // Decrease score as depth increases (decay)
                        let depth_decay = 1.0 / (1.0 + hop as f32 * 0.2);
                        let final_score = new_score * depth_decay;

                        visited.insert(edge.target_id);
                        next_level.push((edge.target_id, final_score));
                    }
                }
            }

            if next_level.is_empty() || visited.len() >= max_total_nodes {
                break;
            }

            // Sort by score and limit count
            next_level.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let max_per_level = (max_total_nodes - visited.len()).min(100);
            next_level.truncate(max_per_level);
            
            current_level = next_level;
        }

        visited.into_iter().collect()
    }

    /// Calculate relevance scores for expanded nodes (based on distance to seed nodes and path weights)
    pub fn compute_relevance_scores(
        graph: &Graph,
        seed_nodes: &[u64],
        _expanded_nodes: &[u64],
    ) -> HashMap<u64, f32> {
        let mut scores = HashMap::new();
        
        // Initialize scores for each seed node
        for &seed in seed_nodes {
            scores.insert(seed, 1.0);
        }

        // Use BFS to calculate distance and path weight from each node to nearest seed node
        let mut queue: VecDeque<(u64, f32, usize)> = seed_nodes.iter()
            .map(|&id| (id, 1.0, 0))
            .collect();
        
        let mut visited = HashSet::new();
        for &seed in seed_nodes {
            visited.insert(seed);
        }

        while let Some((node_id, path_score, depth)) = queue.pop_front() {
            if let Some(edges) = graph.get_neighbors(node_id) {
                for edge in edges {
                    if visited.contains(&edge.target_id) {
                        continue;
                    }

                    let new_score = path_score * edge.weight * (0.9_f32).powi(depth as i32);
                    
                    // Update score (take maximum)
                    let current_score = scores.get(&edge.target_id).copied().unwrap_or(0.0);
                    if new_score > current_score {
                        scores.insert(edge.target_id, new_score);
                    }

                    visited.insert(edge.target_id);
                    queue.push_back((edge.target_id, new_score, depth + 1));
                }
            }
        }

        scores
    }
}
