/// Vector Index Module: Implements HNSW (Hierarchical Navigable Small World) algorithm
/// 
/// Algorithm:
/// HNSW is a graph-based approximate nearest neighbor search algorithm with the following features:
/// 1. Multi-layer graph structure: bottom layer contains all nodes, upper layers are sparse fast navigation layers
/// 2. Greedy search: start from top layer, search downward layer by layer to find nearest neighbors
/// 3. Dynamic insertion: new nodes connect to most similar neighbors through heuristic methods
/// 
/// Due to lifetime complexity in PyO3 bindings with hnsw_rs library, we implement a simplified but complete HNSW here.
/// For production environments, it's recommended to use specialized vector databases (e.g., Milvus, Qdrant) or properly wrap hnsw_rs

use anyhow::{Result, anyhow};
use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;

/// HNSW Vector Index
pub struct VectorIndex {
    /// Node ID -> Vector
    vectors: HashMap<u64, Vec<f32>>,
    /// Maximum number of elements
    max_elements: usize,
    /// Maximum connections per layer
    m: usize,
    /// Layer count control parameter (affects top layer sparsity)
    m_max: usize,
    /// Number of layers (grows dynamically)
    max_level: usize,
    /// Adjacency list per layer: level -> (node_id -> neighbor_ids)
    layers: Vec<HashMap<u64, Vec<u64>>>,
    /// Highest layer each node belongs to
    node_levels: HashMap<u64, usize>,
}

impl VectorIndex {
    /// Create a new vector index
    /// 
    /// # Arguments
    /// - `max_elements`: Maximum number of elements
    /// - `m`: Maximum connections per layer (default 16)
    /// - `m_max`: Parameter controlling number of layers (default 64)
    pub fn new(max_elements: usize) -> Self {
        Self {
            vectors: HashMap::new(),
            max_elements,
            m: 16,  // Maximum connections per layer
            m_max: 64,  // Control layer count
            max_level: 0,
            layers: vec![HashMap::new()],  // At least one layer (layer 0)
            node_levels: HashMap::new(),
        }
    }

    /// Insert a vector
    /// 
    /// # Algorithm
    /// 1. Randomly assign node level (exponential distribution)
    /// 2. Search for nearest neighbors starting from top layer
    /// 3. Connect to m most similar neighbors at each layer
    /// 4. Prune connections to maintain at most m connections per layer
    pub fn insert(&mut self, node_id: u64, vector: Vec<f32>) -> Result<()> {
        if self.vectors.len() >= self.max_elements {
            return Err(anyhow!("Index is full"));
        }

        // Normalize vector (L2 normalization for cosine similarity)
        let normalized = Self::normalize(&vector)?;
        self.vectors.insert(node_id, normalized.clone());

        // If index is empty, insert directly
        if self.vectors.len() == 1 {
            let level = self.random_level();
            self.node_levels.insert(node_id, level);
            
            // Ensure enough layers exist
            while self.layers.len() <= level {
                self.layers.push(HashMap::new());
            }
            
            // Add node to all layers
            for l in 0..=level {
                self.layers[l].insert(node_id, Vec::new());
            }
            
            if level > self.max_level {
                self.max_level = level;
            }
            
            return Ok(());
        }

        // Randomly assign level (exponential distribution, similar to skip list)
        let level = self.random_level();
        self.node_levels.insert(node_id, level);

        // Ensure enough layers exist
        while self.layers.len() <= level {
            self.layers.push(HashMap::new());
        }

        // Search for entry point starting from top layer
        let mut entry_point = self.find_entry_point();
        let mut candidates = vec![(entry_point, Self::cosine_distance(&normalized, self.vectors.get(&entry_point).unwrap()))];

        // Search for nearest neighbors in top layers
        for l in (level + 1..=self.max_level).rev() {
            candidates = self.search_layer(&normalized, &candidates, 1, l);
            if let Some((ep, _)) = candidates.first() {
                entry_point = *ep;
            }
        }

        // Insert layer by layer from top to node level
        let mut neighbors: Vec<(u64, f32)> = Vec::new();
        for l in (0..=level.min(self.max_level)).rev() {
            // Search for candidate neighbors in current layer
            let search_candidates = if l == self.max_level {
                vec![(entry_point, Self::cosine_distance(&normalized, self.vectors.get(&entry_point).unwrap()))]
            } else {
                neighbors.clone()
            };

            let layer_candidates = self.search_layer(&normalized, &search_candidates, self.ef_construction(l), l);
            
            // Select m most similar neighbors (keep as Vec<(u64, f32)> for next iteration)
            neighbors = layer_candidates.into_iter().take(self.m).collect();
            
            // Extract just the node IDs for storing in layers
            let neighbor_ids: Vec<u64> = neighbors.iter().map(|(id, _)| *id).collect();
            
            // Add node to current layer
            self.layers[l].insert(node_id, neighbor_ids.clone());
            
            // Build bidirectional connections
            for &neighbor_id in &neighbor_ids {
                if let Some(neighbor_conns) = self.layers[l].get_mut(&neighbor_id) {
                    neighbor_conns.push(node_id);
                    // Prune neighbor connections to maintain at most m connections
                    if neighbor_conns.len() > self.m {
                        let neighbor_vec = self.vectors.get(&neighbor_id).unwrap();
                        let mut neighbor_candidates: Vec<(u64, f32)> = neighbor_conns.iter()
                            .map(|&id| (id, Self::cosine_distance(neighbor_vec, self.vectors.get(&id).unwrap())))
                            .collect();
                        neighbor_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                        *neighbor_conns = neighbor_candidates.into_iter().take(self.m).map(|(id, _)| id).collect();
                    }
                }
            }
        }

        // Update maximum level
        if level > self.max_level {
            self.max_level = level;
        }

        Ok(())
    }

    /// Search for nearest neighbors
    /// 
    /// # Arguments
    /// - `query`: Query vector
    /// - `ef`: Search width (exploration factor)
    /// - `k`: Return top-k results
    /// 
    /// # Algorithm
    /// 1. Start from top layer, find entry point
    /// 2. Search downward layer by layer, find ef candidates at each layer
    /// 3. Perform exact search at bottom layer, return top-k
    pub fn search(&self, query: &[f32], ef: usize, k: usize) -> Result<Vec<(u64, f32)>> {
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let normalized_query = Self::normalize(query)?;
        
        // Find entry point (any node in top layer)
        let entry_point = self.find_entry_point();
        let mut candidates = vec![(entry_point, Self::cosine_distance(&normalized_query, self.vectors.get(&entry_point).unwrap()))];

        // Search downward from top layer
        for l in (1..=self.max_level).rev() {
            candidates = self.search_layer(&normalized_query, &candidates, 1, l);
        }

        // Perform exact search at bottom layer
        let results = self.search_layer(&normalized_query, &candidates, ef.max(k), 0);
        
        // Return top-k
        let mut sorted_results = results;
        sorted_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        
        Ok(sorted_results.into_iter().take(k).map(|(id, dist)| (id, dist)).collect())
    }

    /// Search in specified layer
    fn search_layer(
        &self,
        query: &[f32],
        entry_candidates: &[(u64, f32)],
        ef: usize,
        level: usize,
    ) -> Vec<(u64, f32)> {
        if level >= self.layers.len() {
            return entry_candidates.to_vec();
        }

        let mut visited = HashSet::new();
        let mut candidates: Vec<(u64, f32)> = entry_candidates.to_vec();
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        
        let mut dynamic_candidates = candidates.clone();

        for &(entry_id, _) in entry_candidates {
            visited.insert(entry_id);
        }

        while !candidates.is_empty() {
            let (current_id, _current_dist) = candidates.remove(0);
            
            // Get neighbors of current node
            if let Some(neighbors) = self.layers[level].get(&current_id) {
                for &neighbor_id in neighbors {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }
                    visited.insert(neighbor_id);
                    
                    if let Some(neighbor_vec) = self.vectors.get(&neighbor_id) {
                        let dist = Self::cosine_distance(query, neighbor_vec);
                        
                        // Add if candidate list is not full or distance is closer
                        if dynamic_candidates.len() < ef || dist < dynamic_candidates.last().unwrap().1 {
                            dynamic_candidates.push((neighbor_id, dist));
                            dynamic_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                            
                            if dynamic_candidates.len() > ef {
                                dynamic_candidates.pop();
                            }
                            
                            // Add to candidate list to continue search
                            candidates.push((neighbor_id, dist));
                            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                        }
                    }
                }
            }
        }

        dynamic_candidates
    }

    /// Select neighbors (heuristic: select m most similar)
    /// Note: This function is kept for potential future use, but currently we use direct sorting
    fn _select_neighbors(
        &self,
        _query: &[f32],
        candidates: &[(u64, f32)],
        m: usize,
        _level: usize,
    ) -> Vec<u64> {
        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        sorted.into_iter().take(m).map(|(id, _)| id).collect()
    }

    /// Find entry point (any node in top layer)
    fn find_entry_point(&self) -> u64 {
        if self.max_level >= self.layers.len() || self.layers[self.max_level].is_empty() {
            // If top layer is empty, return any node
            *self.vectors.keys().next().unwrap()
        } else {
            *self.layers[self.max_level].keys().next().unwrap()
        }
    }

    /// Random level (exponential distribution)
    /// Uses hash of node ID to generate pseudo-random numbers for reproducibility
    fn random_level(&self) -> usize {
        // Use current vector count as seed to generate pseudo-random level
        let seed = self.vectors.len() as u64;
        let mut level = 0;
        let mut hash = seed;
        
        // Simple linear congruential generator
        while level < 16 {
            hash = hash.wrapping_mul(1103515245).wrapping_add(12345);
            let rand_val = (hash >> 16) as f32 / 65536.0;
            if rand_val < 0.5 {
                level += 1;
            } else {
                break;
            }
        }
        level
    }

    /// Exploration factor during construction
    fn ef_construction(&self, _level: usize) -> usize {
        self.m * 2  // Usually set to 2-4 times m
    }

    /// Cosine distance (1 - cosine similarity)
    fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        1.0 - dot_product  // Assuming vectors are normalized
    }

    /// L2 distance
    fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Normalize vector (L2 normalization)
    fn normalize(vec: &[f32]) -> Result<Vec<f32>> {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            return Err(anyhow!("Zero vector cannot be normalized"));
        }
        Ok(vec.iter().map(|x| x / norm).collect())
    }
}
