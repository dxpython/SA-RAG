// Retrieval Consistency Checker
// Detects drift and ensures reproducible retrieval

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Consistency check result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct ConsistencyReport {
    #[pyo3(get, set)]
    pub query_hash: String,
    #[pyo3(get, set)]
    pub previous_results: Vec<u64>, // Previous node IDs
    #[pyo3(get, set)]
    pub current_results: Vec<u64>, // Current node IDs
    #[pyo3(get, set)]
    pub similarity_score: f32, // Jaccard similarity
    #[pyo3(get, set)]
    pub drift_detected: bool,
    #[pyo3(get, set)]
    pub drift_reason: Option<String>,
    #[pyo3(get, set)]
    pub timestamp: u64,
}

#[pymethods]
impl ConsistencyReport {
    #[new]
    fn new(query_hash: String) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        Self {
            query_hash,
            previous_results: Vec::new(),
            current_results: Vec::new(),
            similarity_score: 1.0,
            drift_detected: false,
            drift_reason: None,
            timestamp: now,
        }
    }
}

/// Consistency checker
#[pyclass]
pub struct ConsistencyChecker {
    // query_hash -> (timestamp, results)
    query_history: HashMap<String, Vec<(u64, Vec<u64>)>>,
    // Drift threshold (0.0-1.0)
    drift_threshold: f32,
}

#[pymethods]
impl ConsistencyChecker {
    #[new]
    pub fn new() -> Self {
        Self {
            query_history: HashMap::new(),
            drift_threshold: 0.7, // If similarity < 0.7, drift detected
        }
    }

    /// Check consistency for a query
    pub fn check_consistency(
        &mut self,
        query: &str,
        current_results: Vec<u64>,
    ) -> PyResult<ConsistencyReport> {
        let query_hash = self.hash_query(query);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut report = ConsistencyReport::new(query_hash.clone());
        report.current_results = current_results.clone();

        if let Some(history) = self.query_history.get(&query_hash) {
            if let Some((_, previous_results)) = history.last() {
                report.previous_results = previous_results.clone();
                
                // Calculate Jaccard similarity
                let similarity = self.jaccard_similarity(previous_results, &current_results);
                report.similarity_score = similarity;

                // Detect drift
                if similarity < self.drift_threshold {
                    report.drift_detected = true;
                    report.drift_reason = Some(format!(
                        "Result similarity ({:.2}) below threshold ({:.2})",
                        similarity, self.drift_threshold
                    ));
                }
            }
        } else {
            // First time seeing this query
            report.similarity_score = 1.0;
        }

        // Update history
        let history = self.query_history.entry(query_hash).or_insert_with(Vec::new);
        history.push((now, current_results));
        
        // Keep only last 10 entries per query
        if history.len() > 10 {
            history.remove(0);
        }

        Ok(report)
    }

    /// Set drift threshold
    fn set_drift_threshold(&mut self, threshold: f32) {
        self.drift_threshold = threshold.max(0.0).min(1.0);
    }

    /// Get query history
    fn get_history(&self, query: &str) -> PyResult<Vec<(u64, Vec<u64>)>> {
        let query_hash = self.hash_query(query);
        Ok(self.query_history.get(&query_hash)
            .cloned()
            .unwrap_or_default())
    }
}

impl ConsistencyChecker {
    fn hash_query(&self, query: &str) -> String {
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    fn jaccard_similarity(&self, a: &[u64], b: &[u64]) -> f32 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let set_a: std::collections::HashSet<u64> = a.iter().copied().collect();
        let set_b: std::collections::HashSet<u64> = b.iter().copied().collect();

        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();

        intersection as f32 / union as f32
    }
}

impl Default for ConsistencyChecker {
    fn default() -> Self {
        Self::new()
    }
}

