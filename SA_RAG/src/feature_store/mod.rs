// High-Dimensional Feature Store
// Manages embedding vectors with versioning and TTL

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use dashmap::DashMap;
use parking_lot::RwLock;

/// Feature vector with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct FeatureVector {
    #[pyo3(get, set)]
    pub feature_id: String,
    #[pyo3(get, set)]
    pub vector: Vec<f32>,
    #[pyo3(get, set)]
    pub model_version: String,
    #[pyo3(get, set)]
    pub created_at: u64,
    #[pyo3(get, set)]
    pub ttl: Option<u64>, // Time to live in seconds
    #[pyo3(get, set)]
    pub metadata: HashMap<String, String>,
}

#[pymethods]
impl FeatureVector {
    #[new]
    fn new(feature_id: String, vector: Vec<f32>, model_version: String) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        Self {
            feature_id,
            vector,
            model_version,
            created_at: now,
            ttl: None,
            metadata: HashMap::new(),
        }
    }
}

/// Feature store with versioning and TTL
#[pyclass]
pub struct FeatureStore {
    // feature_id -> version -> FeatureVector
    features: Arc<DashMap<String, HashMap<String, FeatureVector>>>,
    // Current model version
    current_model_version: Arc<RwLock<String>>,
    // Default TTL in seconds
    default_ttl: u64,
}

use std::sync::Arc;

#[pymethods]
impl FeatureStore {
    #[new]
    pub fn new() -> Self {
        Self {
            features: Arc::new(DashMap::new()),
            current_model_version: Arc::new(RwLock::new("v1.0".to_string())),
            default_ttl: 86400 * 30, // 30 days default
        }
    }

    /// Store feature vector
    pub fn store(
        &self,
        feature_id: String,
        vector: Vec<f32>,
        model_version: Option<String>,
        ttl: Option<u64>,
    ) -> PyResult<()> {
        let version = model_version.unwrap_or_else(|| {
            self.current_model_version.read().clone()
        });
        
        let feature = FeatureVector::new(feature_id.clone(), vector, version.clone());
        let mut feature_with_ttl = feature;
        feature_with_ttl.ttl = ttl.or(Some(self.default_ttl));

        let mut versions = self.features
            .entry(feature_id)
            .or_insert_with(HashMap::new);
        versions.insert(version, feature_with_ttl);

        Ok(())
    }

    /// Retrieve feature vector (latest version)
    pub fn get(&self, feature_id: &str) -> PyResult<Option<FeatureVector>> {
        if let Some(versions) = self.features.get(feature_id) {
            // Get latest version
            let latest = versions.value()
                .values()
                .max_by_key(|f| f.created_at)
                .cloned();
            
            // Check TTL
            if let Some(ref feature) = latest {
                if let Some(ttl) = feature.ttl {
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs();
                    if now - feature.created_at > ttl {
                        return Ok(None); // Expired
                    }
                }
            }
            
            Ok(latest)
        } else {
            Ok(None)
        }
    }

    /// Retrieve feature vector by version
    fn get_by_version(&self, feature_id: &str, version: &str) -> PyResult<Option<FeatureVector>> {
        if let Some(versions) = self.features.get(feature_id) {
            Ok(versions.value().get(version).cloned())
        } else {
            Ok(None)
        }
    }

    /// Update model version
    fn set_model_version(&self, version: String) {
        *self.current_model_version.write() = version;
    }

    /// Get current model version
    fn get_model_version(&self) -> String {
        self.current_model_version.read().clone()
    }

    /// Clean expired features
    fn clean_expired(&self) -> PyResult<usize> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let mut cleaned = 0;
        let mut to_remove = Vec::new();

        for entry in self.features.iter() {
            let feature_id = entry.key().clone();
            let versions = entry.value();
            
            let mut expired_versions = Vec::new();
            for (version, feature) in versions.iter() {
                if let Some(ttl) = feature.ttl {
                    if now - feature.created_at > ttl {
                        expired_versions.push(version.clone());
                    }
                }
            }

            if !expired_versions.is_empty() {
                if let Some(mut versions) = self.features.get_mut(&feature_id) {
                    for version in expired_versions {
                        versions.remove(&version);
                        cleaned += 1;
                    }
                    
                    // Remove feature_id if no versions left
                    if versions.is_empty() {
                        to_remove.push(feature_id);
                    }
                }
            }
        }

        for feature_id in to_remove {
            self.features.remove(&feature_id);
        }

        Ok(cleaned)
    }

    /// Batch store features
    fn batch_store(
        &self,
        features: Vec<(String, Vec<f32>)>,
        model_version: Option<String>,
    ) -> PyResult<()> {
        let version = model_version.unwrap_or_else(|| {
            self.current_model_version.read().clone()
        });

        for (feature_id, vector) in features {
            self.store(feature_id, vector, Some(version.clone()), None)?;
        }

        Ok(())
    }

    /// Get all versions of a feature
    fn get_versions(&self, feature_id: &str) -> PyResult<Vec<String>> {
        if let Some(versions) = self.features.get(feature_id) {
            Ok(versions.value().keys().cloned().collect())
        } else {
            Ok(Vec::new())
        }
    }
}

impl Default for FeatureStore {
    fn default() -> Self {
        Self::new()
    }
}

