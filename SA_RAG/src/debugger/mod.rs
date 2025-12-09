// Autonomous Debugger Module
// Automatically analyzes retrieval failures and suggests optimizations

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Debug analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct DebugAnalysis {
    #[pyo3(get, set)]
    pub query: String,
    #[pyo3(get, set)]
    pub success: bool,
    #[pyo3(get, set)]
    pub failure_stage: Option<String>,
    #[pyo3(get, set)]
    pub issues: Vec<DebugIssue>,
    #[pyo3(get, set)]
    pub suggestions: Vec<OptimizationSuggestion>,
    #[pyo3(get, set)]
    pub metrics: HashMap<String, f32>,
}

#[pymethods]
impl DebugAnalysis {
    #[new]
    fn new(query: String) -> Self {
        Self {
            query,
            success: true,
            failure_stage: None,
            issues: Vec::new(),
            suggestions: Vec::new(),
            metrics: HashMap::new(),
        }
    }
}

/// Debug issue
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct DebugIssue {
    #[pyo3(get, set)]
    pub stage: String,
    #[pyo3(get, set)]
    pub severity: String, // "error", "warning", "info"
    #[pyo3(get, set)]
    pub description: String,
    #[pyo3(get, set)]
    pub details: HashMap<String, String>,
}

#[pymethods]
impl DebugIssue {
    #[new]
    fn new(stage: String, severity: String, description: String) -> Self {
        Self {
            stage,
            severity,
            description,
            details: HashMap::new(),
        }
    }
}

/// Optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct OptimizationSuggestion {
    #[pyo3(get, set)]
    pub category: String, // "ranking", "indexing", "query", "graph"
    #[pyo3(get, set)]
    pub action: String,
    #[pyo3(get, set)]
    pub description: String,
    #[pyo3(get, set)]
    pub expected_improvement: f32, // Expected improvement (0.0-1.0)
    #[pyo3(get, set)]
    pub parameters: HashMap<String, f32>,
}

#[pymethods]
impl OptimizationSuggestion {
    #[new]
    fn new(category: String, action: String, description: String) -> Self {
        Self {
            category,
            action,
            description,
            expected_improvement: 0.0,
            parameters: HashMap::new(),
        }
    }
}

/// Autonomous debugger
#[pyclass]
pub struct AutonomousDebugger {
    // Query analysis history
    query_history: HashMap<String, Vec<DebugAnalysis>>,
    // Performance baselines
    baselines: HashMap<String, f32>,
}

#[pymethods]
impl AutonomousDebugger {
    #[new]
    pub fn new() -> Self {
        Self {
            query_history: HashMap::new(),
            baselines: HashMap::new(),
        }
    }

    /// Analyze retrieval failure
    #[pyo3(signature = (query, vector_results, bm25_results, graph_results, final_results, *, query_analysis_result = None, answer_quality = None))]
    pub fn analyze_retrieval(
        &mut self,
        query: String,
        vector_results: Vec<(u64, f32)>,
        bm25_results: Vec<(u64, f32)>,
        graph_results: Vec<(u64, f32)>,
        final_results: Vec<(u64, f32)>,
        query_analysis_result: Option<HashMap<String, String>>,
        answer_quality: Option<f32>,
    ) -> PyResult<DebugAnalysis> {
        let mut analysis = DebugAnalysis::new(query.clone());

        // Check query analysis stage
        if query_analysis_result.is_none() {
            analysis.issues.push(DebugIssue::new(
                "query_analysis".to_string(),
                "error".to_string(),
                "Query analysis failed or returned empty result".to_string(),
            ));
            analysis.failure_stage = Some("query_analysis".to_string());
            analysis.success = false;
        }

        // Check vector search results
        if vector_results.is_empty() {
            analysis.issues.push(DebugIssue::new(
                "vector_search".to_string(),
                "warning".to_string(),
                "Vector search returned no results".to_string(),
            ));
            analysis.suggestions.push(OptimizationSuggestion::new(
                "indexing".to_string(),
                "regenerate_embeddings".to_string(),
                "Regenerate embeddings with a different model".to_string(),
            ));
        } else {
            let avg_score: f32 = vector_results.iter().map(|(_, s)| *s).sum::<f32>() / vector_results.len() as f32;
            analysis.metrics.insert("vector_avg_score".to_string(), avg_score);
            
            if avg_score < 0.3 {
                analysis.issues.push(DebugIssue::new(
                    "vector_search".to_string(),
                    "warning".to_string(),
                    format!("Low vector search scores (avg: {:.2})", avg_score),
                ));
            }
        }

        // Check BM25 results
        if bm25_results.is_empty() {
            analysis.issues.push(DebugIssue::new(
                "bm25_search".to_string(),
                "warning".to_string(),
                "BM25 search returned no results".to_string(),
            ));
        } else {
            let avg_score: f32 = bm25_results.iter().map(|(_, s)| *s).sum::<f32>() / bm25_results.len() as f32;
            analysis.metrics.insert("bm25_avg_score".to_string(), avg_score);
        }

        // Check graph expansion
        if graph_results.is_empty() {
            analysis.issues.push(DebugIssue::new(
                "graph_expansion".to_string(),
                "info".to_string(),
                "Graph expansion returned no additional results".to_string(),
            ));
        }

        // Check result diversity
        let result_count = final_results.len();
        analysis.metrics.insert("result_count".to_string(), result_count as f32);
        
        if result_count < 3 {
            analysis.issues.push(DebugIssue::new(
                "ranking".to_string(),
                "warning".to_string(),
                format!("Too few results returned: {}", result_count),
            ));
            analysis.suggestions.push(OptimizationSuggestion::new(
                "ranking".to_string(),
                "increase_top_k".to_string(),
                "Increase top_k parameter or adjust ranking weights".to_string(),
            ));
        }

        // Check answer quality
        if let Some(quality) = answer_quality {
            analysis.metrics.insert("answer_quality".to_string(), quality);
            
            if quality < 0.5 {
                analysis.success = false;
                analysis.issues.push(DebugIssue::new(
                    "answer_generation".to_string(),
                    "error".to_string(),
                    format!("Low answer quality score: {:.2}", quality),
                ));
                
                // Suggest ranking weight adjustments
                let mut params = HashMap::new();
                params.insert("vector_weight".to_string(), 0.5);
                params.insert("bm25_weight".to_string(), 0.3);
                params.insert("graph_weight".to_string(), 0.2);
                
                let mut suggestion = OptimizationSuggestion::new(
                    "ranking".to_string(),
                    "adjust_weights".to_string(),
                    "Adjust ranking weights to improve retrieval quality".to_string(),
                );
                suggestion.expected_improvement = 0.2;
                suggestion.parameters = params;
                analysis.suggestions.push(suggestion);
            }
        }

        // Store in history
        let history = self.query_history.entry(query).or_insert_with(Vec::new);
        history.push(analysis.clone());
        if history.len() > 10 {
            history.remove(0);
        }

        Ok(analysis)
    }

    /// Get optimization suggestions based on history
    fn get_optimization_suggestions(&self) -> PyResult<Vec<OptimizationSuggestion>> {
        let mut suggestions = Vec::new();

        // Analyze common issues across queries
        let mut stage_failures: HashMap<String, usize> = HashMap::new();
        for analyses in self.query_history.values() {
            for analysis in analyses {
                if let Some(ref stage) = analysis.failure_stage {
                    *stage_failures.entry(stage.clone()).or_insert(0) += 1;
                }
            }
        }

        // Generate suggestions based on common failures
        for (stage, count) in stage_failures {
            if count > 3 {
                match stage.as_str() {
                    "vector_search" => {
                        suggestions.push(OptimizationSuggestion::new(
                            "indexing".to_string(),
                            "improve_embeddings".to_string(),
                            "Vector search frequently fails. Consider improving embedding quality.".to_string(),
                        ));
                    }
                    "bm25_search" => {
                        suggestions.push(OptimizationSuggestion::new(
                            "indexing".to_string(),
                            "rebuild_bm25_index".to_string(),
                            "BM25 search frequently fails. Consider rebuilding the index.".to_string(),
                        ));
                    }
                    "graph_expansion" => {
                        suggestions.push(OptimizationSuggestion::new(
                            "graph".to_string(),
                            "expand_graph".to_string(),
                            "Graph expansion frequently fails. Consider adding more graph edges.".to_string(),
                        ));
                    }
                    _ => {}
                }
            }
        }

        Ok(suggestions)
    }
}

impl Default for AutonomousDebugger {
    fn default() -> Self {
        Self::new()
    }
}

