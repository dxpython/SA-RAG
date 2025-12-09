// Multimodal-Ready Node Engine
// Supports image, table, code block, and formula nodes

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use crate::semantic_node::{SemanticNode, NodeType};

/// Multimodal content type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[pyclass]
pub enum MultimodalType {
    #[pyo3(name = "IMAGE")]
    Image,
    #[pyo3(name = "TABLE")]
    Table,
    #[pyo3(name = "CODE")]
    Code,
    #[pyo3(name = "FORMULA")]
    Formula,
    #[pyo3(name = "TEXT")]
    Text, // Default text node
}

/// Multimodal node metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct MultimodalMetadata {
    #[pyo3(get, set)]
    pub content_type: String, // MultimodalType as string
    #[pyo3(get, set)]
    pub mime_type: Option<String>, // e.g., "image/png", "text/csv"
    #[pyo3(get, set)]
    pub width: Option<u32>, // For images/tables
    #[pyo3(get, set)]
    pub height: Option<u32>, // For images/tables
    #[pyo3(get, set)]
    pub language: Option<String>, // For code blocks
    #[pyo3(get, set)]
    pub formula_type: Option<String>, // e.g., "latex", "mathml"
    #[pyo3(get, set)]
    pub raw_data: Option<Vec<u8>>, // Raw binary data (for images)
    #[pyo3(get, set)]
    pub alt_text: Option<String>, // Alternative text description
}

#[pymethods]
impl MultimodalMetadata {
    #[new]
    fn new(content_type: String) -> Self {
        Self {
            content_type,
            mime_type: None,
            width: None,
            height: None,
            language: None,
            formula_type: None,
            raw_data: None,
            alt_text: None,
        }
    }
}

/// Multimodal node parser
pub struct MultimodalParser;

impl MultimodalParser {
    pub fn new() -> Self {
        Self
    }

    /// Create semantic node from image
    pub fn from_image(
        doc_id: u64,
        node_id: u64,
        image_data: Vec<u8>,
        alt_text: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
    ) -> SemanticNode {
        let metadata = serde_json::json!({
            "content_type": "IMAGE",
            "mime_type": "image/png", // Default, should be detected
            "width": width,
            "height": height,
            "alt_text": alt_text,
        });

        SemanticNode::new(
            node_id,
            doc_id,
            0,
            0,
            0,
            NodeType::Code, // Use Code type for now, could add Image type
            alt_text.unwrap_or_else(|| "Image".to_string()),
        )
    }

    /// Create semantic node from table
    pub fn from_table(
        doc_id: u64,
        node_id: u64,
        table_data: String,
        rows: usize,
        cols: usize,
    ) -> SemanticNode {
        let metadata = serde_json::json!({
            "content_type": "TABLE",
            "rows": rows,
            "cols": cols,
        });

        SemanticNode::new(
            node_id,
            doc_id,
            0,
            table_data.len(),
            0,
            NodeType::Table,
            table_data,
        )
    }

    /// Create semantic node from code block
    pub fn from_code(
        doc_id: u64,
        node_id: u64,
        code: String,
        language: Option<String>,
    ) -> SemanticNode {
        let metadata = serde_json::json!({
            "content_type": "CODE",
            "language": language,
        });

        SemanticNode::new(
            node_id,
            doc_id,
            0,
            code.len(),
            0,
            NodeType::Code,
            code,
        )
    }

    /// Create semantic node from formula
    pub fn from_formula(
        doc_id: u64,
        node_id: u64,
        formula: String,
        formula_type: Option<String>,
    ) -> SemanticNode {
        let metadata = serde_json::json!({
            "content_type": "FORMULA",
            "formula_type": formula_type.unwrap_or_else(|| "latex".to_string()),
        });

        SemanticNode::new(
            node_id,
            doc_id,
            0,
            formula.len(),
            0,
            NodeType::Theorem, // Use Theorem type for formulas
            formula,
        )
    }

    /// Detect multimodal type from content
    pub fn detect_type(content: &str) -> MultimodalType {
        // Simple heuristics for detection
        if content.trim().starts_with("```") || content.contains("def ") || content.contains("function ") {
            MultimodalType::Code
        } else if content.contains("\\(") || content.contains("$$") || content.contains("\\[") {
            MultimodalType::Formula
        } else if content.contains("|") && content.matches("|").count() > 3 {
            MultimodalType::Table
        } else {
            MultimodalType::Text
        }
    }
}

impl Default for MultimodalParser {
    fn default() -> Self {
        Self::new()
    }
}

