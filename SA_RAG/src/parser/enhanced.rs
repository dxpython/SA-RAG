/// Enhanced Semantic Parser: Engineering-grade parsing with support for structured elements
/// 
/// Supports:
/// - Multi-level semantic node tree: Section → Subsection → Paragraph → Sentence
/// - Structured elements: headings, lists, code blocks, tables, quotes, references
/// - Node types: Definition, Example, Conclusion, Procedure, Code, Table, etc.
/// - Semantic edges: NEXT, PARENT_OF, REFERS_TO, CITES

use crate::semantic_node::{SemanticNode, NodeType};
use regex::Regex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

static NODE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_node_id() -> u64 {
    NODE_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Enhanced semantic parser
pub struct EnhancedSemanticParser;

impl EnhancedSemanticParser {
    /// Parse document with full structured element support
    pub fn parse(doc_id: u64, text: &str) -> Vec<SemanticNode> {
        let mut nodes = Vec::new();
        let len = text.len();

        // Create root node
        let root_id = next_node_id();
        let mut root_node = SemanticNode::new(
            root_id,
            doc_id,
            0,
            len,
            0,
            NodeType::Root,
            format!("Document Root #{}", doc_id),
        );

        // Step 1: Identify all structured elements
        let structured_elements = Self::identify_structured_elements(text);

        // Step 2: Build hierarchical structure
        let mut last_pos = 0;
        let mut heading_stack: VecDeque<(u64, u8)> = VecDeque::new();

        for element in structured_elements {
            // Process content before this element
            if last_pos < element.start {
                let content = &text[last_pos..element.start];
                let parent_id = heading_stack.front().map(|(id, _)| *id).unwrap_or(root_id);
                let content_nodes = Self::parse_content(
                    doc_id,
                    last_pos,
                    element.start,
                    content,
                    parent_id,
                );
                nodes.extend(content_nodes);
            }

            // Process the structured element
            let element_type = element.element_type.clone();
            let element_node = Self::create_structured_node(
                doc_id,
                element.start,
                element.end,
                &text[element.start..element.end],
                element_type,
                element.level,
                heading_stack.front().map(|(id, _)| *id).unwrap_or(root_id),
            );

            // Update heading stack for headings
            if matches!(element.element_type, NodeType::Heading) {
                while let Some((_, parent_level)) = heading_stack.front() {
                    if *parent_level >= element.level {
                        heading_stack.pop_front();
                    } else {
                        break;
                    }
                }
                heading_stack.push_front((element_node.node_id, element.level));
                root_node.add_child(element_node.node_id);
            }

            nodes.push(element_node);
            last_pos = element.end;
        }

        // Process remaining content
        if last_pos < len {
            let content = &text[last_pos..len];
            let parent_id = heading_stack.front().map(|(id, _)| *id).unwrap_or(root_id);
            let content_nodes = Self::parse_content(doc_id, last_pos, len, content, parent_id);
            nodes.extend(content_nodes);
        }

        // Add root node
        nodes.insert(0, root_node);

        // Step 3: Build semantic relationships
        Self::build_semantic_relationships(&mut nodes);

        nodes
    }

    /// Identify all structured elements in text
    fn identify_structured_elements(text: &str) -> Vec<StructuredElement> {
        let mut elements = Vec::new();

        // Headings (Markdown style)
        let heading_re = Regex::new(r"(?m)^(#{1,6}\s+.+?)$").unwrap();
        for cap in heading_re.find_iter(text) {
            let level = Self::detect_heading_level(&text[cap.start()..cap.end()]);
            elements.push(StructuredElement {
                start: cap.start(),
                end: cap.end(),
                element_type: NodeType::Heading,
                level,
            });
        }

        // Code blocks
        let code_block_re = Regex::new(r"(?s)```[\s\S]*?```").unwrap();
        for cap in code_block_re.find_iter(text) {
            elements.push(StructuredElement {
                start: cap.start(),
                end: cap.end(),
                element_type: NodeType::Code,
                level: 0,
            });
        }

        // Lists (ordered and unordered)
        let list_re = Regex::new(r"(?m)^(\s*[-*+]|\s*\d+\.)\s+.+$").unwrap();
        for cap in list_re.find_iter(text) {
            elements.push(StructuredElement {
                start: cap.start(),
                end: cap.end(),
                element_type: NodeType::List,
                level: 0,
            });
        }

        // Tables (Markdown style)
        let table_re = Regex::new(r"(?m)^\|.+\|$").unwrap();
        for cap in table_re.find_iter(text) {
            elements.push(StructuredElement {
                start: cap.start(),
                end: cap.end(),
                element_type: NodeType::Table,
                level: 0,
            });
        }

        // Quotes
        let quote_re = Regex::new(r"(?m)^>\s+.+$").unwrap();
        for cap in quote_re.find_iter(text) {
            elements.push(StructuredElement {
                start: cap.start(),
                end: cap.end(),
                element_type: NodeType::Quote,
                level: 0,
            });
        }

        // Sort by position
        elements.sort_by_key(|e| e.start);
        elements
    }

    /// Parse regular content (paragraphs, sentences)
    fn parse_content(
        doc_id: u64,
        start: usize,
        end: usize,
        text: &str,
        parent_id: u64,
    ) -> Vec<SemanticNode> {
        let mut nodes = Vec::new();

        // Split into paragraphs
        let para_re = Regex::new(r"\n\s*\n+").unwrap();
        let mut last_pos = 0;

        for mat in para_re.find_iter(text) {
            let para_end = mat.start();
            if para_end > last_pos {
                let para_text = &text[last_pos..para_end];
                let para_node = Self::create_paragraph_node(
                    doc_id,
                    start + last_pos,
                    start + para_end,
                    para_text,
                    parent_id,
                );
                nodes.push(para_node);
            }
            last_pos = mat.end();
        }

        // Last paragraph
        if last_pos < text.len() {
            let para_text = &text[last_pos..];
            if !para_text.trim().is_empty() {
                let para_node = Self::create_paragraph_node(
                    doc_id,
                    start + last_pos,
                    end,
                    para_text,
                    parent_id,
                );
                nodes.push(para_node);
            }
        }

        nodes
    }

    /// Create structured node
    fn create_structured_node(
        doc_id: u64,
        start: usize,
        end: usize,
        text: &str,
        node_type: NodeType,
        level: u8,
        parent_id: u64,
    ) -> SemanticNode {
        let node_id = next_node_id();
        let mut node = SemanticNode::new(
            node_id,
            doc_id,
            start,
            end,
            level,
            node_type,
            text.trim().to_string(),
        );
        node.add_parent(parent_id);
        node
    }

    /// Create paragraph node
    fn create_paragraph_node(
        doc_id: u64,
        start: usize,
        end: usize,
        text: &str,
        parent_id: u64,
    ) -> SemanticNode {
        // Detect paragraph type
        let node_type = Self::detect_paragraph_type(text);
        
        let node_id = next_node_id();
        let mut node = SemanticNode::new(
            node_id,
            doc_id,
            start,
            end,
            2, // Paragraph level
            node_type,
            text.trim().to_string(),
        );
        node.add_parent(parent_id);
        node
    }

    /// Detect paragraph type based on content
    fn detect_paragraph_type(text: &str) -> NodeType {
        let text_lower = text.to_lowercase();
        
        // Definition patterns
        if text_lower.contains("is defined as") || text_lower.contains("refers to") 
            || text_lower.contains("means") || text_lower.contains("denotes") {
            return NodeType::Definition;
        }
        
        // Example patterns
        if text_lower.contains("for example") || text_lower.contains("such as")
            || text_lower.contains("e.g.") || text_lower.starts_with("example:") {
            return NodeType::Example;
        }
        
        // Conclusion patterns
        if text_lower.contains("in conclusion") || text_lower.contains("to summarize")
            || text_lower.contains("in summary") || text_lower.contains("therefore") {
            return NodeType::Conclusion;
        }
        
        // Procedure patterns
        if text_lower.contains("step") || text_lower.contains("first") && text_lower.contains("then")
            || text_lower.contains("procedure") || text_lower.contains("algorithm") {
            return NodeType::Procedure;
        }
        
        NodeType::Paragraph
    }

    /// Detect heading level
    fn detect_heading_level(heading_text: &str) -> u8 {
        if heading_text.starts_with("######") {
            6
        } else if heading_text.starts_with("#####") {
            5
        } else if heading_text.starts_with("####") {
            4
        } else if heading_text.starts_with("###") {
            3
        } else if heading_text.starts_with("##") {
            2
        } else if heading_text.starts_with("#") {
            1
        } else {
            1
        }
    }

    /// Build semantic relationships (NEXT, REFERS_TO, CITES)
    fn build_semantic_relationships(nodes: &mut [SemanticNode]) {
        // Build NEXT relationships (sequential)
        for i in 0..nodes.len().saturating_sub(1) {
            if nodes[i].doc_id == nodes[i + 1].doc_id {
                // Add NEXT relationship metadata
                nodes[i].metadata.insert("next_node".to_string(), nodes[i + 1].node_id.to_string());
            }
        }

        // Detect REFERS_TO relationships (simple pattern matching)
        for i in 0..nodes.len() {
            let text = &nodes[i].text;
            for j in 0..nodes.len() {
                if i != j {
                    // Check if node i mentions concepts from node j
                    let keywords: Vec<&str> = nodes[j].text.split_whitespace().take(5).collect();
                    let mention_count = keywords.iter()
                        .filter(|kw| text.contains(*kw))
                        .count();
                    
                    if mention_count >= 2 {
                        nodes[i].metadata.insert(
                            format!("refers_to_{}", nodes[j].node_id),
                            nodes[j].node_id.to_string(),
                        );
                    }
                }
            }
        }
    }
}

/// Structured element information
struct StructuredElement {
    start: usize,
    end: usize,
    element_type: NodeType,
    level: u8,
}

