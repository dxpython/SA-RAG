use crate::semantic_node::{NodeType, SemanticNode};
use regex::Regex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::VecDeque;

// Global counter for node IDs (thread-safe atomic counter)
static NODE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_node_id() -> u64 {
    NODE_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Semantic Parser: Splits document text into hierarchical semantic nodes
/// 
/// Algorithm:
/// 1. Paragraph segmentation based on punctuation, line breaks, and heading markers
/// 2. Identify heading levels (Markdown style #, ##, ### or numbered headings)
/// 3. Build parent-child relationships: heading -> paragraph -> sentence -> chunk
/// 4. Use weak rules to identify semantic boundaries (periods, question marks, exclamation marks, etc.)
pub struct SemanticParser;

impl SemanticParser {
    /// Parse document and return a list of semantic nodes (with complete hierarchy)
    pub fn parse(doc_id: u64, text: &str) -> Vec<SemanticNode> {
        let mut nodes = Vec::new();
        let len = text.len();

        // Create document root node
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

        // Step 1: Identify headings (Markdown style or numbered)
        let heading_re = Regex::new(r"(?m)^(#{1,6}\s+.+?)$|^(第[一二三四五六七八九十\d]+[章节部分]\s+.+?)$|^(\d+\.\d*\s+.+?)$").unwrap();
        let mut heading_positions: Vec<(usize, usize, u8)> = Vec::new(); // (start, end, level)
        
        for cap in heading_re.find_iter(text) {
            let start = cap.start();
            let end = cap.end();
            let heading_text = &text[start..end];
            let level = Self::detect_heading_level(heading_text);
            heading_positions.push((start, end, level));
        }

        // Step 2: Split document into sections by headings
        let mut last_end = 0;
        let mut heading_stack: VecDeque<(u64, u8)> = VecDeque::new(); // (node_id, level) for maintaining heading hierarchy

        for (h_start, h_end, h_level) in heading_positions.iter().copied() {
            // Process paragraphs before heading
            if last_end < h_start {
                let section_text = &text[last_end..h_start];
                let parent_id = heading_stack.front().map(|(id, _)| *id).unwrap_or(root_id);
                let section_nodes = Self::parse_section(doc_id, last_end, h_start, section_text, parent_id, &mut heading_stack);
                nodes.extend(section_nodes);
            }

            // Create heading node
            let heading_id = next_node_id();
            let heading_text = text[h_start..h_end].trim().to_string();
            let mut heading_node = SemanticNode::new(
                heading_id,
                doc_id,
                h_start,
                h_end,
                h_level,
                NodeType::Heading,
                heading_text,
            );

            // Build heading hierarchy relationships
            while let Some((_, parent_level)) = heading_stack.front() {
                if *parent_level >= h_level {
                    heading_stack.pop_front();
                } else {
                    break;
                }
            }
            if let Some((parent_id, _)) = heading_stack.front() {
                heading_node.add_parent(*parent_id);
            } else {
                heading_node.add_parent(root_id);
            }
            heading_stack.push_front((heading_id, h_level));
            root_node.add_child(heading_id);
            nodes.push(heading_node);

            last_end = h_end;
        }

        // Process content after the last heading
        if last_end < len {
            let section_text = &text[last_end..len];
            let parent_id = heading_stack.front().map(|(id, _)| *id).unwrap_or(root_id);
            let section_nodes = Self::parse_section(doc_id, last_end, len, section_text, parent_id, &mut heading_stack);
            nodes.extend(section_nodes);
        }

        // If no headings found, treat entire document as paragraphs
        if nodes.len() <= 1 || heading_positions.is_empty() {
            let paragraph_nodes = Self::parse_paragraphs(doc_id, 0, len, text, root_id);
            nodes.extend(paragraph_nodes);
        }

        // Add root node
        nodes.insert(0, root_node);

        // Step 3: Build parent-child relationships for all nodes
        Self::build_hierarchical_relationships(&mut nodes, root_id);

        nodes
    }

    /// Detect heading level (1-6)
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
        } else if heading_text.starts_with("第") {
            // Chinese numbering: 第一章, 第一节, etc.
            if heading_text.contains("章") {
                1
            } else if heading_text.contains("节") {
                2
            } else {
                3
            }
        } else {
            // Numbered headings: 1., 1.1, etc.
            let dot_count = heading_text.matches('.').count();
            (dot_count.min(6) + 1) as u8
        }
    }

    /// Parse section content (paragraphs and sentences)
    fn parse_section(
        doc_id: u64,
        section_start: usize,
        section_end: usize,
        text: &str,
        parent_id: u64,
        _heading_stack: &mut VecDeque<(u64, u8)>,
    ) -> Vec<SemanticNode> {
        let mut nodes = Vec::new();
        
        // Split paragraphs by double newlines
        let para_re = Regex::new(r"\n\s*\n+").unwrap();
        let mut last_pos = 0;
        let mut para_start = 0;

        for mat in para_re.find_iter(text) {
            let para_end = mat.start();
            if para_end > para_start {
                let para_text = &text[para_start..para_end];
                let para_node = Self::create_paragraph_node(
                    doc_id,
                    section_start + para_start,
                    section_start + para_end,
                    para_text,
                    parent_id,
                );
                nodes.push(para_node);
            }
            para_start = mat.end();
            last_pos = para_start;
        }

        // Process last paragraph
        if last_pos < text.len() {
            let para_text = &text[last_pos..];
            if !para_text.trim().is_empty() {
                let para_node = Self::create_paragraph_node(
                    doc_id,
                    section_start + last_pos,
                    section_end,
                    para_text,
                    parent_id,
                );
                nodes.push(para_node);
            }
        }

        nodes
    }

    /// Parse paragraphs (when no headings are present)
    fn parse_paragraphs(
        doc_id: u64,
        start: usize,
        end: usize,
        text: &str,
        parent_id: u64,
    ) -> Vec<SemanticNode> {
        let mut nodes = Vec::new();
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

    /// Create paragraph node and further split into sentences
    fn create_paragraph_node(
        doc_id: u64,
        start: usize,
        end: usize,
        text: &str,
        parent_id: u64,
    ) -> SemanticNode {
        let trimmed = text.trim();
        let para_id = next_node_id();
        let mut para_node = SemanticNode::new(
            para_id,
            doc_id,
            start,
            end,
            2,
            NodeType::Paragraph,
            trimmed.to_string(),
        );
        para_node.add_parent(parent_id);

        // Further split into sentences (based on periods, question marks, exclamation marks)
        let sentence_re = Regex::new(r"[。！？.!?]\s*").unwrap();
        let mut last_pos = 0;
        let mut sentence_nodes = Vec::new();

        for mat in sentence_re.find_iter(trimmed) {
            let sent_end = mat.end();
            if sent_end > last_pos {
                let sent_text = &trimmed[last_pos..sent_end].trim();
                if !sent_text.is_empty() && sent_text.len() > 5 {
                    let sent_id = next_node_id();
                    let mut sent_node = SemanticNode::new(
                        sent_id,
                        doc_id,
                        start + last_pos,
                        start + sent_end,
                        3,
                        NodeType::Sentence,
                        sent_text.to_string(),
                    );
                    sent_node.add_parent(para_id);
                    sentence_nodes.push(sent_node);
                }
            }
            last_pos = sent_end;
        }

        // Process last sentence (if not ending with punctuation)
        if last_pos < trimmed.len() {
            let sent_text = &trimmed[last_pos..].trim();
            if !sent_text.is_empty() && sent_text.len() > 5 {
                let sent_id = next_node_id();
                let mut sent_node = SemanticNode::new(
                    sent_id,
                    doc_id,
                    start + last_pos,
                    end,
                    3,
                    NodeType::Sentence,
                    sent_text.to_string(),
                );
                sent_node.add_parent(para_id);
                sentence_nodes.push(sent_node);
            }
        }

        // Add sentence nodes to paragraph node's children
        for sent_node in sentence_nodes {
            para_node.add_child(sent_node.node_id);
        }

        para_node
    }

    /// Build complete hierarchical relationships (ensure bidirectional links)
    fn build_hierarchical_relationships(nodes: &mut Vec<SemanticNode>, _root_id: u64) {
        // Create node ID to index mapping
        let mut node_map: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
        for (idx, node) in nodes.iter().enumerate() {
            node_map.insert(node.node_id, idx);
        }

        // Build bidirectional relationships
        for i in 0..nodes.len() {
            let node_id = nodes[i].node_id;
            
            // Add current node as child to each parent
            for &parent_id in &nodes[i].parents.clone() {
                if let Some(&parent_idx) = node_map.get(&parent_id) {
                    if parent_idx < nodes.len() {
                        nodes[parent_idx].add_child(node_id);
                    }
                }
            }

            // Add current node as parent to each child
            for &child_id in &nodes[i].children.clone() {
                if let Some(&child_idx) = node_map.get(&child_id) {
                    if child_idx < nodes.len() {
                        nodes[child_idx].add_parent(node_id);
                    }
                }
            }
        }
    }
}
