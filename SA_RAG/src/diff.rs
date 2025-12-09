/// Differential Indexing Module: Implements document version management and incremental updates
/// 
/// Algorithm:
/// 1. Use Myers diff algorithm to calculate text differences
/// 2. Identify added, deleted, and modified text segments
/// 3. Only regenerate embeddings and index for changed paragraphs
/// 4. Maintain document version history, support rollback
/// 
/// This avoids full index rebuilds and significantly improves update efficiency

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// Rolling hash types (defined inline for now)
mod rolling_hash {
    use std::collections::HashMap;

    const BASE: u64 = 256;
    const MOD: u64 = 1_000_000_007;

    pub struct RollingHash;

    impl RollingHash {
        pub fn hash_segment(text: &str) -> u64 {
            let mut hash = 0u64;
            for byte in text.bytes() {
                hash = (hash * BASE + byte as u64) % MOD;
            }
            hash
        }

        pub fn find_matching_segments(
            old_text: &str,
            new_text: &str,
            segment_size: usize,
        ) -> Vec<(usize, usize)> {
            // Simplified: calculate hashes for segments
            let mut matches = Vec::new();
            let old_segments: HashMap<u64, usize> = (0..old_text.len().saturating_sub(segment_size))
                .step_by(segment_size)
                .map(|i| {
                    let end = (i + segment_size).min(old_text.len());
                    (Self::hash_segment(&old_text[i..end]), i)
                })
                .collect();

            for i in (0..new_text.len().saturating_sub(segment_size)).step_by(segment_size) {
                let end = (i + segment_size).min(new_text.len());
                let hash = Self::hash_segment(&new_text[i..end]);
                if let Some(&old_pos) = old_segments.get(&hash) {
                    // Verify match
                    if old_text[old_pos..old_pos + segment_size.min(old_text.len() - old_pos)] ==
                       new_text[i..end] {
                        matches.push((old_pos, i));
                    }
                }
            }
            matches
        }
    }

    pub struct TextSegment {
        pub start: usize,
        pub end: usize,
        pub hash: u64,
        pub content: String,
    }

    impl TextSegment {
        pub fn new(start: usize, end: usize, content: &str) -> Self {
            Self {
                start,
                end,
                hash: RollingHash::hash_segment(content),
                content: content.to_string(),
            }
        }
    }
}

use rolling_hash::{RollingHash, TextSegment};

/// Document version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentVersion {
    pub doc_id: u64,
    pub version: u64,
    pub text: String,
    pub timestamp: u64,
}

/// Text difference operation types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffOperation {
    /// Unchanged
    Equal,
    /// Delete (position in old text)
    Delete { start: usize, end: usize },
    /// Insert (position in new text)
    Insert { start: usize, end: usize },
    /// Replace
    Replace { old_start: usize, old_end: usize, new_start: usize, new_end: usize },
}

/// Diff result
#[derive(Debug, Clone)]
pub struct DiffResult {
    pub operations: Vec<DiffOperation>,
    pub changed_ranges: Vec<(usize, usize)>, // (start, end) changed ranges in new text
}

/// Document version manager with rolling hash support
pub struct DocumentVersionManager {
    /// doc_id -> version list (sorted by time, newest first)
    versions: HashMap<u64, Vec<DocumentVersion>>,
    /// Current version number counter
    version_counter: u64,
    /// doc_id -> rolling hash of current version
    document_hashes: HashMap<u64, u64>,
    /// doc_id -> text segments with hashes (for incremental updates)
    document_segments: HashMap<u64, Vec<TextSegment>>,
}

impl DocumentVersionManager {
    pub fn new() -> Self {
        Self {
            versions: HashMap::new(),
            version_counter: 1,
            document_hashes: HashMap::new(),
            document_segments: HashMap::new(),
        }
    }

    /// Add new version with rolling hash calculation
    pub fn add_version(&mut self, doc_id: u64, text: String, timestamp: u64) -> u64 {
        let version = self.version_counter;
        self.version_counter += 1;

        // Calculate rolling hash for the document
        let doc_hash = RollingHash::hash_segment(&text);
        self.document_hashes.insert(doc_id, doc_hash);

        // Calculate text segments (for incremental updates)
        let segment_size = 100; // 100 characters per segment
        let segments: Vec<TextSegment> = (0..text.len())
            .step_by(segment_size)
            .map(|start| {
                let end = (start + segment_size).min(text.len());
                TextSegment::new(start, end, &text[start..end])
            })
            .collect();
        self.document_segments.insert(doc_id, segments);

        let doc_version = DocumentVersion {
            doc_id,
            version,
            text,
            timestamp,
        };

        self.versions.entry(doc_id).or_insert_with(Vec::new).push(doc_version);
        version
    }

    /// Get document hash (for quick change detection)
    pub fn get_document_hash(&self, doc_id: u64) -> Option<u64> {
        self.document_hashes.get(&doc_id).copied()
    }

    /// Check if document has changed (by comparing hashes)
    pub fn has_changed(&self, doc_id: u64, new_text: &str) -> bool {
        let new_hash = RollingHash::hash_segment(new_text);
        self.document_hashes.get(&doc_id)
            .map(|&old_hash| old_hash != new_hash)
            .unwrap_or(true)
    }

    /// Get changed segments using rolling hash (for incremental updates)
    pub fn get_changed_segments_rolling_hash(
        &self,
        doc_id: u64,
        new_text: &str,
    ) -> Vec<(usize, usize)> {
        if let Some(_old_segments) = self.document_segments.get(&doc_id) {
            if let Some(old_version) = self.get_latest_version(doc_id) {
                // Use rolling hash to find matching segments
                let segment_size = 100;
                let matches = RollingHash::find_matching_segments(
                    &old_version.text,
                    new_text,
                    segment_size,
                );

                // Find unmatched segments (changed regions)
                let new_segments = (0..new_text.len())
                    .step_by(segment_size)
                    .map(|start| (start, (start + segment_size).min(new_text.len())))
                    .collect::<Vec<_>>();

                let mut changed = Vec::new();
                let mut matched_positions = std::collections::HashSet::new();
                for (_, new_pos) in &matches {
                    matched_positions.insert(*new_pos);
                }

                for (start, end) in new_segments {
                    if !matched_positions.contains(&start) {
                        changed.push((start, end));
                    }
                }

                changed
            } else {
                // No old version, entire document is new
                vec![(0, new_text.len())]
            }
        } else {
            // No segments stored, entire document is new
            vec![(0, new_text.len())]
        }
    }

    /// Get latest version of document
    pub fn get_latest_version(&self, doc_id: u64) -> Option<&DocumentVersion> {
        self.versions.get(&doc_id)?.last()
    }

    /// Get specified version of document
    pub fn get_version(&self, doc_id: u64, version: u64) -> Option<&DocumentVersion> {
        self.versions.get(&doc_id)?.iter().find(|v| v.version == version)
    }

    /// Calculate difference between two versions
    pub fn calculate_diff(&self, doc_id: u64, old_version: u64, new_version: u64) -> Option<DiffResult> {
        let old_ver = self.get_version(doc_id, old_version)?;
        let new_ver = self.get_version(doc_id, new_version)?;

        Some(calculate_text_diff(&old_ver.text, &new_ver.text))
    }
}

impl Default for DocumentVersionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate text difference (simplified Myers algorithm)
/// 
/// Algorithm:
/// Myers algorithm uses dynamic programming to find the shortest edit path
/// Here we implement a simplified version using greedy strategy to find main differences
pub fn calculate_text_diff(old_text: &str, new_text: &str) -> DiffResult {
    if old_text == new_text {
        return DiffResult {
            operations: vec![DiffOperation::Equal],
            changed_ranges: Vec::new(),
        };
    }

    let old_chars: Vec<char> = old_text.chars().collect();
    let new_chars: Vec<char> = new_text.chars().collect();
    let old_len = old_chars.len();
    let new_len = new_chars.len();

    // Use Longest Common Subsequence (LCS) to find common parts
    let lcs = longest_common_subsequence(&old_chars, &new_chars);
    
    // Generate diff operations based on LCS
    let mut operations = Vec::new();
    let mut changed_ranges = Vec::new();
    
    let mut old_idx = 0;
    let mut new_idx = 0;
    let mut lcs_idx = 0;

    while old_idx < old_len || new_idx < new_len {
        // Find next common subsequence position
        let (next_old, next_new) = if lcs_idx < lcs.len() {
            (lcs[lcs_idx].0, lcs[lcs_idx].1)
        } else {
            (old_len, new_len)
        };

        // Handle deletion
        if old_idx < next_old {
            let delete_start = old_idx;
            let delete_end = next_old;
            operations.push(DiffOperation::Delete {
                start: delete_start,
                end: delete_end,
            });
            old_idx = next_old;
        }

        // Handle insertion
        if new_idx < next_new {
            let insert_start = new_idx;
            let insert_end = next_new;
            operations.push(DiffOperation::Insert {
                start: insert_start,
                end: insert_end,
            });
            
            // Record changed range
            changed_ranges.push((insert_start, insert_end));
            new_idx = next_new;
        }

        // Handle common parts
        if old_idx < old_len && new_idx < new_len && lcs_idx < lcs.len() {
            let (lcs_old, lcs_new) = lcs[lcs_idx];
            if old_idx == lcs_old && new_idx == lcs_new {
                // Found matching character, skip
                old_idx += 1;
                new_idx += 1;
                lcs_idx += 1;
            } else {
                // Mismatch, continue
                old_idx += 1;
                new_idx += 1;
            }
        } else if lcs_idx < lcs.len() {
            lcs_idx += 1;
        } else {
            break;
        }
    }

    // Merge adjacent changed ranges
    changed_ranges = merge_ranges(changed_ranges);

    DiffResult {
        operations,
        changed_ranges,
    }
}

/// Calculate Longest Common Subsequence (LCS)
/// Returns list of (old_index, new_index)
fn longest_common_subsequence(old: &[char], new: &[char]) -> Vec<(usize, usize)> {
    let old_len = old.len();
    let new_len = new.len();

    // Dynamic programming table
    let mut dp = vec![vec![0; new_len + 1]; old_len + 1];

    // Fill DP table
    for i in 1..=old_len {
        for j in 1..=new_len {
            if old[i - 1] == new[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    // Backtrack to find LCS
    let mut lcs = Vec::new();
    let mut i = old_len;
    let mut j = new_len;

    while i > 0 && j > 0 {
        if old[i - 1] == new[j - 1] {
            lcs.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }

    lcs.reverse();
    lcs
}

/// Merge overlapping or adjacent ranges
fn merge_ranges(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if ranges.is_empty() {
        return ranges;
    }

    ranges.sort_by_key(|r| r.0);
    let mut merged = Vec::new();
    let mut current = ranges[0];

    for range in ranges.into_iter().skip(1) {
        if range.0 <= current.1 {
            // Overlapping or adjacent, merge
            current.1 = current.1.max(range.1);
        } else {
            // Not overlapping, save current and start new
            merged.push(current);
            current = range;
        }
    }
    merged.push(current);

    merged
}

/// Identify text segments that need re-indexing based on diff result
/// 
/// Returns text fragments that need reprocessing (positions in new text)
pub fn get_changed_segments(new_text: &str, diff_result: &DiffResult) -> Vec<String> {
    let mut segments = Vec::new();
    let chars: Vec<char> = new_text.chars().collect();

    for (start, end) in &diff_result.changed_ranges {
        if *end <= chars.len() {
            let segment: String = chars[*start..*end].iter().collect();
            segments.push(segment);
        }
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_diff() {
        let old = "Hello world";
        let new = "Hello Rust world";
        let diff = calculate_text_diff(old, new);
        assert!(!diff.changed_ranges.is_empty());
    }

    #[test]
    fn test_identical_text() {
        let text = "Same text";
        let diff = calculate_text_diff(text, text);
        assert_eq!(diff.operations.len(), 1);
        assert!(matches!(diff.operations[0], DiffOperation::Equal));
    }
}
