use std::collections::HashMap;
use rust_stemmers::{Algorithm, Stemmer};

/// A simple in-memory inverted index with BM25 scoring.
pub struct BM25Index {
    // Term -> List of (NodeID, TermFreq)
    inverted_index: HashMap<String, Vec<(u64, u32)>>,
    // NodeID -> Document Length (number of tokens)
    doc_lengths: HashMap<u64, u32>,
    // Average Document Length
    avg_dl: f32,
    // Total documents
    num_docs: usize,
    // BM25 parameters
    k1: f32,
    b: f32,
    stemmer: Stemmer,
}

impl BM25Index {
    pub fn new() -> Self {
        Self {
            inverted_index: HashMap::new(),
            doc_lengths: HashMap::new(),
            avg_dl: 0.0,
            num_docs: 0,
            k1: 1.5,
            b: 0.75,
            stemmer: Stemmer::create(Algorithm::English),
        }
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| {
                let cleaned: String = s.chars().filter(|c| c.is_alphanumeric()).collect();
                self.stemmer.stem(&cleaned).to_string()
            })
            .filter(|s| !s.is_empty())
            .collect()
    }

    pub fn insert(&mut self, node_id: u64, text: &str) {
        let tokens = self.tokenize(text);
        let doc_len = tokens.len() as u32;
        
        self.doc_lengths.insert(node_id, doc_len);
        
        let mut term_freqs: HashMap<String, u32> = HashMap::new();
        for token in tokens {
            *term_freqs.entry(token).or_insert(0) += 1;
        }

        for (term, freq) in term_freqs {
            self.inverted_index
                .entry(term)
                .or_insert_with(Vec::new)
                .push((node_id, freq));
        }

        // Update stats
        let total_len: u64 = self.doc_lengths.values().map(|&x| x as u64).sum();
        self.num_docs += 1;
        self.avg_dl = total_len as f32 / self.num_docs as f32;
    }

    /// Score a specific document for a query (used in multi-stage retrieval)
    pub fn score_document(&self, query: &str, node_id: u64) -> f32 {
        let query_tokens = self.tokenize(query);
        let mut score = 0.0;
        
        let doc_len = self.doc_lengths.get(&node_id).copied().unwrap_or(0) as f32;
        
        for term in query_tokens {
            if let Some(postings) = self.inverted_index.get(&term) {
                // Find term frequency in this document
                let tf = postings
                    .iter()
                    .find(|(id, _)| *id == node_id)
                    .map(|(_, tf)| *tf as f32)
                    .unwrap_or(0.0);
                
                if tf > 0.0 {
                    // Calculate IDF
                    let df = postings.len() as f32;
                    let idf = ((self.num_docs as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();
                    
                    // Calculate BM25 score component
                    let numerator = tf * (self.k1 + 1.0);
                    let denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_dl));
                    score += idf * (numerator / denominator);
                }
            }
        }
        
        score
    }

    pub fn search(&self, query: &str, k: usize) -> Vec<(u64, f32)> {
        let tokens = self.tokenize(query);
        let mut scores: HashMap<u64, f32> = HashMap::new();

        for term in tokens {
            if let Some(postings) = self.inverted_index.get(&term) {
                // IDF calculation
                let n_q = postings.len() as f32; // Number of docs containing term
                let idf = ((self.num_docs as f32 - n_q + 0.5) / (n_q + 0.5) + 1.0).ln();

                for &(node_id, freq) in postings {
                    let doc_len = *self.doc_lengths.get(&node_id).unwrap_or(&0) as f32;
                    let tf = freq as f32;
                    let num = tf * (self.k1 + 1.0);
                    let denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_dl));
                    
                    let score = idf * (num / denom);
                    *scores.entry(node_id).or_insert(0.0) += score;
                }
            }
        }

        let mut results: Vec<(u64, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.into_iter().take(k).collect()
    }
}
