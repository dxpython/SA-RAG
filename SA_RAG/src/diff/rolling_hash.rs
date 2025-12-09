/// Rolling Hash for efficient text change detection
/// 
/// Uses polynomial rolling hash (Rabin-Karp style) for fast segment comparison

use std::collections::HashMap;

const BASE: u64 = 256;
const MOD: u64 = 1_000_000_007;

/// Rolling hash calculator
pub struct RollingHash {
    /// Hash value
    hash: u64,
    /// Base power for current position
    base_power: u64,
    /// Window size
    window_size: usize,
}

impl RollingHash {
    /// Create new rolling hash with window size
    pub fn new(window_size: usize) -> Self {
        let mut base_power = 1;
        for _ in 0..window_size - 1 {
            base_power = (base_power * BASE) % MOD;
        }
        
        Self {
            hash: 0,
            base_power,
            window_size,
        }
    }

    /// Calculate hash for a text segment
    pub fn hash_segment(text: &str) -> u64 {
        let mut hash = 0u64;
        for byte in text.bytes() {
            hash = (hash * BASE + byte as u64) % MOD;
        }
        hash
    }

    /// Calculate rolling hashes for all segments of given size
    pub fn calculate_segment_hashes(text: &str, segment_size: usize) -> Vec<(usize, u64)> {
        if text.len() < segment_size {
            return vec![(0, Self::hash_segment(text))];
        }

        let mut hashes = Vec::new();
        let mut hash = Self::hash_segment(&text[0..segment_size]);
        hashes.push((0, hash));

        let mut base_power = 1;
        for _ in 0..segment_size - 1 {
            base_power = (base_power * BASE) % MOD;
        }

        for i in 1..=text.len() - segment_size {
            // Remove first character
            let first_char = text.as_bytes()[i - 1] as u64;
            hash = (hash + MOD - (first_char * base_power) % MOD) % MOD;
            
            // Add new character
            let new_char = text.as_bytes()[i + segment_size - 1] as u64;
            hash = (hash * BASE + new_char) % MOD;
            
            hashes.push((i, hash));
        }

        hashes
    }

    /// Find matching segments between two texts
    pub fn find_matching_segments(
        old_text: &str,
        new_text: &str,
        segment_size: usize,
    ) -> Vec<(usize, usize)> {
        // Calculate hashes for both texts
        let old_hashes: HashMap<u64, usize> = Self::calculate_segment_hashes(old_text, segment_size)
            .into_iter()
            .map(|(pos, hash)| (hash, pos))
            .collect();
        
        let new_hashes = Self::calculate_segment_hashes(new_text, segment_size);
        
        // Find matches
        let mut matches = Vec::new();
        for (new_pos, hash) in new_hashes {
            if let Some(&old_pos) = old_hashes.get(&hash) {
                // Verify it's actually a match (hash collision check)
                if old_text[old_pos..old_pos + segment_size.min(old_text.len() - old_pos)] ==
                   new_text[new_pos..new_pos + segment_size.min(new_text.len() - new_pos)] {
                    matches.push((old_pos, new_pos));
                }
            }
        }
        
        matches
    }
}

/// Text segment with hash
#[derive(Debug, Clone)]
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

