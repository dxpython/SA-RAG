/// Long-term Memory Module: Implements memory management based on importance and time decay
/// 
/// Enhanced with Cognitive Memory System (Memory-RAG 2.0)
/// 
/// Algorithm:
/// 1. Based on Ebbinghaus forgetting curve: memory strength decays exponentially over time
/// 2. Importance weighting: important memories decay slower
/// 3. Access frequency: frequently accessed memories decay slower (spaced repetition effect)
/// 4. Relevance retrieval: combines keyword matching and vector similarity
/// 5. Memory consolidation: important memories move from short-term to long-term to semantic
/// 
/// Memory scoring formula:
/// score = importance * retention * relevance
/// retention = exp(-decay_rate * time_passed / importance_factor)
/// 
/// where importance_factor makes important memories decay slower

use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Memory type: short-term, long-term, or semantic (consolidated)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryType {
    ShortTerm,   // Recent memories, fast decay
    LongTerm,    // Important memories, slow decay
    Semantic,    // Consolidated knowledge, minimal decay
}

/// Memory source: where the memory comes from
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemorySource {
    /// User dialogue/interaction
    UserDialogue,
    /// System-generated summary
    Summary,
    /// User preference or setting
    Preference,
    /// External knowledge
    External,
    /// Query-answer pair
    QAPair,
}

/// Memory node with enhanced metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    pub memory_id: u64,
    pub content: String,
    /// Importance score (0.0 - 1.0)
    pub importance: f32,
    /// Creation time (Unix timestamp)
    pub created_at: u64,
    /// Last access time (Unix timestamp)
    pub last_accessed: u64,
    /// Access count
    pub access_count: u64,
    /// Associated keywords (for fast retrieval)
    pub keywords: Vec<String>,
    /// Vector representation of memory (optional, for semantic retrieval)
    pub embedding: Option<Vec<f32>>,
    /// Memory source
    pub source: MemorySource,
    /// Recency score (calculated based on time since creation/access)
    pub recency: u64,
    /// Memory type
    pub memory_type: MemoryType,
    /// Emotional weight (model-scored, 0.0-1.0)
    pub emotional_weight: f32,
    /// Frequency score (based on access count)
    pub frequency: f32,
}

/// Long-term memory store with cognitive memory system
pub struct MemoryStore {
    /// All memory nodes
    memories: Vec<MemoryNode>,
    /// Decay rate (per hour)
    decay_rate: f32,
    /// Importance factor (affects decay speed)
    importance_factor: f32,
    /// Maximum number of memories (low-scoring memories are deleted when exceeded)
    max_memories: usize,
    /// Memory ID counter
    memory_counter: u64,
    /// Keyword index: keyword -> memory_ids
    keyword_index: HashMap<String, Vec<u64>>,
    /// Consolidation threshold: memories above this importance become semantic
    consolidation_threshold: f32,
}

impl MemoryStore {
    /// Create a new memory store
    /// 
    /// # Arguments
    /// - `decay_rate`: Decay rate (per hour), recommended value 0.1-0.5
    /// - `importance_factor`: Importance factor, larger values make important memories decay slower
    /// - `max_memories`: Maximum number of memories
    pub fn new(decay_rate: f32) -> Self {
        Self {
            memories: Vec::new(),
            decay_rate,
            importance_factor: 2.0,  // Important memories decay at half speed
            max_memories: 10000,
            memory_counter: 1,
            keyword_index: HashMap::new(),
            consolidation_threshold: 0.8, // Memories with importance > 0.8 become semantic
        }
    }

    /// Add memory with cognitive classification
    /// 
    /// # Arguments
    /// - `content`: Memory content
    /// - `importance`: Importance score (0.0-1.0)
    /// - `keywords`: Associated keywords (optional)
    /// - `embedding`: Vector representation (optional)
    /// - `emotional_weight`: Emotional weight (optional, model-scored)
    pub fn add_memory(
        &mut self,
        content: String,
        importance: f32,
        keywords: Option<Vec<String>>,
        embedding: Option<Vec<f32>>,
        emotional_weight: Option<f32>,
    ) -> u64 {
        let memory_id = self.memory_counter;
        self.memory_counter += 1;

        let now = Self::current_timestamp();
        let keywords = keywords.unwrap_or_default();
        let emotional_weight = emotional_weight.unwrap_or(0.5);

        // Classify memory type based on importance
        let memory_type = if importance >= self.consolidation_threshold {
            MemoryType::Semantic
        } else if importance >= 0.5 {
            MemoryType::LongTerm
        } else {
            MemoryType::ShortTerm
        };

        let memory = MemoryNode {
            memory_id,
            content: content.clone(),
            importance,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            keywords: keywords.clone(),
            embedding,
            source: MemorySource::UserDialogue,
            recency: now,
            memory_type,
            emotional_weight,
            frequency: 1.0,
        };

        // Index by keywords
        for keyword in keywords {
            self.keyword_index
                .entry(keyword.to_lowercase())
                .or_insert_with(Vec::new)
                .push(memory_id);
        }

        self.memories.push(memory);

        // Auto-forget low-scoring memories if exceeded max
        if self.memories.len() > self.max_memories {
            self.auto_forget();
        }

        memory_id
    }

    /// Retrieve relevant memories with cognitive scoring
    pub fn retrieve_relevant(
        &mut self,
        query: &str,
        top_k: usize,
        query_embedding: Option<&[f32]>,
    ) -> Vec<&MemoryNode> {
        let now = Self::current_timestamp();
        let query_keywords = Self::extract_keywords(query);

        // Update recency and access counts
        self.update_recency();

        // Calculate score for each memory
        let mut scored_memories: Vec<(u64, f32)> = self.memories.iter()
            .map(|memory| {
                // 1. Calculate retention score (Ebbinghaus forgetting curve)
                let hours_passed = (now - memory.last_accessed) as f32 / 3600.0;
                let importance_adjusted_decay = self.decay_rate / (1.0 + memory.importance * self.importance_factor);
                
                // Different decay rates for different memory types
                let decay_rate = match memory.memory_type {
                    MemoryType::ShortTerm => importance_adjusted_decay * 2.0, // Fast decay
                    MemoryType::LongTerm => importance_adjusted_decay,        // Normal decay
                    MemoryType::Semantic => importance_adjusted_decay * 0.1,  // Very slow decay
                };
                
                let retention = (-decay_rate * hours_passed).exp();
                
                // 2. Access frequency boost (spaced repetition effect)
                let access_boost = 1.0 + (memory.access_count as f32 * 0.1).min(1.0);
                let frequency_score = memory.frequency;
                
                // 3. Memory type boost
                let type_boost = match memory.memory_type {
                    MemoryType::ShortTerm => 0.5,
                    MemoryType::LongTerm => 1.0,
                    MemoryType::Semantic => 1.5, // Semantic memories are prioritized
                };
                
                let retention_score = retention * access_boost * frequency_score * type_boost * memory.importance;

                // 4. Calculate relevance score
                let keyword_score = Self::calculate_keyword_relevance(&memory.keywords, &query_keywords);
                let semantic_score = if let (Some(mem_emb), Some(query_emb)) = (&memory.embedding, query_embedding) {
                    Self::cosine_similarity(mem_emb, query_emb)
                } else {
                    0.0
                };
                
                let relevance = (keyword_score * 0.4 + semantic_score * 0.6).min(1.0);

                // 5. Emotional weight boost
                let emotional_boost = 1.0 + memory.emotional_weight * 0.2;

                // 6. Final score
                let final_score = retention_score * relevance * emotional_boost;

                (memory.memory_id, final_score)
            })
            .collect();

        // Sort by score
        scored_memories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Update access counts for retrieved memories
        for (memory_id, _) in scored_memories.iter().take(top_k) {
            if let Some(memory) = self.memories.iter_mut().find(|m| m.memory_id == *memory_id) {
                memory.last_accessed = now;
                memory.access_count += 1;
                memory.frequency = (memory.access_count as f32 / 100.0).min(1.0);
            }
        }

        // Return top-k memories
        scored_memories.iter()
            .take(top_k)
            .filter_map(|(id, _)| self.memories.iter().find(|m| m.memory_id == *id))
            .collect()
    }

    /// Consolidate memories: move important memories to semantic type
    pub fn consolidate_memories(&mut self) {
        for memory in &mut self.memories {
            if memory.importance >= self.consolidation_threshold && 
               memory.memory_type != MemoryType::Semantic &&
               memory.access_count > 5 {
                memory.memory_type = MemoryType::Semantic;
            }
        }
    }

    /// Auto-forget low-scoring memories
    fn auto_forget(&mut self) {
        let now = Self::current_timestamp();
        let forget_threshold = 0.1;

        // Calculate scores for all memories
        let mut scored: Vec<(usize, f32)> = self.memories.iter()
            .enumerate()
            .map(|(i, memory)| {
                let hours_passed = (now - memory.last_accessed) as f32 / 3600.0;
                let decay_rate = match memory.memory_type {
                    MemoryType::ShortTerm => self.decay_rate * 2.0,
                    MemoryType::LongTerm => self.decay_rate,
                    MemoryType::Semantic => self.decay_rate * 0.1,
                };
                let retention = (-decay_rate * hours_passed).exp();
                let score = retention * memory.importance;
                (i, score)
            })
            .collect();

        // Sort by score (lowest first)
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Remove lowest-scoring memories (but never remove semantic memories)
        let to_remove: Vec<usize> = scored.iter()
            .filter(|(i, score)| {
                *score < forget_threshold && 
                self.memories[*i].memory_type != MemoryType::Semantic
            })
            .map(|(i, _)| *i)
            .take(self.memories.len() - self.max_memories)
            .collect();

        // Remove in reverse order to maintain indices
        for &idx in to_remove.iter().rev() {
            let memory_id = self.memories[idx].memory_id;
            self.memories.remove(idx);
            
            // Remove from keyword index
            self.keyword_index.values_mut().for_each(|ids| {
                ids.retain(|&id| id != memory_id);
            });
        }
    }

    /// Update recency scores
    pub fn update_recency(&mut self) {
        let now = Self::current_timestamp();
        for memory in &mut self.memories {
            memory.recency = now;
        }
    }

    /// Get memory by ID
    pub fn get(&self, memory_id: u64) -> Option<&MemoryNode> {
        self.memories.iter().find(|m| m.memory_id == memory_id)
    }

    /// Get all memories of a specific type
    pub fn get_by_type(&self, memory_type: MemoryType) -> Vec<&MemoryNode> {
        self.memories.iter()
            .filter(|m| m.memory_type == memory_type)
            .collect()
    }

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn extract_keywords(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.chars().filter(|c| c.is_alphanumeric()).collect())
            .filter(|s: &String| s.len() > 2)
            .collect()
    }

    fn calculate_keyword_relevance(memory_keywords: &[String], query_keywords: &[String]) -> f32 {
        if query_keywords.is_empty() {
            return 0.0;
        }

        let memory_set: std::collections::HashSet<&String> = memory_keywords.iter().collect();
        let query_set: std::collections::HashSet<&String> = query_keywords.iter().collect();

        let intersection = memory_set.intersection(&query_set).count();
        intersection as f32 / query_keywords.len() as f32
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    pub fn len(&self) -> usize {
        self.memories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }
}
