//! Natural Language Processing and LLM integration kernels.
//!
//! This module provides GPU-accelerated NLP algorithms:
//! - EmbeddingGeneration - Text to vector embeddings
//! - SemanticSimilarity - Document/entity similarity matching

use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Embedding Generation Kernel
// ============================================================================

/// Configuration for embedding generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding dimension.
    pub dimension: usize,
    /// Maximum sequence length.
    pub max_seq_length: usize,
    /// Whether to normalize embeddings.
    pub normalize: bool,
    /// Pooling strategy for sequence embeddings.
    pub pooling: PoolingStrategy,
    /// Vocabulary size for hash-based embeddings.
    pub vocab_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            dimension: 384,
            max_seq_length: 512,
            normalize: true,
            pooling: PoolingStrategy::Mean,
            vocab_size: 50000,
        }
    }
}

/// Pooling strategy for combining token embeddings.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Average of all token embeddings.
    Mean,
    /// Max pooling across tokens.
    Max,
    /// Use CLS token embedding (first token).
    CLS,
    /// Weighted average by attention.
    AttentionWeighted,
}

/// Result of embedding generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResult {
    /// Generated embeddings (one per input text).
    pub embeddings: Vec<Vec<f64>>,
    /// Token counts per input.
    pub token_counts: Vec<usize>,
    /// Embedding dimension.
    pub dimension: usize,
}

/// Embedding Generation kernel.
///
/// Generates dense vector embeddings from text using hash-based
/// token embeddings with configurable pooling strategies.
/// Suitable for semantic search, clustering, and similarity tasks.
#[derive(Debug, Clone)]
pub struct EmbeddingGeneration {
    metadata: KernelMetadata,
}

impl Default for EmbeddingGeneration {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingGeneration {
    /// Create a new Embedding Generation kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/embedding-generation", Domain::StatisticalML)
                .with_description("GPU-accelerated text embedding generation")
                .with_throughput(10_000)
                .with_latency_us(50.0),
        }
    }

    /// Generate embeddings for a batch of texts.
    pub fn compute(texts: &[&str], config: &EmbeddingConfig) -> EmbeddingResult {
        if texts.is_empty() {
            return EmbeddingResult {
                embeddings: Vec::new(),
                token_counts: Vec::new(),
                dimension: config.dimension,
            };
        }

        let mut embeddings = Vec::with_capacity(texts.len());
        let mut token_counts = Vec::with_capacity(texts.len());

        for text in texts {
            let tokens = Self::tokenize(text, config.max_seq_length);
            token_counts.push(tokens.len());

            let token_embeddings: Vec<Vec<f64>> = tokens
                .iter()
                .map(|token| Self::hash_embedding(token, config.dimension, config.vocab_size))
                .collect();

            let pooled = Self::pool_embeddings(&token_embeddings, config);

            let final_embedding = if config.normalize {
                Self::normalize_vector(&pooled)
            } else {
                pooled
            };

            embeddings.push(final_embedding);
        }

        EmbeddingResult {
            embeddings,
            token_counts,
            dimension: config.dimension,
        }
    }

    /// Simple whitespace tokenization with lowercasing.
    fn tokenize(text: &str, max_length: usize) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .take(max_length)
            .map(|s| s.chars().filter(|c| c.is_alphanumeric()).collect())
            .filter(|s: &String| !s.is_empty())
            .collect()
    }

    /// Generate embedding from token using hash-based approach.
    #[allow(clippy::needless_range_loop)]
    fn hash_embedding(token: &str, dimension: usize, vocab_size: usize) -> Vec<f64> {
        let mut embedding = vec![0.0; dimension];

        // Use multiple hash functions for better distribution
        let hash1 = Self::hash_token(token, 0) as usize;
        let hash2 = Self::hash_token(token, 1) as usize;
        let hash3 = Self::hash_token(token, 2) as usize;

        // Sparse embedding based on hashes
        for i in 0..dimension {
            let idx1 = (hash1 + i * 31) % vocab_size;
            let idx2 = (hash2 + i * 37) % vocab_size;
            let idx3 = (hash3 + i * 41) % vocab_size;

            // Combine hashes to create embedding value
            let sign1 = if (idx1 % 2) == 0 { 1.0 } else { -1.0 };
            let sign2 = if (idx2 % 2) == 0 { 1.0 } else { -1.0 };

            embedding[i] = sign1 * ((idx1 as f64 / vocab_size as f64) - 0.5)
                + sign2 * ((idx2 as f64 / vocab_size as f64) - 0.5) * 0.5
                + ((idx3 as f64 / vocab_size as f64) - 0.5) * 0.25;
        }

        embedding
    }

    /// Simple hash function for tokens.
    fn hash_token(token: &str, seed: u64) -> u64 {
        let mut hash: u64 = seed.wrapping_mul(0x517cc1b727220a95);
        for byte in token.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        hash
    }

    /// Pool token embeddings according to strategy.
    fn pool_embeddings(embeddings: &[Vec<f64>], config: &EmbeddingConfig) -> Vec<f64> {
        if embeddings.is_empty() {
            return vec![0.0; config.dimension];
        }

        match config.pooling {
            PoolingStrategy::Mean => {
                let mut result = vec![0.0; config.dimension];
                for emb in embeddings {
                    for (i, &v) in emb.iter().enumerate() {
                        result[i] += v;
                    }
                }
                let n = embeddings.len() as f64;
                result.iter_mut().for_each(|v| *v /= n);
                result
            }
            PoolingStrategy::Max => {
                let mut result = vec![f64::NEG_INFINITY; config.dimension];
                for emb in embeddings {
                    for (i, &v) in emb.iter().enumerate() {
                        result[i] = result[i].max(v);
                    }
                }
                result
            }
            PoolingStrategy::CLS => embeddings[0].clone(),
            PoolingStrategy::AttentionWeighted => {
                // Simple attention: weight by position (earlier = higher weight)
                let mut result = vec![0.0; config.dimension];
                let mut total_weight = 0.0;

                for (pos, emb) in embeddings.iter().enumerate() {
                    let weight = 1.0 / (1.0 + pos as f64 * 0.1);
                    total_weight += weight;
                    for (i, &v) in emb.iter().enumerate() {
                        result[i] += v * weight;
                    }
                }

                result.iter_mut().for_each(|v| *v /= total_weight);
                result
            }
        }
    }

    /// Normalize vector to unit length.
    fn normalize_vector(v: &[f64]) -> Vec<f64> {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            v.to_vec()
        } else {
            v.iter().map(|x| x / norm).collect()
        }
    }
}

impl GpuKernel for EmbeddingGeneration {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Semantic Similarity Kernel
// ============================================================================

/// Configuration for semantic similarity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityConfig {
    /// Similarity metric to use.
    pub metric: SimilarityMetric,
    /// Minimum similarity threshold for matches.
    pub threshold: f64,
    /// Maximum number of matches to return per query.
    pub top_k: usize,
    /// Whether to include self-matches.
    pub include_self: bool,
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            metric: SimilarityMetric::Cosine,
            threshold: 0.5,
            top_k: 10,
            include_self: false,
        }
    }
}

/// Similarity metric.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SimilarityMetric {
    /// Cosine similarity (dot product of normalized vectors).
    Cosine,
    /// Euclidean distance (converted to similarity).
    Euclidean,
    /// Dot product (unnormalized).
    DotProduct,
    /// Manhattan distance (converted to similarity).
    Manhattan,
}

/// A similarity match result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityMatch {
    /// Index of the query item.
    pub query_idx: usize,
    /// Index of the matched item.
    pub match_idx: usize,
    /// Similarity score.
    pub score: f64,
}

/// Result of semantic similarity computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    /// All matches above threshold.
    pub matches: Vec<SimilarityMatch>,
    /// Full similarity matrix (if computed).
    pub similarity_matrix: Option<Vec<Vec<f64>>>,
    /// Query embeddings used.
    pub query_count: usize,
    /// Corpus embeddings used.
    pub corpus_count: usize,
}

/// Semantic Similarity kernel.
///
/// Computes semantic similarity between text embeddings for
/// document matching, entity resolution, and semantic search.
#[derive(Debug, Clone)]
pub struct SemanticSimilarity {
    metadata: KernelMetadata,
}

impl Default for SemanticSimilarity {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticSimilarity {
    /// Create a new Semantic Similarity kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/semantic-similarity", Domain::StatisticalML)
                .with_description("Semantic similarity matching for documents and entities")
                .with_throughput(50_000)
                .with_latency_us(20.0),
        }
    }

    /// Compute similarity between query embeddings and corpus embeddings.
    pub fn compute(
        queries: &[Vec<f64>],
        corpus: &[Vec<f64>],
        config: &SimilarityConfig,
    ) -> SimilarityResult {
        if queries.is_empty() || corpus.is_empty() {
            return SimilarityResult {
                matches: Vec::new(),
                similarity_matrix: None,
                query_count: queries.len(),
                corpus_count: corpus.len(),
            };
        }

        let mut all_matches: Vec<SimilarityMatch> = Vec::new();
        let mut similarity_matrix: Vec<Vec<f64>> = Vec::with_capacity(queries.len());

        for (q_idx, query) in queries.iter().enumerate() {
            let mut row_scores: Vec<(usize, f64)> = Vec::with_capacity(corpus.len());

            for (c_idx, doc) in corpus.iter().enumerate() {
                if !config.include_self && q_idx == c_idx {
                    continue;
                }

                let score = Self::compute_similarity(query, doc, config.metric);
                row_scores.push((c_idx, score));
            }

            // Sort by score descending
            row_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top-k above threshold
            for (c_idx, score) in row_scores.iter().take(config.top_k) {
                if *score >= config.threshold {
                    all_matches.push(SimilarityMatch {
                        query_idx: q_idx,
                        match_idx: *c_idx,
                        score: *score,
                    });
                }
            }

            // Build full row for matrix
            let mut full_row = vec![0.0; corpus.len()];
            for (c_idx, score) in row_scores {
                full_row[c_idx] = score;
            }
            similarity_matrix.push(full_row);
        }

        SimilarityResult {
            matches: all_matches,
            similarity_matrix: Some(similarity_matrix),
            query_count: queries.len(),
            corpus_count: corpus.len(),
        }
    }

    /// Find most similar documents for each query.
    pub fn find_similar(
        queries: &[Vec<f64>],
        corpus: &[Vec<f64>],
        labels: Option<&[String]>,
        config: &SimilarityConfig,
    ) -> Vec<Vec<(usize, f64, Option<String>)>> {
        let result = Self::compute(queries, corpus, config);

        let mut grouped: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
        for m in result.matches {
            grouped
                .entry(m.query_idx)
                .or_default()
                .push((m.match_idx, m.score));
        }

        // Sort each group by score descending
        for matches in grouped.values_mut() {
            matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        queries
            .iter()
            .enumerate()
            .map(|(q_idx, _)| {
                grouped
                    .get(&q_idx)
                    .map(|matches| {
                        matches
                            .iter()
                            .map(|(idx, score)| {
                                let label = labels.and_then(|l| l.get(*idx).cloned());
                                (*idx, *score, label)
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect()
    }

    /// Compute pairwise similarity between two vectors.
    fn compute_similarity(a: &[f64], b: &[f64], metric: SimilarityMetric) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        match metric {
            SimilarityMetric::Cosine => {
                let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
                let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm_a < 1e-10 || norm_b < 1e-10 {
                    0.0
                } else {
                    dot / (norm_a * norm_b)
                }
            }
            SimilarityMetric::Euclidean => {
                let dist: f64 = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f64>()
                    .sqrt();
                1.0 / (1.0 + dist)
            }
            SimilarityMetric::DotProduct => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
            SimilarityMetric::Manhattan => {
                let dist: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
                1.0 / (1.0 + dist)
            }
        }
    }

    /// Deduplicate a corpus based on similarity threshold.
    pub fn deduplicate(embeddings: &[Vec<f64>], threshold: f64) -> Vec<usize> {
        if embeddings.is_empty() {
            return Vec::new();
        }

        let mut keep: Vec<usize> = vec![0]; // Always keep first

        for i in 1..embeddings.len() {
            let is_duplicate = keep.iter().any(|&j| {
                let sim = Self::compute_similarity(
                    &embeddings[i],
                    &embeddings[j],
                    SimilarityMetric::Cosine,
                );
                sim >= threshold
            });

            if !is_duplicate {
                keep.push(i);
            }
        }

        keep
    }
}

impl GpuKernel for SemanticSimilarity {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_generation_metadata() {
        let kernel = EmbeddingGeneration::new();
        assert_eq!(kernel.metadata().id, "ml/embedding-generation");
    }

    #[test]
    fn test_embedding_generation_basic() {
        let config = EmbeddingConfig::default();
        let texts = vec!["hello world", "machine learning"];

        let result = EmbeddingGeneration::compute(&texts, &config);

        assert_eq!(result.embeddings.len(), 2);
        assert_eq!(result.embeddings[0].len(), config.dimension);
        assert_eq!(result.token_counts, vec![2, 2]);
    }

    #[test]
    fn test_embedding_normalization() {
        let config = EmbeddingConfig {
            normalize: true,
            ..Default::default()
        };

        let result = EmbeddingGeneration::compute(&["test text"], &config);

        let norm: f64 = result.embeddings[0]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_embedding_empty() {
        let config = EmbeddingConfig::default();
        let result = EmbeddingGeneration::compute(&[], &config);
        assert!(result.embeddings.is_empty());
    }

    #[test]
    fn test_pooling_strategies() {
        let texts = vec!["a b c d e"];

        for pooling in [
            PoolingStrategy::Mean,
            PoolingStrategy::Max,
            PoolingStrategy::CLS,
            PoolingStrategy::AttentionWeighted,
        ] {
            let config = EmbeddingConfig {
                pooling,
                ..Default::default()
            };
            let result = EmbeddingGeneration::compute(&texts, &config);
            assert_eq!(result.embeddings.len(), 1);
            assert_eq!(result.embeddings[0].len(), config.dimension);
        }
    }

    #[test]
    fn test_semantic_similarity_metadata() {
        let kernel = SemanticSimilarity::new();
        assert_eq!(kernel.metadata().id, "ml/semantic-similarity");
    }

    #[test]
    fn test_semantic_similarity_basic() {
        let queries = vec![vec![1.0, 0.0, 0.0]];
        let corpus = vec![
            vec![1.0, 0.0, 0.0], // Same as query
            vec![0.0, 1.0, 0.0], // Orthogonal
            vec![0.7, 0.7, 0.0], // Partially similar
        ];

        let config = SimilarityConfig {
            threshold: 0.0,
            include_self: true,
            ..Default::default()
        };

        let result = SemanticSimilarity::compute(&queries, &corpus, &config);

        assert!(!result.matches.is_empty());
        // First match should be the identical vector
        assert_eq!(result.matches[0].match_idx, 0);
        assert!((result.matches[0].score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_similarity_metrics() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];

        for metric in [
            SimilarityMetric::Cosine,
            SimilarityMetric::Euclidean,
            SimilarityMetric::DotProduct,
            SimilarityMetric::Manhattan,
        ] {
            let sim = SemanticSimilarity::compute_similarity(&a, &b, metric);
            assert!(
                sim > 0.0,
                "Identical vectors should have positive similarity for {:?}",
                metric
            );
        }
    }

    #[test]
    fn test_deduplicate() {
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.99, 0.01], // Very similar to first
            vec![0.0, 1.0],   // Different
            vec![0.01, 0.99], // Very similar to third
        ];

        let kept = SemanticSimilarity::deduplicate(&embeddings, 0.95);

        assert_eq!(kept.len(), 2);
        assert!(kept.contains(&0));
        assert!(kept.contains(&2));
    }

    #[test]
    fn test_find_similar_with_labels() {
        let queries = vec![vec![1.0, 0.0]];
        let corpus = vec![vec![0.9, 0.1], vec![0.0, 1.0]];
        let labels = vec!["doc_a".to_string(), "doc_b".to_string()];

        let config = SimilarityConfig {
            threshold: 0.0,
            include_self: true, // Include all comparisons since query != corpus
            ..Default::default()
        };

        let results = SemanticSimilarity::find_similar(&queries, &corpus, Some(&labels), &config);

        assert_eq!(results.len(), 1);
        assert!(!results[0].is_empty());
        // The highest similarity should come first (doc_a has higher cosine sim to query)
        assert_eq!(results[0][0].2, Some("doc_a".to_string()));
    }
}
