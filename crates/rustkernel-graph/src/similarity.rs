//! Graph similarity kernels.
//!
//! This module provides similarity measures for graph nodes:
//! - Jaccard similarity (neighbor set overlap)
//! - Cosine similarity (normalized dot product)
//! - Adamic-Adar index (weighted common neighbors)

use crate::types::{CsrGraph, SimilarityScore};
use rustkernel_core::{
    domain::Domain,
    kernel::KernelMetadata,
    traits::GpuKernel,
};
use std::collections::HashSet;

// ============================================================================
// Jaccard Similarity Kernel
// ============================================================================

/// Jaccard similarity kernel.
///
/// Computes Jaccard similarity: |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
/// where N(x) is the neighbor set of node x.
#[derive(Debug, Clone)]
pub struct JaccardSimilarity {
    metadata: KernelMetadata,
}

impl JaccardSimilarity {
    /// Create a new Jaccard similarity kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/jaccard-similarity", Domain::GraphAnalytics)
                .with_description("Jaccard similarity (neighbor set overlap)")
                .with_throughput(100_000)
                .with_latency_us(10.0),
        }
    }

    /// Compute Jaccard similarity between two nodes.
    pub fn compute_pair(graph: &CsrGraph, node_a: u64, node_b: u64) -> f64 {
        let neighbors_a: HashSet<u64> = graph.neighbors(node_a).iter().copied().collect();
        let neighbors_b: HashSet<u64> = graph.neighbors(node_b).iter().copied().collect();

        let intersection = neighbors_a.intersection(&neighbors_b).count();
        let union = neighbors_a.union(&neighbors_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Compute Jaccard similarity for all pairs of nodes above a threshold.
    ///
    /// # Arguments
    /// * `graph` - Input graph
    /// * `min_similarity` - Only return pairs with similarity >= this threshold
    /// * `max_pairs` - Maximum number of pairs to return
    pub fn compute_all_pairs(
        graph: &CsrGraph,
        min_similarity: f64,
        max_pairs: usize,
    ) -> Vec<SimilarityScore> {
        let n = graph.num_nodes;
        let mut results = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let similarity = Self::compute_pair(graph, i as u64, j as u64);

                if similarity >= min_similarity {
                    results.push(SimilarityScore {
                        id_a: i as u64,
                        id_b: j as u64,
                        similarity,
                    });

                    if results.len() >= max_pairs {
                        return results;
                    }
                }
            }
        }

        // Sort by similarity descending
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Compute top-k most similar pairs using Jaccard similarity.
    pub fn top_k_pairs(graph: &CsrGraph, k: usize) -> Vec<SimilarityScore> {
        Self::compute_all_pairs(graph, 0.0, k)
    }
}

impl Default for JaccardSimilarity {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for JaccardSimilarity {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Cosine Similarity Kernel
// ============================================================================

/// Cosine similarity kernel.
///
/// Computes cosine similarity: |N(u) ∩ N(v)| / sqrt(|N(u)| * |N(v)|)
/// This is the normalized version of common neighbors.
#[derive(Debug, Clone)]
pub struct CosineSimilarity {
    metadata: KernelMetadata,
}

impl CosineSimilarity {
    /// Create a new cosine similarity kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/cosine-similarity", Domain::GraphAnalytics)
                .with_description("Cosine similarity (normalized dot product)")
                .with_throughput(100_000)
                .with_latency_us(10.0),
        }
    }

    /// Compute cosine similarity between two nodes.
    pub fn compute_pair(graph: &CsrGraph, node_a: u64, node_b: u64) -> f64 {
        let neighbors_a: HashSet<u64> = graph.neighbors(node_a).iter().copied().collect();
        let neighbors_b: HashSet<u64> = graph.neighbors(node_b).iter().copied().collect();

        let intersection = neighbors_a.intersection(&neighbors_b).count() as f64;
        let norm = (neighbors_a.len() as f64 * neighbors_b.len() as f64).sqrt();

        if norm == 0.0 {
            0.0
        } else {
            intersection / norm
        }
    }

    /// Compute cosine similarity for all pairs above a threshold.
    pub fn compute_all_pairs(
        graph: &CsrGraph,
        min_similarity: f64,
        max_pairs: usize,
    ) -> Vec<SimilarityScore> {
        let n = graph.num_nodes;
        let mut results = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let similarity = Self::compute_pair(graph, i as u64, j as u64);

                if similarity >= min_similarity {
                    results.push(SimilarityScore {
                        id_a: i as u64,
                        id_b: j as u64,
                        similarity,
                    });

                    if results.len() >= max_pairs {
                        return results;
                    }
                }
            }
        }

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

impl Default for CosineSimilarity {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for CosineSimilarity {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Adamic-Adar Index Kernel
// ============================================================================

/// Adamic-Adar index kernel.
///
/// Computes Adamic-Adar index: Σ 1/log(|N(z)|) for all z ∈ N(u) ∩ N(v)
/// This weights common neighbors inversely by their degree.
#[derive(Debug, Clone)]
pub struct AdamicAdarIndex {
    metadata: KernelMetadata,
}

impl AdamicAdarIndex {
    /// Create a new Adamic-Adar index kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/adamic-adar", Domain::GraphAnalytics)
                .with_description("Adamic-Adar index (weighted common neighbors)")
                .with_throughput(100_000)
                .with_latency_us(10.0),
        }
    }

    /// Compute Adamic-Adar index between two nodes.
    pub fn compute_pair(graph: &CsrGraph, node_a: u64, node_b: u64) -> f64 {
        let neighbors_a: HashSet<u64> = graph.neighbors(node_a).iter().copied().collect();
        let neighbors_b: HashSet<u64> = graph.neighbors(node_b).iter().copied().collect();

        let common_neighbors = neighbors_a.intersection(&neighbors_b);

        common_neighbors
            .map(|&z| {
                let degree = graph.out_degree(z) as f64;
                if degree > 1.0 {
                    1.0 / degree.ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Compute Adamic-Adar for all pairs, returning those above threshold.
    pub fn compute_all_pairs(
        graph: &CsrGraph,
        min_score: f64,
        max_pairs: usize,
    ) -> Vec<SimilarityScore> {
        let n = graph.num_nodes;
        let mut results = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let score = Self::compute_pair(graph, i as u64, j as u64);

                if score >= min_score {
                    results.push(SimilarityScore {
                        id_a: i as u64,
                        id_b: j as u64,
                        similarity: score,
                    });

                    if results.len() >= max_pairs {
                        return results;
                    }
                }
            }
        }

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Find top-k most similar pairs for link prediction.
    pub fn top_k_pairs(graph: &CsrGraph, k: usize) -> Vec<SimilarityScore> {
        Self::compute_all_pairs(graph, 0.0, k)
    }
}

impl Default for AdamicAdarIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for AdamicAdarIndex {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Common Neighbors Kernel
// ============================================================================

/// Common neighbors kernel.
///
/// Simply counts the number of common neighbors: |N(u) ∩ N(v)|
#[derive(Debug, Clone)]
pub struct CommonNeighbors {
    metadata: KernelMetadata,
}

impl CommonNeighbors {
    /// Create a new common neighbors kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/common-neighbors", Domain::GraphAnalytics)
                .with_description("Common neighbors count")
                .with_throughput(200_000)
                .with_latency_us(5.0),
        }
    }

    /// Count common neighbors between two nodes.
    pub fn compute_pair(graph: &CsrGraph, node_a: u64, node_b: u64) -> usize {
        let neighbors_a: HashSet<u64> = graph.neighbors(node_a).iter().copied().collect();
        let neighbors_b: HashSet<u64> = graph.neighbors(node_b).iter().copied().collect();

        neighbors_a.intersection(&neighbors_b).count()
    }

    /// Compute common neighbors for all pairs with count >= min_count.
    pub fn compute_all_pairs(
        graph: &CsrGraph,
        min_count: usize,
        max_pairs: usize,
    ) -> Vec<SimilarityScore> {
        let n = graph.num_nodes;
        let mut results = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let count = Self::compute_pair(graph, i as u64, j as u64);

                if count >= min_count {
                    results.push(SimilarityScore {
                        id_a: i as u64,
                        id_b: j as u64,
                        similarity: count as f64,
                    });

                    if results.len() >= max_pairs {
                        return results;
                    }
                }
            }
        }

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

impl Default for CommonNeighbors {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for CommonNeighbors {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> CsrGraph {
        // Graph with known overlapping neighbors:
        //     0 -- 1 -- 2
        //     |    |    |
        //     3 -- 4 -- 5
        CsrGraph::from_edges(6, &[
            (0, 1), (1, 0), (1, 2), (2, 1),
            (0, 3), (3, 0), (1, 4), (4, 1), (2, 5), (5, 2),
            (3, 4), (4, 3), (4, 5), (5, 4),
        ])
    }

    #[test]
    fn test_jaccard_similarity_metadata() {
        let kernel = JaccardSimilarity::new();
        assert_eq!(kernel.metadata().id, "graph/jaccard-similarity");
        assert_eq!(kernel.metadata().domain, Domain::GraphAnalytics);
    }

    #[test]
    fn test_jaccard_similarity_pair() {
        let graph = create_test_graph();

        // Nodes 0 and 2: neighbors of 0 = {1, 3}, neighbors of 2 = {1, 5}
        // Intersection = {1}, Union = {1, 3, 5}
        // Jaccard = 1/3
        let sim = JaccardSimilarity::compute_pair(&graph, 0, 2);
        assert!((sim - 1.0 / 3.0).abs() < 0.01, "Expected ~0.33, got {}", sim);

        // Self-comparison: identical neighbor sets should give 1.0 if same node
        // But for different nodes with identical neighbors, it's 1.0
    }

    #[test]
    fn test_cosine_similarity_pair() {
        let graph = create_test_graph();

        // Nodes 0 and 2: common = 1, |N(0)| = 2, |N(2)| = 2
        // Cosine = 1 / sqrt(2*2) = 0.5
        let sim = CosineSimilarity::compute_pair(&graph, 0, 2);
        assert!((sim - 0.5).abs() < 0.01, "Expected 0.5, got {}", sim);
    }

    #[test]
    fn test_adamic_adar_pair() {
        let graph = create_test_graph();

        // Nodes 0 and 2 share neighbor 1
        // Node 1 has degree = neighbors in CSR format
        let aa = AdamicAdarIndex::compute_pair(&graph, 0, 2);

        // Adamic-Adar should be positive for nodes with common neighbors
        assert!(aa > 0.0, "Expected positive Adamic-Adar score, got {}", aa);

        // Nodes with no common neighbors should have 0
        let aa_no_common = AdamicAdarIndex::compute_pair(&graph, 0, 5);
        assert_eq!(aa_no_common, 0.0);
    }

    #[test]
    fn test_common_neighbors_pair() {
        let graph = create_test_graph();

        // Nodes 0 and 2 share neighbor 1
        let count = CommonNeighbors::compute_pair(&graph, 0, 2);
        assert_eq!(count, 1);

        // Nodes 0 and 1 are connected, check their common neighbors
        let count = CommonNeighbors::compute_pair(&graph, 0, 1);
        // N(0) = {1, 3}, N(1) = {0, 2, 4}, intersection = {} (0 and 1 are neighbors, not common neighbors of each other)
        assert_eq!(count, 0);
    }

    #[test]
    fn test_jaccard_all_pairs() {
        let graph = create_test_graph();
        let pairs = JaccardSimilarity::compute_all_pairs(&graph, 0.0, 100);

        // Should have pairs for all node combinations
        assert!(!pairs.is_empty());

        // Should be sorted by similarity descending
        for i in 1..pairs.len() {
            assert!(pairs[i - 1].similarity >= pairs[i].similarity);
        }
    }
}
