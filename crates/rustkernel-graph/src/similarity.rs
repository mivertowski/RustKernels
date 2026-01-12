//! Graph similarity kernels.
//!
//! This module provides similarity measures for graph nodes:
//! - Jaccard similarity (neighbor set overlap)
//! - Cosine similarity (normalized dot product)
//! - Adamic-Adar index (weighted common neighbors)

use crate::types::{CsrGraph, SimilarityScore};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
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
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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
                if degree > 1.0 { 1.0 / degree.ln() } else { 0.0 }
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

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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
        CsrGraph::from_edges(
            6,
            &[
                (0, 1),
                (1, 0),
                (1, 2),
                (2, 1),
                (0, 3),
                (3, 0),
                (1, 4),
                (4, 1),
                (2, 5),
                (5, 2),
                (3, 4),
                (4, 3),
                (4, 5),
                (5, 4),
            ],
        )
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
        assert!(
            (sim - 1.0 / 3.0).abs() < 0.01,
            "Expected ~0.33, got {}",
            sim
        );

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

// ============================================================================
// Value Similarity Kernel
// ============================================================================

/// Value distribution for similarity calculation.
#[derive(Debug, Clone)]
pub struct ValueDistribution {
    /// Number of nodes.
    pub node_count: usize,
    /// Number of histogram bins.
    pub bin_count: usize,
    /// Probability distributions in row-major format [node_count × bin_count].
    /// Each row must sum to 1.0 (normalized histogram).
    pub distributions: Vec<f64>,
    /// Bin edges for interpreting distributions (bin_count + 1 values).
    pub bin_edges: Vec<f64>,
    /// Binning strategy used.
    pub strategy: BinningStrategy,
}

/// Binning strategy for value distributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinningStrategy {
    /// Equal-width bins (uniform spacing).
    EqualWidth,
    /// Logarithmic bins (geometric spacing).
    Logarithmic,
    /// Quantile bins (equal probability mass).
    Quantile,
}

impl ValueDistribution {
    /// Create a new value distribution.
    pub fn new(node_count: usize, bin_count: usize) -> Self {
        Self {
            node_count,
            bin_count,
            distributions: vec![0.0; node_count * bin_count],
            bin_edges: vec![0.0; bin_count + 1],
            strategy: BinningStrategy::EqualWidth,
        }
    }

    /// Create from raw values using equal-width binning.
    pub fn from_values(values: &[Vec<f64>], bin_count: usize) -> Self {
        let node_count = values.len();

        // Find global min/max
        let (min_val, max_val) = values
            .iter()
            .flat_map(|v| v.iter())
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &v| {
                (min.min(v), max.max(v))
            });

        let range = max_val - min_val;
        let bin_width = if range > 0.0 { range / bin_count as f64 } else { 1.0 };

        let mut dist = Self::new(node_count, bin_count);

        // Set bin edges
        for i in 0..=bin_count {
            dist.bin_edges[i] = min_val + i as f64 * bin_width;
        }
        dist.bin_edges[bin_count] = max_val + 0.001; // Ensure max value is included

        // Compute histograms
        for (node, node_values) in values.iter().enumerate() {
            if node_values.is_empty() {
                continue;
            }

            for &v in node_values {
                let bin = ((v - min_val) / bin_width).floor() as usize;
                let bin = bin.min(bin_count - 1);
                dist.distributions[node * bin_count + bin] += 1.0;
            }

            // Normalize
            let sum: f64 = dist.distributions[node * bin_count..(node + 1) * bin_count]
                .iter()
                .sum();
            if sum > 0.0 {
                for b in 0..bin_count {
                    dist.distributions[node * bin_count + b] /= sum;
                }
            }
        }

        dist
    }

    /// Get distribution for a node.
    pub fn get_distribution(&self, node: usize) -> &[f64] {
        let start = node * self.bin_count;
        &self.distributions[start..start + self.bin_count]
    }
}

/// Value similarity result.
#[derive(Debug, Clone)]
pub struct ValueSimilarityResult {
    /// Node A index.
    pub node_a: usize,
    /// Node B index.
    pub node_b: usize,
    /// Similarity score [0, 1].
    pub similarity: f64,
    /// Distance metric value.
    pub distance: f64,
}

/// Value similarity kernel.
///
/// Compares probability distributions using statistical distance metrics:
/// - Jensen-Shannon Divergence (JSD)
/// - Wasserstein Distance (Earth Mover's)
#[derive(Debug, Clone)]
pub struct ValueSimilarity {
    metadata: KernelMetadata,
}

impl Default for ValueSimilarity {
    fn default() -> Self {
        Self::new()
    }
}

impl ValueSimilarity {
    /// Create a new value similarity kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/value-similarity", Domain::GraphAnalytics)
                .with_description("Value distribution similarity via JSD/Wasserstein")
                .with_throughput(25_000)
                .with_latency_us(800.0),
        }
    }

    /// Compute Jensen-Shannon Divergence between two distributions.
    ///
    /// JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    /// where M = 0.5 * (P + Q)
    pub fn jensen_shannon_divergence(p: &[f64], q: &[f64]) -> f64 {
        assert_eq!(p.len(), q.len(), "Distributions must have same length");

        let epsilon = 1e-10;

        let mut kl_pm = 0.0;
        let mut kl_qm = 0.0;

        for i in 0..p.len() {
            let m = 0.5 * (p[i] + q[i]);

            if p[i] > epsilon && m > epsilon {
                kl_pm += p[i] * (p[i] / m).ln();
            }
            if q[i] > epsilon && m > epsilon {
                kl_qm += q[i] * (q[i] / m).ln();
            }
        }

        0.5 * kl_pm + 0.5 * kl_qm
    }

    /// Compute similarity from JSD (normalized to [0, 1]).
    pub fn jsd_similarity(p: &[f64], q: &[f64]) -> f64 {
        let jsd = Self::jensen_shannon_divergence(p, q);
        // JSD is in [0, ln(2)], normalize to [0, 1] similarity
        1.0 - (jsd / 2.0_f64.ln()).sqrt()
    }

    /// Compute Wasserstein-1 distance (Earth Mover's Distance) for 1D distributions.
    ///
    /// For 1D sorted bins: W1(P,Q) = Σ|CDF_P[i] - CDF_Q[i]|
    pub fn wasserstein_distance(p: &[f64], q: &[f64]) -> f64 {
        assert_eq!(p.len(), q.len(), "Distributions must have same length");

        let mut cdf_p = 0.0;
        let mut cdf_q = 0.0;
        let mut w1 = 0.0;

        for i in 0..p.len() {
            cdf_p += p[i];
            cdf_q += q[i];
            w1 += (cdf_p - cdf_q).abs();
        }

        w1
    }

    /// Compute similarity from Wasserstein distance.
    pub fn wasserstein_similarity(p: &[f64], q: &[f64]) -> f64 {
        let w1 = Self::wasserstein_distance(p, q);
        // W1 is in [0, n_bins], normalize to [0, 1] similarity
        1.0 / (1.0 + w1)
    }

    /// Compute pairwise similarities using JSD.
    pub fn compute_all_pairs_jsd(
        distributions: &ValueDistribution,
        min_similarity: f64,
        max_pairs: usize,
    ) -> Vec<ValueSimilarityResult> {
        let n = distributions.node_count;
        let mut results = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let p = distributions.get_distribution(i);
                let q = distributions.get_distribution(j);

                let jsd = Self::jensen_shannon_divergence(p, q);
                let similarity = 1.0 - (jsd / 2.0_f64.ln()).sqrt();

                if similarity >= min_similarity {
                    results.push(ValueSimilarityResult {
                        node_a: i,
                        node_b: j,
                        similarity,
                        distance: jsd,
                    });

                    if results.len() >= max_pairs {
                        return results;
                    }
                }
            }
        }

        // Sort by similarity descending
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Compute pairwise similarities using Wasserstein distance.
    pub fn compute_all_pairs_wasserstein(
        distributions: &ValueDistribution,
        min_similarity: f64,
        max_pairs: usize,
    ) -> Vec<ValueSimilarityResult> {
        let n = distributions.node_count;
        let mut results = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let p = distributions.get_distribution(i);
                let q = distributions.get_distribution(j);

                let w1 = Self::wasserstein_distance(p, q);
                let similarity = 1.0 / (1.0 + w1);

                if similarity >= min_similarity {
                    results.push(ValueSimilarityResult {
                        node_a: i,
                        node_b: j,
                        similarity,
                        distance: w1,
                    });

                    if results.len() >= max_pairs {
                        return results;
                    }
                }
            }
        }

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Find nodes with similar value distributions.
    pub fn find_similar_nodes(
        distributions: &ValueDistribution,
        target_node: usize,
        min_similarity: f64,
        top_k: usize,
    ) -> Vec<ValueSimilarityResult> {
        let n = distributions.node_count;
        let p = distributions.get_distribution(target_node);
        let mut results = Vec::new();

        for i in 0..n {
            if i == target_node {
                continue;
            }

            let q = distributions.get_distribution(i);
            let similarity = Self::jsd_similarity(p, q);

            if similarity >= min_similarity {
                results.push(ValueSimilarityResult {
                    node_a: target_node,
                    node_b: i,
                    similarity,
                    distance: Self::jensen_shannon_divergence(p, q),
                });
            }
        }

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.into_iter().take(top_k).collect()
    }
}

impl GpuKernel for ValueSimilarity {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod value_similarity_tests {
    use super::*;

    #[test]
    fn test_value_similarity_metadata() {
        let kernel = ValueSimilarity::new();
        assert_eq!(kernel.metadata().id, "graph/value-similarity");
        assert_eq!(kernel.metadata().domain, Domain::GraphAnalytics);
    }

    #[test]
    fn test_jsd_identical_distributions() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.25, 0.25, 0.25, 0.25];

        let jsd = ValueSimilarity::jensen_shannon_divergence(&p, &q);
        assert!(jsd.abs() < 0.001, "JSD of identical distributions should be 0");
    }

    #[test]
    fn test_jsd_different_distributions() {
        let p = vec![1.0, 0.0, 0.0, 0.0];
        let q = vec![0.0, 0.0, 0.0, 1.0];

        let jsd = ValueSimilarity::jensen_shannon_divergence(&p, &q);
        // JSD should be close to ln(2) for maximally different distributions
        assert!(jsd > 0.6, "JSD should be high for very different distributions");
    }

    #[test]
    fn test_jsd_similarity() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.25, 0.25, 0.25, 0.25];

        let sim = ValueSimilarity::jsd_similarity(&p, &q);
        assert!((sim - 1.0).abs() < 0.01, "Identical distributions should have similarity 1.0");
    }

    #[test]
    fn test_wasserstein_identical() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.25, 0.25, 0.25, 0.25];

        let w1 = ValueSimilarity::wasserstein_distance(&p, &q);
        assert!(w1.abs() < 0.001, "Wasserstein of identical distributions should be 0");
    }

    #[test]
    fn test_wasserstein_shifted() {
        let p = vec![1.0, 0.0, 0.0, 0.0];
        let q = vec![0.0, 1.0, 0.0, 0.0];

        let w1 = ValueSimilarity::wasserstein_distance(&p, &q);
        // Should be 1.0 (one bin of "earth" moved one position)
        assert!((w1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_value_distribution_from_values() {
        let values = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
        ];

        let dist = ValueDistribution::from_values(&values, 4);

        assert_eq!(dist.node_count, 2);
        assert_eq!(dist.bin_count, 4);

        // Check normalization
        let sum0: f64 = dist.get_distribution(0).iter().sum();
        let sum1: f64 = dist.get_distribution(1).iter().sum();
        assert!((sum0 - 1.0).abs() < 0.01);
        assert!((sum1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_find_similar_nodes() {
        let values = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0], // Same as node 0
            vec![10.0, 11.0, 12.0], // Different
        ];

        let dist = ValueDistribution::from_values(&values, 5);
        let similar = ValueSimilarity::find_similar_nodes(&dist, 0, 0.5, 5);

        // Node 1 should be most similar to node 0
        assert!(!similar.is_empty());
        assert_eq!(similar[0].node_b, 1);
    }
}
