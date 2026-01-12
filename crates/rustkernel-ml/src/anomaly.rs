//! Anomaly detection kernels.
//!
//! This module provides anomaly detection algorithms:
//! - Isolation Forest (ensemble of isolation trees)
//! - Local Outlier Factor (k-NN density-based)

use crate::types::{AnomalyResult, DataMatrix, DistanceMetric};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use rand::prelude::*;

// ============================================================================
// Isolation Forest Kernel
// ============================================================================

/// Isolation Forest kernel.
///
/// Anomaly detection using ensemble of isolation trees.
/// Anomalies are isolated quickly (short path lengths) while normal
/// points require more splits.
#[derive(Debug, Clone)]
pub struct IsolationForest {
    metadata: KernelMetadata,
}

impl Default for IsolationForest {
    fn default() -> Self {
        Self::new()
    }
}

impl IsolationForest {
    /// Create a new Isolation Forest kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/isolation-forest", Domain::StatisticalML)
                .with_description("Isolation Forest ensemble anomaly detection")
                .with_throughput(10_000)
                .with_latency_us(100.0),
        }
    }

    /// Train and score using Isolation Forest.
    ///
    /// # Arguments
    /// * `data` - Input data matrix
    /// * `n_trees` - Number of isolation trees
    /// * `sample_size` - Subsample size for each tree
    /// * `contamination` - Expected proportion of anomalies (for threshold)
    pub fn compute(
        data: &DataMatrix,
        n_trees: usize,
        sample_size: usize,
        contamination: f64,
    ) -> AnomalyResult {
        let n = data.n_samples;

        if n == 0 {
            return AnomalyResult {
                scores: Vec::new(),
                labels: Vec::new(),
                threshold: 0.5,
            };
        }

        let actual_sample_size = sample_size.min(n);
        let max_depth = (actual_sample_size as f64).log2().ceil() as usize;

        // Build isolation trees
        let trees: Vec<IsolationTree> = (0..n_trees)
            .map(|_| IsolationTree::build(data, actual_sample_size, max_depth))
            .collect();

        // Compute anomaly scores for each point
        let scores: Vec<f64> = (0..n)
            .map(|i| {
                let point = data.row(i);
                let avg_path_length: f64 = trees
                    .iter()
                    .map(|tree| tree.path_length(point) as f64)
                    .sum::<f64>()
                    / n_trees as f64;

                // Anomaly score: s(x, n) = 2^(-E[h(x)] / c(n))
                // where c(n) is the average path length in a random tree
                let c_n = Self::average_path_length(actual_sample_size);
                (2.0_f64).powf(-avg_path_length / c_n)
            })
            .collect();

        // Determine threshold based on contamination
        let mut sorted_scores = scores.clone();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx = ((n as f64 * contamination) as usize).max(1).min(n - 1);
        let threshold = sorted_scores[threshold_idx];

        // Label anomalies
        let labels: Vec<i32> = scores
            .iter()
            .map(|&s| if s >= threshold { -1 } else { 1 })
            .collect();

        AnomalyResult {
            scores,
            labels,
            threshold,
        }
    }

    /// Average path length in a random binary search tree.
    fn average_path_length(n: usize) -> f64 {
        if n <= 1 {
            return 0.0;
        }
        if n == 2 {
            return 1.0;
        }
        // c(n) = 2 * H(n-1) - 2(n-1)/n
        // H(i) ≈ ln(i) + 0.5772156649 (Euler's constant)
        let euler = 0.5772156649;
        2.0 * ((n as f64 - 1.0).ln() + euler) - 2.0 * (n as f64 - 1.0) / n as f64
    }
}

impl GpuKernel for IsolationForest {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// A single isolation tree node.
#[derive(Debug, Clone)]
enum IsolationTreeNode {
    Internal {
        feature: usize,
        threshold: f64,
        left: Box<IsolationTreeNode>,
        right: Box<IsolationTreeNode>,
    },
    Leaf {
        size: usize,
    },
}

/// An isolation tree.
#[derive(Debug, Clone)]
struct IsolationTree {
    root: IsolationTreeNode,
    max_depth: usize,
}

impl IsolationTree {
    /// Build an isolation tree from a random subsample.
    fn build(data: &DataMatrix, sample_size: usize, max_depth: usize) -> Self {
        let mut rng = rand::rng();
        let n = data.n_samples;
        let d = data.n_features;

        // Random subsample
        let indices: Vec<usize> = (0..n).collect();
        let sample_indices: Vec<usize> = indices
            .choose_multiple(&mut rng, sample_size.min(n))
            .copied()
            .collect();

        // Extract subsample data
        let sample_data: Vec<Vec<f64>> = sample_indices
            .iter()
            .map(|&i| data.row(i).to_vec())
            .collect();

        let root = Self::build_node(&sample_data, d, 0, max_depth, &mut rng);

        IsolationTree { root, max_depth }
    }

    fn build_node(
        data: &[Vec<f64>],
        n_features: usize,
        depth: usize,
        max_depth: usize,
        rng: &mut impl Rng,
    ) -> IsolationTreeNode {
        let n = data.len();

        // Base case: leaf node
        if depth >= max_depth || n <= 1 {
            return IsolationTreeNode::Leaf { size: n };
        }

        // Random feature
        let feature = rng.random_range(0..n_features);

        // Find min/max for this feature
        let values: Vec<f64> = data.iter().map(|row| row[feature]).collect();
        let min_val = values.iter().copied().fold(f64::MAX, f64::min);
        let max_val = values.iter().copied().fold(f64::MIN, f64::max);

        if (max_val - min_val).abs() < 1e-10 {
            return IsolationTreeNode::Leaf { size: n };
        }

        // Random threshold
        let threshold = min_val + rng.random::<f64>() * (max_val - min_val);

        // Split data
        let (left_data, right_data): (Vec<Vec<f64>>, Vec<Vec<f64>>) = data
            .iter()
            .cloned()
            .partition(|row| row[feature] < threshold);

        if left_data.is_empty() || right_data.is_empty() {
            return IsolationTreeNode::Leaf { size: n };
        }

        IsolationTreeNode::Internal {
            feature,
            threshold,
            left: Box::new(Self::build_node(&left_data, n_features, depth + 1, max_depth, rng)),
            right: Box::new(Self::build_node(&right_data, n_features, depth + 1, max_depth, rng)),
        }
    }

    /// Compute path length to isolate a point.
    fn path_length(&self, point: &[f64]) -> usize {
        Self::path_length_node(&self.root, point, 0)
    }

    fn path_length_node(node: &IsolationTreeNode, point: &[f64], depth: usize) -> usize {
        match node {
            IsolationTreeNode::Leaf { size } => {
                // Add expected path length for remaining points
                depth + IsolationForest::average_path_length(*size) as usize
            }
            IsolationTreeNode::Internal {
                feature,
                threshold,
                left,
                right,
            } => {
                if point[*feature] < *threshold {
                    Self::path_length_node(left, point, depth + 1)
                } else {
                    Self::path_length_node(right, point, depth + 1)
                }
            }
        }
    }
}

// ============================================================================
// Local Outlier Factor (LOF) Kernel
// ============================================================================

/// Local Outlier Factor (LOF) kernel.
///
/// Density-based anomaly detection using k-nearest neighbors.
/// Points with substantially lower density than their neighbors are anomalies.
#[derive(Debug, Clone)]
pub struct LocalOutlierFactor {
    metadata: KernelMetadata,
}

impl Default for LocalOutlierFactor {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalOutlierFactor {
    /// Create a new LOF kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/local-outlier-factor", Domain::StatisticalML)
                .with_description("Local Outlier Factor (k-NN density estimation)")
                .with_throughput(5_000)
                .with_latency_us(200.0),
        }
    }

    /// Compute LOF scores.
    ///
    /// # Arguments
    /// * `data` - Input data matrix
    /// * `k` - Number of neighbors
    /// * `contamination` - Expected proportion of anomalies (for threshold)
    /// * `metric` - Distance metric
    pub fn compute(
        data: &DataMatrix,
        k: usize,
        contamination: f64,
        metric: DistanceMetric,
    ) -> AnomalyResult {
        let n = data.n_samples;

        if n == 0 || k == 0 {
            return AnomalyResult {
                scores: Vec::new(),
                labels: Vec::new(),
                threshold: 1.0,
            };
        }

        let k = k.min(n - 1);

        // Compute k-distances and k-nearest neighbors for all points
        let (k_distances, k_neighbors) = Self::compute_knn(data, k, metric);

        // Compute reachability distances
        let reach_dists: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                k_neighbors[i]
                    .iter()
                    .map(|&j| {
                        let dist = metric.compute(data.row(i), data.row(j));
                        dist.max(k_distances[j])
                    })
                    .collect()
            })
            .collect();

        // Compute local reachability density (LRD)
        let lrd: Vec<f64> = (0..n)
            .map(|i| {
                let sum_reach: f64 = reach_dists[i].iter().sum();
                if sum_reach == 0.0 {
                    f64::MAX
                } else {
                    k as f64 / sum_reach
                }
            })
            .collect();

        // Compute LOF scores
        let scores: Vec<f64> = (0..n)
            .map(|i| {
                if lrd[i] == f64::MAX {
                    return 1.0;
                }
                let avg_neighbor_lrd: f64 =
                    k_neighbors[i].iter().map(|&j| lrd[j]).sum::<f64>() / k as f64;
                avg_neighbor_lrd / lrd[i]
            })
            .collect();

        // Determine threshold
        let mut sorted_scores = scores.clone();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx = ((n as f64 * contamination) as usize).max(1).min(n - 1);
        let threshold = sorted_scores[threshold_idx];

        // Label anomalies (LOF > threshold means anomaly)
        let labels: Vec<i32> = scores
            .iter()
            .map(|&s| if s > threshold { -1 } else { 1 })
            .collect();

        AnomalyResult {
            scores,
            labels,
            threshold,
        }
    }

    /// Compute k-nearest neighbors and k-distances for all points.
    fn compute_knn(
        data: &DataMatrix,
        k: usize,
        metric: DistanceMetric,
    ) -> (Vec<f64>, Vec<Vec<usize>>) {
        let n = data.n_samples;

        let mut k_distances = vec![0.0f64; n];
        let mut k_neighbors = vec![Vec::new(); n];

        for i in 0..n {
            // Compute distances to all other points
            let mut distances: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, metric.compute(data.row(i), data.row(j))))
                .collect();

            // Sort by distance
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take k nearest
            let k_nearest: Vec<(usize, f64)> = distances.into_iter().take(k).collect();

            k_distances[i] = k_nearest.last().map(|(_, d)| *d).unwrap_or(0.0);
            k_neighbors[i] = k_nearest.iter().map(|(j, _)| *j).collect();
        }

        (k_distances, k_neighbors)
    }
}

impl GpuKernel for LocalOutlierFactor {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_anomaly_data() -> DataMatrix {
        // Normal cluster around origin
        // One clear outlier at (100, 100)
        DataMatrix::from_rows(&[
            &[0.0, 0.0],
            &[0.1, 0.1],
            &[0.2, 0.0],
            &[0.0, 0.2],
            &[-0.1, 0.1],
            &[0.1, -0.1],
            &[100.0, 100.0], // Anomaly
        ])
    }

    #[test]
    fn test_isolation_forest_metadata() {
        let kernel = IsolationForest::new();
        assert_eq!(kernel.metadata().id, "ml/isolation-forest");
        assert_eq!(kernel.metadata().domain, Domain::StatisticalML);
    }

    #[test]
    fn test_isolation_forest_detects_anomaly() {
        let data = create_anomaly_data();
        let result = IsolationForest::compute(&data, 100, 256, 0.15);

        // The outlier (index 6) should have high anomaly score
        assert!(result.scores[6] > result.scores[0]);
        assert!(result.scores[6] > result.scores[1]);
        assert!(result.scores[6] > result.scores[2]);

        // Outlier should be labeled as anomaly (-1)
        assert_eq!(result.labels[6], -1);
    }

    #[test]
    fn test_lof_metadata() {
        let kernel = LocalOutlierFactor::new();
        assert_eq!(kernel.metadata().id, "ml/local-outlier-factor");
        assert_eq!(kernel.metadata().domain, Domain::StatisticalML);
    }

    #[test]
    fn test_lof_detects_anomaly() {
        let data = create_anomaly_data();
        let result = LocalOutlierFactor::compute(&data, 3, 0.15, DistanceMetric::Euclidean);

        // The outlier (index 6) should have high LOF score
        // LOF > 1 indicates an outlier
        assert!(result.scores[6] > 1.0);

        // Normal points should have LOF close to 1
        for i in 0..6 {
            assert!(result.scores[i] < result.scores[6]);
        }
    }

    #[test]
    fn test_isolation_forest_empty() {
        let data = DataMatrix::zeros(0, 2);
        let result = IsolationForest::compute(&data, 10, 256, 0.1);
        assert!(result.scores.is_empty());
        assert!(result.labels.is_empty());
    }

    #[test]
    fn test_lof_empty() {
        let data = DataMatrix::zeros(0, 2);
        let result = LocalOutlierFactor::compute(&data, 3, 0.1, DistanceMetric::Euclidean);
        assert!(result.scores.is_empty());
        assert!(result.labels.is_empty());
    }
}
