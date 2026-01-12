//! Clustering kernels.
//!
//! This module provides machine learning clustering algorithms:
//! - K-Means (Lloyd's algorithm with K-Means++ initialization)
//! - DBSCAN (density-based clustering)
//! - Hierarchical clustering (agglomerative)

use crate::types::{ClusteringResult, DataMatrix, DistanceMetric};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use rand::prelude::*;

// ============================================================================
// K-Means Clustering Kernel
// ============================================================================

/// K-Means clustering kernel.
///
/// Implements Lloyd's algorithm with K-Means++ initialization.
#[derive(Debug, Clone)]
pub struct KMeans {
    metadata: KernelMetadata,
}

impl Default for KMeans {
    fn default() -> Self {
        Self::new()
    }
}

impl KMeans {
    /// Create a new K-Means kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/kmeans-cluster", Domain::StatisticalML)
                .with_description("K-Means clustering with K-Means++ initialization")
                .with_throughput(20_000)
                .with_latency_us(50.0),
        }
    }

    /// Run K-Means clustering.
    ///
    /// # Arguments
    /// * `data` - Input data matrix (n_samples x n_features)
    /// * `k` - Number of clusters
    /// * `max_iterations` - Maximum number of iterations
    /// * `tolerance` - Convergence threshold for centroid movement
    pub fn compute(
        data: &DataMatrix,
        k: usize,
        max_iterations: u32,
        tolerance: f64,
    ) -> ClusteringResult {
        let n = data.n_samples;
        let d = data.n_features;

        if n == 0 || k == 0 || k > n {
            return ClusteringResult {
                labels: Vec::new(),
                n_clusters: 0,
                centroids: Vec::new(),
                inertia: 0.0,
                iterations: 0,
                converged: true,
            };
        }

        // K-Means++ initialization
        let mut centroids = Self::kmeans_plus_plus_init(data, k);
        let mut labels = vec![0usize; n];
        let mut converged = false;
        let mut iterations = 0u32;

        for iter in 0..max_iterations {
            iterations = iter + 1;

            // Assignment step: assign each point to nearest centroid
            for i in 0..n {
                let point = data.row(i);
                let mut min_dist = f64::MAX;
                let mut min_cluster = 0;

                for (c, centroid) in centroids.chunks(d).enumerate() {
                    let dist = Self::euclidean_distance(point, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        min_cluster = c;
                    }
                }
                labels[i] = min_cluster;
            }

            // Update step: recalculate centroids
            let mut new_centroids = vec![0.0f64; k * d];
            let mut counts = vec![0usize; k];

            for i in 0..n {
                let cluster = labels[i];
                counts[cluster] += 1;
                let point = data.row(i);
                for j in 0..d {
                    new_centroids[cluster * d + j] += point[j];
                }
            }

            // Normalize centroids
            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..d {
                        new_centroids[c * d + j] /= counts[c] as f64;
                    }
                }
            }

            // Check convergence
            let max_shift = centroids
                .chunks(d)
                .zip(new_centroids.chunks(d))
                .map(|(old, new)| Self::euclidean_distance(old, new))
                .fold(0.0f64, f64::max);

            centroids = new_centroids;

            if max_shift < tolerance {
                converged = true;
                break;
            }
        }

        // Calculate inertia (sum of squared distances to centroids)
        let inertia: f64 = (0..n)
            .map(|i| {
                let point = data.row(i);
                let centroid_start = labels[i] * d;
                let centroid = &centroids[centroid_start..centroid_start + d];
                let dist = Self::euclidean_distance(point, centroid);
                dist * dist
            })
            .sum();

        ClusteringResult {
            labels,
            n_clusters: k,
            centroids,
            inertia,
            iterations,
            converged,
        }
    }

    /// K-Means++ initialization.
    fn kmeans_plus_plus_init(data: &DataMatrix, k: usize) -> Vec<f64> {
        let n = data.n_samples;
        let d = data.n_features;
        let mut rng = rand::rng();
        let mut centroids = Vec::with_capacity(k * d);

        // Choose first centroid randomly
        let first_idx = rng.random_range(0..n);
        centroids.extend_from_slice(data.row(first_idx));

        let mut distances = vec![f64::MAX; n];

        // Choose remaining centroids
        for _ in 1..k {
            // Update distances to nearest centroid
            for i in 0..n {
                let point = data.row(i);
                let last_centroid = &centroids[centroids.len() - d..];
                let dist = Self::euclidean_distance(point, last_centroid);
                distances[i] = distances[i].min(dist);
            }

            // Choose next centroid with probability proportional to D^2
            let total: f64 = distances.iter().map(|d| d * d).sum();
            let threshold = rng.random::<f64>() * total;

            let mut cumsum = 0.0;
            let mut next_idx = 0;
            for (i, &dist) in distances.iter().enumerate() {
                cumsum += dist * dist;
                if cumsum >= threshold {
                    next_idx = i;
                    break;
                }
            }

            centroids.extend_from_slice(data.row(next_idx));
        }

        centroids
    }

    /// Euclidean distance between two vectors.
    fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl GpuKernel for KMeans {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// DBSCAN Clustering Kernel
// ============================================================================

/// DBSCAN clustering kernel.
///
/// Density-based spatial clustering of applications with noise.
#[derive(Debug, Clone)]
pub struct DBSCAN {
    metadata: KernelMetadata,
}

impl Default for DBSCAN {
    fn default() -> Self {
        Self::new()
    }
}

impl DBSCAN {
    /// Create a new DBSCAN kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/dbscan-cluster", Domain::StatisticalML)
                .with_description("Density-based clustering with GPU union-find")
                .with_throughput(1_000)
                .with_latency_us(10_000.0),
        }
    }

    /// Run DBSCAN clustering.
    ///
    /// # Arguments
    /// * `data` - Input data matrix
    /// * `eps` - Maximum distance for neighborhood
    /// * `min_samples` - Minimum points to form a dense region
    /// * `metric` - Distance metric to use
    pub fn compute(
        data: &DataMatrix,
        eps: f64,
        min_samples: usize,
        metric: DistanceMetric,
    ) -> ClusteringResult {
        let n = data.n_samples;

        if n == 0 {
            return ClusteringResult {
                labels: Vec::new(),
                n_clusters: 0,
                centroids: Vec::new(),
                inertia: 0.0,
                iterations: 1,
                converged: true,
            };
        }

        // -1 = unvisited, -2 = noise, >= 0 = cluster label
        let mut labels = vec![-1i64; n];
        let mut current_cluster = 0i64;

        // Precompute neighborhoods (for efficiency)
        let neighborhoods: Vec<Vec<usize>> = (0..n)
            .map(|i| Self::get_neighbors(data, i, eps, metric))
            .collect();

        for i in 0..n {
            if labels[i] != -1 {
                continue; // Already processed
            }

            let neighbors = &neighborhoods[i];

            if neighbors.len() < min_samples {
                labels[i] = -2; // Mark as noise
                continue;
            }

            // Start new cluster
            labels[i] = current_cluster;
            let mut seed_set: Vec<usize> = neighbors.clone();
            let mut j = 0;

            while j < seed_set.len() {
                let q = seed_set[j];
                j += 1;

                if labels[q] == -2 {
                    labels[q] = current_cluster; // Change noise to border
                }

                if labels[q] != -1 {
                    continue; // Already processed
                }

                labels[q] = current_cluster;

                let q_neighbors = &neighborhoods[q];
                if q_neighbors.len() >= min_samples {
                    // Add new neighbors to seed set
                    for &neighbor in q_neighbors {
                        if !seed_set.contains(&neighbor) {
                            seed_set.push(neighbor);
                        }
                    }
                }
            }

            current_cluster += 1;
        }

        // Convert labels to usize (noise stays as max value)
        let n_clusters = current_cluster as usize;
        let labels: Vec<usize> = labels
            .iter()
            .map(|&l| if l < 0 { usize::MAX } else { l as usize })
            .collect();

        // Calculate centroids for each cluster
        let d = data.n_features;
        let mut centroids = vec![0.0f64; n_clusters * d];
        let mut counts = vec![0usize; n_clusters];

        for i in 0..n {
            if labels[i] < n_clusters {
                let cluster = labels[i];
                counts[cluster] += 1;
                for j in 0..d {
                    centroids[cluster * d + j] += data.row(i)[j];
                }
            }
        }

        for c in 0..n_clusters {
            if counts[c] > 0 {
                for j in 0..d {
                    centroids[c * d + j] /= counts[c] as f64;
                }
            }
        }

        ClusteringResult {
            labels,
            n_clusters,
            centroids,
            inertia: 0.0,
            iterations: 1,
            converged: true,
        }
    }

    /// Get neighbors within eps distance.
    fn get_neighbors(
        data: &DataMatrix,
        point_idx: usize,
        eps: f64,
        metric: DistanceMetric,
    ) -> Vec<usize> {
        let n = data.n_samples;
        let point = data.row(point_idx);

        (0..n)
            .filter(|&i| {
                let other = data.row(i);
                let dist = metric.compute(point, other);
                dist <= eps
            })
            .collect()
    }
}

impl GpuKernel for DBSCAN {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Hierarchical Clustering Kernel
// ============================================================================

/// Linkage method for hierarchical clustering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinkageMethod {
    /// Single linkage (minimum distance)
    Single,
    /// Complete linkage (maximum distance)
    Complete,
    /// Average linkage (UPGMA)
    Average,
    /// Ward's method (minimize variance)
    Ward,
}

/// Hierarchical clustering kernel.
///
/// Agglomerative hierarchical clustering with various linkage methods.
#[derive(Debug, Clone)]
pub struct HierarchicalClustering {
    metadata: KernelMetadata,
}

impl Default for HierarchicalClustering {
    fn default() -> Self {
        Self::new()
    }
}

impl HierarchicalClustering {
    /// Create a new hierarchical clustering kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/hierarchical-cluster", Domain::StatisticalML)
                .with_description("Agglomerative hierarchical clustering")
                .with_throughput(500)
                .with_latency_us(50_000.0),
        }
    }

    /// Run hierarchical clustering.
    ///
    /// # Arguments
    /// * `data` - Input data matrix
    /// * `n_clusters` - Number of clusters to form
    /// * `linkage` - Linkage method
    /// * `metric` - Distance metric
    pub fn compute(
        data: &DataMatrix,
        n_clusters: usize,
        linkage: LinkageMethod,
        metric: DistanceMetric,
    ) -> ClusteringResult {
        let n = data.n_samples;

        if n == 0 || n_clusters == 0 {
            return ClusteringResult {
                labels: Vec::new(),
                n_clusters: 0,
                centroids: Vec::new(),
                inertia: 0.0,
                iterations: 0,
                converged: true,
            };
        }

        // Initialize each point as its own cluster
        let mut labels: Vec<usize> = (0..n).collect();
        let mut active_clusters: Vec<bool> = vec![true; n];
        let mut cluster_sizes: Vec<usize> = vec![1; n];

        // Compute initial distance matrix
        let mut distances = Self::compute_distance_matrix(data, metric);

        // Merge clusters until we have n_clusters
        let mut current_n_clusters = n;

        while current_n_clusters > n_clusters {
            // Find closest pair of clusters
            let (c1, c2) = Self::find_closest_clusters(&distances, &active_clusters, n);

            if c1 == c2 {
                break;
            }

            // Merge c2 into c1
            for label in &mut labels {
                if *label == c2 {
                    *label = c1;
                }
            }

            // Update distances based on linkage
            Self::update_distances(
                &mut distances,
                c1,
                c2,
                n,
                linkage,
                &cluster_sizes,
                &active_clusters,
            );

            cluster_sizes[c1] += cluster_sizes[c2];
            active_clusters[c2] = false;
            current_n_clusters -= 1;
        }

        // Renumber labels to be contiguous
        let mut label_map = std::collections::HashMap::new();
        let mut next_label = 0usize;

        for label in &mut labels {
            let new_label = *label_map.entry(*label).or_insert_with(|| {
                let l = next_label;
                next_label += 1;
                l
            });
            *label = new_label;
        }

        // Calculate centroids
        let d = data.n_features;
        let final_n_clusters = next_label;
        let mut centroids = vec![0.0f64; final_n_clusters * d];
        let mut counts = vec![0usize; final_n_clusters];

        for i in 0..n {
            let cluster = labels[i];
            counts[cluster] += 1;
            for j in 0..d {
                centroids[cluster * d + j] += data.row(i)[j];
            }
        }

        for c in 0..final_n_clusters {
            if counts[c] > 0 {
                for j in 0..d {
                    centroids[c * d + j] /= counts[c] as f64;
                }
            }
        }

        ClusteringResult {
            labels,
            n_clusters: final_n_clusters,
            centroids,
            inertia: 0.0,
            iterations: (n - n_clusters) as u32,
            converged: true,
        }
    }

    fn compute_distance_matrix(data: &DataMatrix, metric: DistanceMetric) -> Vec<f64> {
        let n = data.n_samples;
        let mut distances = vec![f64::MAX; n * n];

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    distances[i * n + j] = metric.compute(data.row(i), data.row(j));
                }
            }
        }

        distances
    }

    fn find_closest_clusters(distances: &[f64], active: &[bool], n: usize) -> (usize, usize) {
        let mut min_dist = f64::MAX;
        let mut min_i = 0;
        let mut min_j = 0;

        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                let dist = distances[i * n + j];
                if dist < min_dist {
                    min_dist = dist;
                    min_i = i;
                    min_j = j;
                }
            }
        }

        (min_i, min_j)
    }

    fn update_distances(
        distances: &mut [f64],
        c1: usize,
        c2: usize,
        n: usize,
        linkage: LinkageMethod,
        cluster_sizes: &[usize],
        active: &[bool],
    ) {
        for k in 0..n {
            if !active[k] || k == c1 || k == c2 {
                continue;
            }

            let d1 = distances[c1 * n + k];
            let d2 = distances[c2 * n + k];

            let new_dist = match linkage {
                LinkageMethod::Single => d1.min(d2),
                LinkageMethod::Complete => d1.max(d2),
                LinkageMethod::Average => {
                    let n1 = cluster_sizes[c1] as f64;
                    let n2 = cluster_sizes[c2] as f64;
                    (n1 * d1 + n2 * d2) / (n1 + n2)
                }
                LinkageMethod::Ward => {
                    let n1 = cluster_sizes[c1] as f64;
                    let n2 = cluster_sizes[c2] as f64;
                    let nk = cluster_sizes[k] as f64;
                    let total = n1 + n2 + nk;
                    ((n1 + nk) * d1 * d1 + (n2 + nk) * d2 * d2
                        - nk * distances[c1 * n + c2].powi(2))
                        / total
                }
            };

            distances[c1 * n + k] = new_dist;
            distances[k * n + c1] = new_dist;
        }
    }
}

impl GpuKernel for HierarchicalClustering {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_two_clusters() -> DataMatrix {
        // Two clear clusters
        DataMatrix::from_rows(&[
            &[0.0, 0.0],
            &[0.1, 0.1],
            &[0.2, 0.0],
            &[10.0, 10.0],
            &[10.1, 10.1],
            &[10.2, 10.0],
        ])
    }

    #[test]
    fn test_kmeans_metadata() {
        let kernel = KMeans::new();
        assert_eq!(kernel.metadata().id, "ml/kmeans-cluster");
        assert_eq!(kernel.metadata().domain, Domain::StatisticalML);
    }

    #[test]
    fn test_kmeans_two_clusters() {
        let data = create_two_clusters();
        let result = KMeans::compute(&data, 2, 100, 1e-6);

        assert_eq!(result.n_clusters, 2);
        assert!(result.converged);

        // Points 0,1,2 should be in one cluster, 3,4,5 in another
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[1], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[4], result.labels[5]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_dbscan_two_clusters() {
        let data = create_two_clusters();
        let result = DBSCAN::compute(&data, 1.0, 2, DistanceMetric::Euclidean);

        assert_eq!(result.n_clusters, 2);

        // Points should be grouped correctly
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_hierarchical_two_clusters() {
        let data = create_two_clusters();
        let result = HierarchicalClustering::compute(
            &data,
            2,
            LinkageMethod::Complete,
            DistanceMetric::Euclidean,
        );

        assert_eq!(result.n_clusters, 2);

        // Points should be grouped correctly
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[1], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }
}
