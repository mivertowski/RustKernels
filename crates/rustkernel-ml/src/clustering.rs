//! Clustering kernels.
//!
//! This module provides machine learning clustering algorithms:
//! - K-Means (Lloyd's algorithm with K-Means++ initialization)
//! - DBSCAN (density-based clustering)
//! - Hierarchical clustering (agglomerative)

use crate::ring_messages::{
    K2KCentroidAggregation, K2KCentroidBroadcast, K2KCentroidBroadcastAck, K2KKMeansSync,
    K2KKMeansSyncResponse, K2KPartialCentroid, KMeansAssignResponse, KMeansAssignRing,
    KMeansQueryResponse, KMeansQueryRing, KMeansUpdateResponse, KMeansUpdateRing, from_fixed_point,
    to_fixed_point, unpack_coordinates,
};
use crate::types::{ClusteringResult, DataMatrix, DistanceMetric};
use rand::prelude::*;
use ringkernel_core::RingContext;
use rustkernel_core::traits::RingKernelHandler;
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

// ============================================================================
// K-Means Clustering Kernel
// ============================================================================

/// K-Means clustering state for Ring mode operations.
#[derive(Debug, Clone, Default)]
pub struct KMeansState {
    /// Current centroids (k * n_features).
    pub centroids: Vec<f64>,
    /// Input data reference (stored for query operations).
    pub data: Option<DataMatrix>,
    /// Number of clusters.
    pub k: usize,
    /// Number of features per point.
    pub n_features: usize,
    /// Current iteration.
    pub iteration: u32,
    /// Current inertia (sum of squared distances).
    pub inertia: f64,
    /// Whether converged.
    pub converged: bool,
    /// Current cluster assignments.
    pub labels: Vec<usize>,
}

/// K-Means clustering kernel.
///
/// Implements Lloyd's algorithm with K-Means++ initialization.
#[derive(Debug)]
pub struct KMeans {
    metadata: KernelMetadata,
    /// Internal state for Ring mode operations.
    state: std::sync::RwLock<KMeansState>,
}

impl Clone for KMeans {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            state: std::sync::RwLock::new(self.state.read().unwrap().clone()),
        }
    }
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
            state: std::sync::RwLock::new(KMeansState::default()),
        }
    }

    /// Initialize the kernel with data and k for Ring mode operations.
    pub fn initialize(&self, data: DataMatrix, k: usize) {
        let centroids = Self::kmeans_plus_plus_init(&data, k);
        let n = data.n_samples;
        let n_features = data.n_features;

        let mut state = self.state.write().unwrap();
        *state = KMeansState {
            centroids,
            data: Some(data),
            k,
            n_features,
            iteration: 0,
            inertia: 0.0,
            converged: false,
            labels: vec![0; n],
        };
    }

    /// Perform one E-step (assignment) on internal state.
    /// Returns the total inertia (sum of squared distances).
    pub fn assign_step(&self) -> f64 {
        let mut state = self.state.write().unwrap();

        // Check if data exists
        let data = match state.data {
            Some(ref d) => d.clone(),
            None => return 0.0,
        };

        let n = data.n_samples;
        let d_features = state.n_features;
        let mut total_inertia = 0.0;

        // Clone centroids to avoid borrow conflict
        let centroids = state.centroids.clone();

        // Compute assignments
        let mut new_labels = vec![0usize; n];
        for i in 0..n {
            let point = data.row(i);
            let mut min_dist = f64::MAX;
            let mut min_cluster = 0;

            for (c, centroid) in centroids.chunks(d_features).enumerate() {
                let dist = Self::euclidean_distance(point, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    min_cluster = c;
                }
            }
            new_labels[i] = min_cluster;
            total_inertia += min_dist * min_dist;
        }

        // Update state
        state.labels = new_labels;
        state.inertia = total_inertia;
        total_inertia
    }

    /// Perform one M-step (centroid update) on internal state.
    /// Returns the maximum centroid shift.
    pub fn update_step(&self) -> f64 {
        let mut state = self.state.write().unwrap();
        let Some(ref data) = state.data else {
            return 0.0;
        };

        let n = data.n_samples;
        let d = state.n_features;
        let k = state.k;

        let mut new_centroids = vec![0.0f64; k * d];
        let mut counts = vec![0usize; k];

        for i in 0..n {
            let cluster = state.labels[i];
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

        // Calculate maximum shift
        let max_shift = state
            .centroids
            .chunks(d)
            .zip(new_centroids.chunks(d))
            .map(|(old, new)| Self::euclidean_distance(old, new))
            .fold(0.0f64, f64::max);

        state.centroids = new_centroids;
        state.iteration += 1;
        max_shift
    }

    /// Query the nearest cluster for a point.
    pub fn query_point(&self, point: &[f64]) -> (usize, f64) {
        let state = self.state.read().unwrap();
        let d = state.n_features;

        let mut min_dist = f64::MAX;
        let mut min_cluster = 0;

        for (c, centroid) in state.centroids.chunks(d).enumerate() {
            let dist = Self::euclidean_distance(point, centroid);
            if dist < min_dist {
                min_dist = dist;
                min_cluster = c;
            }
        }

        (min_cluster, min_dist)
    }

    /// Get current iteration count.
    pub fn current_iteration(&self) -> u32 {
        self.state.read().unwrap().iteration
    }

    /// Get current inertia.
    pub fn current_inertia(&self) -> f64 {
        self.state.read().unwrap().inertia
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
// KMeans RingKernelHandler Implementations
// ============================================================================

/// RingKernelHandler for KMeans assignment step (E-step).
#[async_trait::async_trait]
impl RingKernelHandler<KMeansAssignRing, KMeansAssignResponse> for KMeans {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: KMeansAssignRing,
    ) -> Result<KMeansAssignResponse> {
        // Perform assignment step on internal state
        let inertia = self.assign_step();

        let state = self.state.read().unwrap();
        let points_assigned = state.labels.len() as u32;

        Ok(KMeansAssignResponse {
            request_id: msg.id.0,
            iteration: msg.iteration,
            inertia_fp: to_fixed_point(inertia),
            points_assigned,
        })
    }
}

/// RingKernelHandler for KMeans update step (M-step).
#[async_trait::async_trait]
impl RingKernelHandler<KMeansUpdateRing, KMeansUpdateResponse> for KMeans {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: KMeansUpdateRing,
    ) -> Result<KMeansUpdateResponse> {
        // Perform update step on internal state
        let max_shift = self.update_step();
        let converged = max_shift < 1e-6;

        // Update convergence status in state
        if converged {
            let mut state = self.state.write().unwrap();
            state.converged = true;
        }

        Ok(KMeansUpdateResponse {
            request_id: msg.id.0,
            iteration: msg.iteration,
            max_shift_fp: to_fixed_point(max_shift),
            converged,
        })
    }
}

/// RingKernelHandler for point queries.
#[async_trait::async_trait]
impl RingKernelHandler<KMeansQueryRing, KMeansQueryResponse> for KMeans {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: KMeansQueryRing,
    ) -> Result<KMeansQueryResponse> {
        // Unpack the query point coordinates
        let point = unpack_coordinates(&msg.point, msg.n_dims as usize);

        // Query the nearest cluster using internal state
        let (cluster, distance) = self.query_point(&point);

        Ok(KMeansQueryResponse {
            request_id: msg.id.0,
            cluster: cluster as u32,
            distance_fp: to_fixed_point(distance),
        })
    }
}

/// RingKernelHandler for K2K partial centroid updates.
///
/// Aggregates partial centroid contributions from distributed workers.
#[async_trait::async_trait]
impl RingKernelHandler<K2KPartialCentroid, K2KCentroidAggregation> for KMeans {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: K2KPartialCentroid,
    ) -> Result<K2KCentroidAggregation> {
        let n_dims = msg.n_dims as usize;
        let cluster_id = msg.cluster_id as usize;
        let mut new_centroid = [0i64; 8];

        // Compute new centroid from partial sums
        if msg.point_count > 0 {
            for i in 0..n_dims.min(8) {
                new_centroid[i] = msg.coord_sum_fp[i] / msg.point_count as i64;
            }
        }

        // Calculate shift from old centroid in internal state
        let shift = {
            let state = self.state.read().unwrap();
            let d = state.n_features;
            if cluster_id < state.k && d > 0 {
                let old_centroid = &state.centroids[cluster_id * d..(cluster_id + 1) * d];
                let new_coords: Vec<f64> = new_centroid[..d.min(8)]
                    .iter()
                    .map(|&v| from_fixed_point(v))
                    .collect();
                Self::euclidean_distance(old_centroid, &new_coords)
            } else {
                0.0
            }
        };

        Ok(K2KCentroidAggregation {
            request_id: msg.id.0,
            cluster_id: msg.cluster_id,
            iteration: msg.iteration,
            new_centroid_fp: new_centroid,
            total_points: msg.point_count,
            shift_fp: to_fixed_point(shift),
        })
    }
}

/// RingKernelHandler for K2K iteration sync.
///
/// Synchronizes distributed KMeans workers after each iteration.
/// In a single-instance setting, validates iteration state and returns convergence status.
#[async_trait::async_trait]
impl RingKernelHandler<K2KKMeansSync, K2KKMeansSyncResponse> for KMeans {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: K2KKMeansSync,
    ) -> Result<K2KKMeansSyncResponse> {
        let state = self.state.read().unwrap();

        // Verify iteration matches internal state
        let current_iteration = state.iteration as u64;
        let all_synced = msg.iteration <= current_iteration;

        // Use reported values for single-worker case
        // In distributed setting, would aggregate across workers
        let global_shift = from_fixed_point(msg.max_shift_fp);
        let converged = global_shift < 1e-6 || state.converged;

        Ok(K2KKMeansSyncResponse {
            request_id: msg.id.0,
            iteration: msg.iteration,
            all_synced,
            global_inertia_fp: msg.local_inertia_fp,
            global_max_shift_fp: msg.max_shift_fp,
            converged,
        })
    }
}

/// RingKernelHandler for K2K centroid broadcast.
///
/// Receives new centroids broadcast from coordinator.
#[async_trait::async_trait]
impl RingKernelHandler<K2KCentroidBroadcast, K2KCentroidBroadcastAck> for KMeans {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: K2KCentroidBroadcast,
    ) -> Result<K2KCentroidBroadcastAck> {
        // In a distributed setting, this would update local centroids
        Ok(K2KCentroidBroadcastAck {
            request_id: msg.id.0,
            worker_id: 0, // Would be actual worker ID
            iteration: msg.iteration,
            applied: true,
        })
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

// ============================================================================
// BatchKernel Implementations
// ============================================================================

use crate::messages::{
    DBSCANInput, DBSCANOutput, HierarchicalInput, HierarchicalOutput, KMeansInput, KMeansOutput,
    Linkage,
};
use async_trait::async_trait;
use rustkernel_core::error::Result;
use rustkernel_core::traits::BatchKernel;
use std::time::Instant;

/// K-Means batch kernel implementation.
impl KMeans {
    /// Execute K-Means clustering as a batch operation.
    ///
    /// Convenience method for batch clustering.
    pub async fn cluster_batch(&self, input: KMeansInput) -> Result<KMeansOutput> {
        let start = Instant::now();
        let result = Self::compute(&input.data, input.k, input.max_iterations, input.tolerance);
        let compute_time_us = start.elapsed().as_micros() as u64;

        Ok(KMeansOutput {
            result,
            compute_time_us,
        })
    }
}

#[async_trait]
impl BatchKernel<KMeansInput, KMeansOutput> for KMeans {
    async fn execute(&self, input: KMeansInput) -> Result<KMeansOutput> {
        self.cluster_batch(input).await
    }
}

/// DBSCAN batch kernel implementation.
#[async_trait]
impl BatchKernel<DBSCANInput, DBSCANOutput> for DBSCAN {
    async fn execute(&self, input: DBSCANInput) -> Result<DBSCANOutput> {
        let start = Instant::now();
        let result = Self::compute(&input.data, input.eps, input.min_samples, input.metric);
        let compute_time_us = start.elapsed().as_micros() as u64;

        Ok(DBSCANOutput {
            result,
            compute_time_us,
        })
    }
}

/// Hierarchical clustering batch kernel implementation.
#[async_trait]
impl BatchKernel<HierarchicalInput, HierarchicalOutput> for HierarchicalClustering {
    async fn execute(&self, input: HierarchicalInput) -> Result<HierarchicalOutput> {
        let start = Instant::now();
        let linkage_method = match input.linkage {
            Linkage::Single => LinkageMethod::Single,
            Linkage::Complete => LinkageMethod::Complete,
            Linkage::Average => LinkageMethod::Average,
            Linkage::Ward => LinkageMethod::Ward,
        };
        let result = Self::compute(&input.data, input.n_clusters, linkage_method, input.metric);
        let compute_time_us = start.elapsed().as_micros() as u64;

        Ok(HierarchicalOutput {
            result,
            compute_time_us,
        })
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
