//! Ring message types for Statistical ML kernels.
//!
//! This module defines zero-copy Ring messages for GPU-native persistent actors.
//! Type IDs 200-299 are reserved for Statistical ML domain.
//!
//! ## Type ID Allocation
//!
//! - 200-219: KMeans messages
//! - 220-239: DBSCAN messages
//! - 240-259: Anomaly detection messages
//! - 260-279: K2K parallel coordination messages

use ringkernel_derive::RingMessage;
use rkyv::{Archive, Deserialize, Serialize};
use rustkernel_core::messages::MessageId;

// ============================================================================
// KMeans Ring Messages (200-219)
// ============================================================================

/// Initialize KMeans with centroids.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 200)]
pub struct KMeansInitRing {
    /// Message ID.
    pub id: MessageId,
    /// Number of clusters (K).
    pub k: u32,
    /// Number of features per point.
    pub n_features: u32,
    /// Initial centroids (packed: k * n_features values, fixed-point).
    pub centroids_packed: [i64; 32], // Support up to k=8, n_features=4
}

/// KMeans initialization response.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 201)]
pub struct KMeansInitResponse {
    /// Original message ID.
    pub request_id: u64,
    /// Whether initialization succeeded.
    pub success: bool,
    /// Number of clusters configured.
    pub k: u32,
}

/// Assign points to clusters (E-step).
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 202)]
pub struct KMeansAssignRing {
    /// Message ID.
    pub id: MessageId,
    /// Iteration number.
    pub iteration: u32,
}

/// Assignment response.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 203)]
pub struct KMeansAssignResponse {
    /// Original message ID.
    pub request_id: u64,
    /// Iteration number.
    pub iteration: u32,
    /// Total inertia (sum of squared distances, fixed-point).
    pub inertia_fp: i64,
    /// Number of points assigned.
    pub points_assigned: u32,
}

/// Update centroids (M-step).
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 204)]
pub struct KMeansUpdateRing {
    /// Message ID.
    pub id: MessageId,
    /// Iteration number.
    pub iteration: u32,
}

/// Update response with new centroids.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 205)]
pub struct KMeansUpdateResponse {
    /// Original message ID.
    pub request_id: u64,
    /// Iteration number.
    pub iteration: u32,
    /// Maximum centroid shift (fixed-point).
    pub max_shift_fp: i64,
    /// Whether converged.
    pub converged: bool,
}

/// Query cluster assignment for a point.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 206)]
pub struct KMeansQueryRing {
    /// Message ID.
    pub id: MessageId,
    /// Point coordinates (fixed-point).
    pub point: [i64; 8], // Up to 8 dimensions
    /// Number of dimensions.
    pub n_dims: u8,
}

/// Query response.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 207)]
pub struct KMeansQueryResponse {
    /// Original message ID.
    pub request_id: u64,
    /// Assigned cluster.
    pub cluster: u32,
    /// Distance to centroid (fixed-point).
    pub distance_fp: i64,
}

// ============================================================================
// K2K Parallel Centroid Update Messages (260-279)
// ============================================================================

/// K2K partial centroid update from a worker.
///
/// In distributed KMeans, each worker computes partial sums for centroids
/// from its data partition.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 260)]
pub struct K2KPartialCentroid {
    /// Message ID.
    pub id: MessageId,
    /// Worker ID.
    pub worker_id: u64,
    /// Iteration number.
    pub iteration: u64,
    /// Cluster ID this update is for.
    pub cluster_id: u32,
    /// Number of points assigned to this cluster on this worker.
    pub point_count: u32,
    /// Partial sum of coordinates (fixed-point, up to 8 dimensions).
    pub coord_sum_fp: [i64; 8],
    /// Number of dimensions.
    pub n_dims: u8,
}

/// K2K centroid aggregation response.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 261)]
pub struct K2KCentroidAggregation {
    /// Original request ID.
    pub request_id: u64,
    /// Cluster ID.
    pub cluster_id: u32,
    /// Iteration number.
    pub iteration: u64,
    /// New centroid coordinates (fixed-point).
    pub new_centroid_fp: [i64; 8],
    /// Total points in cluster.
    pub total_points: u32,
    /// Centroid shift from previous iteration (fixed-point).
    pub shift_fp: i64,
}

/// K2K iteration sync for distributed KMeans.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 262)]
pub struct K2KKMeansSync {
    /// Message ID.
    pub id: MessageId,
    /// Worker ID.
    pub worker_id: u64,
    /// Iteration number.
    pub iteration: u64,
    /// Local inertia (fixed-point).
    pub local_inertia_fp: i64,
    /// Points processed on this worker.
    pub points_processed: u32,
    /// Maximum local centroid shift (fixed-point).
    pub max_shift_fp: i64,
}

/// K2K sync response.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 263)]
pub struct K2KKMeansSyncResponse {
    /// Original request ID.
    pub request_id: u64,
    /// Iteration number.
    pub iteration: u64,
    /// All workers synced.
    pub all_synced: bool,
    /// Global inertia (fixed-point).
    pub global_inertia_fp: i64,
    /// Global maximum shift (fixed-point).
    pub global_max_shift_fp: i64,
    /// Global converged.
    pub converged: bool,
}

/// K2K broadcast new centroids to workers.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 264)]
pub struct K2KCentroidBroadcast {
    /// Message ID.
    pub id: MessageId,
    /// Iteration number.
    pub iteration: u64,
    /// Number of clusters.
    pub k: u32,
    /// Number of dimensions.
    pub n_dims: u8,
    /// Packed centroids (up to k=4 clusters, 8 dims each).
    pub centroids_packed: [i64; 32],
}

/// K2K broadcast acknowledgment.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 265)]
pub struct K2KCentroidBroadcastAck {
    /// Original message ID.
    pub request_id: u64,
    /// Worker ID.
    pub worker_id: u64,
    /// Iteration received.
    pub iteration: u64,
    /// Centroids applied.
    pub applied: bool,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert f64 to fixed-point i64 (8 decimal places).
#[inline]
pub fn to_fixed_point(value: f64) -> i64 {
    (value * 100_000_000.0) as i64
}

/// Convert fixed-point i64 to f64.
#[inline]
pub fn from_fixed_point(fp: i64) -> f64 {
    fp as f64 / 100_000_000.0
}

/// Pack coordinates into fixed-point array.
pub fn pack_coordinates(coords: &[f64], output: &mut [i64; 8]) {
    for (i, &c) in coords.iter().take(8).enumerate() {
        output[i] = to_fixed_point(c);
    }
}

/// Unpack coordinates from fixed-point array.
pub fn unpack_coordinates(input: &[i64; 8], n_dims: usize) -> Vec<f64> {
    input
        .iter()
        .take(n_dims)
        .map(|&fp| from_fixed_point(fp))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_conversion() {
        let value = 2.5;
        let fp = to_fixed_point(value);
        let back = from_fixed_point(fp);
        assert!((value - back).abs() < 1e-8);
    }

    #[test]
    fn test_pack_unpack_coordinates() {
        let coords = vec![1.5, 2.5, 3.5];
        let mut packed = [0i64; 8];
        pack_coordinates(&coords, &mut packed);

        let unpacked = unpack_coordinates(&packed, 3);
        assert_eq!(unpacked.len(), 3);
        for (a, b) in coords.iter().zip(unpacked.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn test_kmeans_init_ring() {
        let msg = KMeansInitRing {
            id: MessageId(1),
            k: 3,
            n_features: 2,
            centroids_packed: [0; 32],
        };
        assert_eq!(msg.k, 3);
    }

    #[test]
    fn test_k2k_partial_centroid() {
        let mut coord_sum = [0i64; 8];
        pack_coordinates(&[10.0, 20.0], &mut coord_sum);

        let msg = K2KPartialCentroid {
            id: MessageId(2),
            worker_id: 1,
            iteration: 5,
            cluster_id: 0,
            point_count: 100,
            coord_sum_fp: coord_sum,
            n_dims: 2,
        };
        assert_eq!(msg.point_count, 100);
        assert_eq!(msg.iteration, 5);
    }

    #[test]
    fn test_k2k_kmeans_sync() {
        let msg = K2KKMeansSync {
            id: MessageId(3),
            worker_id: 2,
            iteration: 10,
            local_inertia_fp: to_fixed_point(1234.5),
            points_processed: 5000,
            max_shift_fp: to_fixed_point(0.001),
        };
        assert_eq!(msg.iteration, 10);
        assert_eq!(msg.points_processed, 5000);
    }
}
