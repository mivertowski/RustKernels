//! Ring message types for Graph Analytics kernels.
//!
//! This module defines zero-copy Ring messages for GPU-native persistent actors.
//! Type IDs 100-199 are reserved for Graph Analytics domain.
//!
//! ## Type ID Allocation
//!
//! - 100-109: PageRank messages
//! - 110-119: Community detection messages
//! - 120-129: Centrality messages
//! - 130-139: K2K coordination messages

use ringkernel_derive::RingMessage;
use rkyv::{Archive, Deserialize, Serialize};
use rustkernel_core::messages::MessageId;

// ============================================================================
// PageRank Ring Messages (100-109)
// ============================================================================

/// PageRank query request - get score for a specific node.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 100)]
pub struct PageRankQueryRing {
    /// Message ID.
    pub id: MessageId,
    /// Node ID to query.
    pub node_id: u64,
}

/// PageRank query response.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 101)]
pub struct PageRankQueryResponse {
    /// Original message ID.
    pub request_id: u64,
    /// Node ID queried.
    pub node_id: u64,
    /// PageRank score (fixed-point: value * 100_000_000).
    pub score_fp: i64,
    /// Current iteration count.
    pub iteration: u32,
    /// Whether algorithm has converged.
    pub converged: bool,
}

/// PageRank iterate request - perform one power iteration step.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 102)]
pub struct PageRankIterateRing {
    /// Message ID.
    pub id: MessageId,
}

/// PageRank iterate response.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 103)]
pub struct PageRankIterateResponse {
    /// Original message ID.
    pub request_id: u64,
    /// Iteration number.
    pub iteration: u32,
    /// Maximum delta in this iteration (fixed-point: value * 100_000_000).
    pub max_delta_fp: i64,
    /// Whether algorithm has converged.
    pub converged: bool,
}

/// PageRank converge request - iterate until threshold or max iterations.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 104)]
pub struct PageRankConvergeRing {
    /// Message ID.
    pub id: MessageId,
    /// Convergence threshold (fixed-point: value * 100_000_000).
    pub threshold_fp: i64,
    /// Maximum iterations.
    pub max_iterations: u32,
}

/// PageRank convergence response.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 105)]
pub struct PageRankConvergeResponse {
    /// Original message ID.
    pub request_id: u64,
    /// Total iterations executed.
    pub iterations: u32,
    /// Final maximum delta (fixed-point: value * 100_000_000).
    pub final_delta_fp: i64,
    /// Whether algorithm converged (vs hit max iterations).
    pub converged: bool,
}

// ============================================================================
// K2K Coordination Messages (130-139)
// ============================================================================

/// K2K iteration synchronization request.
///
/// Used for coordinating distributed PageRank across graph partitions.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 130)]
pub struct K2KIterationSync {
    /// Message ID.
    pub id: MessageId,
    /// Worker/partition ID (hashed KernelId).
    pub worker_id: u64,
    /// Current iteration number.
    pub iteration: u64,
    /// Local delta from this partition (fixed-point: value * 100_000_000).
    pub local_delta_fp: i64,
    /// Number of nodes processed.
    pub nodes_processed: u64,
}

/// K2K iteration sync response.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 131)]
pub struct K2KIterationSyncResponse {
    /// Original message ID.
    pub request_id: u64,
    /// Acknowledged iteration.
    pub iteration: u64,
    /// Whether all workers have synced.
    pub all_synced: bool,
    /// Global delta (max across all partitions, fixed-point).
    pub global_delta_fp: i64,
    /// Whether global convergence achieved.
    pub global_converged: bool,
}

/// K2K boundary node update.
///
/// When graph is partitioned, boundary nodes need score updates from other partitions.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 132)]
pub struct K2KBoundaryUpdate {
    /// Message ID.
    pub id: MessageId,
    /// Source partition ID.
    pub source_partition: u64,
    /// Target partition ID.
    pub target_partition: u64,
    /// Iteration number.
    pub iteration: u64,
    /// Number of boundary node updates.
    pub update_count: u32,
    /// Boundary node IDs (serialized array).
    pub node_ids_packed: [u64; 8],
    /// Boundary node scores (fixed-point, serialized array).
    pub scores_packed: [i64; 8],
}

/// K2K boundary update acknowledgment.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 133)]
pub struct K2KBoundaryUpdateAck {
    /// Original message ID.
    pub request_id: u64,
    /// Iteration number.
    pub iteration: u64,
    /// Updates applied.
    pub updates_applied: u32,
}

/// K2K barrier synchronization.
///
/// Used to synchronize all workers before proceeding to next iteration.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 134)]
pub struct K2KBarrier {
    /// Message ID.
    pub id: MessageId,
    /// Barrier ID (iteration-based).
    pub barrier_id: u64,
    /// Worker ID.
    pub worker_id: u64,
    /// Workers ready count (from worker's perspective).
    pub ready_count: u32,
    /// Total workers expected.
    pub total_workers: u32,
}

/// K2K barrier release.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 135)]
pub struct K2KBarrierRelease {
    /// Original barrier ID.
    pub barrier_id: u64,
    /// All workers synchronized.
    pub all_ready: bool,
    /// Next iteration number.
    pub next_iteration: u64,
}

/// K2K worker heartbeat.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 136)]
pub struct K2KHeartbeat {
    /// Message ID.
    pub id: MessageId,
    /// Worker ID.
    pub worker_id: u64,
    /// Sequence number.
    pub sequence: u64,
    /// Timestamp in microseconds.
    pub timestamp_us: u64,
    /// Current state: 0=idle, 1=computing, 2=syncing, 3=converged.
    pub state: u8,
}

// ============================================================================
// Community Detection Ring Messages (110-119)
// ============================================================================

/// Request to compute modularity for current community assignment.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 110)]
pub struct ComputeModularityRing {
    /// Message ID.
    pub id: MessageId,
}

/// Modularity computation response.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 111)]
pub struct ModularityResponse {
    /// Original message ID.
    pub request_id: u64,
    /// Modularity score (fixed-point: value * 100_000_000).
    pub modularity_fp: i64,
    /// Number of communities.
    pub num_communities: u32,
}

/// K2K community merge proposal.
///
/// Used in distributed Louvain for proposing community merges across partitions.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 112)]
pub struct K2KCommunityMerge {
    /// Message ID.
    pub id: MessageId,
    /// Source partition.
    pub source_partition: u64,
    /// Community A ID.
    pub community_a: u64,
    /// Community B ID.
    pub community_b: u64,
    /// Delta modularity from merge (fixed-point: value * 100_000_000).
    pub delta_q_fp: i64,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_conversion() {
        let value = 0.85;
        let fp = to_fixed_point(value);
        let back = from_fixed_point(fp);
        assert!((value - back).abs() < 1e-8);
    }

    #[test]
    fn test_pagerank_query_ring() {
        let msg = PageRankQueryRing {
            id: MessageId(1),
            node_id: 42,
        };
        assert_eq!(msg.node_id, 42);
    }

    #[test]
    fn test_k2k_iteration_sync() {
        let msg = K2KIterationSync {
            id: MessageId(2),
            worker_id: 1,
            iteration: 5,
            local_delta_fp: to_fixed_point(0.001),
            nodes_processed: 1000,
        };
        assert_eq!(msg.iteration, 5);
        let delta = from_fixed_point(msg.local_delta_fp);
        assert!((delta - 0.001).abs() < 1e-8);
    }

    #[test]
    fn test_k2k_barrier() {
        let msg = K2KBarrier {
            id: MessageId(3),
            barrier_id: 10,
            worker_id: 2,
            ready_count: 3,
            total_workers: 4,
        };
        assert_eq!(msg.barrier_id, 10);
        assert_eq!(msg.ready_count, 3);
    }
}
