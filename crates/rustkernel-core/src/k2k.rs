//! K2K (Kernel-to-Kernel) coordination utilities.
//!
//! This module provides higher-level abstractions for K2K communication patterns
//! commonly used in financial analytics pipelines.
//!
//! ## Coordination Patterns
//!
//! - **Fan-out**: One kernel broadcasting to multiple downstream kernels
//! - **Fan-in**: Multiple kernels sending to one aggregator kernel
//! - **Pipeline**: Sequential multi-stage processing
//! - **Scatter-Gather**: Parallel processing with result aggregation
//! - **Iterative**: Convergence-based algorithms (PageRank, K-Means)

use ringkernel_core::runtime::KernelId;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

// ============================================================================
// Kernel ID Utilities
// ============================================================================

/// Convert a KernelId to a u64 hash for message envelope addressing.
pub fn kernel_id_to_u64(id: &KernelId) -> u64 {
    let mut hasher = DefaultHasher::new();
    id.as_str().hash(&mut hasher);
    hasher.finish()
}

// ============================================================================
// Iterative Convergence Coordinator
// ============================================================================

/// State for tracking iterative algorithm convergence.
///
/// Used for algorithms like PageRank, K-Means, GARCH that iterate until convergence.
#[derive(Debug, Clone)]
pub struct IterativeState {
    /// Current iteration number.
    pub iteration: u64,
    /// Last computed delta/error value.
    pub last_delta: f64,
    /// Convergence threshold.
    pub convergence_threshold: f64,
    /// Maximum allowed iterations.
    pub max_iterations: u64,
    /// Whether algorithm has converged.
    pub converged: bool,
}

impl IterativeState {
    /// Create a new iterative state.
    pub fn new(convergence_threshold: f64, max_iterations: u64) -> Self {
        Self {
            iteration: 0,
            last_delta: f64::MAX,
            convergence_threshold,
            max_iterations,
            converged: false,
        }
    }

    /// Update state with new delta from an iteration.
    pub fn update(&mut self, delta: f64) -> bool {
        self.iteration += 1;
        self.last_delta = delta;
        self.converged = delta < self.convergence_threshold || self.iteration >= self.max_iterations;
        self.converged
    }

    /// Check if should continue iterating.
    pub fn should_continue(&self) -> bool {
        !self.converged && self.iteration < self.max_iterations
    }

    /// Reset state for a new run.
    pub fn reset(&mut self) {
        self.iteration = 0;
        self.last_delta = f64::MAX;
        self.converged = false;
    }

    /// Get convergence summary.
    pub fn summary(&self) -> IterativeConvergenceSummary {
        IterativeConvergenceSummary {
            iterations: self.iteration,
            final_delta: self.last_delta,
            converged: self.converged,
            reached_max: self.iteration >= self.max_iterations,
        }
    }
}

/// Summary of iterative algorithm convergence.
#[derive(Debug, Clone)]
pub struct IterativeConvergenceSummary {
    /// Number of iterations executed.
    pub iterations: u64,
    /// Final delta/error value.
    pub final_delta: f64,
    /// Whether convergence was achieved.
    pub converged: bool,
    /// Whether max iterations was reached.
    pub reached_max: bool,
}

// ============================================================================
// Pipeline Stage Tracker
// ============================================================================

/// Tracks progress through a multi-stage pipeline.
#[derive(Debug, Clone)]
pub struct PipelineTracker {
    stages: Vec<String>,
    current_stage: usize,
    stage_timings_us: HashMap<String, u64>,
    total_items_processed: u64,
}

impl PipelineTracker {
    /// Create a new pipeline tracker with the given stages.
    pub fn new(stages: Vec<String>) -> Self {
        Self {
            stages,
            current_stage: 0,
            stage_timings_us: HashMap::new(),
            total_items_processed: 0,
        }
    }

    /// Get the current stage name.
    pub fn current_stage(&self) -> Option<&str> {
        self.stages.get(self.current_stage).map(|s| s.as_str())
    }

    /// Get the next stage name.
    pub fn next_stage(&self) -> Option<&str> {
        self.stages.get(self.current_stage + 1).map(|s| s.as_str())
    }

    /// Advance to the next stage, recording timing for the completed stage.
    pub fn advance(&mut self, elapsed_us: u64) -> bool {
        if let Some(stage) = self.stages.get(self.current_stage) {
            self.stage_timings_us.insert(stage.clone(), elapsed_us);
        }
        if self.current_stage + 1 < self.stages.len() {
            self.current_stage += 1;
            true
        } else {
            false
        }
    }

    /// Record items processed in current stage.
    pub fn record_items(&mut self, count: u64) {
        self.total_items_processed += count;
    }

    /// Check if pipeline is complete.
    pub fn is_complete(&self) -> bool {
        self.current_stage >= self.stages.len().saturating_sub(1)
            && self.stage_timings_us.len() >= self.stages.len()
    }

    /// Get total pipeline timing.
    pub fn total_time_us(&self) -> u64 {
        self.stage_timings_us.values().sum()
    }

    /// Get timing for a specific stage.
    pub fn stage_timing(&self, stage: &str) -> Option<u64> {
        self.stage_timings_us.get(stage).copied()
    }

    /// Reset pipeline for new processing.
    pub fn reset(&mut self) {
        self.current_stage = 0;
        self.stage_timings_us.clear();
        self.total_items_processed = 0;
    }
}

// ============================================================================
// Scatter-Gather State
// ============================================================================

/// Tracks scatter-gather operation state.
#[derive(Debug)]
pub struct ScatterGatherState<T> {
    /// Number of workers to scatter to.
    pub worker_count: usize,
    /// Results received so far.
    pub results: Vec<T>,
    /// Workers that have responded.
    pub responded_workers: Vec<KernelId>,
    /// Start timestamp (microseconds).
    pub start_time_us: u64,
}

impl<T> ScatterGatherState<T> {
    /// Create new scatter-gather state.
    pub fn new(worker_count: usize, start_time_us: u64) -> Self {
        Self {
            worker_count,
            results: Vec::with_capacity(worker_count),
            responded_workers: Vec::with_capacity(worker_count),
            start_time_us,
        }
    }

    /// Record a result from a worker.
    pub fn receive_result(&mut self, worker: KernelId, result: T) {
        if !self.responded_workers.contains(&worker) {
            self.responded_workers.push(worker);
            self.results.push(result);
        }
    }

    /// Check if all workers have responded.
    pub fn is_complete(&self) -> bool {
        self.responded_workers.len() >= self.worker_count
    }

    /// Get count of pending responses.
    pub fn pending_count(&self) -> usize {
        self.worker_count.saturating_sub(self.responded_workers.len())
    }

    /// Get the results (consumes the state).
    pub fn take_results(self) -> Vec<T> {
        self.results
    }
}

// ============================================================================
// Fan-Out Destination Tracker
// ============================================================================

/// Tracks fan-out broadcast destinations and delivery status.
#[derive(Debug, Clone)]
pub struct FanOutTracker {
    destinations: Vec<KernelId>,
    delivery_status: HashMap<String, bool>,
    broadcast_count: u64,
}

impl FanOutTracker {
    /// Create new fan-out tracker.
    pub fn new() -> Self {
        Self {
            destinations: Vec::new(),
            delivery_status: HashMap::new(),
            broadcast_count: 0,
        }
    }

    /// Add a destination kernel.
    pub fn add_destination(&mut self, dest: KernelId) {
        if !self.destinations.iter().any(|d| d.as_str() == dest.as_str()) {
            self.destinations.push(dest);
        }
    }

    /// Remove a destination kernel.
    pub fn remove_destination(&mut self, dest: &KernelId) {
        self.destinations.retain(|d| d.as_str() != dest.as_str());
        self.delivery_status.remove(dest.as_str());
    }

    /// Get all destination IDs.
    pub fn destinations(&self) -> &[KernelId] {
        &self.destinations
    }

    /// Record broadcast attempt.
    pub fn record_broadcast(&mut self) {
        self.broadcast_count += 1;
        // Reset delivery status for new broadcast
        for dest in &self.destinations {
            self.delivery_status.insert(dest.as_str().to_string(), false);
        }
    }

    /// Mark delivery to a destination as successful.
    pub fn mark_delivered(&mut self, dest: &KernelId) {
        self.delivery_status.insert(dest.as_str().to_string(), true);
    }

    /// Get delivery success count for last broadcast.
    pub fn delivery_count(&self) -> usize {
        self.delivery_status.values().filter(|&&v| v).count()
    }

    /// Get total broadcast count.
    pub fn broadcast_count(&self) -> u64 {
        self.broadcast_count
    }

    /// Get destination count.
    pub fn destination_count(&self) -> usize {
        self.destinations.len()
    }
}

impl Default for FanOutTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// K2K Control Messages
// ============================================================================

/// Control messages for K2K coordination between kernels.
#[derive(Debug, Clone)]
pub enum K2KControlMessage {
    /// Signal to start processing.
    Start {
        /// Correlation ID for tracking.
        correlation_id: u64,
    },
    /// Signal to stop processing.
    Stop {
        /// Reason for stopping.
        reason: String,
    },
    /// Request current state/status.
    GetStatus {
        /// Correlation ID for response.
        correlation_id: u64,
    },
    /// Signal iteration complete.
    IterationComplete {
        /// Iteration number.
        iteration: u64,
        /// Delta/error from this iteration.
        delta: f64,
        /// Worker ID that completed.
        worker_id: u64,
    },
    /// Signal convergence reached.
    Converged {
        /// Total iterations.
        iterations: u64,
        /// Final delta/error.
        final_delta: f64,
    },
    /// Signal processing error.
    Error {
        /// Error message.
        message: String,
        /// Error code.
        code: u32,
    },
    /// Heartbeat for liveness checking.
    Heartbeat {
        /// Sequence number.
        sequence: u64,
        /// Timestamp (microseconds).
        timestamp_us: u64,
    },
    /// Barrier synchronization.
    Barrier {
        /// Barrier ID.
        barrier_id: u64,
        /// Worker that reached barrier.
        worker_id: u64,
    },
}

// ============================================================================
// K2K Aggregation Result
// ============================================================================

/// Result from a worker in a scatter-gather operation.
#[derive(Debug, Clone)]
pub struct K2KWorkerResult<T> {
    /// Worker that produced this result.
    pub worker_id: KernelId,
    /// Correlation ID linking to original request.
    pub correlation_id: u64,
    /// The result data.
    pub result: T,
    /// Processing time in microseconds.
    pub processing_time_us: u64,
}

impl<T> K2KWorkerResult<T> {
    /// Create a new worker result.
    pub fn new(
        worker_id: KernelId,
        correlation_id: u64,
        result: T,
        processing_time_us: u64,
    ) -> Self {
        Self {
            worker_id,
            correlation_id,
            result,
            processing_time_us,
        }
    }
}

// ============================================================================
// K2K Message Priority
// ============================================================================

/// Priority levels for K2K messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum K2KPriority {
    /// Low priority - background processing.
    Low = 0,
    /// Normal priority - default.
    Normal = 64,
    /// High priority - time-sensitive operations.
    High = 128,
    /// Critical priority - must process immediately.
    Critical = 192,
    /// Real-time priority - latency-critical paths.
    RealTime = 255,
}

impl Default for K2KPriority {
    fn default() -> Self {
        Self::Normal
    }
}

impl From<K2KPriority> for u8 {
    fn from(p: K2KPriority) -> u8 {
        p as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iterative_state_convergence() {
        let mut state = IterativeState::new(1e-6, 100);

        assert!(state.should_continue());
        assert!(!state.converged);

        // Simulate iterations
        state.update(0.1);
        assert!(!state.converged);
        assert_eq!(state.iteration, 1);

        state.update(0.01);
        assert!(!state.converged);

        state.update(1e-7); // Below threshold
        assert!(state.converged);

        let summary = state.summary();
        assert_eq!(summary.iterations, 3);
        assert!(summary.converged);
    }

    #[test]
    fn test_iterative_state_max_iterations() {
        let mut state = IterativeState::new(1e-6, 3);

        state.update(0.1);
        state.update(0.05);
        state.update(0.01); // Reaches max iterations

        assert!(state.converged);
        let summary = state.summary();
        assert!(summary.reached_max);
    }

    #[test]
    fn test_pipeline_tracker() {
        let stages = vec!["ingest".to_string(), "transform".to_string(), "output".to_string()];
        let mut tracker = PipelineTracker::new(stages);

        assert_eq!(tracker.current_stage(), Some("ingest"));
        assert_eq!(tracker.next_stage(), Some("transform"));

        tracker.advance(1000);
        assert_eq!(tracker.current_stage(), Some("transform"));

        tracker.advance(2000);
        assert_eq!(tracker.current_stage(), Some("output"));

        tracker.advance(500);
        assert!(tracker.is_complete());
        assert_eq!(tracker.total_time_us(), 3500);
    }

    #[test]
    fn test_scatter_gather_state() {
        let mut state: ScatterGatherState<i32> = ScatterGatherState::new(3, 0);

        assert!(!state.is_complete());
        assert_eq!(state.pending_count(), 3);

        state.receive_result(KernelId::new("worker1"), 10);
        state.receive_result(KernelId::new("worker2"), 20);
        assert_eq!(state.pending_count(), 1);

        state.receive_result(KernelId::new("worker3"), 30);
        assert!(state.is_complete());

        let results = state.take_results();
        assert_eq!(results, vec![10, 20, 30]);
    }

    #[test]
    fn test_fan_out_tracker() {
        let mut tracker = FanOutTracker::new();

        tracker.add_destination(KernelId::new("dest1"));
        tracker.add_destination(KernelId::new("dest2"));
        tracker.add_destination(KernelId::new("dest1")); // Duplicate

        assert_eq!(tracker.destination_count(), 2);

        tracker.record_broadcast();
        assert_eq!(tracker.broadcast_count(), 1);
        assert_eq!(tracker.delivery_count(), 0);

        tracker.mark_delivered(&KernelId::new("dest1"));
        assert_eq!(tracker.delivery_count(), 1);
    }

    #[test]
    fn test_kernel_id_to_u64() {
        let id1 = KernelId::new("kernel-a");
        let id2 = KernelId::new("kernel-b");
        let id1_copy = KernelId::new("kernel-a");

        let hash1 = kernel_id_to_u64(&id1);
        let hash2 = kernel_id_to_u64(&id2);
        let hash1_copy = kernel_id_to_u64(&id1_copy);

        assert_ne!(hash1, hash2);
        assert_eq!(hash1, hash1_copy);
    }
}
