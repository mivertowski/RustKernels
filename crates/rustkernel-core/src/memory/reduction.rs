//! Reduction Primitives
//!
//! Provides GPU reduction operations and multi-phase kernel synchronization
//! for iterative algorithms like PageRank, K-Means, and graph analytics.
//!
//! # Reduction Modes
//!
//! - **Single-pass**: Simple reductions completed in one kernel launch
//! - **Multi-phase**: Complex reductions requiring intermediate storage
//! - **Cooperative**: GPU-wide synchronization using cooperative groups
//!
//! # Sync Modes
//!
//! - **Cooperative**: Use CUDA cooperative groups for grid-wide sync
//! - **SoftwareBarrier**: Software-based barrier with atomic operations
//! - **MultiLaunch**: Separate kernel launches per phase
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_core::memory::reduction::{InterPhaseReduction, SyncMode};
//!
//! let reduction = InterPhaseReduction::<f64>::new(1000, SyncMode::Cooperative);
//!
//! // Phase 1: Local reduction
//! reduction.phase_start(0);
//! // ... kernel execution ...
//! reduction.phase_complete(0);
//!
//! // Phase 2: Global reduction
//! reduction.phase_start(1);
//! // ... kernel execution ...
//! let result = reduction.finalize();
//! ```

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// Synchronization mode for multi-phase reductions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SyncMode {
    /// Use cooperative groups for GPU-wide synchronization
    Cooperative,
    /// Software barrier using atomic operations
    SoftwareBarrier,
    /// Separate kernel launches per phase
    #[default]
    MultiLaunch,
}

/// Reduction operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    /// Sum all values
    Sum,
    /// Product of all values
    Product,
    /// Maximum value
    Max,
    /// Minimum value
    Min,
    /// Count of true values
    Count,
    /// Logical AND
    All,
    /// Logical OR
    Any,
}

/// Phase state for multi-phase reductions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseState {
    /// Phase not started
    Pending,
    /// Phase in progress
    Running,
    /// Phase completed
    Complete,
    /// Phase failed
    Failed,
}

/// Configuration for inter-phase reduction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReductionConfig {
    /// Synchronization mode
    pub sync_mode: SyncMode,
    /// Number of phases
    pub num_phases: u32,
    /// Elements per block for block-level reduction
    pub block_size: u32,
    /// Number of blocks
    pub grid_size: u32,
    /// Enable convergence checking between phases
    pub convergence_check: bool,
    /// Convergence threshold
    pub convergence_threshold: f64,
}

impl Default for ReductionConfig {
    fn default() -> Self {
        Self {
            sync_mode: SyncMode::MultiLaunch,
            num_phases: 2,
            block_size: 256,
            grid_size: 1024,
            convergence_check: false,
            convergence_threshold: 1e-6,
        }
    }
}

/// Inter-phase reduction state
pub struct InterPhaseReduction<T> {
    /// Configuration
    config: ReductionConfig,
    /// Input size
    input_size: usize,
    /// Phase buffers
    phase_buffers: Vec<Vec<T>>,
    /// Current phase
    current_phase: AtomicU32,
    /// Phase states
    phase_states: Vec<AtomicU32>,
    /// Whether reduction is complete
    is_complete: AtomicBool,
    /// Convergence value (if tracked)
    convergence_value: AtomicU64,
}

impl<T: Default + Clone + Copy> InterPhaseReduction<T> {
    /// Create a new inter-phase reduction
    pub fn new(input_size: usize, sync_mode: SyncMode) -> Self {
        Self::with_config(
            input_size,
            ReductionConfig {
                sync_mode,
                ..Default::default()
            },
        )
    }

    /// Create with full configuration
    pub fn with_config(input_size: usize, config: ReductionConfig) -> Self {
        let num_phases = config.num_phases as usize;

        // Calculate buffer sizes for each phase
        let mut phase_buffers = Vec::with_capacity(num_phases);
        let mut size = input_size;
        for _ in 0..num_phases {
            phase_buffers.push(vec![T::default(); size]);
            // Each phase reduces by block_size factor
            size = size.div_ceil(config.block_size as usize);
            size = size.max(1);
        }

        let phase_states: Vec<_> = (0..num_phases)
            .map(|_| AtomicU32::new(PhaseState::Pending as u32))
            .collect();

        Self {
            config,
            input_size,
            phase_buffers,
            current_phase: AtomicU32::new(0),
            phase_states,
            is_complete: AtomicBool::new(false),
            convergence_value: AtomicU64::new(0),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &ReductionConfig {
        &self.config
    }

    /// Get input size
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get current phase
    pub fn current_phase(&self) -> u32 {
        self.current_phase.load(Ordering::Relaxed)
    }

    /// Start a phase
    pub fn phase_start(&self, phase: u32) -> Result<(), ReductionError> {
        if phase >= self.config.num_phases {
            return Err(ReductionError::InvalidPhase {
                phase,
                max_phases: self.config.num_phases,
            });
        }

        let expected = PhaseState::Pending as u32;
        let new = PhaseState::Running as u32;

        match self.phase_states[phase as usize].compare_exchange(
            expected,
            new,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => {
                self.current_phase.store(phase, Ordering::Relaxed);
                Ok(())
            }
            Err(current) => Err(ReductionError::InvalidPhaseState {
                phase,
                current: phase_state_from_u32(current),
            }),
        }
    }

    /// Complete a phase
    pub fn phase_complete(&self, phase: u32) -> Result<(), ReductionError> {
        if phase >= self.config.num_phases {
            return Err(ReductionError::InvalidPhase {
                phase,
                max_phases: self.config.num_phases,
            });
        }

        let expected = PhaseState::Running as u32;
        let new = PhaseState::Complete as u32;

        match self.phase_states[phase as usize].compare_exchange(
            expected,
            new,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => {
                // Check if all phases complete
                if phase == self.config.num_phases - 1 {
                    self.is_complete.store(true, Ordering::Release);
                }
                Ok(())
            }
            Err(current) => Err(ReductionError::InvalidPhaseState {
                phase,
                current: phase_state_from_u32(current),
            }),
        }
    }

    /// Mark a phase as failed
    pub fn phase_failed(&self, phase: u32) {
        if (phase as usize) < self.phase_states.len() {
            self.phase_states[phase as usize].store(PhaseState::Failed as u32, Ordering::Release);
        }
    }

    /// Get phase state
    pub fn phase_state(&self, phase: u32) -> PhaseState {
        if phase >= self.config.num_phases {
            return PhaseState::Pending;
        }
        phase_state_from_u32(self.phase_states[phase as usize].load(Ordering::Acquire))
    }

    /// Check if reduction is complete
    pub fn is_complete(&self) -> bool {
        self.is_complete.load(Ordering::Acquire)
    }

    /// Get buffer for a phase (for reading previous phase results)
    pub fn get_buffer(&self, phase: u32) -> Option<&[T]> {
        self.phase_buffers.get(phase as usize).map(|v| v.as_slice())
    }

    /// Get mutable buffer for a phase (for writing current phase results)
    pub fn get_buffer_mut(&mut self, phase: u32) -> Option<&mut [T]> {
        self.phase_buffers
            .get_mut(phase as usize)
            .map(|v| v.as_mut_slice())
    }

    /// Get buffer size for a phase
    pub fn buffer_size(&self, phase: u32) -> usize {
        self.phase_buffers
            .get(phase as usize)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Set convergence value (as bits)
    pub fn set_convergence(&self, value: f64) {
        self.convergence_value
            .store(value.to_bits(), Ordering::Release);
    }

    /// Get convergence value
    pub fn convergence(&self) -> f64 {
        f64::from_bits(self.convergence_value.load(Ordering::Acquire))
    }

    /// Check if converged (if convergence checking enabled)
    pub fn is_converged(&self) -> bool {
        if !self.config.convergence_check {
            return false;
        }
        self.convergence() < self.config.convergence_threshold
    }

    /// Reset for reuse
    pub fn reset(&mut self) {
        self.current_phase.store(0, Ordering::Relaxed);
        self.is_complete.store(false, Ordering::Release);
        self.convergence_value.store(0, Ordering::Release);

        for state in &self.phase_states {
            state.store(PhaseState::Pending as u32, Ordering::Release);
        }

        for buffer in &mut self.phase_buffers {
            for item in buffer.iter_mut() {
                *item = T::default();
            }
        }
    }
}

fn phase_state_from_u32(value: u32) -> PhaseState {
    match value {
        0 => PhaseState::Pending,
        1 => PhaseState::Running,
        2 => PhaseState::Complete,
        _ => PhaseState::Failed,
    }
}

/// Reduction errors
#[derive(Debug, thiserror::Error)]
pub enum ReductionError {
    /// Invalid phase number
    #[error("Invalid phase {phase}, max phases: {max_phases}")]
    InvalidPhase {
        /// Requested phase
        phase: u32,
        /// Maximum phases
        max_phases: u32,
    },

    /// Invalid phase state transition
    #[error("Invalid phase state for phase {phase}: {current:?}")]
    InvalidPhaseState {
        /// Phase number
        phase: u32,
        /// Current state
        current: PhaseState,
    },

    /// Reduction not complete
    #[error("Reduction not complete, current phase: {current_phase}")]
    NotComplete {
        /// Current phase
        current_phase: u32,
    },

    /// Buffer size mismatch
    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch {
        /// Expected size
        expected: usize,
        /// Actual size
        actual: usize,
    },
}

/// Global reduction tracker for K2K coordination
pub struct GlobalReduction {
    /// Total number of participants
    pub total_participants: u32,
    /// Number of participants that have completed
    pub completed: AtomicU32,
    /// Whether all participants are done
    pub all_complete: AtomicBool,
    /// Partial results (one per participant)
    pub partial_results: Vec<AtomicU64>,
}

impl GlobalReduction {
    /// Create a new global reduction
    pub fn new(participants: u32) -> Self {
        let partial_results = (0..participants).map(|_| AtomicU64::new(0)).collect();

        Self {
            total_participants: participants,
            completed: AtomicU32::new(0),
            all_complete: AtomicBool::new(false),
            partial_results,
        }
    }

    /// Submit a partial result
    pub fn submit(&self, participant_id: u32, value: f64) -> bool {
        if participant_id >= self.total_participants {
            return false;
        }

        self.partial_results[participant_id as usize].store(value.to_bits(), Ordering::Release);

        let count = self.completed.fetch_add(1, Ordering::AcqRel) + 1;
        if count == self.total_participants {
            self.all_complete.store(true, Ordering::Release);
            return true;
        }

        false
    }

    /// Check if all participants have submitted
    pub fn is_complete(&self) -> bool {
        self.all_complete.load(Ordering::Acquire)
    }

    /// Get completion count
    pub fn completion_count(&self) -> u32 {
        self.completed.load(Ordering::Acquire)
    }

    /// Compute final result (sum of partials)
    pub fn finalize_sum(&self) -> Option<f64> {
        if !self.is_complete() {
            return None;
        }

        let sum: f64 = self
            .partial_results
            .iter()
            .map(|v| f64::from_bits(v.load(Ordering::Acquire)))
            .sum();

        Some(sum)
    }

    /// Compute final result (max of partials)
    pub fn finalize_max(&self) -> Option<f64> {
        if !self.is_complete() {
            return None;
        }

        self.partial_results
            .iter()
            .map(|v| f64::from_bits(v.load(Ordering::Acquire)))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Compute final result (min of partials)
    pub fn finalize_min(&self) -> Option<f64> {
        if !self.is_complete() {
            return None;
        }

        self.partial_results
            .iter()
            .map(|v| f64::from_bits(v.load(Ordering::Acquire)))
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Reset for reuse
    pub fn reset(&self) {
        self.completed.store(0, Ordering::Release);
        self.all_complete.store(false, Ordering::Release);
        for partial in &self.partial_results {
            partial.store(0, Ordering::Release);
        }
    }
}

/// Helper to create a cooperative sync barrier
pub struct CooperativeBarrier {
    /// Expected thread count
    expected: u32,
    /// Arrived count
    arrived: AtomicU32,
    /// Generation counter
    generation: AtomicU32,
}

impl CooperativeBarrier {
    /// Create a new barrier
    pub fn new(expected: u32) -> Self {
        Self {
            expected,
            arrived: AtomicU32::new(0),
            generation: AtomicU32::new(0),
        }
    }

    /// Wait at the barrier
    pub fn wait(&self) -> u32 {
        let generation_num = self.generation.load(Ordering::Acquire);
        let arrived = self.arrived.fetch_add(1, Ordering::AcqRel) + 1;

        if arrived == self.expected {
            // Last one to arrive, reset and advance generation
            self.arrived.store(0, Ordering::Release);
            self.generation.fetch_add(1, Ordering::Release);
        } else {
            // Spin wait for generation change
            while self.generation.load(Ordering::Acquire) == generation_num {
                std::hint::spin_loop();
            }
        }

        generation_num
    }

    /// Reset the barrier
    pub fn reset(&self) {
        self.arrived.store(0, Ordering::Release);
        self.generation.store(0, Ordering::Release);
    }
}

/// Builder for reduction operations
pub struct ReductionBuilder {
    config: ReductionConfig,
}

impl ReductionBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: ReductionConfig::default(),
        }
    }

    /// Set sync mode
    pub fn sync_mode(mut self, mode: SyncMode) -> Self {
        self.config.sync_mode = mode;
        self
    }

    /// Set number of phases
    pub fn phases(mut self, num: u32) -> Self {
        self.config.num_phases = num;
        self
    }

    /// Set block size
    pub fn block_size(mut self, size: u32) -> Self {
        self.config.block_size = size;
        self
    }

    /// Set grid size
    pub fn grid_size(mut self, size: u32) -> Self {
        self.config.grid_size = size;
        self
    }

    /// Enable convergence checking
    pub fn with_convergence(mut self, threshold: f64) -> Self {
        self.config.convergence_check = true;
        self.config.convergence_threshold = threshold;
        self
    }

    /// Build the configuration
    pub fn build(self) -> ReductionConfig {
        self.config
    }

    /// Build an InterPhaseReduction
    pub fn build_reduction<T: Default + Clone + Copy>(
        self,
        input_size: usize,
    ) -> InterPhaseReduction<T> {
        InterPhaseReduction::with_config(input_size, self.config)
    }
}

impl Default for ReductionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inter_phase_reduction() {
        let reduction = InterPhaseReduction::<f64>::new(1024, SyncMode::MultiLaunch);

        assert_eq!(reduction.current_phase(), 0);
        assert!(!reduction.is_complete());

        // Phase 0
        reduction.phase_start(0).unwrap();
        assert_eq!(reduction.phase_state(0), PhaseState::Running);
        reduction.phase_complete(0).unwrap();
        assert_eq!(reduction.phase_state(0), PhaseState::Complete);

        // Phase 1
        reduction.phase_start(1).unwrap();
        reduction.phase_complete(1).unwrap();

        assert!(reduction.is_complete());
    }

    #[test]
    fn test_phase_buffers() {
        let mut reduction = InterPhaseReduction::<f64>::with_config(
            1000,
            ReductionConfig {
                block_size: 256,
                num_phases: 3,
                ..Default::default()
            },
        );

        // First phase buffer should be 1000 elements
        assert_eq!(reduction.buffer_size(0), 1000);

        // Subsequent buffers are reduced
        assert!(reduction.buffer_size(1) < reduction.buffer_size(0));

        // Can write to buffers
        if let Some(buf) = reduction.get_buffer_mut(0) {
            buf[0] = 42.0;
        }

        assert_eq!(reduction.get_buffer(0).unwrap()[0], 42.0);
    }

    #[test]
    fn test_global_reduction() {
        let reduction = GlobalReduction::new(4);

        assert!(!reduction.is_complete());

        reduction.submit(0, 1.0);
        reduction.submit(1, 2.0);
        reduction.submit(2, 3.0);

        assert!(!reduction.is_complete());
        assert_eq!(reduction.completion_count(), 3);

        reduction.submit(3, 4.0);

        assert!(reduction.is_complete());
        assert_eq!(reduction.finalize_sum(), Some(10.0));
    }

    #[test]
    fn test_cooperative_barrier() {
        use std::sync::Arc;
        use std::thread;

        let barrier = Arc::new(CooperativeBarrier::new(3));
        let handles: Vec<_> = (0..3)
            .map(|_| {
                let b = barrier.clone();
                thread::spawn(move || b.wait())
            })
            .collect();

        for h in handles {
            let generation_num = h.join().unwrap();
            assert_eq!(generation_num, 0);
        }
    }

    #[test]
    fn test_reduction_builder() {
        let config = ReductionBuilder::new()
            .sync_mode(SyncMode::Cooperative)
            .phases(3)
            .block_size(512)
            .with_convergence(1e-8)
            .build();

        assert_eq!(config.sync_mode, SyncMode::Cooperative);
        assert_eq!(config.num_phases, 3);
        assert_eq!(config.block_size, 512);
        assert!(config.convergence_check);
    }

    #[test]
    fn test_convergence_tracking() {
        let reduction = InterPhaseReduction::<f64>::with_config(
            100,
            ReductionConfig {
                convergence_check: true,
                convergence_threshold: 1e-6,
                ..Default::default()
            },
        );

        reduction.set_convergence(1e-3);
        assert!(!reduction.is_converged());

        reduction.set_convergence(1e-8);
        assert!(reduction.is_converged());
    }
}
