//! Core kernel traits.
//!
//! This module defines the fundamental traits that all kernels implement:
//! - `GpuKernel`: Base trait for all GPU kernels
//! - `BatchKernel`: Trait for batch (CPU-orchestrated) kernels
//! - `RingKernelHandler`: Trait for ring (persistent actor) kernels

use crate::error::Result;
use crate::kernel::KernelMetadata;
use async_trait::async_trait;
use ringkernel_core::{RingContext, RingMessage};
use std::fmt::Debug;

/// Base trait for all GPU kernels.
///
/// Provides access to kernel metadata and input validation.
pub trait GpuKernel: Send + Sync + Debug {
    /// Returns the kernel metadata.
    fn metadata(&self) -> &KernelMetadata;

    /// Validate kernel configuration.
    ///
    /// Called before kernel launch to ensure configuration is valid.
    fn validate(&self) -> Result<()> {
        Ok(())
    }

    /// Returns the kernel ID.
    fn id(&self) -> &str {
        &self.metadata().id
    }

    /// Returns true if this kernel requires GPU-native execution.
    fn requires_gpu_native(&self) -> bool {
        self.metadata().requires_gpu_native
    }
}

/// Trait for batch (CPU-orchestrated) kernels.
///
/// Batch kernels are launched on-demand with CPU orchestration.
/// They have 10-50μs launch overhead and state resides in CPU memory.
///
/// # Type Parameters
///
/// - `I`: Input type
/// - `O`: Output type
#[async_trait]
pub trait BatchKernel<I, O>: GpuKernel
where
    I: Send + Sync,
    O: Send + Sync,
{
    /// Execute the kernel with the given input.
    ///
    /// # Arguments
    ///
    /// * `input` - The input data for the kernel
    ///
    /// # Returns
    ///
    /// The kernel output or an error.
    async fn execute(&self, input: I) -> Result<O>;

    /// Validate the input before execution.
    ///
    /// Override to provide custom input validation.
    fn validate_input(&self, _input: &I) -> Result<()> {
        Ok(())
    }
}

/// Trait for ring (persistent actor) kernels.
///
/// Ring kernels are persistent GPU actors with 100-500ns message latency.
/// State resides permanently in GPU memory.
///
/// # Type Parameters
///
/// - `M`: Request message type
/// - `R`: Response message type
#[async_trait]
pub trait RingKernelHandler<M, R>: GpuKernel
where
    M: RingMessage + Send + Sync,
    R: RingMessage + Send + Sync,
{
    /// Handle an incoming message.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The ring kernel context with GPU intrinsics
    /// * `msg` - The incoming message
    ///
    /// # Returns
    ///
    /// The response message or an error.
    async fn handle(&self, ctx: &mut RingContext, msg: M) -> Result<R>;

    /// Initialize the kernel state.
    ///
    /// Called once when the kernel is first activated.
    async fn initialize(&self, _ctx: &mut RingContext) -> Result<()> {
        Ok(())
    }

    /// Called when the kernel is being shut down.
    ///
    /// Use this to clean up resources.
    async fn shutdown(&self, _ctx: &mut RingContext) -> Result<()> {
        Ok(())
    }
}

/// Trait for iterative (multi-pass) kernels.
///
/// Provides support for algorithms that require multiple iterations
/// to converge (e.g., PageRank, K-Means).
///
/// # Type Parameters
///
/// - `S`: State type
/// - `I`: Input type
/// - `O`: Output type
#[async_trait]
pub trait IterativeKernel<S, I, O>: GpuKernel
where
    S: Send + Sync + 'static,
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    /// Create the initial state.
    fn initial_state(&self, input: &I) -> S;

    /// Perform one iteration.
    ///
    /// # Arguments
    ///
    /// * `state` - The current state (mutable)
    /// * `input` - The input data
    ///
    /// # Returns
    ///
    /// The iteration result.
    async fn iterate(&self, state: &mut S, input: &I) -> Result<IterationResult<O>>;

    /// Check if the algorithm has converged.
    ///
    /// # Arguments
    ///
    /// * `state` - The current state
    /// * `threshold` - The convergence threshold
    ///
    /// # Returns
    ///
    /// `true` if converged, `false` otherwise.
    fn converged(&self, state: &S, threshold: f64) -> bool;

    /// Maximum number of iterations.
    fn max_iterations(&self) -> usize {
        100
    }

    /// Default convergence threshold.
    fn default_threshold(&self) -> f64 {
        1e-6
    }

    /// Run the iterative algorithm to convergence.
    async fn run_to_convergence(&self, input: I) -> Result<O> {
        self.run_to_convergence_with_threshold(input, self.default_threshold())
            .await
    }

    /// Run the iterative algorithm with a custom threshold.
    async fn run_to_convergence_with_threshold(&self, input: I, threshold: f64) -> Result<O> {
        let mut state = self.initial_state(&input);
        let max_iter = self.max_iterations();

        for _ in 0..max_iter {
            let result = self.iterate(&mut state, &input).await?;

            if let IterationResult::Converged(output) = result {
                return Ok(output);
            }

            if self.converged(&state, threshold) {
                if let IterationResult::Continue(output) = result {
                    return Ok(output);
                }
            }
        }

        // Return final state even if not converged
        match self.iterate(&mut state, &input).await? {
            IterationResult::Converged(output) | IterationResult::Continue(output) => Ok(output),
        }
    }
}

/// Result of a single iteration.
#[derive(Debug, Clone)]
pub enum IterationResult<O> {
    /// Algorithm has converged with final output.
    Converged(O),
    /// Algorithm should continue; current intermediate output.
    Continue(O),
}

impl<O> IterationResult<O> {
    /// Returns true if converged.
    #[must_use]
    pub fn is_converged(&self) -> bool {
        matches!(self, IterationResult::Converged(_))
    }

    /// Extract the output.
    #[must_use]
    pub fn into_output(self) -> O {
        match self {
            IterationResult::Converged(o) | IterationResult::Continue(o) => o,
        }
    }
}

/// Type-erased batch kernel for registry storage.
#[async_trait]
pub trait BatchKernelDyn: GpuKernel {
    /// Execute with type-erased input/output.
    async fn execute_dyn(&self, input: &[u8]) -> Result<Vec<u8>>;
}

/// Type-erased ring kernel for registry storage.
#[async_trait]
pub trait RingKernelDyn: GpuKernel {
    /// Handle with type-erased messages.
    async fn handle_dyn(&self, ctx: &mut RingContext, msg: &[u8]) -> Result<Vec<u8>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iteration_result() {
        let converged: IterationResult<i32> = IterationResult::Converged(42);
        assert!(converged.is_converged());
        assert_eq!(converged.into_output(), 42);

        let continuing: IterationResult<i32> = IterationResult::Continue(0);
        assert!(!continuing.is_converged());
    }
}
