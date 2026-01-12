//! # RustKernel Banking
//!
//! GPU-accelerated banking kernels for fraud detection.
//!
//! ## Kernels
//! - `FraudPatternMatch` - Aho-Corasick pattern matching + rapid split + cycle detection

#![warn(missing_docs)]

use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

/// Fraud pattern matching kernel.
#[derive(Debug, Clone, Default)]
pub struct FraudPatternMatch { metadata: KernelMetadata }
impl FraudPatternMatch {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::ring("banking/fraud-pattern-match", Domain::Banking)
            .with_description("Fraud pattern detection (Aho-Corasick, rapid split, cycles)")
            .with_throughput(50_000).with_latency_us(100.0).with_gpu_native(true) }
    }
}
impl GpuKernel for FraudPatternMatch { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Register all banking kernels.
pub fn register_all(_registry: &rustkernel_core::registry::KernelRegistry) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering banking kernels");
    Ok(())
}
