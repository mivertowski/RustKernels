//! # RustKernel Behavioral Analytics
//!
//! GPU-accelerated behavioral analytics kernels for profiling and forensics.
//!
//! ## Kernels
//! - `BehavioralProfiling` - Feature extraction for user behavior
//! - `AnomalyProfiling` - Deviation scoring
//! - `FraudSignatureDetection` - Known fraud pattern matching
//! - `CausalGraphConstruction` - DAG inference
//! - `ForensicQueryExecution` - Historical pattern search
//! - `EventCorrelation` - Temporal event correlation

#![warn(missing_docs)]

use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

/// Behavioral profiling kernel.
#[derive(Debug, Clone, Default)]
pub struct BehavioralProfiling { metadata: KernelMetadata }
impl BehavioralProfiling {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::ring("behavioral/profiling", Domain::BehavioralAnalytics)
            .with_description("User behavioral feature extraction")
            .with_throughput(100_000).with_latency_us(10.0) }
    }
}
impl GpuKernel for BehavioralProfiling { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Register all behavioral kernels.
pub fn register_all(_registry: &rustkernel_core::registry::KernelRegistry) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering behavioral analytics kernels");
    Ok(())
}
