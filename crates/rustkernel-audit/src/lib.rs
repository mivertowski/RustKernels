//! # RustKernel Financial Audit
//!
//! GPU-accelerated financial audit kernels.
//!
//! ## Kernels
//! - `FeatureExtraction` - Audit feature vectors
//! - `HypergraphConstruction` - Multi-way relationship analysis

#![warn(missing_docs)]

use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

/// Feature extraction kernel.
#[derive(Debug, Clone, Default)]
pub struct FeatureExtraction { metadata: KernelMetadata }
impl FeatureExtraction {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::batch("audit/feature-extraction", Domain::FinancialAudit)
            .with_description("Audit feature vector extraction")
            .with_throughput(50_000).with_latency_us(50.0) }
    }
}
impl GpuKernel for FeatureExtraction { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Hypergraph construction kernel.
#[derive(Debug, Clone, Default)]
pub struct HypergraphConstruction { metadata: KernelMetadata }
impl HypergraphConstruction {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::batch("audit/hypergraph", Domain::FinancialAudit)
            .with_description("Multi-way relationship hypergraph")
            .with_throughput(10_000).with_latency_us(500.0) }
    }
}
impl GpuKernel for HypergraphConstruction { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Register all audit kernels.
pub fn register_all(_registry: &rustkernel_core::registry::KernelRegistry) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering financial audit kernels");
    Ok(())
}
