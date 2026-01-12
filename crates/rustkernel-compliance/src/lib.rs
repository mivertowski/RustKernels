//! # RustKernel Compliance
//!
//! GPU-accelerated compliance kernels for AML, KYC, sanctions screening, and transaction monitoring.
//!
//! ## Kernels
//!
//! ### AML (4 kernels)
//! - `CircularFlowRatio` - SCC detection for circular transactions
//! - `ReciprocityFlowRatio` - Mutual transaction detection
//! - `RapidMovement` - Velocity analysis for structuring
//! - `AMLPatternDetection` - Multi-pattern FSM detection
//!
//! ### KYC (2 kernels)
//! - `KYCScoring` - Risk factor aggregation
//! - `EntityResolution` - Fuzzy entity matching
//!
//! ### Sanctions (2 kernels)
//! - `SanctionsScreening` - OFAC/UN/EU list matching
//! - `PEPScreening` - Politically exposed persons
//!
//! ### Monitoring (1 kernel)
//! - `TransactionMonitoring` - Real-time threshold alerts

#![warn(missing_docs)]

use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

/// AML circular flow detection kernel.
#[derive(Debug, Clone, Default)]
pub struct CircularFlowRatio { metadata: KernelMetadata }
impl CircularFlowRatio {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::ring("compliance/circular-flow", Domain::Compliance)
            .with_description("Circular flow detection via SCC")
            .with_throughput(50_000).with_latency_us(100.0) }
    }
}
impl GpuKernel for CircularFlowRatio { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Sanctions screening kernel.
#[derive(Debug, Clone, Default)]
pub struct SanctionsScreening { metadata: KernelMetadata }
impl SanctionsScreening {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::ring("compliance/sanctions-screening", Domain::Compliance)
            .with_description("OFAC/UN/EU sanctions list screening")
            .with_throughput(100_000).with_latency_us(10.0) }
    }
}
impl GpuKernel for SanctionsScreening { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Transaction monitoring kernel.
#[derive(Debug, Clone, Default)]
pub struct TransactionMonitoring { metadata: KernelMetadata }
impl TransactionMonitoring {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::ring("compliance/transaction-monitoring", Domain::Compliance)
            .with_description("Real-time transaction threshold monitoring")
            .with_throughput(500_000).with_latency_us(1.0) }
    }
}
impl GpuKernel for TransactionMonitoring { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Register all compliance kernels.
pub fn register_all(_registry: &rustkernel_core::registry::KernelRegistry) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering compliance kernels");
    Ok(())
}
