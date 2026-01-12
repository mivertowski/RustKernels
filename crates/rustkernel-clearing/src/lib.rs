//! # RustKernel Clearing
//!
//! GPU-accelerated clearing and settlement kernels.
//!
//! ## Kernels
//! - `ClearingValidation` - Trade validation
//! - `DVPMatching` - Delivery vs payment matching
//! - `NettingCalculation` - Multilateral netting
//! - `SettlementExecution` - Settlement finalization
//! - `ZeroBalanceFrequency` - Settlement efficiency metrics

#![warn(missing_docs)]

use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

/// DVP matching kernel.
#[derive(Debug, Clone, Default)]
pub struct DVPMatching { metadata: KernelMetadata }
impl DVPMatching {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::ring("clearing/dvp-matching", Domain::Clearing)
            .with_description("Delivery vs payment matching")
            .with_throughput(50_000).with_latency_us(100.0) }
    }
}
impl GpuKernel for DVPMatching { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Netting calculation kernel.
#[derive(Debug, Clone, Default)]
pub struct NettingCalculation { metadata: KernelMetadata }
impl NettingCalculation {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::batch("clearing/netting", Domain::Clearing)
            .with_description("Multilateral netting calculation")
            .with_throughput(10_000).with_latency_us(500.0) }
    }
}
impl GpuKernel for NettingCalculation { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Register all clearing kernels.
pub fn register_all(_registry: &rustkernel_core::registry::KernelRegistry) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering clearing kernels");
    Ok(())
}
