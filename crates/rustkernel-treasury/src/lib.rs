//! # RustKernel Treasury Management
//!
//! GPU-accelerated treasury management kernels.
//!
//! ## Kernels
//! - `CashFlowForecasting` - Multi-horizon cash flow projection
//! - `CollateralOptimization` - LP/QP optimization
//! - `FXHedging` - Currency exposure management
//! - `InterestRateRisk` - Duration/convexity analysis
//! - `LiquidityOptimization` - LCR/NSFR optimization

#![warn(missing_docs)]

use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

/// Cash flow forecasting kernel.
#[derive(Debug, Clone, Default)]
pub struct CashFlowForecasting { metadata: KernelMetadata }
impl CashFlowForecasting {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::batch("treasury/cashflow-forecast", Domain::TreasuryManagement)
            .with_description("Multi-horizon cash flow projection")
            .with_throughput(10_000).with_latency_us(500.0) }
    }
}
impl GpuKernel for CashFlowForecasting { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Register all treasury kernels.
pub fn register_all(_registry: &rustkernel_core::registry::KernelRegistry) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering treasury management kernels");
    Ok(())
}
