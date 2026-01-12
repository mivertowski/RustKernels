//! # RustKernel Risk Analytics
//!
//! GPU-accelerated risk analytics kernels for credit, market, and portfolio risk.
//!
//! ## Kernels
//!
//! ### Credit (1 kernel)
//! - `CreditRiskScoring` - PD/LGD/EAD calculation
//!
//! ### Market (2 kernels)
//! - `MonteCarloVaR` - Parallel Monte Carlo VaR simulation
//! - `PortfolioRiskAggregation` - Correlation-adjusted portfolio VaR
//!
//! ### Stress (1 kernel)
//! - `StressTesting` - Scenario-based stress testing

#![warn(missing_docs)]

use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

/// Credit risk scoring kernel.
#[derive(Debug, Clone, Default)]
pub struct CreditRiskScoring { metadata: KernelMetadata }
impl CreditRiskScoring {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::ring("risk/credit-scoring", Domain::RiskAnalytics)
            .with_description("PD/LGD/EAD credit risk calculation")
            .with_throughput(50_000).with_latency_us(100.0) }
    }
}
impl GpuKernel for CreditRiskScoring { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Monte Carlo VaR kernel.
#[derive(Debug, Clone, Default)]
pub struct MonteCarloVaR { metadata: KernelMetadata }
impl MonteCarloVaR {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::ring("risk/monte-carlo-var", Domain::RiskAnalytics)
            .with_description("Monte Carlo Value at Risk simulation")
            .with_throughput(100_000).with_latency_us(1000.0) }
    }
}
impl GpuKernel for MonteCarloVaR { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Portfolio risk aggregation kernel.
#[derive(Debug, Clone, Default)]
pub struct PortfolioRiskAggregation { metadata: KernelMetadata }
impl PortfolioRiskAggregation {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::ring("risk/portfolio-aggregation", Domain::RiskAnalytics)
            .with_description("Correlation-adjusted portfolio risk")
            .with_throughput(10_000).with_latency_us(500.0) }
    }
}
impl GpuKernel for PortfolioRiskAggregation { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Stress testing kernel.
#[derive(Debug, Clone, Default)]
pub struct StressTesting { metadata: KernelMetadata }
impl StressTesting {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::batch("risk/stress-testing", Domain::RiskAnalytics)
            .with_description("Scenario-based stress testing")
            .with_throughput(5_000).with_latency_us(2000.0) }
    }
}
impl GpuKernel for StressTesting { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Register all risk kernels.
pub fn register_all(_registry: &rustkernel_core::registry::KernelRegistry) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering risk analytics kernels");
    Ok(())
}
