//! # RustKernel Risk Analytics
//!
//! GPU-accelerated risk analytics kernels for credit, market, and portfolio risk.
//!
//! ## Kernels
//!
//! ### Credit (1 kernel)
//! - `CreditRiskScoring` - PD/LGD/EAD calculation and credit scoring
//!
//! ### Market (2 kernels)
//! - `MonteCarloVaR` - Monte Carlo Value at Risk simulation
//! - `PortfolioRiskAggregation` - Correlation-adjusted portfolio VaR
//!
//! ### Stress (1 kernel)
//! - `StressTesting` - Scenario-based stress testing

#![warn(missing_docs)]

pub mod credit;
pub mod market;
pub mod messages;
pub mod ring_messages;
pub mod stress;
pub mod types;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::credit::*;
    pub use crate::market::*;
    pub use crate::messages::*;
    pub use crate::ring_messages::*;
    pub use crate::stress::*;
    pub use crate::types::*;
}

// Re-export main kernels
pub use credit::CreditRiskScoring;
pub use market::{MonteCarloVaR, PortfolioRiskAggregation};
pub use stress::StressTesting;

// Re-export key types
pub use types::{
    CreditExposure, CreditFactors, CreditRiskResult, Portfolio, PortfolioRiskResult, RiskFactor,
    RiskFactorType, Sensitivity, StressScenario, StressTestResult, VaRParams, VaRResult,
};

/// Register all risk kernels with a registry.
pub fn register_all(
    registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    use rustkernel_core::traits::GpuKernel;

    tracing::info!("Registering risk analytics kernels");

    // Credit kernel (1)
    registry.register_metadata(credit::CreditRiskScoring::new().metadata().clone())?;

    // Market kernels (2)
    registry.register_metadata(market::MonteCarloVaR::new().metadata().clone())?;
    registry.register_metadata(market::PortfolioRiskAggregation::new().metadata().clone())?;

    // Stress kernel (1)
    registry.register_metadata(stress::StressTesting::new().metadata().clone())?;

    tracing::info!("Registered 4 risk analytics kernels");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustkernel_core::registry::KernelRegistry;

    #[test]
    fn test_register_all() {
        let registry = KernelRegistry::new();
        register_all(&registry).expect("Failed to register risk kernels");
        assert_eq!(registry.total_count(), 4);
    }
}
