//! # RustKernel Risk Analytics
//!
//! GPU-accelerated risk analytics kernels for credit, market, and portfolio risk.
//!
//! ## Kernels
//!
//! ### Credit (1 kernel)
//! - `CreditRiskScoring` - PD/LGD/EAD calculation and credit scoring
//!
//! ### Market (3 kernels)
//! - `MonteCarloVaR` - Monte Carlo Value at Risk simulation
//! - `PortfolioRiskAggregation` - Correlation-adjusted portfolio VaR
//! - `RealTimeCorrelation` - Streaming correlation matrix updates
//!
//! ### Stress (1 kernel)
//! - `StressTesting` - Scenario-based stress testing

#![warn(missing_docs)]

pub mod correlation;
pub mod credit;
pub mod market;
pub mod messages;
pub mod ring_messages;
pub mod stress;
pub mod types;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::correlation::*;
    pub use crate::credit::*;
    pub use crate::market::*;
    pub use crate::messages::*;
    pub use crate::ring_messages::*;
    pub use crate::stress::*;
    pub use crate::types::*;
}

// Re-export main kernels
pub use correlation::RealTimeCorrelation;
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
    tracing::info!("Registering risk analytics kernels");

    // Credit kernel (1) - Ring
    registry.register_ring_metadata_from(credit::CreditRiskScoring::new)?;

    // Market kernels (3) - Ring
    registry.register_ring_metadata_from(market::MonteCarloVaR::new)?;
    registry.register_ring_metadata_from(market::PortfolioRiskAggregation::new)?;
    registry.register_ring_metadata_from(correlation::RealTimeCorrelation::new)?;

    // Stress kernel (1) - Batch
    registry.register_batch_typed::<StressTesting, messages::StressTestingInput, messages::StressTestingOutput>(stress::StressTesting::new)?;

    tracing::info!("Registered 5 risk analytics kernels");
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
        assert_eq!(registry.total_count(), 5);
    }
}
