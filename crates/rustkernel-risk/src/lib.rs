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
    CreditExposure, CreditFactors, CreditRiskResult, Portfolio, PortfolioRiskResult,
    RiskFactor, RiskFactorType, Sensitivity, StressScenario, StressTestResult, VaRParams,
    VaRResult,
};

/// Register all risk kernels with a registry.
pub fn register_all(
    _registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering risk analytics kernels");
    Ok(())
}
