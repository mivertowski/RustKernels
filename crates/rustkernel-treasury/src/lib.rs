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

pub mod types;
pub mod cashflow;
pub mod collateral;
pub mod fx;
pub mod interest_rate;
pub mod liquidity;

pub use cashflow::CashFlowForecasting;
pub use collateral::CollateralOptimization;
pub use fx::FXHedging;
pub use interest_rate::InterestRateRisk;
pub use liquidity::LiquidityOptimization;

/// Register all treasury kernels.
pub fn register_all(_registry: &rustkernel_core::registry::KernelRegistry) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering treasury management kernels");
    Ok(())
}
