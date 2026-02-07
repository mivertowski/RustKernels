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

pub mod cashflow;
pub mod collateral;
pub mod fx;
pub mod interest_rate;
pub mod liquidity;
pub mod types;

pub use cashflow::CashFlowForecasting;
pub use collateral::CollateralOptimization;
pub use fx::FXHedging;
pub use interest_rate::InterestRateRisk;
pub use liquidity::LiquidityOptimization;

/// Register all treasury kernels.
pub fn register_all(
    registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering treasury management kernels");

    // Cash flow kernel (1) — Batch
    registry.register_ring_metadata_from(cashflow::CashFlowForecasting::new)?;

    // Collateral kernel (1) — Batch
    registry.register_ring_metadata_from(collateral::CollateralOptimization::new)?;

    // FX kernel (1) — Batch
    registry.register_ring_metadata_from(fx::FXHedging::new)?;

    // Interest rate kernel (1) — Batch
    registry.register_ring_metadata_from(interest_rate::InterestRateRisk::new)?;

    // Liquidity kernel (1) — Batch
    registry.register_ring_metadata_from(liquidity::LiquidityOptimization::new)?;

    tracing::info!("Registered 5 treasury management kernels");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustkernel_core::registry::KernelRegistry;

    #[test]
    fn test_register_all() {
        let registry = KernelRegistry::new();
        register_all(&registry).expect("Failed to register treasury kernels");
        assert_eq!(registry.total_count(), 5);
    }
}
