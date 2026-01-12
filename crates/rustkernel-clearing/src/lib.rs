//! # RustKernel Clearing
//!
//! GPU-accelerated clearing and settlement kernels.
//!
//! ## Kernels
//! - `ClearingValidation` - Trade validation for clearing eligibility
//! - `DVPMatching` - Delivery vs payment matching
//! - `NettingCalculation` - Multilateral netting calculation
//! - `SettlementExecution` - Settlement instruction execution
//! - `ZeroBalanceFrequency` - Settlement efficiency metrics
//!
//! ## Features
//! - Trade validation with counterparty/security eligibility checks
//! - DVP instruction matching with tolerance-based scoring
//! - Multilateral netting to reduce gross obligations
//! - Settlement execution with priority and partial settlement support
//! - Zero balance frequency and efficiency metrics

#![warn(missing_docs)]

pub mod dvp;
pub mod efficiency;
pub mod netting;
pub mod settlement;
pub mod types;
pub mod validation;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::dvp::*;
    pub use crate::efficiency::*;
    pub use crate::netting::*;
    pub use crate::settlement::*;
    pub use crate::types::*;
    pub use crate::validation::*;
}

// Re-export main kernels
pub use dvp::DVPMatching;
pub use efficiency::ZeroBalanceFrequency;
pub use netting::NettingCalculation;
pub use settlement::SettlementExecution;
pub use validation::ClearingValidation;

// Re-export key types
pub use types::{
    DVPInstruction, DVPMatchResult, DVPStatus, ErrorSeverity, InstructionType, NetPosition,
    NettingConfig, NettingResult, SettlementEfficiency, SettlementExecutionResult,
    SettlementInstruction, SettlementStatus, Trade, TradeStatus, TradeType, ValidationConfig,
    ValidationError, ValidationResult, ZeroBalanceMetrics,
};

/// Register all clearing kernels with a registry.
pub fn register_all(
    registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    use rustkernel_core::traits::GpuKernel;

    tracing::info!("Registering clearing kernels");

    // Validation kernel (1)
    registry.register_metadata(validation::ClearingValidation::new().metadata().clone())?;

    // DVP kernel (1)
    registry.register_metadata(dvp::DVPMatching::new().metadata().clone())?;

    // Netting kernel (1)
    registry.register_metadata(netting::NettingCalculation::new().metadata().clone())?;

    // Settlement kernel (1)
    registry.register_metadata(settlement::SettlementExecution::new().metadata().clone())?;

    // Efficiency kernel (1)
    registry.register_metadata(efficiency::ZeroBalanceFrequency::new().metadata().clone())?;

    tracing::info!("Registered 5 clearing kernels");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustkernel_core::registry::KernelRegistry;

    #[test]
    fn test_register_all() {
        let registry = KernelRegistry::new();
        register_all(&registry).expect("Failed to register clearing kernels");
        assert_eq!(registry.total_count(), 5);
    }
}
