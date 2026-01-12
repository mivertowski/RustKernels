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

pub mod aml;
pub mod kyc;
pub mod messages;
pub mod monitoring;
pub mod ring_messages;
pub mod sanctions;
pub mod types;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::aml::*;
    pub use crate::kyc::*;
    pub use crate::messages::*;
    pub use crate::monitoring::*;
    pub use crate::ring_messages::*;
    pub use crate::sanctions::*;
    pub use crate::types::*;
}

// Re-export main types for convenience
pub use aml::{AMLPatternDetection, CircularFlowRatio, RapidMovement, ReciprocityFlowRatio};
pub use kyc::{EntityResolution, KYCScoring};
pub use monitoring::TransactionMonitoring;
pub use sanctions::{PEPScreening, SanctionsScreening};

/// Register all compliance kernels with a registry.
pub fn register_all(
    registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    use rustkernel_core::traits::GpuKernel;

    tracing::info!("Registering compliance kernels");

    // AML kernels (4)
    registry.register_metadata(aml::CircularFlowRatio::new().metadata().clone())?;
    registry.register_metadata(aml::ReciprocityFlowRatio::new().metadata().clone())?;
    registry.register_metadata(aml::RapidMovement::new().metadata().clone())?;
    registry.register_metadata(aml::AMLPatternDetection::new().metadata().clone())?;

    // KYC kernels (2)
    registry.register_metadata(kyc::KYCScoring::new().metadata().clone())?;
    registry.register_metadata(kyc::EntityResolution::new().metadata().clone())?;

    // Sanctions kernels (2)
    registry.register_metadata(sanctions::SanctionsScreening::new().metadata().clone())?;
    registry.register_metadata(sanctions::PEPScreening::new().metadata().clone())?;

    // Monitoring kernel (1)
    registry.register_metadata(monitoring::TransactionMonitoring::new().metadata().clone())?;

    tracing::info!("Registered 9 compliance kernels");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustkernel_core::registry::KernelRegistry;

    #[test]
    fn test_register_all() {
        let registry = KernelRegistry::new();
        register_all(&registry).expect("Failed to register compliance kernels");
        assert_eq!(registry.total_count(), 9);
    }
}
