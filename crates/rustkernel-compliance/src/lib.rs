//! # RustKernel Compliance
//!
//! GPU-accelerated compliance kernels for AML, KYC, sanctions screening, and transaction monitoring.
//!
//! ## Kernels
//!
//! ### AML (6 kernels)
//! - `CircularFlowRatio` - SCC detection for circular transactions
//! - `ReciprocityFlowRatio` - Mutual transaction detection
//! - `RapidMovement` - Velocity analysis for structuring
//! - `AMLPatternDetection` - Multi-pattern FSM detection
//! - `FlowReversalPattern` - Transaction reversal detection (wash trading, round-tripping)
//! - `FlowSplitRatio` - Transaction splitting/structuring detection
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
pub use aml::{
    AMLPatternDetection, CircularFlowRatio, FlowReversalConfig, FlowReversalPattern,
    FlowSplitConfig, FlowSplitRatio, RapidMovement, ReciprocityFlowRatio,
};
pub use kyc::{EntityResolution, KYCScoring};
pub use monitoring::TransactionMonitoring;
pub use sanctions::{PEPScreening, SanctionsScreening};

/// Register all compliance kernels with a registry.
pub fn register_all(
    registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering compliance kernels");

    // AML kernels (6)
    registry.register_ring_metadata_from(aml::CircularFlowRatio::new)?;
    registry.register_ring_metadata_from(aml::ReciprocityFlowRatio::new)?;
    registry.register_ring_metadata_from(aml::RapidMovement::new)?;
    registry.register_ring_metadata_from(aml::AMLPatternDetection::new)?;
    registry.register_batch_metadata_from(aml::FlowReversalPattern::new)?;
    registry.register_batch_metadata_from(aml::FlowSplitRatio::new)?;

    // KYC kernels (2)
    registry.register_batch_typed(kyc::KYCScoring::new)?;
    registry.register_batch_typed(kyc::EntityResolution::new)?;

    // Sanctions kernels (2)
    registry.register_ring_metadata_from(sanctions::SanctionsScreening::new)?;
    registry.register_ring_metadata_from(sanctions::PEPScreening::new)?;

    // Monitoring kernel (1)
    registry.register_ring_metadata_from(monitoring::TransactionMonitoring::new)?;

    tracing::info!("Registered 11 compliance kernels");
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
        assert_eq!(registry.total_count(), 11);
    }
}
