//! # RustKernel Accounting
//!
//! GPU-accelerated accounting kernels.
//!
//! ## Kernels
//! - `ChartOfAccountsMapping` - Entity-specific CoA mapping
//! - `JournalTransformation` - GL mapping
//! - `GLReconciliation` - Account matching
//! - `NetworkAnalysis` - Intercompany analysis
//! - `TemporalCorrelation` - Account correlations
//! - `NetworkGeneration` - Journal entry to accounting network transformation
//! - `NetworkGenerationRing` - Streaming network generation
//! - `SuspenseAccountDetection` - Centrality-based suspense account detection
//! - `GaapViolationDetection` - GAAP prohibited flow pattern detection

#![warn(missing_docs)]

pub mod coa_mapping;
pub mod detection;
pub mod journal;
pub mod network;
pub mod network_generation;
pub mod reconciliation;
pub mod temporal;
pub mod types;

pub use coa_mapping::ChartOfAccountsMapping;
pub use detection::{
    GaapDetectionConfig, GaapViolationDetection, SuspenseAccountDetection, SuspenseDetectionConfig,
};
pub use journal::JournalTransformation;
pub use network::NetworkAnalysis;
pub use network_generation::{
    AccountingFlow, AccountingNetwork, FixedPoint128, NetworkGeneration, NetworkGenerationConfig,
    NetworkGenerationRing, NetworkGenerationStats, SolvingMethod,
};
pub use reconciliation::GLReconciliation;
pub use temporal::TemporalCorrelation;

/// Register all accounting kernels.
pub fn register_all(
    registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    use rustkernel_core::traits::GpuKernel;

    tracing::info!("Registering accounting kernels");

    // CoA mapping kernel (1)
    registry.register_metadata(
        coa_mapping::ChartOfAccountsMapping::new()
            .metadata()
            .clone(),
    )?;

    // Journal kernel (1)
    registry.register_metadata(journal::JournalTransformation::new().metadata().clone())?;

    // Reconciliation kernel (1)
    registry.register_metadata(reconciliation::GLReconciliation::new().metadata().clone())?;

    // Network analysis kernel (1)
    registry.register_metadata(network::NetworkAnalysis::new().metadata().clone())?;

    // Temporal kernel (1)
    registry.register_metadata(temporal::TemporalCorrelation::new().metadata().clone())?;

    // Network generation batch kernel (1)
    registry.register_metadata(
        network_generation::NetworkGeneration::new()
            .metadata()
            .clone(),
    )?;

    // Network generation ring kernel (1)
    registry.register_metadata(
        network_generation::NetworkGenerationRing::new()
            .metadata()
            .clone(),
    )?;

    // Detection kernels (2)
    registry.register_metadata(
        detection::SuspenseAccountDetection::new()
            .metadata()
            .clone(),
    )?;
    registry.register_metadata(
        detection::GaapViolationDetection::new()
            .metadata()
            .clone(),
    )?;

    tracing::info!("Registered 9 accounting kernels");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustkernel_core::registry::KernelRegistry;

    #[test]
    fn test_register_all() {
        let registry = KernelRegistry::new();
        register_all(&registry).expect("Failed to register accounting kernels");
        assert_eq!(registry.total_count(), 9);
    }
}
