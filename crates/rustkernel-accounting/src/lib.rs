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
    tracing::info!("Registering accounting kernels");

    // CoA mapping kernel (1) — Batch
    registry.register_ring_metadata_from(coa_mapping::ChartOfAccountsMapping::new)?;

    // Journal kernel (1) — Batch
    registry.register_ring_metadata_from(journal::JournalTransformation::new)?;

    // Reconciliation kernel (1) — Batch
    registry.register_ring_metadata_from(reconciliation::GLReconciliation::new)?;

    // Network analysis kernel (1) — Batch
    registry.register_ring_metadata_from(network::NetworkAnalysis::new)?;

    // Temporal kernel (1) — Batch
    registry.register_ring_metadata_from(temporal::TemporalCorrelation::new)?;

    // Network generation batch kernel (1) — Batch
    registry.register_batch_typed(network_generation::NetworkGeneration::new)?;

    // Network generation ring kernel (1) — Ring
    registry.register_ring_metadata_from(network_generation::NetworkGenerationRing::new)?;

    // Detection kernels (2) — Batch
    registry.register_ring_metadata_from(detection::SuspenseAccountDetection::new)?;
    registry.register_ring_metadata_from(detection::GaapViolationDetection::new)?;

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
