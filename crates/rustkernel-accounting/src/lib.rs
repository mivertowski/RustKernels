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

#![warn(missing_docs)]

pub mod coa_mapping;
pub mod journal;
pub mod network;
pub mod reconciliation;
pub mod temporal;
pub mod types;

pub use coa_mapping::ChartOfAccountsMapping;
pub use journal::JournalTransformation;
pub use network::NetworkAnalysis;
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

    // Network kernel (1)
    registry.register_metadata(network::NetworkAnalysis::new().metadata().clone())?;

    // Temporal kernel (1)
    registry.register_metadata(temporal::TemporalCorrelation::new().metadata().clone())?;

    tracing::info!("Registered 5 accounting kernels");
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
        assert_eq!(registry.total_count(), 5);
    }
}
