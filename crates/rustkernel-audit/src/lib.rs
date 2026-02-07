//! # RustKernel Financial Audit
//!
//! GPU-accelerated financial audit kernels.
//!
//! ## Kernels
//! - `FeatureExtraction` - Audit feature vector extraction for ML analysis
//! - `HypergraphConstruction` - Multi-way relationship hypergraph construction

#![warn(missing_docs)]

pub mod feature_extraction;
pub mod hypergraph;
pub mod types;

pub use feature_extraction::{FeatureConfig, FeatureExtraction};
pub use hypergraph::{HypergraphConfig, HypergraphConstruction};
pub use types::*;

/// Register all audit kernels.
pub fn register_all(
    registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering financial audit kernels");

    // Feature extraction kernel (1) - Batch
    registry.register_batch_metadata_from(feature_extraction::FeatureExtraction::new)?;

    // Hypergraph kernel (1) - Batch
    registry.register_batch_metadata_from(hypergraph::HypergraphConstruction::new)?;

    tracing::info!("Registered 2 financial audit kernels");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustkernel_core::{domain::Domain, registry::KernelRegistry, traits::GpuKernel};

    #[test]
    fn test_register_all() {
        let registry = KernelRegistry::new();
        register_all(&registry).expect("Failed to register audit kernels");
        assert_eq!(registry.total_count(), 2);
    }

    #[test]
    fn test_feature_extraction_metadata() {
        let kernel = FeatureExtraction::new();
        let metadata = kernel.metadata();
        assert!(metadata.id.contains("feature"));
        assert_eq!(metadata.domain, Domain::FinancialAudit);
    }

    #[test]
    fn test_hypergraph_construction_metadata() {
        let kernel = HypergraphConstruction::new();
        let metadata = kernel.metadata();
        assert!(metadata.id.contains("hypergraph"));
        assert_eq!(metadata.domain, Domain::FinancialAudit);
    }
}
