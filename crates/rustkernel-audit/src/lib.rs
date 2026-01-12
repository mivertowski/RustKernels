//! # RustKernel Financial Audit
//!
//! GPU-accelerated financial audit kernels.
//!
//! ## Kernels
//! - `FeatureExtraction` - Audit feature vector extraction for ML analysis
//! - `HypergraphConstruction` - Multi-way relationship hypergraph construction

#![warn(missing_docs)]

pub mod types;
pub mod feature_extraction;
pub mod hypergraph;

pub use types::*;
pub use feature_extraction::{FeatureExtraction, FeatureConfig};
pub use hypergraph::{HypergraphConstruction, HypergraphConfig};

/// Register all audit kernels.
pub fn register_all(_registry: &rustkernel_core::registry::KernelRegistry) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering financial audit kernels");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustkernel_core::{domain::Domain, traits::GpuKernel};

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
