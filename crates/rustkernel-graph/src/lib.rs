//! # RustKernel Graph Analytics
//!
//! GPU-accelerated graph analytics kernels including centrality measures,
//! community detection, motif analysis, and similarity metrics.
//!
//! ## Kernels
//!
//! ### Centrality (6 kernels)
//! - `DegreeCentrality` - Ring kernel, O(1) query
//! - `BetweennessCentrality` - Ring kernel, Brandes algorithm
//! - `ClosenessCentrality` - Ring kernel, BFS-based
//! - `EigenvectorCentrality` - Ring kernel, power iteration
//! - `PageRank` - Ring kernel, power iteration with teleport
//! - `KatzCentrality` - Ring kernel, attenuated paths
//!
//! ### Community Detection (2 kernels)
//! - `ModularityScore` - Batch kernel
//! - `LouvainCommunity` - Batch kernel, multi-level optimization
//!
//! ### Motif Detection (2 kernels)
//! - `TriangleCounting` - Ring kernel
//! - `MotifDetection` - Batch kernel, k-node subgraph census
//!
//! ### Similarity (3 kernels)
//! - `JaccardSimilarity` - Batch kernel
//! - `CosineSimilarity` - Batch kernel
//! - `AdamicAdarIndex` - Batch kernel
//!
//! ### Metrics (2 kernels)
//! - `GraphDensity` - Batch kernel
//! - `ClusteringCoefficient` - Batch kernel

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod centrality;
pub mod community;
pub mod messages;
pub mod metrics;
pub mod motif;
pub mod ring_messages;
pub mod similarity;

// Common graph types
pub mod types;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::centrality::*;
    pub use crate::community::*;
    pub use crate::messages::*;
    pub use crate::metrics::*;
    pub use crate::motif::*;
    pub use crate::ring_messages::*;
    pub use crate::similarity::*;
    pub use crate::types::*;
}

/// Register all graph kernels with a registry.
pub fn register_all(
    registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    use rustkernel_core::traits::GpuKernel;

    tracing::info!("Registering graph analytics kernels");

    // Centrality kernels (6)
    registry.register_metadata(centrality::PageRank::new().metadata().clone())?;
    registry.register_metadata(centrality::DegreeCentrality::new().metadata().clone())?;
    registry.register_metadata(centrality::BetweennessCentrality::new().metadata().clone())?;
    registry.register_metadata(centrality::ClosenessCentrality::new().metadata().clone())?;
    registry.register_metadata(centrality::EigenvectorCentrality::new().metadata().clone())?;
    registry.register_metadata(centrality::KatzCentrality::new().metadata().clone())?;

    // Community detection kernels (3)
    registry.register_metadata(community::ModularityScore::new().metadata().clone())?;
    registry.register_metadata(community::LouvainCommunity::new().metadata().clone())?;
    registry.register_metadata(community::LabelPropagation::new().metadata().clone())?;

    // Similarity kernels (4)
    registry.register_metadata(similarity::JaccardSimilarity::new().metadata().clone())?;
    registry.register_metadata(similarity::CosineSimilarity::new().metadata().clone())?;
    registry.register_metadata(similarity::AdamicAdarIndex::new().metadata().clone())?;
    registry.register_metadata(similarity::CommonNeighbors::new().metadata().clone())?;

    // Metrics kernels (5)
    registry.register_metadata(metrics::GraphDensity::new().metadata().clone())?;
    registry.register_metadata(metrics::AveragePathLength::new().metadata().clone())?;
    registry.register_metadata(metrics::ClusteringCoefficient::new().metadata().clone())?;
    registry.register_metadata(metrics::ConnectedComponents::new().metadata().clone())?;
    registry.register_metadata(metrics::FullGraphMetrics::new().metadata().clone())?;

    // Motif detection kernels (3)
    registry.register_metadata(motif::TriangleCounting::new().metadata().clone())?;
    registry.register_metadata(motif::MotifDetection::new().metadata().clone())?;
    registry.register_metadata(motif::KCliqueDetection::new().metadata().clone())?;

    tracing::info!("Registered 21 graph analytics kernels");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustkernel_core::registry::KernelRegistry;

    #[test]
    fn test_register_all() {
        let registry = KernelRegistry::new();
        register_all(&registry).expect("Failed to register graph kernels");
        assert_eq!(registry.total_count(), 21);
    }

    #[test]
    fn test_register_all_by_domain() {
        use rustkernel_core::domain::Domain;

        let registry = KernelRegistry::new();
        register_all(&registry).expect("Failed to register graph kernels");

        let graph_kernels = registry.by_domain(Domain::GraphAnalytics);
        assert_eq!(graph_kernels.len(), 21);
    }
}
