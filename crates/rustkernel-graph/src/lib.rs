//! # RustKernel Graph Analytics
//!
//! GPU-accelerated graph analytics kernels including centrality measures,
//! community detection, motif analysis, similarity metrics, and AML-focused analytics.
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
//! ### Community Detection (3 kernels)
//! - `ModularityScore` - Batch kernel
//! - `LouvainCommunity` - Batch kernel, multi-level optimization
//! - `LabelPropagation` - Batch kernel
//!
//! ### Motif Detection (3 kernels)
//! - `TriangleCounting` - Ring kernel
//! - `MotifDetection` - Batch kernel, k-node subgraph census
//! - `KCliqueDetection` - Batch kernel
//!
//! ### Similarity (5 kernels)
//! - `JaccardSimilarity` - Batch kernel
//! - `CosineSimilarity` - Batch kernel
//! - `AdamicAdarIndex` - Batch kernel
//! - `CommonNeighbors` - Batch kernel
//! - `ValueSimilarity` - Batch kernel (JSD/Wasserstein)
//!
//! ### Metrics (5 kernels)
//! - `GraphDensity` - Batch kernel
//! - `AveragePathLength` - Batch kernel
//! - `ClusteringCoefficient` - Batch kernel
//! - `ConnectedComponents` - Batch kernel
//! - `FullGraphMetrics` - Batch kernel
//!
//! ### Topology (2 kernels)
//! - `DegreeRatio` - Ring kernel, source/sink classification
//! - `StarTopologyScore` - Batch kernel, hub-and-spoke detection
//!
//! ### Cycles (1 kernel)
//! - `ShortCycleParticipation` - Batch kernel, 2-4 hop cycle detection (AML)
//!
//! ### Paths (1 kernel)
//! - `ShortestPath` - Batch kernel, BFS/Delta-Stepping SSSP/APSP
//!
//! ### Graph Neural Networks (2 kernels)
//! - `GNNInference` - Message passing neural network inference
//! - `GraphAttention` - Graph Attention Network (GAT) with multi-head attention

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod centrality;
pub mod community;
pub mod cycles;
pub mod gnn;
pub mod messages;
pub mod metrics;
pub mod motif;
pub mod paths;
pub mod ring_messages;
pub mod similarity;
pub mod topology;

// Common graph types
pub mod types;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::centrality::*;
    pub use crate::community::*;
    pub use crate::cycles::*;
    pub use crate::gnn::*;
    pub use crate::messages::*;
    pub use crate::metrics::*;
    pub use crate::motif::*;
    pub use crate::paths::*;
    pub use crate::ring_messages::*;
    pub use crate::similarity::*;
    pub use crate::topology::*;
    pub use crate::types::*;
}

/// Register all graph kernels with a registry.
///
/// Batch kernels are registered with factories for direct execution via REST/gRPC.
/// Ring kernels are registered as metadata for discovery (require Ring runtime for execution).
pub fn register_all(
    registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering graph analytics kernels");

    // Centrality kernels (6) — Ring: PageRank, DegreeCentrality; Batch: rest
    registry.register_ring_metadata_from(centrality::PageRank::new)?;
    registry.register_ring_metadata_from(centrality::DegreeCentrality::new)?;
    registry.register_batch_typed(centrality::BetweennessCentrality::new)?;
    registry.register_batch_typed(centrality::ClosenessCentrality::new)?;
    registry.register_batch_typed(centrality::EigenvectorCentrality::new)?;
    registry.register_batch_typed(centrality::KatzCentrality::new)?;

    // Community detection kernels (3) — Batch (GpuKernel only)
    registry.register_batch_metadata_from(community::ModularityScore::new)?;
    registry.register_batch_metadata_from(community::LouvainCommunity::new)?;
    registry.register_batch_metadata_from(community::LabelPropagation::new)?;

    // Similarity kernels (5) — Batch (GpuKernel only)
    registry.register_batch_metadata_from(similarity::JaccardSimilarity::new)?;
    registry.register_batch_metadata_from(similarity::CosineSimilarity::new)?;
    registry.register_batch_metadata_from(similarity::AdamicAdarIndex::new)?;
    registry.register_batch_metadata_from(similarity::CommonNeighbors::new)?;
    registry.register_batch_metadata_from(similarity::ValueSimilarity::new)?;

    // Metrics kernels (5) — Batch (GpuKernel only)
    registry.register_batch_metadata_from(metrics::GraphDensity::new)?;
    registry.register_batch_metadata_from(metrics::AveragePathLength::new)?;
    registry.register_batch_metadata_from(metrics::ClusteringCoefficient::new)?;
    registry.register_batch_metadata_from(metrics::ConnectedComponents::new)?;
    registry.register_batch_metadata_from(metrics::FullGraphMetrics::new)?;

    // Motif detection kernels (3) — Ring: TriangleCounting; Batch: rest
    registry.register_ring_metadata_from(motif::TriangleCounting::new)?;
    registry.register_batch_metadata_from(motif::MotifDetection::new)?;
    registry.register_batch_metadata_from(motif::KCliqueDetection::new)?;

    // Topology kernels (2) — Ring: DegreeRatio; Batch: StarTopologyScore
    registry.register_ring_metadata_from(topology::DegreeRatio::new)?;
    registry.register_batch_metadata_from(topology::StarTopologyScore::new)?;

    // Cycle detection kernels (1) — Batch
    registry.register_batch_metadata_from(cycles::ShortCycleParticipation::new)?;

    // Path kernels (1) — Batch
    registry.register_batch_metadata_from(paths::ShortestPath::new)?;

    // GNN kernels (2) — Batch (GpuKernel only)
    registry.register_batch_metadata_from(gnn::GNNInference::new)?;
    registry.register_batch_metadata_from(gnn::GraphAttention::new)?;

    tracing::info!("Registered 28 graph analytics kernels");
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
        assert_eq!(registry.total_count(), 28);
    }

    #[test]
    fn test_register_all_by_domain() {
        use rustkernel_core::domain::Domain;

        let registry = KernelRegistry::new();
        register_all(&registry).expect("Failed to register graph kernels");

        let graph_kernels = registry.by_domain(Domain::GraphAnalytics);
        assert_eq!(graph_kernels.len(), 28);
    }
}
