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
    pub use crate::similarity::*;
    pub use crate::types::*;
}

/// Register all graph kernels with a registry.
pub fn register_all(registry: &rustkernel_core::registry::KernelRegistry) -> rustkernel_core::error::Result<()> {
    // TODO: Register all kernels
    tracing::info!("Registering graph analytics kernels");
    Ok(())
}
