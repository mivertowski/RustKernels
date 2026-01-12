//! # RustKernel Statistical ML
//!
//! GPU-accelerated machine learning kernels for clustering, anomaly detection, and regression.
//!
//! ## Kernels
//!
//! ### Clustering (3 kernels)
//! - `KMeans` - Lloyd's algorithm with K-Means++ initialization
//! - `DBSCAN` - Density-based clustering with GPU union-find
//! - `HierarchicalClustering` - Agglomerative clustering
//!
//! ### Anomaly Detection (2 kernels)
//! - `IsolationForest` - Ensemble of isolation trees
//! - `LocalOutlierFactor` - k-NN density estimation
//!
//! ### Ensemble (1 kernel)
//! - `EnsembleVoting` - Weighted majority voting
//!
//! ### Regression (2 kernels)
//! - `LinearRegression` - OLS via normal equations
//! - `RidgeRegression` - L2 regularization

#![warn(missing_docs)]

pub mod anomaly;
pub mod clustering;
pub mod ensemble;
pub mod messages;
pub mod regression;
pub mod types;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::anomaly::*;
    pub use crate::clustering::*;
    pub use crate::ensemble::*;
    pub use crate::messages::*;
    pub use crate::regression::*;
    pub use crate::types::*;
}

/// Register all ML kernels with a registry.
pub fn register_all(registry: &rustkernel_core::registry::KernelRegistry) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering statistical ML kernels");
    Ok(())
}
