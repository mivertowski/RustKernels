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
//! ### Streaming Anomaly Detection (2 kernels)
//! - `StreamingIsolationForest` - Online anomaly detection with sliding window
//! - `AdaptiveThreshold` - Self-adjusting thresholds with drift detection
//!
//! ### Ensemble (1 kernel)
//! - `EnsembleVoting` - Weighted majority voting
//!
//! ### Regression (2 kernels)
//! - `LinearRegression` - OLS via normal equations
//! - `RidgeRegression` - L2 regularization
//!
//! ### Explainability (2 kernels)
//! - `SHAPValues` - Kernel SHAP for feature explanations
//! - `FeatureImportance` - Permutation-based feature importance
//!
//! ### NLP / LLM Integration (2 kernels)
//! - `EmbeddingGeneration` - Text embedding via hash-based features
//! - `SemanticSimilarity` - Multi-metric semantic similarity
//!
//! ### Federated Learning (1 kernel)
//! - `SecureAggregation` - Privacy-preserving model aggregation
//!
//! ### Healthcare Analytics (2 kernels)
//! - `DrugInteractionPrediction` - Multi-drug interaction prediction
//! - `ClinicalPathwayConformance` - Treatment guideline checking

#![warn(missing_docs)]

pub mod anomaly;
pub mod clustering;
pub mod ensemble;
pub mod explainability;
pub mod federated;
pub mod healthcare;
pub mod messages;
pub mod nlp;
pub mod regression;
pub mod ring_messages;
pub mod streaming;
pub mod types;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::anomaly::*;
    pub use crate::clustering::*;
    pub use crate::ensemble::*;
    pub use crate::explainability::*;
    pub use crate::federated::*;
    pub use crate::healthcare::*;
    pub use crate::messages::*;
    pub use crate::nlp::*;
    pub use crate::regression::*;
    pub use crate::ring_messages::*;
    pub use crate::streaming::*;
    pub use crate::types::*;
}

/// Register all ML kernels with a registry.
pub fn register_all(
    registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering statistical ML kernels");

    // Clustering kernels (3) - implement BatchKernel<I, O>
    registry.register_batch_typed(clustering::KMeans::new)?;
    registry.register_batch_typed(clustering::DBSCAN::new)?;
    registry.register_batch_typed(clustering::HierarchicalClustering::new)?;

    // Anomaly detection kernels (2)
    registry.register_batch_metadata_from(anomaly::IsolationForest::new)?;
    registry.register_batch_metadata_from(anomaly::LocalOutlierFactor::new)?;

    // Streaming anomaly detection kernels (2)
    registry.register_batch_metadata_from(streaming::StreamingIsolationForest::new)?;
    registry.register_batch_metadata_from(streaming::AdaptiveThreshold::new)?;

    // Ensemble kernel (1)
    registry.register_batch_metadata_from(ensemble::EnsembleVoting::new)?;

    // Regression kernels (2)
    registry.register_batch_metadata_from(regression::LinearRegression::new)?;
    registry.register_batch_metadata_from(regression::RidgeRegression::new)?;

    // Explainability kernels (2)
    registry.register_batch_metadata_from(explainability::SHAPValues::new)?;
    registry.register_batch_metadata_from(explainability::FeatureImportance::new)?;

    // NLP / LLM Integration kernels (2)
    registry.register_batch_metadata_from(nlp::EmbeddingGeneration::new)?;
    registry.register_batch_metadata_from(nlp::SemanticSimilarity::new)?;

    // Federated Learning kernels (1)
    registry.register_batch_metadata_from(federated::SecureAggregation::new)?;

    // Healthcare Analytics kernels (2)
    registry.register_batch_metadata_from(healthcare::DrugInteractionPrediction::new)?;
    registry.register_batch_metadata_from(healthcare::ClinicalPathwayConformance::new)?;

    tracing::info!("Registered 17 statistical ML kernels");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustkernel_core::registry::KernelRegistry;

    #[test]
    fn test_register_all() {
        let registry = KernelRegistry::new();
        register_all(&registry).expect("Failed to register ML kernels");
        assert_eq!(registry.total_count(), 17);
    }
}
