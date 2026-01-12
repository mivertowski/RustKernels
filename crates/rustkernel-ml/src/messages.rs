//! Ring message types for ML kernels.
//!
//! This module defines request/response message types for GPU-native
//! persistent actor communication for machine learning algorithms.

use crate::types::{ClusteringResult, DataMatrix, DistanceMetric};
use rustkernel_core::messages::CorrelationId;
use serde::{Deserialize, Serialize};

// ============================================================================
// K-Means Messages
// ============================================================================

/// K-Means clustering input for batch execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansInput {
    /// Input data matrix (n_samples x n_features).
    pub data: DataMatrix,
    /// Number of clusters.
    pub k: usize,
    /// Maximum number of iterations.
    pub max_iterations: u32,
    /// Convergence tolerance for centroid movement.
    pub tolerance: f64,
}

impl KMeansInput {
    /// Create a new K-Means input.
    pub fn new(data: DataMatrix, k: usize) -> Self {
        Self {
            data,
            k,
            max_iterations: 100,
            tolerance: 1e-4,
        }
    }

    /// Set maximum iterations.
    pub fn with_max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set convergence tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }
}

/// K-Means clustering output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansOutput {
    /// The clustering result.
    pub result: ClusteringResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// DBSCAN Messages
// ============================================================================

/// DBSCAN clustering input for batch execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DBSCANInput {
    /// Input data matrix.
    pub data: DataMatrix,
    /// Maximum distance for neighborhood (epsilon).
    pub eps: f64,
    /// Minimum points to form a dense region.
    pub min_samples: usize,
    /// Distance metric to use.
    pub metric: DistanceMetric,
}

impl DBSCANInput {
    /// Create a new DBSCAN input.
    pub fn new(data: DataMatrix, eps: f64, min_samples: usize) -> Self {
        Self {
            data,
            eps,
            min_samples,
            metric: DistanceMetric::Euclidean,
        }
    }

    /// Set the distance metric.
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }
}

/// DBSCAN clustering output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DBSCANOutput {
    /// The clustering result.
    pub result: ClusteringResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Hierarchical Clustering Messages
// ============================================================================

/// Linkage method for hierarchical clustering.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Linkage {
    /// Single linkage (minimum distance).
    Single,
    /// Complete linkage (maximum distance).
    Complete,
    /// Average linkage (UPGMA).
    Average,
    /// Ward's method (minimize variance).
    Ward,
}

/// Hierarchical clustering input for batch execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalInput {
    /// Input data matrix.
    pub data: DataMatrix,
    /// Number of clusters to form.
    pub n_clusters: usize,
    /// Linkage method.
    pub linkage: Linkage,
    /// Distance metric.
    pub metric: DistanceMetric,
}

impl HierarchicalInput {
    /// Create a new hierarchical clustering input.
    pub fn new(data: DataMatrix, n_clusters: usize) -> Self {
        Self {
            data,
            n_clusters,
            linkage: Linkage::Complete,
            metric: DistanceMetric::Euclidean,
        }
    }

    /// Set the linkage method.
    pub fn with_linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }

    /// Set the distance metric.
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }
}

/// Hierarchical clustering output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalOutput {
    /// The clustering result.
    pub result: ClusteringResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Anomaly Detection Messages
// ============================================================================

/// Isolation Forest input for batch execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationForestInput {
    /// Input data matrix.
    pub data: DataMatrix,
    /// Number of trees in the ensemble.
    pub n_trees: usize,
    /// Contamination proportion (expected fraction of outliers).
    pub contamination: f64,
}

impl IsolationForestInput {
    /// Create a new Isolation Forest input.
    pub fn new(data: DataMatrix) -> Self {
        Self {
            data,
            n_trees: 100,
            contamination: 0.1,
        }
    }

    /// Set the number of trees.
    pub fn with_n_trees(mut self, n_trees: usize) -> Self {
        self.n_trees = n_trees;
        self
    }

    /// Set the contamination proportion.
    pub fn with_contamination(mut self, contamination: f64) -> Self {
        self.contamination = contamination;
        self
    }
}

/// Anomaly detection output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyOutput {
    /// Anomaly scores for each sample (higher = more anomalous).
    pub scores: Vec<f64>,
    /// Labels (1 = anomaly, 0 = normal) based on threshold.
    pub labels: Vec<i32>,
    /// The threshold used for classification.
    pub threshold: f64,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

/// Local Outlier Factor input for batch execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LOFInput {
    /// Input data matrix.
    pub data: DataMatrix,
    /// Number of neighbors.
    pub n_neighbors: usize,
    /// Contamination proportion.
    pub contamination: f64,
    /// Distance metric.
    pub metric: DistanceMetric,
}

impl LOFInput {
    /// Create a new LOF input.
    pub fn new(data: DataMatrix) -> Self {
        Self {
            data,
            n_neighbors: 20,
            contamination: 0.1,
            metric: DistanceMetric::Euclidean,
        }
    }

    /// Set the number of neighbors.
    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }
}

// ============================================================================
// Regression Messages
// ============================================================================

/// Regression input for batch execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionInput {
    /// Feature matrix X (n_samples x n_features).
    pub x: DataMatrix,
    /// Target vector y (n_samples).
    pub y: Vec<f64>,
    /// Whether to fit intercept.
    pub fit_intercept: bool,
    /// Regularization parameter (for Ridge regression).
    pub alpha: Option<f64>,
}

impl RegressionInput {
    /// Create a new regression input.
    pub fn new(x: DataMatrix, y: Vec<f64>) -> Self {
        Self {
            x,
            y,
            fit_intercept: true,
            alpha: None,
        }
    }

    /// Enable Ridge regularization with given alpha.
    pub fn with_ridge(mut self, alpha: f64) -> Self {
        self.alpha = Some(alpha);
        self
    }
}

/// Regression output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionOutput {
    /// Coefficient vector.
    pub coefficients: Vec<f64>,
    /// Intercept (if fit_intercept was true).
    pub intercept: Option<f64>,
    /// R-squared score.
    pub r_squared: f64,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_input_builder() {
        let data = DataMatrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let input = KMeansInput::new(data, 2).with_max_iterations(50).with_tolerance(1e-6);
        assert_eq!(input.k, 2);
        assert_eq!(input.max_iterations, 50);
    }

    #[test]
    fn test_dbscan_input_builder() {
        let data = DataMatrix::from_rows(&[&[1.0, 2.0]]);
        let input = DBSCANInput::new(data, 0.5, 3).with_metric(DistanceMetric::Manhattan);
        assert_eq!(input.eps, 0.5);
        assert_eq!(input.min_samples, 3);
    }
}
