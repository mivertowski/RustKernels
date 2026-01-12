//! Common ML types and data structures.

use serde::{Deserialize, Serialize};

/// Dataset for ML operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    /// Feature matrix (rows: samples, cols: features).
    pub features: Vec<Vec<f32>>,
    /// Number of samples.
    pub n_samples: usize,
    /// Number of features.
    pub n_features: usize,
    /// Optional labels.
    pub labels: Option<Vec<i32>>,
}

impl Dataset {
    /// Create a new dataset.
    #[must_use]
    pub fn new(features: Vec<Vec<f32>>) -> Self {
        let n_samples = features.len();
        let n_features = features.first().map(|f| f.len()).unwrap_or(0);
        Self {
            features,
            n_samples,
            n_features,
            labels: None,
        }
    }
}

/// Clustering result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringResult {
    /// Cluster assignment per sample.
    pub labels: Vec<i32>,
    /// Cluster centroids.
    pub centroids: Option<Vec<Vec<f32>>>,
    /// Number of clusters.
    pub n_clusters: usize,
    /// Inertia/WCSS score.
    pub inertia: Option<f64>,
    /// Silhouette score.
    pub silhouette: Option<f64>,
}

/// Anomaly detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    /// Anomaly scores per sample.
    pub scores: Vec<f64>,
    /// Binary labels (-1 for anomaly, 1 for normal).
    pub labels: Vec<i32>,
    /// Threshold used for classification.
    pub threshold: f64,
}

/// Regression result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    /// Model coefficients.
    pub coefficients: Vec<f64>,
    /// Intercept.
    pub intercept: f64,
    /// R² score.
    pub r2_score: f64,
}
