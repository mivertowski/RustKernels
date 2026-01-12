//! Common ML types and data structures.

use serde::{Deserialize, Serialize};

// ============================================================================
// Data Matrix
// ============================================================================

/// A dense matrix for ML data (row-major storage).
///
/// Each row represents a sample, each column represents a feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataMatrix {
    /// Flat storage of all values (row-major).
    pub data: Vec<f64>,
    /// Number of samples (rows).
    pub n_samples: usize,
    /// Number of features (columns).
    pub n_features: usize,
}

impl DataMatrix {
    /// Create a new data matrix from flat data.
    #[must_use]
    pub fn new(data: Vec<f64>, n_samples: usize, n_features: usize) -> Self {
        assert_eq!(data.len(), n_samples * n_features);
        Self {
            data,
            n_samples,
            n_features,
        }
    }

    /// Create a data matrix from row vectors.
    #[must_use]
    pub fn from_rows(rows: &[&[f64]]) -> Self {
        let n_samples = rows.len();
        let n_features = rows.first().map(|r| r.len()).unwrap_or(0);
        let mut data = Vec::with_capacity(n_samples * n_features);

        for row in rows {
            assert_eq!(row.len(), n_features, "All rows must have same length");
            data.extend_from_slice(row);
        }

        Self {
            data,
            n_samples,
            n_features,
        }
    }

    /// Get a row (sample) as a slice.
    #[must_use]
    pub fn row(&self, idx: usize) -> &[f64] {
        let start = idx * self.n_features;
        &self.data[start..start + self.n_features]
    }

    /// Get a mutable row.
    pub fn row_mut(&mut self, idx: usize) -> &mut [f64] {
        let start = idx * self.n_features;
        let end = start + self.n_features;
        &mut self.data[start..end]
    }

    /// Get element at (row, col).
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.n_features + col]
    }

    /// Set element at (row, col).
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.n_features + col] = value;
    }

    /// Create a zero matrix.
    #[must_use]
    pub fn zeros(n_samples: usize, n_features: usize) -> Self {
        Self {
            data: vec![0.0; n_samples * n_features],
            n_samples,
            n_features,
        }
    }
}

// ============================================================================
// Distance Metrics
// ============================================================================

/// Distance metric for similarity calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Euclidean distance (L2 norm).
    #[default]
    Euclidean,
    /// Manhattan distance (L1 norm).
    Manhattan,
    /// Cosine distance (1 - cosine similarity).
    Cosine,
    /// Chebyshev distance (L∞ norm).
    Chebyshev,
}

impl DistanceMetric {
    /// Compute distance between two vectors.
    #[must_use]
    pub fn compute(&self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt(),
            DistanceMetric::Manhattan => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
            DistanceMetric::Cosine => {
                let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f64 = a.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
                let norm_b: f64 = b.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
                if norm_a == 0.0 || norm_b == 0.0 {
                    1.0
                } else {
                    1.0 - (dot / (norm_a * norm_b))
                }
            }
            DistanceMetric::Chebyshev => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max),
        }
    }
}

// ============================================================================
// Dataset (Legacy/Compatibility)
// ============================================================================

/// Dataset for ML operations (f32 version for compatibility).
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

    /// Convert to DataMatrix (f64).
    #[must_use]
    pub fn to_data_matrix(&self) -> DataMatrix {
        let data: Vec<f64> = self
            .features
            .iter()
            .flat_map(|row| row.iter().map(|&x| x as f64))
            .collect();
        DataMatrix::new(data, self.n_samples, self.n_features)
    }
}

// ============================================================================
// Clustering Result
// ============================================================================

/// Clustering result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringResult {
    /// Cluster assignment per sample.
    pub labels: Vec<usize>,
    /// Number of clusters found.
    pub n_clusters: usize,
    /// Cluster centroids (flattened, row-major).
    pub centroids: Vec<f64>,
    /// Inertia/WCSS (within-cluster sum of squares).
    pub inertia: f64,
    /// Number of iterations run.
    pub iterations: u32,
    /// Whether the algorithm converged.
    pub converged: bool,
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
