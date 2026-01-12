//! Explainability kernels for model interpretation.
//!
//! This module provides GPU-accelerated explainability algorithms:
//! - SHAPValues - Kernel SHAP approximation for feature importance
//! - FeatureImportance - Permutation-based feature importance

use crate::types::DataMatrix;
use rand::prelude::*;
use rand::{rng, Rng, SeedableRng};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use serde::{Deserialize, Serialize};

// ============================================================================
// SHAP Values Kernel
// ============================================================================

/// Configuration for SHAP computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SHAPConfig {
    /// Number of samples for approximation.
    pub n_samples: usize,
    /// Whether to use kernel SHAP (vs sampling SHAP).
    pub use_kernel_shap: bool,
    /// Regularization for weighted least squares.
    pub regularization: f64,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for SHAPConfig {
    fn default() -> Self {
        Self {
            n_samples: 100,
            use_kernel_shap: true,
            regularization: 0.01,
            seed: None,
        }
    }
}

/// SHAP explanation for a single prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SHAPExplanation {
    /// Base value (expected prediction over training data).
    pub base_value: f64,
    /// SHAP values for each feature.
    pub shap_values: Vec<f64>,
    /// Feature names if provided.
    pub feature_names: Option<Vec<String>>,
    /// The prediction being explained.
    pub prediction: f64,
    /// Sum of SHAP values (should equal prediction - base_value).
    pub shap_sum: f64,
}

/// Batch SHAP results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SHAPBatchResult {
    /// Base value.
    pub base_value: f64,
    /// SHAP values matrix (samples x features).
    pub shap_values: Vec<Vec<f64>>,
    /// Feature names.
    pub feature_names: Option<Vec<String>>,
    /// Mean absolute SHAP values per feature.
    pub feature_importance: Vec<f64>,
}

/// SHAP Values kernel.
///
/// Computes SHAP (SHapley Additive exPlanations) values for model predictions.
/// Uses Kernel SHAP approximation which is model-agnostic and works with any
/// prediction function.
///
/// SHAP values satisfy:
/// - Local accuracy: f(x) = base_value + sum(shap_values)
/// - Missingness: Missing features have 0 contribution
/// - Consistency: If a feature's contribution increases, its SHAP value increases
#[derive(Debug, Clone)]
pub struct SHAPValues {
    metadata: KernelMetadata,
}

impl Default for SHAPValues {
    fn default() -> Self {
        Self::new()
    }
}

impl SHAPValues {
    /// Create a new SHAP Values kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/shap-values", Domain::StatisticalML)
                .with_description("Kernel SHAP for model-agnostic feature explanations")
                .with_throughput(1_000)
                .with_latency_us(500.0),
        }
    }

    /// Compute SHAP values for a single instance.
    ///
    /// # Arguments
    /// * `instance` - The instance to explain
    /// * `background` - Background dataset for baseline
    /// * `predict_fn` - Model prediction function
    /// * `config` - SHAP configuration
    pub fn explain<F>(
        instance: &[f64],
        background: &DataMatrix,
        predict_fn: F,
        config: &SHAPConfig,
    ) -> SHAPExplanation
    where
        F: Fn(&[f64]) -> f64,
    {
        let n_features = instance.len();

        if n_features == 0 || background.n_samples == 0 {
            return SHAPExplanation {
                base_value: 0.0,
                shap_values: Vec::new(),
                feature_names: None,
                prediction: 0.0,
                shap_sum: 0.0,
            };
        }

        // Compute base value as expected prediction over background
        let base_value: f64 = (0..background.n_samples)
            .map(|i| predict_fn(background.row(i)))
            .sum::<f64>()
            / background.n_samples as f64;

        let prediction = predict_fn(instance);

        // Use Kernel SHAP
        let shap_values = if config.use_kernel_shap {
            Self::kernel_shap(instance, background, &predict_fn, config)
        } else {
            Self::sampling_shap(instance, background, &predict_fn, config)
        };

        let shap_sum: f64 = shap_values.iter().sum();

        SHAPExplanation {
            base_value,
            shap_values,
            feature_names: None,
            prediction,
            shap_sum,
        }
    }

    /// Kernel SHAP implementation using weighted linear regression.
    fn kernel_shap<F>(
        instance: &[f64],
        background: &DataMatrix,
        predict_fn: &F,
        config: &SHAPConfig,
    ) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n_features = instance.len();
        let n_samples = config.n_samples;

        let mut rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut rng()),
        };

        // Generate coalition samples
        let mut coalitions: Vec<Vec<bool>> = Vec::with_capacity(n_samples);
        let mut predictions: Vec<f64> = Vec::with_capacity(n_samples);
        let mut weights: Vec<f64> = Vec::with_capacity(n_samples);

        // Always include full and empty coalitions
        coalitions.push(vec![true; n_features]);
        coalitions.push(vec![false; n_features]);

        for coalition in &coalitions[..2] {
            let masked = Self::create_masked_instance(instance, background, coalition, &mut rng);
            predictions.push(predict_fn(&masked));
        }

        weights.push(1e6); // High weight for full coalition
        weights.push(1e6); // High weight for empty coalition

        // Sample random coalitions
        for _ in 2..n_samples {
            let coalition: Vec<bool> = (0..n_features).map(|_| rng.gen_bool(0.5)).collect();

            let z: usize = coalition.iter().filter(|&&b| b).count();
            let weight = Self::kernel_shap_weight(n_features, z);

            let masked = Self::create_masked_instance(instance, background, &coalition, &mut rng);
            let pred = predict_fn(&masked);

            coalitions.push(coalition);
            predictions.push(pred);
            weights.push(weight);
        }

        // Solve weighted least squares: (X^T W X + λI)^-1 X^T W y
        Self::solve_weighted_regression(&coalitions, &predictions, &weights, config.regularization)
    }

    /// Sampling SHAP implementation (simpler, faster, less accurate).
    fn sampling_shap<F>(
        instance: &[f64],
        background: &DataMatrix,
        predict_fn: &F,
        config: &SHAPConfig,
    ) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n_features = instance.len();
        let mut shap_values = vec![0.0; n_features];
        let samples_per_feature = config.n_samples / n_features;

        let mut rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut rng()),
        };

        for feature_idx in 0..n_features {
            let mut contributions = Vec::with_capacity(samples_per_feature);

            for _ in 0..samples_per_feature {
                // Random permutation
                let mut perm: Vec<usize> = (0..n_features).collect();
                perm.shuffle(&mut rng);

                let feature_pos = perm.iter().position(|&i| i == feature_idx).unwrap();

                // Features before this one in permutation
                let before: Vec<bool> = (0..n_features)
                    .map(|i| {
                        let pos = perm.iter().position(|&p| p == i).unwrap();
                        pos < feature_pos
                    })
                    .collect();

                // Include current feature
                let mut with_feature = before.clone();
                with_feature[feature_idx] = true;

                // Sample background
                let bg_idx = rng.random_range(0..background.n_samples);
                let bg = background.row(bg_idx);

                // Create masked instances
                let x_with: Vec<f64> = (0..n_features)
                    .map(|i| {
                        if with_feature[i] {
                            instance[i]
                        } else {
                            bg[i]
                        }
                    })
                    .collect();

                let x_without: Vec<f64> = (0..n_features)
                    .map(|i| if before[i] { instance[i] } else { bg[i] })
                    .collect();

                let contribution = predict_fn(&x_with) - predict_fn(&x_without);
                contributions.push(contribution);
            }

            shap_values[feature_idx] =
                contributions.iter().sum::<f64>() / contributions.len() as f64;
        }

        shap_values
    }

    /// Kernel SHAP weight function.
    fn kernel_shap_weight(n_features: usize, coalition_size: usize) -> f64 {
        if coalition_size == 0 || coalition_size == n_features {
            return 1e6; // Very high weight for full/empty coalitions
        }

        let m = n_features as f64;
        let z = coalition_size as f64;

        // SHAP kernel weight: (M-1) / (C(M,z) * z * (M-z))
        let binomial = Self::binomial(n_features, coalition_size);
        if binomial == 0.0 {
            return 0.0;
        }

        (m - 1.0) / (binomial * z * (m - z))
    }

    /// Binomial coefficient.
    fn binomial(n: usize, k: usize) -> f64 {
        if k > n {
            return 0.0;
        }
        let k = k.min(n - k);
        let mut result = 1.0;
        for i in 0..k {
            result *= (n - i) as f64 / (i + 1) as f64;
        }
        result
    }

    /// Create masked instance using background data.
    fn create_masked_instance(
        instance: &[f64],
        background: &DataMatrix,
        coalition: &[bool],
        rng: &mut StdRng,
    ) -> Vec<f64> {
        let bg_idx = rng.random_range(0..background.n_samples);
        let bg = background.row(bg_idx);

        coalition
            .iter()
            .enumerate()
            .map(|(i, &included)| {
                if included {
                    instance[i]
                } else {
                    bg[i]
                }
            })
            .collect()
    }

    /// Solve weighted least squares regression.
    fn solve_weighted_regression(
        coalitions: &[Vec<bool>],
        predictions: &[f64],
        weights: &[f64],
        regularization: f64,
    ) -> Vec<f64> {
        if coalitions.is_empty() {
            return Vec::new();
        }

        let n_features = coalitions[0].len();
        let n_samples = coalitions.len();

        // Build design matrix X (coalitions as 0/1)
        let mut x: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
        for coalition in coalitions {
            let row: Vec<f64> = coalition.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
            x.push(row);
        }

        // Compute X^T W X
        let mut xtw_x = vec![vec![0.0; n_features]; n_features];
        for i in 0..n_features {
            for j in 0..n_features {
                for k in 0..n_samples {
                    xtw_x[i][j] += x[k][i] * weights[k] * x[k][j];
                }
            }
        }

        // Add regularization
        for i in 0..n_features {
            xtw_x[i][i] += regularization;
        }

        // Compute X^T W y
        let mut xtw_y = vec![0.0; n_features];
        for i in 0..n_features {
            for k in 0..n_samples {
                xtw_y[i] += x[k][i] * weights[k] * predictions[k];
            }
        }

        // Solve using simple Cholesky-like approach
        Self::solve_linear_system(&xtw_x, &xtw_y)
    }

    /// Simple linear system solver.
    fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = b.len();
        if n == 0 {
            return Vec::new();
        }

        // Gaussian elimination with partial pivoting
        let mut aug: Vec<Vec<f64>> = a
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut new_row = row.clone();
                new_row.push(b[i]);
                new_row
            })
            .collect();

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_idx = i;
            let mut max_val = aug[i][i].abs();
            for k in (i + 1)..n {
                if aug[k][i].abs() > max_val {
                    max_val = aug[k][i].abs();
                    max_idx = k;
                }
            }

            aug.swap(i, max_idx);

            if aug[i][i].abs() < 1e-10 {
                continue;
            }

            for k in (i + 1)..n {
                let factor = aug[k][i] / aug[i][i];
                for j in i..=n {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            if aug[i][i].abs() < 1e-10 {
                x[i] = 0.0;
                continue;
            }
            x[i] = aug[i][n];
            for j in (i + 1)..n {
                x[i] -= aug[i][j] * x[j];
            }
            x[i] /= aug[i][i];
        }

        x
    }

    /// Explain multiple instances.
    pub fn explain_batch<F>(
        instances: &DataMatrix,
        background: &DataMatrix,
        predict_fn: F,
        config: &SHAPConfig,
        feature_names: Option<Vec<String>>,
    ) -> SHAPBatchResult
    where
        F: Fn(&[f64]) -> f64,
    {
        if instances.n_samples == 0 {
            return SHAPBatchResult {
                base_value: 0.0,
                shap_values: Vec::new(),
                feature_names: None,
                feature_importance: Vec::new(),
            };
        }

        // Compute base value
        let base_value: f64 = (0..background.n_samples)
            .map(|i| predict_fn(background.row(i)))
            .sum::<f64>()
            / background.n_samples.max(1) as f64;

        // Compute SHAP values for each instance
        let mut shap_values: Vec<Vec<f64>> = Vec::with_capacity(instances.n_samples);

        for i in 0..instances.n_samples {
            let instance = instances.row(i);
            let explanation = Self::explain(instance, background, &predict_fn, config);
            shap_values.push(explanation.shap_values);
        }

        // Compute feature importance as mean absolute SHAP values
        let n_features = instances.n_features;
        let mut feature_importance = vec![0.0; n_features];

        for values in &shap_values {
            for (i, &v) in values.iter().enumerate() {
                feature_importance[i] += v.abs();
            }
        }

        for imp in &mut feature_importance {
            *imp /= shap_values.len() as f64;
        }

        SHAPBatchResult {
            base_value,
            shap_values,
            feature_names,
            feature_importance,
        }
    }
}

impl GpuKernel for SHAPValues {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Feature Importance Kernel
// ============================================================================

/// Configuration for permutation feature importance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportanceConfig {
    /// Number of permutations per feature.
    pub n_permutations: usize,
    /// Random seed.
    pub seed: Option<u64>,
    /// Metric to use (higher is better).
    pub metric: ImportanceMetric,
}

impl Default for FeatureImportanceConfig {
    fn default() -> Self {
        Self {
            n_permutations: 10,
            seed: None,
            metric: ImportanceMetric::Accuracy,
        }
    }
}

/// Metric for measuring importance.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImportanceMetric {
    /// Classification accuracy.
    Accuracy,
    /// Mean squared error (for regression).
    MSE,
    /// Mean absolute error.
    MAE,
    /// R-squared score.
    R2,
}

/// Feature importance result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportanceResult {
    /// Importance scores per feature.
    pub importances: Vec<f64>,
    /// Standard deviations of importance scores.
    pub std_devs: Vec<f64>,
    /// Feature names if provided.
    pub feature_names: Option<Vec<String>>,
    /// Baseline score (without permutation).
    pub baseline_score: f64,
    /// Ranked feature indices (most important first).
    pub ranking: Vec<usize>,
}

/// Permutation Feature Importance kernel.
///
/// Computes feature importance by measuring how much model performance
/// degrades when each feature is randomly shuffled. Features that cause
/// larger degradation are more important.
#[derive(Debug, Clone)]
pub struct FeatureImportance {
    metadata: KernelMetadata,
}

impl Default for FeatureImportance {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureImportance {
    /// Create a new Feature Importance kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/feature-importance", Domain::StatisticalML)
                .with_description("Permutation-based feature importance")
                .with_throughput(5_000)
                .with_latency_us(200.0),
        }
    }

    /// Compute permutation feature importance.
    ///
    /// # Arguments
    /// * `data` - Input features
    /// * `targets` - True labels/values
    /// * `predict_fn` - Model prediction function
    /// * `config` - Configuration
    /// * `feature_names` - Optional feature names
    pub fn compute<F>(
        data: &DataMatrix,
        targets: &[f64],
        predict_fn: F,
        config: &FeatureImportanceConfig,
        feature_names: Option<Vec<String>>,
    ) -> FeatureImportanceResult
    where
        F: Fn(&[f64]) -> f64,
    {
        if data.n_samples == 0 || data.n_features == 0 {
            return FeatureImportanceResult {
                importances: Vec::new(),
                std_devs: Vec::new(),
                feature_names: None,
                baseline_score: 0.0,
                ranking: Vec::new(),
            };
        }

        let mut rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut rng()),
        };

        // Compute baseline score
        let predictions: Vec<f64> = (0..data.n_samples)
            .map(|i| predict_fn(data.row(i)))
            .collect();
        let baseline_score = Self::compute_score(&predictions, targets, config.metric);

        // Compute importance for each feature
        let mut importances = Vec::with_capacity(data.n_features);
        let mut std_devs = Vec::with_capacity(data.n_features);

        for feature_idx in 0..data.n_features {
            let mut scores = Vec::with_capacity(config.n_permutations);

            for _ in 0..config.n_permutations {
                // Create permuted data
                let mut perm_data = data.data.clone();
                let mut perm_indices: Vec<usize> = (0..data.n_samples).collect();
                perm_indices.shuffle(&mut rng);

                // Shuffle feature values
                for (i, &perm_idx) in perm_indices.iter().enumerate() {
                    perm_data[i * data.n_features + feature_idx] =
                        data.data[perm_idx * data.n_features + feature_idx];
                }

                let perm_matrix = DataMatrix::new(perm_data, data.n_samples, data.n_features);

                // Compute predictions with permuted feature
                let perm_predictions: Vec<f64> = (0..perm_matrix.n_samples)
                    .map(|i| predict_fn(perm_matrix.row(i)))
                    .collect();

                let score = Self::compute_score(&perm_predictions, targets, config.metric);
                scores.push(score);
            }

            // Importance = baseline - mean(permuted scores)
            let mean_score: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
            let importance = baseline_score - mean_score;

            let variance: f64 = scores.iter().map(|s| (s - mean_score).powi(2)).sum::<f64>()
                / scores.len() as f64;
            let std_dev = variance.sqrt();

            importances.push(importance);
            std_devs.push(std_dev);
        }

        // Compute ranking
        let mut ranking: Vec<usize> = (0..data.n_features).collect();
        ranking.sort_by(|&a, &b| {
            importances[b]
                .partial_cmp(&importances[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        FeatureImportanceResult {
            importances,
            std_devs,
            feature_names,
            baseline_score,
            ranking,
        }
    }

    /// Compute score based on metric.
    fn compute_score(predictions: &[f64], targets: &[f64], metric: ImportanceMetric) -> f64 {
        if predictions.is_empty() || targets.is_empty() {
            return 0.0;
        }

        match metric {
            ImportanceMetric::Accuracy => {
                let correct: usize = predictions
                    .iter()
                    .zip(targets.iter())
                    .filter(|&(p, t)| (p.round() - t.round()).abs() < 0.5)
                    .count();
                correct as f64 / predictions.len() as f64
            }
            ImportanceMetric::MSE => {
                let mse: f64 = predictions
                    .iter()
                    .zip(targets.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
                    / predictions.len() as f64;
                -mse // Negative because higher is better
            }
            ImportanceMetric::MAE => {
                let mae: f64 = predictions
                    .iter()
                    .zip(targets.iter())
                    .map(|(p, t)| (p - t).abs())
                    .sum::<f64>()
                    / predictions.len() as f64;
                -mae // Negative because higher is better
            }
            ImportanceMetric::R2 => {
                let mean_target: f64 = targets.iter().sum::<f64>() / targets.len() as f64;
                let ss_res: f64 = predictions
                    .iter()
                    .zip(targets.iter())
                    .map(|(p, t)| (t - p).powi(2))
                    .sum();
                let ss_tot: f64 = targets.iter().map(|t| (t - mean_target).powi(2)).sum();
                if ss_tot.abs() < 1e-10 {
                    0.0
                } else {
                    1.0 - ss_res / ss_tot
                }
            }
        }
    }
}

impl GpuKernel for FeatureImportance {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shap_values_metadata() {
        let kernel = SHAPValues::new();
        assert_eq!(kernel.metadata().id, "ml/shap-values");
    }

    #[test]
    fn test_shap_basic() {
        // Simple linear model: f(x) = x[0] + 2*x[1]
        let predict_fn = |x: &[f64]| x[0] + 2.0 * x[1];

        let background = DataMatrix::new(
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            4,
            2,
        );

        let config = SHAPConfig {
            n_samples: 50,
            use_kernel_shap: true,
            regularization: 0.1,
            seed: Some(42),
        };

        let instance = vec![1.0, 1.0];
        let explanation = SHAPValues::explain(&instance, &background, predict_fn, &config);

        // For linear model, SHAP values should approximate coefficients
        assert!(explanation.shap_values.len() == 2);
        assert!(explanation.prediction > 0.0);
    }

    #[test]
    fn test_shap_batch() {
        let predict_fn = |x: &[f64]| x[0] * 2.0;

        let background = DataMatrix::new(vec![0.0, 0.5, 1.0, 1.5], 4, 1);
        let instances = DataMatrix::new(vec![0.5, 1.0, 2.0], 3, 1);

        let config = SHAPConfig {
            n_samples: 20,
            seed: Some(42),
            ..Default::default()
        };

        let result =
            SHAPValues::explain_batch(&instances, &background, predict_fn, &config, None);

        assert_eq!(result.shap_values.len(), 3);
        assert_eq!(result.feature_importance.len(), 1);
    }

    #[test]
    fn test_shap_empty() {
        let predict_fn = |x: &[f64]| x.iter().sum();
        let background = DataMatrix::new(vec![], 0, 0);
        let config = SHAPConfig::default();

        let explanation = SHAPValues::explain(&[], &background, predict_fn, &config);
        assert!(explanation.shap_values.is_empty());
    }

    #[test]
    fn test_kernel_shap_weight() {
        // Edge cases
        assert!(SHAPValues::kernel_shap_weight(5, 0) > 1000.0);
        assert!(SHAPValues::kernel_shap_weight(5, 5) > 1000.0);

        // Middle values should have finite weights
        let w = SHAPValues::kernel_shap_weight(5, 2);
        assert!(w > 0.0 && w < 1000.0);
    }

    #[test]
    fn test_feature_importance_metadata() {
        let kernel = FeatureImportance::new();
        assert_eq!(kernel.metadata().id, "ml/feature-importance");
    }

    #[test]
    fn test_feature_importance_basic() {
        // Model that only uses first feature
        let predict_fn = |x: &[f64]| x[0];

        let data = DataMatrix::new(
            vec![
                1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
            ],
            4,
            3,
        );
        let targets = vec![1.0, 2.0, 3.0, 4.0];

        let config = FeatureImportanceConfig {
            n_permutations: 5,
            seed: Some(42),
            metric: ImportanceMetric::MSE,
        };

        let result = FeatureImportance::compute(&data, &targets, predict_fn, &config, None);

        // First feature should be most important
        assert_eq!(result.importances.len(), 3);
        assert!(result.importances[0].abs() > result.importances[1].abs());
        assert!(result.importances[0].abs() > result.importances[2].abs());
        assert_eq!(result.ranking[0], 0);
    }

    #[test]
    fn test_feature_importance_empty() {
        let predict_fn = |_: &[f64]| 0.0;
        let data = DataMatrix::new(vec![], 0, 0);
        let targets: Vec<f64> = vec![];
        let config = FeatureImportanceConfig::default();

        let result = FeatureImportance::compute(&data, &targets, predict_fn, &config, None);
        assert!(result.importances.is_empty());
    }

    #[test]
    fn test_metrics() {
        let preds = vec![1.0, 2.0, 3.0];
        let targets = vec![1.0, 2.0, 3.0];

        // Perfect predictions
        let acc = FeatureImportance::compute_score(&preds, &targets, ImportanceMetric::Accuracy);
        assert!((acc - 1.0).abs() < 0.01);

        let mse = FeatureImportance::compute_score(&preds, &targets, ImportanceMetric::MSE);
        assert!((mse - 0.0).abs() < 0.01);

        let r2 = FeatureImportance::compute_score(&preds, &targets, ImportanceMetric::R2);
        assert!((r2 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_binomial() {
        assert!((SHAPValues::binomial(5, 2) - 10.0).abs() < 0.01);
        assert!((SHAPValues::binomial(10, 3) - 120.0).abs() < 0.01);
        assert!((SHAPValues::binomial(5, 0) - 1.0).abs() < 0.01);
        assert!((SHAPValues::binomial(5, 5) - 1.0).abs() < 0.01);
    }
}
