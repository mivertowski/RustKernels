//! Ensemble method kernels.
//!
//! This module provides ensemble methods:
//! - Weighted majority voting
//! - Soft voting (probability averaging)

use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// Ensemble Voting Kernel
// ============================================================================

/// Voting strategy for ensemble.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VotingStrategy {
    /// Hard voting: majority class wins.
    #[default]
    Hard,
    /// Soft voting: average probabilities.
    Soft,
}

/// Ensemble voting kernel.
///
/// Combines predictions from multiple classifiers using
/// majority voting (hard) or probability averaging (soft).
#[derive(Debug, Clone)]
pub struct EnsembleVoting {
    metadata: KernelMetadata,
}

impl Default for EnsembleVoting {
    fn default() -> Self {
        Self::new()
    }
}

impl EnsembleVoting {
    /// Create a new ensemble voting kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/ensemble-voting", Domain::StatisticalML)
                .with_description("Weighted majority voting ensemble")
                .with_throughput(100_000)
                .with_latency_us(10.0),
        }
    }

    /// Compute hard voting (majority vote) for classification.
    ///
    /// # Arguments
    /// * `predictions` - Matrix of predictions (n_classifiers x n_samples)
    /// * `weights` - Optional classifier weights (defaults to equal)
    pub fn hard_vote(predictions: &[Vec<i32>], weights: Option<&[f64]>) -> Vec<i32> {
        if predictions.is_empty() || predictions[0].is_empty() {
            return Vec::new();
        }

        let n_classifiers = predictions.len();
        let n_samples = predictions[0].len();

        // Default to equal weights
        let default_weights: Vec<f64> = vec![1.0 / n_classifiers as f64; n_classifiers];
        let weights = weights.unwrap_or(&default_weights);

        (0..n_samples)
            .map(|i| {
                let mut class_weights: HashMap<i32, f64> = HashMap::new();

                for (j, pred) in predictions.iter().enumerate() {
                    let class = pred[i];
                    *class_weights.entry(class).or_insert(0.0) += weights[j];
                }

                *class_weights
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(class, _)| class)
                    .unwrap_or(&0)
            })
            .collect()
    }

    /// Compute soft voting (probability averaging) for classification.
    ///
    /// # Arguments
    /// * `probabilities` - 3D matrix: (n_classifiers, n_samples, n_classes)
    ///   Outer vec: classifiers, middle vec: samples, inner vec: class probabilities
    /// * `weights` - Optional classifier weights (defaults to equal)
    pub fn soft_vote(probabilities: &[Vec<Vec<f64>>], weights: Option<&[f64]>) -> Vec<usize> {
        if probabilities.is_empty() || probabilities[0].is_empty() {
            return Vec::new();
        }

        let n_classifiers = probabilities.len();
        let n_samples = probabilities[0].len();
        let n_classes = probabilities[0][0].len();

        // Default to equal weights
        let default_weights: Vec<f64> = vec![1.0 / n_classifiers as f64; n_classifiers];
        let weights = weights.unwrap_or(&default_weights);

        (0..n_samples)
            .map(|sample_idx| {
                // Average probabilities across classifiers
                let mut avg_probs = vec![0.0f64; n_classes];

                for (classifier_idx, probs) in probabilities.iter().enumerate() {
                    for (class_idx, &prob) in probs[sample_idx].iter().enumerate() {
                        avg_probs[class_idx] += weights[classifier_idx] * prob;
                    }
                }

                // Return class with highest average probability
                avg_probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Compute weighted average for regression ensemble.
    ///
    /// # Arguments
    /// * `predictions` - Matrix of predictions (n_regressors x n_samples)
    /// * `weights` - Optional regressor weights (defaults to equal)
    pub fn weighted_average(predictions: &[Vec<f64>], weights: Option<&[f64]>) -> Vec<f64> {
        if predictions.is_empty() || predictions[0].is_empty() {
            return Vec::new();
        }

        let n_regressors = predictions.len();
        let n_samples = predictions[0].len();

        // Default to equal weights
        let default_weights: Vec<f64> = vec![1.0; n_regressors];
        let weights = weights.unwrap_or(&default_weights);
        let weight_sum: f64 = weights.iter().sum();

        (0..n_samples)
            .map(|i| {
                let weighted_sum: f64 = predictions
                    .iter()
                    .zip(weights.iter())
                    .map(|(preds, &w)| preds[i] * w)
                    .sum();
                weighted_sum / weight_sum
            })
            .collect()
    }

    /// Compute median for robust regression ensemble.
    ///
    /// # Arguments
    /// * `predictions` - Matrix of predictions (n_regressors x n_samples)
    pub fn median_prediction(predictions: &[Vec<f64>]) -> Vec<f64> {
        if predictions.is_empty() || predictions[0].is_empty() {
            return Vec::new();
        }

        let n_samples = predictions[0].len();

        (0..n_samples)
            .map(|i| {
                let mut values: Vec<f64> = predictions.iter().map(|p| p[i]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let n = values.len();
                if n % 2 == 0 {
                    (values[n / 2 - 1] + values[n / 2]) / 2.0
                } else {
                    values[n / 2]
                }
            })
            .collect()
    }
}

impl GpuKernel for EnsembleVoting {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_voting_metadata() {
        let kernel = EnsembleVoting::new();
        assert_eq!(kernel.metadata().id, "ml/ensemble-voting");
        assert_eq!(kernel.metadata().domain, Domain::StatisticalML);
    }

    #[test]
    fn test_hard_vote() {
        // 3 classifiers, 5 samples
        let predictions = vec![
            vec![0, 1, 0, 1, 0], // Classifier 1
            vec![0, 0, 0, 1, 1], // Classifier 2
            vec![1, 1, 0, 1, 0], // Classifier 3
        ];

        let result = EnsembleVoting::hard_vote(&predictions, None);

        // Majority votes: 0, 1, 0, 1, 0
        assert_eq!(result[0], 0); // 2 votes for 0, 1 vote for 1
        assert_eq!(result[1], 1); // 1 vote for 0, 2 votes for 1
        assert_eq!(result[2], 0); // 3 votes for 0
        assert_eq!(result[3], 1); // 3 votes for 1
        // result[4] is a tie (0: 2, 1: 1), so 0 wins
    }

    #[test]
    fn test_hard_vote_weighted() {
        let predictions = vec![vec![0, 0, 0], vec![1, 1, 1]];

        // Give second classifier higher weight
        let weights = vec![0.3, 0.7];
        let result = EnsembleVoting::hard_vote(&predictions, Some(&weights));

        // Class 1 should win due to higher weight
        assert_eq!(result, vec![1, 1, 1]);
    }

    #[test]
    fn test_soft_vote() {
        // 2 classifiers, 3 samples, 2 classes
        let probabilities = vec![
            // Classifier 1
            vec![
                vec![0.9, 0.1], // Sample 1: strongly class 0
                vec![0.4, 0.6], // Sample 2: slightly class 1
                vec![0.5, 0.5], // Sample 3: tied
            ],
            // Classifier 2
            vec![
                vec![0.8, 0.2], // Sample 1: strongly class 0
                vec![0.3, 0.7], // Sample 2: class 1
                vec![0.2, 0.8], // Sample 3: class 1
            ],
        ];

        let result = EnsembleVoting::soft_vote(&probabilities, None);

        assert_eq!(result[0], 0); // Average: [0.85, 0.15] -> class 0
        assert_eq!(result[1], 1); // Average: [0.35, 0.65] -> class 1
        assert_eq!(result[2], 1); // Average: [0.35, 0.65] -> class 1
    }

    #[test]
    fn test_weighted_average() {
        let predictions = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];

        let result = EnsembleVoting::weighted_average(&predictions, None);

        // Equal weights: average = [2.0, 3.0, 4.0]
        assert!((result[0] - 2.0).abs() < 0.01);
        assert!((result[1] - 3.0).abs() < 0.01);
        assert!((result[2] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_median_prediction() {
        let predictions = vec![
            vec![1.0, 100.0, 3.0],
            vec![2.0, 2.0, 4.0],
            vec![3.0, 3.0, 5.0],
        ];

        let result = EnsembleVoting::median_prediction(&predictions);

        // Median is robust to outliers
        assert!((result[0] - 2.0).abs() < 0.01);
        assert!((result[1] - 3.0).abs() < 0.01); // 100 is outlier
        assert!((result[2] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_predictions() {
        let empty: Vec<Vec<i32>> = vec![];
        assert!(EnsembleVoting::hard_vote(&empty, None).is_empty());

        let empty_probs: Vec<Vec<Vec<f64>>> = vec![];
        assert!(EnsembleVoting::soft_vote(&empty_probs, None).is_empty());

        let empty_reg: Vec<Vec<f64>> = vec![];
        assert!(EnsembleVoting::weighted_average(&empty_reg, None).is_empty());
    }
}
