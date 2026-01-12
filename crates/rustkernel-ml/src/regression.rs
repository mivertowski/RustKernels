//! Regression kernels.
//!
//! This module provides regression algorithms:
//! - Linear regression (OLS via normal equations)
//! - Ridge regression (L2 regularization)

use crate::types::{DataMatrix, RegressionResult};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

// ============================================================================
// Linear Regression Kernel
// ============================================================================

/// Linear regression kernel.
///
/// Ordinary Least Squares regression using the normal equations:
/// β = (X^T X)^(-1) X^T y
#[derive(Debug, Clone)]
pub struct LinearRegression {
    metadata: KernelMetadata,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearRegression {
    /// Create a new linear regression kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/linear-regression", Domain::StatisticalML)
                .with_description("OLS linear regression via normal equations")
                .with_throughput(50_000)
                .with_latency_us(20.0),
        }
    }

    /// Fit linear regression model.
    ///
    /// # Arguments
    /// * `x` - Feature matrix (n_samples x n_features)
    /// * `y` - Target vector (n_samples)
    /// * `fit_intercept` - Whether to fit an intercept term
    pub fn compute(x: &DataMatrix, y: &[f64], fit_intercept: bool) -> RegressionResult {
        let n = x.n_samples;
        let d = x.n_features;

        if n == 0 || d == 0 || y.len() != n {
            return RegressionResult {
                coefficients: Vec::new(),
                intercept: 0.0,
                r2_score: 0.0,
            };
        }

        // Optionally add intercept column
        let (x_aug, d_aug) = if fit_intercept {
            let mut aug_data = Vec::with_capacity(n * (d + 1));
            for i in 0..n {
                aug_data.push(1.0); // Intercept column
                aug_data.extend_from_slice(x.row(i));
            }
            (DataMatrix::new(aug_data, n, d + 1), d + 1)
        } else {
            (x.clone(), d)
        };

        // Compute X^T X
        let xtx = Self::matrix_multiply_transpose_left(&x_aug);

        // Compute X^T y
        let xty = Self::matrix_vector_multiply_transpose(&x_aug, y);

        // Solve (X^T X) β = X^T y using Cholesky decomposition
        let coefficients = match Self::solve_positive_definite(&xtx, &xty, d_aug) {
            Some(c) => c,
            None => vec![0.0; d_aug], // Fall back to zeros if singular
        };

        // Extract intercept if fitted
        let (intercept, coefs) = if fit_intercept {
            (coefficients[0], coefficients[1..].to_vec())
        } else {
            (0.0, coefficients)
        };

        // Compute predictions and R²
        let predictions: Vec<f64> = (0..n)
            .map(|i| {
                let xi = x.row(i);
                intercept + coefs.iter().zip(xi.iter()).map(|(c, x)| c * x).sum::<f64>()
            })
            .collect();

        let r2 = Self::compute_r2(y, &predictions);

        RegressionResult {
            coefficients: coefs,
            intercept,
            r2_score: r2,
        }
    }

    /// Compute X^T X (symmetric positive semi-definite).
    fn matrix_multiply_transpose_left(x: &DataMatrix) -> Vec<f64> {
        let n = x.n_samples;
        let d = x.n_features;
        let mut result = vec![0.0f64; d * d];

        for i in 0..d {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += x.get(k, i) * x.get(k, j);
                }
                result[i * d + j] = sum;
                result[j * d + i] = sum; // Symmetric
            }
        }

        result
    }

    /// Compute X^T y.
    fn matrix_vector_multiply_transpose(x: &DataMatrix, y: &[f64]) -> Vec<f64> {
        let n = x.n_samples;
        let d = x.n_features;
        let mut result = vec![0.0f64; d];

        for j in 0..d {
            for i in 0..n {
                result[j] += x.get(i, j) * y[i];
            }
        }

        result
    }

    /// Solve Ax = b for positive definite A using Cholesky decomposition.
    fn solve_positive_definite(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
        // Cholesky decomposition: A = L L^T
        let mut l = vec![0.0f64; n * n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = a[i * n + j];
                for k in 0..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }

                if i == j {
                    if sum <= 0.0 {
                        // Add regularization for numerical stability
                        let reg_sum = sum + 1e-10;
                        if reg_sum <= 0.0 {
                            return None;
                        }
                        l[i * n + j] = reg_sum.sqrt();
                    } else {
                        l[i * n + j] = sum.sqrt();
                    }
                } else {
                    l[i * n + j] = sum / l[j * n + j];
                }
            }
        }

        // Forward substitution: L y = b
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= l[i * n + j] * y[j];
            }
            y[i] = sum / l[i * n + i];
        }

        // Backward substitution: L^T x = y
        let mut x = vec![0.0f64; n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= l[j * n + i] * x[j]; // L^T[i][j] = L[j][i]
            }
            x[i] = sum / l[i * n + i];
        }

        Some(x)
    }

    /// Compute R² score.
    fn compute_r2(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let n = y_true.len();
        if n == 0 {
            return 0.0;
        }

        let y_mean: f64 = y_true.iter().sum::<f64>() / n as f64;

        let ss_tot: f64 = y_true.iter().map(|&y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&yt, &yp)| (yt - yp).powi(2))
            .sum();

        if ss_tot == 0.0 {
            return if ss_res == 0.0 { 1.0 } else { 0.0 };
        }

        1.0 - ss_res / ss_tot
    }
}

impl GpuKernel for LinearRegression {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Ridge Regression Kernel
// ============================================================================

/// Ridge regression kernel.
///
/// Linear regression with L2 regularization:
/// β = (X^T X + αI)^(-1) X^T y
#[derive(Debug, Clone)]
pub struct RidgeRegression {
    metadata: KernelMetadata,
}

impl Default for RidgeRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl RidgeRegression {
    /// Create a new ridge regression kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/ridge-regression", Domain::StatisticalML)
                .with_description("Ridge regression with L2 regularization")
                .with_throughput(50_000)
                .with_latency_us(20.0),
        }
    }

    /// Fit ridge regression model.
    ///
    /// # Arguments
    /// * `x` - Feature matrix (n_samples x n_features)
    /// * `y` - Target vector (n_samples)
    /// * `alpha` - Regularization strength
    /// * `fit_intercept` - Whether to fit an intercept term
    pub fn compute(
        x: &DataMatrix,
        y: &[f64],
        alpha: f64,
        fit_intercept: bool,
    ) -> RegressionResult {
        let n = x.n_samples;
        let d = x.n_features;

        if n == 0 || d == 0 || y.len() != n {
            return RegressionResult {
                coefficients: Vec::new(),
                intercept: 0.0,
                r2_score: 0.0,
            };
        }

        // Center data if fitting intercept (standard approach for Ridge)
        let (x_centered, y_centered, x_mean, y_mean) = if fit_intercept {
            let x_mean: Vec<f64> = (0..d)
                .map(|j| (0..n).map(|i| x.get(i, j)).sum::<f64>() / n as f64)
                .collect();
            let y_mean: f64 = y.iter().sum::<f64>() / n as f64;

            let mut x_data = Vec::with_capacity(n * d);
            for i in 0..n {
                for j in 0..d {
                    x_data.push(x.get(i, j) - x_mean[j]);
                }
            }

            let y_centered: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();

            (DataMatrix::new(x_data, n, d), y_centered, x_mean, y_mean)
        } else {
            (x.clone(), y.to_vec(), vec![0.0; d], 0.0)
        };

        // Compute X^T X + αI
        let mut xtx = LinearRegression::matrix_multiply_transpose_left(&x_centered);

        // Add regularization to diagonal
        for i in 0..d {
            xtx[i * d + i] += alpha;
        }

        // Compute X^T y
        let xty = LinearRegression::matrix_vector_multiply_transpose(&x_centered, &y_centered);

        // Solve (X^T X + αI) β = X^T y
        let coefficients = match LinearRegression::solve_positive_definite(&xtx, &xty, d) {
            Some(c) => c,
            None => vec![0.0; d],
        };

        // Compute intercept: b = y_mean - Σ(β_j * x_mean_j)
        let intercept = if fit_intercept {
            y_mean
                - coefficients
                    .iter()
                    .zip(x_mean.iter())
                    .map(|(c, m)| c * m)
                    .sum::<f64>()
        } else {
            0.0
        };

        // Compute predictions and R²
        let predictions: Vec<f64> = (0..n)
            .map(|i| {
                let xi = x.row(i);
                intercept
                    + coefficients
                        .iter()
                        .zip(xi.iter())
                        .map(|(c, x)| c * x)
                        .sum::<f64>()
            })
            .collect();

        let r2 = LinearRegression::compute_r2(y, &predictions);

        RegressionResult {
            coefficients,
            intercept,
            r2_score: r2,
        }
    }
}

impl GpuKernel for RidgeRegression {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_linear_data() -> (DataMatrix, Vec<f64>) {
        // y = 2*x1 + 3*x2 + 1
        let x = DataMatrix::from_rows(&[
            &[1.0, 1.0],
            &[2.0, 1.0],
            &[1.0, 2.0],
            &[2.0, 2.0],
            &[3.0, 1.0],
            &[1.0, 3.0],
        ]);
        let y: Vec<f64> = (0..6)
            .map(|i| 2.0 * x.get(i, 0) + 3.0 * x.get(i, 1) + 1.0)
            .collect();
        (x, y)
    }

    #[test]
    fn test_linear_regression_metadata() {
        let kernel = LinearRegression::new();
        assert_eq!(kernel.metadata().id, "ml/linear-regression");
        assert_eq!(kernel.metadata().domain, Domain::StatisticalML);
    }

    #[test]
    fn test_linear_regression_fit() {
        let (x, y) = create_linear_data();
        let result = LinearRegression::compute(&x, &y, true);

        // Should recover coefficients [2, 3] and intercept 1
        assert!(
            (result.coefficients[0] - 2.0).abs() < 0.01,
            "Expected coef[0] ≈ 2.0, got {}",
            result.coefficients[0]
        );
        assert!(
            (result.coefficients[1] - 3.0).abs() < 0.01,
            "Expected coef[1] ≈ 3.0, got {}",
            result.coefficients[1]
        );
        assert!(
            (result.intercept - 1.0).abs() < 0.01,
            "Expected intercept ≈ 1.0, got {}",
            result.intercept
        );

        // R² should be 1.0 for perfect linear data
        assert!(result.r2_score > 0.99, "Expected R² ≈ 1.0, got {}", result.r2_score);
    }

    #[test]
    fn test_ridge_regression_metadata() {
        let kernel = RidgeRegression::new();
        assert_eq!(kernel.metadata().id, "ml/ridge-regression");
        assert_eq!(kernel.metadata().domain, Domain::StatisticalML);
    }

    #[test]
    fn test_ridge_regression_fit() {
        let (x, y) = create_linear_data();

        // With small regularization, should be close to OLS
        let result = RidgeRegression::compute(&x, &y, 0.001, true);

        assert!(
            (result.coefficients[0] - 2.0).abs() < 0.1,
            "Expected coef[0] ≈ 2.0, got {}",
            result.coefficients[0]
        );
        assert!(
            (result.coefficients[1] - 3.0).abs() < 0.1,
            "Expected coef[1] ≈ 3.0, got {}",
            result.coefficients[1]
        );

        assert!(result.r2_score > 0.95, "Expected high R², got {}", result.r2_score);
    }

    #[test]
    fn test_ridge_regularization_effect() {
        let (x, y) = create_linear_data();

        let result_low = RidgeRegression::compute(&x, &y, 0.001, true);
        let result_high = RidgeRegression::compute(&x, &y, 100.0, true);

        // High regularization should shrink coefficients
        let coef_norm_low: f64 = result_low
            .coefficients
            .iter()
            .map(|c| c.powi(2))
            .sum::<f64>()
            .sqrt();
        let coef_norm_high: f64 = result_high
            .coefficients
            .iter()
            .map(|c| c.powi(2))
            .sum::<f64>()
            .sqrt();

        assert!(
            coef_norm_high < coef_norm_low,
            "Higher regularization should shrink coefficients: {} < {}",
            coef_norm_high,
            coef_norm_low
        );
    }

    #[test]
    fn test_linear_regression_empty() {
        let x = DataMatrix::zeros(0, 2);
        let y: Vec<f64> = Vec::new();
        let result = LinearRegression::compute(&x, &y, true);
        assert!(result.coefficients.is_empty());
    }
}
