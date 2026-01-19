//! Time series decomposition kernels.
//!
//! This module provides decomposition algorithms:
//! - Seasonal decomposition (STL-like)
//! - Trend extraction (various moving average methods)

use std::time::Instant;

use async_trait::async_trait;

use crate::messages::{
    SeasonalDecompositionInput, SeasonalDecompositionOutput, TrendExtractionInput,
    TrendExtractionOutput,
};
use crate::types::{DecompositionResult, TimeSeries, TrendMethod, TrendResult};
use rustkernel_core::{
    domain::Domain,
    error::Result,
    kernel::KernelMetadata,
    traits::{BatchKernel, GpuKernel},
};

// ============================================================================
// Seasonal Decomposition Kernel
// ============================================================================

/// Seasonal decomposition kernel (STL-like).
///
/// Decomposes a time series into trend, seasonal, and residual components.
#[derive(Debug, Clone)]
pub struct SeasonalDecomposition {
    metadata: KernelMetadata,
}

impl Default for SeasonalDecomposition {
    fn default() -> Self {
        Self::new()
    }
}

impl SeasonalDecomposition {
    /// Create a new seasonal decomposition kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch(
                "temporal/seasonal-decomposition",
                Domain::TemporalAnalysis,
            )
            .with_description("STL-style seasonal decomposition")
            .with_throughput(10_000)
            .with_latency_us(100.0),
        }
    }

    /// Decompose a time series.
    ///
    /// # Arguments
    /// * `series` - Input time series
    /// * `period` - Seasonal period
    /// * `robust` - Use robust (median-based) estimation
    pub fn compute(series: &TimeSeries, period: usize, robust: bool) -> DecompositionResult {
        let n = series.len();

        if n < 2 * period || period < 2 {
            return DecompositionResult {
                trend: series.values.clone(),
                seasonal: vec![0.0; n],
                residual: vec![0.0; n],
                n,
                period,
            };
        }

        // Step 1: Initial trend estimation using centered moving average
        let trend = Self::centered_moving_average(&series.values, period);

        // Step 2: Detrend the series
        let detrended: Vec<f64> = series
            .values
            .iter()
            .zip(trend.iter())
            .map(|(v, t)| v - t)
            .collect();

        // Step 3: Estimate seasonal component
        let seasonal_pattern = if robust {
            Self::robust_seasonal(&detrended, period)
        } else {
            Self::mean_seasonal(&detrended, period)
        };

        // Extend seasonal pattern to full length
        let seasonal: Vec<f64> = (0..n).map(|i| seasonal_pattern[i % period]).collect();

        // Step 4: Refine trend by removing seasonality first
        let deseasoned: Vec<f64> = series
            .values
            .iter()
            .zip(seasonal.iter())
            .map(|(v, s)| v - s)
            .collect();

        let refined_trend = Self::lowess_trend(&deseasoned, period);

        // Step 5: Calculate residuals
        let residual: Vec<f64> = series
            .values
            .iter()
            .zip(refined_trend.iter())
            .zip(seasonal.iter())
            .map(|((v, t), s)| v - t - s)
            .collect();

        DecompositionResult {
            trend: refined_trend,
            seasonal,
            residual,
            n,
            period,
        }
    }

    /// Additive decomposition (simple version).
    pub fn compute_additive(series: &TimeSeries, period: usize) -> DecompositionResult {
        Self::compute(series, period, false)
    }

    /// Multiplicative decomposition.
    ///
    /// Y = T * S * R
    pub fn compute_multiplicative(series: &TimeSeries, period: usize) -> DecompositionResult {
        let n = series.len();

        if n < 2 * period || period < 2 {
            return DecompositionResult {
                trend: series.values.clone(),
                seasonal: vec![1.0; n],
                residual: vec![1.0; n],
                n,
                period,
            };
        }

        // Convert to log space if all values positive
        let min_val = series.values.iter().cloned().fold(f64::INFINITY, f64::min);

        if min_val <= 0.0 {
            // Fall back to additive for non-positive data
            return Self::compute(series, period, false);
        }

        let log_values: Vec<f64> = series.values.iter().map(|v| v.ln()).collect();
        let log_series = TimeSeries::new(log_values);

        let log_result = Self::compute(&log_series, period, false);

        // Convert back from log space
        DecompositionResult {
            trend: log_result.trend.iter().map(|t| t.exp()).collect(),
            seasonal: log_result.seasonal.iter().map(|s| s.exp()).collect(),
            residual: log_result.residual.iter().map(|r| r.exp()).collect(),
            n,
            period,
        }
    }

    /// Centered moving average.
    #[allow(clippy::needless_range_loop)]
    fn centered_moving_average(values: &[f64], window: usize) -> Vec<f64> {
        let n = values.len();
        let half_w = window / 2;
        let mut result = vec![0.0; n];

        for i in 0..n {
            let start = i.saturating_sub(half_w);
            let end = (i + half_w + 1).min(n);

            // For even windows, use weighted average at boundaries
            if window % 2 == 0 && i >= half_w && i + half_w < n {
                let mut sum = 0.0;
                let mut weight = 0.0;

                for j in start..end {
                    let w = if j == start || j == end - 1 { 0.5 } else { 1.0 };
                    sum += values[j] * w;
                    weight += w;
                }
                result[i] = sum / weight;
            } else {
                result[i] = values[start..end].iter().sum::<f64>() / (end - start) as f64;
            }
        }

        result
    }

    /// Mean-based seasonal estimation.
    fn mean_seasonal(detrended: &[f64], period: usize) -> Vec<f64> {
        let mut seasonal = vec![0.0; period];
        let mut counts = vec![0usize; period];

        for (i, &d) in detrended.iter().enumerate() {
            seasonal[i % period] += d;
            counts[i % period] += 1;
        }

        for i in 0..period {
            if counts[i] > 0 {
                seasonal[i] /= counts[i] as f64;
            }
        }

        // Center the seasonal component
        let mean: f64 = seasonal.iter().sum::<f64>() / period as f64;
        for s in &mut seasonal {
            *s -= mean;
        }

        seasonal
    }

    /// Robust (median-based) seasonal estimation.
    #[allow(clippy::needless_range_loop)]
    fn robust_seasonal(detrended: &[f64], period: usize) -> Vec<f64> {
        let mut seasonal = vec![0.0; period];

        for s in 0..period {
            let mut season_values: Vec<f64> = detrended
                .iter()
                .enumerate()
                .filter(|(i, _)| i % period == s)
                .map(|(_, &v)| v)
                .collect();

            if !season_values.is_empty() {
                season_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                seasonal[s] = season_values[season_values.len() / 2];
            }
        }

        // Center using median
        let mut sorted = seasonal.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[period / 2];

        for s in &mut seasonal {
            *s -= median;
        }

        seasonal
    }

    /// Lowess-style trend extraction (simplified).
    #[allow(clippy::needless_range_loop)]
    fn lowess_trend(values: &[f64], bandwidth: usize) -> Vec<f64> {
        let n = values.len();
        let mut trend = vec![0.0; n];

        for i in 0..n {
            let start = i.saturating_sub(bandwidth);
            let end = (i + bandwidth + 1).min(n);

            // Tricube weights
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for j in start..end {
                let dist = (j as f64 - i as f64).abs() / bandwidth as f64;
                let weight = if dist < 1.0 {
                    (1.0 - dist.powi(3)).powi(3)
                } else {
                    0.0
                };
                weighted_sum += values[j] * weight;
                weight_sum += weight;
            }

            trend[i] = if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                values[i]
            };
        }

        trend
    }
}

impl GpuKernel for SeasonalDecomposition {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[async_trait]
impl BatchKernel<SeasonalDecompositionInput, SeasonalDecompositionOutput>
    for SeasonalDecomposition
{
    async fn execute(
        &self,
        input: SeasonalDecompositionInput,
    ) -> Result<SeasonalDecompositionOutput> {
        let start = Instant::now();
        let result = Self::compute(&input.series, input.period, input.robust);
        Ok(SeasonalDecompositionOutput {
            result,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

// ============================================================================
// Trend Extraction Kernel
// ============================================================================

/// Trend extraction kernel.
///
/// Extracts trend component using various moving average methods.
#[derive(Debug, Clone)]
pub struct TrendExtraction {
    metadata: KernelMetadata,
}

impl Default for TrendExtraction {
    fn default() -> Self {
        Self::new()
    }
}

impl TrendExtraction {
    /// Create a new trend extraction kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("temporal/trend-extraction", Domain::TemporalAnalysis)
                .with_description("Moving average trend extraction")
                .with_throughput(50_000)
                .with_latency_us(20.0),
        }
    }

    /// Extract trend from a time series.
    ///
    /// # Arguments
    /// * `series` - Input time series
    /// * `method` - Trend extraction method
    /// * `window` - Window size for moving average
    pub fn compute(series: &TimeSeries, method: TrendMethod, window: usize) -> TrendResult {
        if series.is_empty() {
            return TrendResult {
                trend: Vec::new(),
                detrended: Vec::new(),
                method,
            };
        }

        let trend = match method {
            TrendMethod::SimpleMovingAverage => Self::simple_ma(&series.values, window),
            TrendMethod::ExponentialMovingAverage => Self::exponential_ma(&series.values, window),
            TrendMethod::CenteredMovingAverage => Self::centered_ma(&series.values, window),
            TrendMethod::Lowess => Self::lowess(&series.values, window),
        };

        let detrended: Vec<f64> = series
            .values
            .iter()
            .zip(trend.iter())
            .map(|(v, t)| v - t)
            .collect();

        TrendResult {
            trend,
            detrended,
            method,
        }
    }

    /// Simple moving average.
    fn simple_ma(values: &[f64], window: usize) -> Vec<f64> {
        let n = values.len();
        let w = window.min(n).max(1);
        let mut result = vec![0.0; n];

        // Cumulative sum for efficient computation
        let mut cumsum = vec![0.0; n + 1];
        for (i, &v) in values.iter().enumerate() {
            cumsum[i + 1] = cumsum[i] + v;
        }

        for i in 0..n {
            let start = i.saturating_sub(w - 1);
            let count = i - start + 1;
            result[i] = (cumsum[i + 1] - cumsum[start]) / count as f64;
        }

        result
    }

    /// Exponential moving average.
    fn exponential_ma(values: &[f64], span: usize) -> Vec<f64> {
        let n = values.len();
        if n == 0 {
            return Vec::new();
        }

        let alpha = 2.0 / (span as f64 + 1.0);
        let mut result = vec![0.0; n];
        result[0] = values[0];

        for i in 1..n {
            result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Centered moving average.
    #[allow(clippy::needless_range_loop)]
    fn centered_ma(values: &[f64], window: usize) -> Vec<f64> {
        let n = values.len();
        let half_w = window / 2;
        let mut result = vec![0.0; n];

        for i in 0..n {
            let start = i.saturating_sub(half_w);
            let end = (i + half_w + 1).min(n);
            result[i] = values[start..end].iter().sum::<f64>() / (end - start) as f64;
        }

        result
    }

    /// Lowess (Locally Weighted Scatterplot Smoothing).
    #[allow(clippy::needless_range_loop)]
    fn lowess(values: &[f64], bandwidth: usize) -> Vec<f64> {
        let n = values.len();
        let mut result = vec![0.0; n];

        for i in 0..n {
            let start = i.saturating_sub(bandwidth);
            let end = (i + bandwidth + 1).min(n);

            // Fit local linear regression with tricube weights
            let mut sum_w = 0.0;
            let mut sum_wx = 0.0;
            let mut sum_wy = 0.0;
            let mut sum_wxx = 0.0;
            let mut sum_wxy = 0.0;

            for j in start..end {
                let x = j as f64;
                let y = values[j];
                let dist = (j as f64 - i as f64).abs() / (bandwidth as f64 + 1.0);
                let w = if dist < 1.0 {
                    (1.0 - dist.powi(3)).powi(3)
                } else {
                    0.0
                };

                sum_w += w;
                sum_wx += w * x;
                sum_wy += w * y;
                sum_wxx += w * x * x;
                sum_wxy += w * x * y;
            }

            // Solve for local linear fit at point i
            let det = sum_w * sum_wxx - sum_wx * sum_wx;
            if det.abs() > 1e-10 {
                let b0 = (sum_wxx * sum_wy - sum_wx * sum_wxy) / det;
                let b1 = (sum_w * sum_wxy - sum_wx * sum_wy) / det;
                result[i] = b0 + b1 * i as f64;
            } else {
                result[i] = if sum_w > 0.0 {
                    sum_wy / sum_w
                } else {
                    values[i]
                };
            }
        }

        result
    }

    /// Double exponential smoothing (Holt's method).
    pub fn holt_smoothing(values: &[f64], alpha: f64, beta: f64) -> (Vec<f64>, Vec<f64>) {
        let n = values.len();
        if n < 2 {
            return (values.to_vec(), vec![0.0; n]);
        }

        let mut level = vec![0.0; n];
        let mut trend = vec![0.0; n];

        // Initialize
        level[0] = values[0];
        trend[0] = values[1] - values[0];

        for i in 1..n {
            level[i] = alpha * values[i] + (1.0 - alpha) * (level[i - 1] + trend[i - 1]);
            trend[i] = beta * (level[i] - level[i - 1]) + (1.0 - beta) * trend[i - 1];
        }

        (level, trend)
    }
}

impl GpuKernel for TrendExtraction {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[async_trait]
impl BatchKernel<TrendExtractionInput, TrendExtractionOutput> for TrendExtraction {
    async fn execute(&self, input: TrendExtractionInput) -> Result<TrendExtractionOutput> {
        let start = Instant::now();
        let result = Self::compute(&input.series, input.method, input.window);
        Ok(TrendExtractionOutput {
            result,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_seasonal_series() -> TimeSeries {
        // Trend + seasonal pattern
        let period = 12;
        let values: Vec<f64> = (0..120)
            .map(|t| {
                let trend = 100.0 + 0.5 * t as f64;
                let seasonal =
                    10.0 * ((2.0 * std::f64::consts::PI * t as f64 / period as f64).sin());
                trend + seasonal
            })
            .collect();
        TimeSeries::new(values)
    }

    fn create_trend_series() -> TimeSeries {
        // Pure trend with some noise
        TimeSeries::new(
            (0..100)
                .map(|t| 10.0 + 2.0 * t as f64 + (t as f64 * 0.3).sin())
                .collect(),
        )
    }

    #[test]
    fn test_decomposition_metadata() {
        let kernel = SeasonalDecomposition::new();
        assert_eq!(kernel.metadata().id, "temporal/seasonal-decomposition");
        assert_eq!(kernel.metadata().domain, Domain::TemporalAnalysis);
    }

    #[test]
    fn test_seasonal_decomposition() {
        let series = create_seasonal_series();
        let result = SeasonalDecomposition::compute(&series, 12, false);

        // Should have correct lengths
        assert_eq!(result.trend.len(), series.len());
        assert_eq!(result.seasonal.len(), series.len());
        assert_eq!(result.residual.len(), series.len());
        assert_eq!(result.period, 12);

        // Seasonal should be periodic
        for i in 0..result.seasonal.len() - 12 {
            let diff = (result.seasonal[i] - result.seasonal[i + 12]).abs();
            assert!(diff < 0.01, "Seasonal not periodic at {}: diff={}", i, diff);
        }
    }

    #[test]
    fn test_robust_decomposition() {
        let series = create_seasonal_series();
        let result = SeasonalDecomposition::compute(&series, 12, true);

        assert_eq!(result.trend.len(), series.len());
        // Robust version should also produce valid decomposition
    }

    #[test]
    fn test_multiplicative_decomposition() {
        // Create multiplicative seasonal pattern
        let values: Vec<f64> = (0..120)
            .map(|t| {
                let trend = 100.0 + 0.5 * t as f64;
                let seasonal = 1.0 + 0.1 * ((2.0 * std::f64::consts::PI * t as f64 / 12.0).sin());
                trend * seasonal
            })
            .collect();
        let series = TimeSeries::new(values);

        let result = SeasonalDecomposition::compute_multiplicative(&series, 12);

        assert_eq!(result.trend.len(), series.len());
        // In multiplicative, seasonal should be multiplicative factors
    }

    #[test]
    fn test_trend_extraction_metadata() {
        let kernel = TrendExtraction::new();
        assert_eq!(kernel.metadata().id, "temporal/trend-extraction");
    }

    #[test]
    fn test_simple_moving_average() {
        let series = create_trend_series();
        let result = TrendExtraction::compute(&series, TrendMethod::SimpleMovingAverage, 5);

        assert_eq!(result.trend.len(), series.len());
        assert_eq!(result.method, TrendMethod::SimpleMovingAverage);

        // Trend should be smoother than original
        let original_var: f64 = series.values.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
        let trend_var: f64 = result.trend.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
        assert!(trend_var <= original_var);
    }

    #[test]
    fn test_exponential_moving_average() {
        let series = create_trend_series();
        let result = TrendExtraction::compute(&series, TrendMethod::ExponentialMovingAverage, 10);

        assert_eq!(result.trend.len(), series.len());
        assert_eq!(result.method, TrendMethod::ExponentialMovingAverage);
    }

    #[test]
    fn test_centered_moving_average() {
        let series = create_trend_series();
        let result = TrendExtraction::compute(&series, TrendMethod::CenteredMovingAverage, 7);

        assert_eq!(result.trend.len(), series.len());
        assert_eq!(result.method, TrendMethod::CenteredMovingAverage);
    }

    #[test]
    fn test_lowess_trend() {
        let series = create_trend_series();
        let result = TrendExtraction::compute(&series, TrendMethod::Lowess, 10);

        assert_eq!(result.trend.len(), series.len());
        assert_eq!(result.method, TrendMethod::Lowess);
    }

    #[test]
    fn test_holt_smoothing() {
        let values: Vec<f64> = (0..50).map(|t| 10.0 + 2.0 * t as f64).collect();
        let (level, trend) = TrendExtraction::holt_smoothing(&values, 0.3, 0.1);

        assert_eq!(level.len(), values.len());
        assert_eq!(trend.len(), values.len());

        // Trend should be approximately 2.0 (the slope)
        assert!((trend.last().unwrap() - 2.0).abs() < 1.0);
    }

    #[test]
    fn test_detrended_sums_to_zero_ish() {
        let series = create_seasonal_series();
        let result = TrendExtraction::compute(&series, TrendMethod::CenteredMovingAverage, 12);

        // Detrended should roughly sum to zero (mean-centered)
        let detrended_mean: f64 =
            result.detrended.iter().sum::<f64>() / result.detrended.len() as f64;
        assert!(
            detrended_mean.abs() < 1.0,
            "Detrended mean: {}",
            detrended_mean
        );
    }

    #[test]
    fn test_empty_series() {
        let empty = TimeSeries::new(Vec::new());

        let decomp = SeasonalDecomposition::compute(&empty, 12, false);
        assert!(decomp.trend.is_empty());

        let trend = TrendExtraction::compute(&empty, TrendMethod::SimpleMovingAverage, 5);
        assert!(trend.trend.is_empty());
    }
}
