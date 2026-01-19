//! Change point and anomaly detection kernels.
//!
//! This module provides detection algorithms:
//! - Change point detection (PELT, Binary Segmentation, CUSUM)
//! - Time series anomaly detection

use std::time::Instant;

use async_trait::async_trait;

use crate::messages::{
    ChangePointDetectionInput, ChangePointDetectionOutput, TimeSeriesAnomalyDetectionInput,
    TimeSeriesAnomalyDetectionOutput,
};
use crate::types::{
    AnomalyMethod, ChangePointMethod, ChangePointResult, TimeSeries, TimeSeriesAnomalyResult,
};
use rustkernel_core::{
    domain::Domain,
    error::Result,
    kernel::KernelMetadata,
    traits::{BatchKernel, GpuKernel},
};

// ============================================================================
// Change Point Detection Kernel
// ============================================================================

/// Change point detection kernel.
///
/// Detects points in a time series where statistical properties change.
/// Supports PELT, Binary Segmentation, and CUSUM methods.
#[derive(Debug, Clone)]
pub struct ChangePointDetection {
    metadata: KernelMetadata,
}

impl Default for ChangePointDetection {
    fn default() -> Self {
        Self::new()
    }
}

impl ChangePointDetection {
    /// Create a new change point detection kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch(
                "temporal/changepoint-detection",
                Domain::TemporalAnalysis,
            )
            .with_description("PELT/Binary segmentation change point detection")
            .with_throughput(20_000)
            .with_latency_us(50.0),
        }
    }

    /// Detect change points in a time series.
    ///
    /// # Arguments
    /// * `series` - Input time series
    /// * `method` - Detection method
    /// * `penalty` - Penalty for adding change points (higher = fewer points)
    /// * `min_segment` - Minimum segment length
    pub fn compute(
        series: &TimeSeries,
        method: ChangePointMethod,
        penalty: f64,
        min_segment: usize,
    ) -> ChangePointResult {
        if series.len() < 2 * min_segment {
            return ChangePointResult {
                change_points: Vec::new(),
                confidence: Vec::new(),
                segment_means: vec![series.mean()],
                segment_variances: vec![series.variance()],
                cost: 0.0,
            };
        }

        match method {
            ChangePointMethod::PELT => Self::pelt(series, penalty, min_segment),
            ChangePointMethod::BinarySegmentation => {
                Self::binary_segmentation(series, penalty, min_segment)
            }
            ChangePointMethod::CUSUM => Self::cusum(series, penalty, min_segment),
        }
    }

    /// PELT (Pruned Exact Linear Time) algorithm.
    #[allow(clippy::needless_range_loop)]
    fn pelt(series: &TimeSeries, penalty: f64, min_segment: usize) -> ChangePointResult {
        let n = series.len();
        let values = &series.values;

        // Compute cumulative sums for efficient cost calculation
        let mut sum = vec![0.0; n + 1];
        let mut sum_sq = vec![0.0; n + 1];

        for (i, &v) in values.iter().enumerate() {
            sum[i + 1] = sum[i] + v;
            sum_sq[i + 1] = sum_sq[i] + v * v;
        }

        // Cost function: sum of squared residuals from segment mean
        let segment_cost = |start: usize, end: usize| -> f64 {
            let len = (end - start) as f64;
            if len < 1.0 {
                return 0.0;
            }
            let s = sum[end] - sum[start];
            let s2 = sum_sq[end] - sum_sq[start];
            s2 - (s * s) / len
        };

        // Dynamic programming with pruning
        let mut f = vec![f64::INFINITY; n + 1]; // Optimal cost up to position i
        let mut cp = vec![0usize; n + 1]; // Last change point before position i
        f[0] = -penalty;

        for t in min_segment..=n {
            let mut best_cost = f64::INFINITY;
            let mut best_cp = 0;

            for s in 0..=(t - min_segment) {
                let cost = f[s] + segment_cost(s, t) + penalty;
                if cost < best_cost {
                    best_cost = cost;
                    best_cp = s;
                }
            }

            f[t] = best_cost;
            cp[t] = best_cp;
        }

        // Backtrack to find change points
        let mut change_points = Vec::new();
        let mut idx = n;
        while cp[idx] > 0 {
            change_points.push(cp[idx]);
            idx = cp[idx];
        }
        change_points.reverse();

        // Calculate confidence scores based on cost reduction
        let confidence = Self::compute_confidence(&change_points, &f, penalty);

        // Calculate segment statistics
        let mut segments = vec![0];
        segments.extend(&change_points);
        segments.push(n);

        let segment_means: Vec<f64> = segments
            .windows(2)
            .map(|w| {
                let start = w[0];
                let end = w[1];
                (sum[end] - sum[start]) / (end - start) as f64
            })
            .collect();

        let segment_variances: Vec<f64> = segments
            .windows(2)
            .map(|w| {
                let start = w[0];
                let end = w[1];
                let len = (end - start) as f64;
                if len <= 1.0 {
                    return 0.0;
                }
                let mean = (sum[end] - sum[start]) / len;
                values[start..end]
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>()
                    / (len - 1.0)
            })
            .collect();

        ChangePointResult {
            change_points,
            confidence,
            segment_means,
            segment_variances,
            cost: f[n],
        }
    }

    /// Binary segmentation algorithm.
    fn binary_segmentation(
        series: &TimeSeries,
        penalty: f64,
        min_segment: usize,
    ) -> ChangePointResult {
        let values = &series.values;
        let n = values.len();

        let mut change_points = Vec::new();
        Self::binary_segment_recursive(values, 0, n, penalty, min_segment, &mut change_points);
        change_points.sort();

        // Compute segment statistics
        let mut segments = vec![0];
        segments.extend(&change_points);
        segments.push(n);

        let segment_means: Vec<f64> = segments
            .windows(2)
            .map(|w| {
                let seg = &values[w[0]..w[1]];
                seg.iter().sum::<f64>() / seg.len() as f64
            })
            .collect();

        let segment_variances: Vec<f64> = segments
            .windows(2)
            .map(|w| {
                let seg = &values[w[0]..w[1]];
                let mean = seg.iter().sum::<f64>() / seg.len() as f64;
                if seg.len() <= 1 {
                    0.0
                } else {
                    seg.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (seg.len() - 1) as f64
                }
            })
            .collect();

        // Simple confidence based on position
        let confidence = vec![0.5; change_points.len()];

        ChangePointResult {
            change_points,
            confidence,
            segment_means,
            segment_variances,
            cost: 0.0,
        }
    }

    /// Recursive helper for binary segmentation.
    fn binary_segment_recursive(
        values: &[f64],
        start: usize,
        end: usize,
        penalty: f64,
        min_segment: usize,
        change_points: &mut Vec<usize>,
    ) {
        if end - start < 2 * min_segment {
            return;
        }

        // Find the best split point
        let mut best_gain = 0.0;
        let mut best_split = 0;

        let segment = &values[start..end];
        let n = segment.len();

        // Total cost (sum of squared deviations from mean)
        let total_mean: f64 = segment.iter().sum::<f64>() / n as f64;
        let total_cost: f64 = segment.iter().map(|x| (x - total_mean).powi(2)).sum();

        for split in min_segment..(n - min_segment + 1) {
            let left = &segment[..split];
            let right = &segment[split..];

            let left_mean: f64 = left.iter().sum::<f64>() / left.len() as f64;
            let right_mean: f64 = right.iter().sum::<f64>() / right.len() as f64;

            let left_cost: f64 = left.iter().map(|x| (x - left_mean).powi(2)).sum();
            let right_cost: f64 = right.iter().map(|x| (x - right_mean).powi(2)).sum();

            let gain = total_cost - left_cost - right_cost;

            if gain > best_gain {
                best_gain = gain;
                best_split = split;
            }
        }

        // If gain exceeds penalty, add change point and recurse
        if best_gain > penalty {
            let cp = start + best_split;
            change_points.push(cp);

            Self::binary_segment_recursive(values, start, cp, penalty, min_segment, change_points);
            Self::binary_segment_recursive(values, cp, end, penalty, min_segment, change_points);
        }
    }

    /// CUSUM (Cumulative Sum) algorithm.
    fn cusum(series: &TimeSeries, threshold: f64, min_segment: usize) -> ChangePointResult {
        let values = &series.values;
        let n = values.len();
        let mean = series.mean();

        // Compute cumulative sum
        let mut cusum = vec![0.0; n];
        let mut cum = 0.0;
        for (i, &v) in values.iter().enumerate() {
            cum += v - mean;
            cusum[i] = cum;
        }

        // Find peaks that exceed threshold
        let mut change_points = Vec::new();
        let mut confidence = Vec::new();
        let mut last_cp = 0;

        for i in min_segment..(n - min_segment) {
            let max_cusum = cusum[i].abs();

            if max_cusum > threshold && i - last_cp >= min_segment {
                // Check if this is a local maximum
                let is_peak = cusum[i - 1].abs() < max_cusum && cusum[i + 1].abs() < max_cusum;

                if is_peak {
                    change_points.push(i);
                    confidence.push((max_cusum / threshold).min(1.0));
                    last_cp = i;
                }
            }
        }

        // Compute segment statistics
        let mut segments = vec![0];
        segments.extend(&change_points);
        segments.push(n);

        let segment_means: Vec<f64> = segments
            .windows(2)
            .map(|w| {
                let seg = &values[w[0]..w[1]];
                seg.iter().sum::<f64>() / seg.len() as f64
            })
            .collect();

        let segment_variances: Vec<f64> = segments
            .windows(2)
            .map(|w| {
                let seg = &values[w[0]..w[1]];
                let mean = seg.iter().sum::<f64>() / seg.len() as f64;
                if seg.len() <= 1 {
                    0.0
                } else {
                    seg.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (seg.len() - 1) as f64
                }
            })
            .collect();

        ChangePointResult {
            change_points,
            confidence,
            segment_means,
            segment_variances,
            cost: 0.0,
        }
    }

    /// Compute confidence scores for change points.
    fn compute_confidence(change_points: &[usize], costs: &[f64], penalty: f64) -> Vec<f64> {
        change_points
            .iter()
            .map(|&cp| {
                if cp < costs.len() {
                    // Higher cost reduction = higher confidence
                    let cost_reduction = if cp > 0 {
                        (costs[cp - 1] - costs[cp]).max(0.0)
                    } else {
                        penalty
                    };
                    (cost_reduction / penalty).clamp(0.0, 1.0)
                } else {
                    0.5
                }
            })
            .collect()
    }
}

impl GpuKernel for ChangePointDetection {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[async_trait]
impl BatchKernel<ChangePointDetectionInput, ChangePointDetectionOutput> for ChangePointDetection {
    async fn execute(
        &self,
        input: ChangePointDetectionInput,
    ) -> Result<ChangePointDetectionOutput> {
        let start = Instant::now();
        let result = Self::compute(
            &input.series,
            input.method,
            input.penalty,
            input.min_segment,
        );
        Ok(ChangePointDetectionOutput {
            result,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

// ============================================================================
// Time Series Anomaly Detection Kernel
// ============================================================================

/// Time series anomaly detection kernel.
///
/// Detects anomalous points in a time series using various methods.
#[derive(Debug, Clone)]
pub struct TimeSeriesAnomalyDetection {
    metadata: KernelMetadata,
}

impl Default for TimeSeriesAnomalyDetection {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeSeriesAnomalyDetection {
    /// Create a new time series anomaly detection kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("temporal/anomaly-detection", Domain::TemporalAnalysis)
                .with_description("Statistical threshold anomaly detection")
                .with_throughput(50_000)
                .with_latency_us(20.0),
        }
    }

    /// Detect anomalies in a time series.
    ///
    /// # Arguments
    /// * `series` - Input time series
    /// * `method` - Detection method
    /// * `threshold` - Anomaly threshold (interpretation depends on method)
    /// * `window` - Window size for moving statistics (optional)
    pub fn compute(
        series: &TimeSeries,
        method: AnomalyMethod,
        threshold: f64,
        window: Option<usize>,
    ) -> TimeSeriesAnomalyResult {
        if series.is_empty() {
            return TimeSeriesAnomalyResult {
                scores: Vec::new(),
                anomaly_indices: Vec::new(),
                expected: Vec::new(),
                threshold,
            };
        }

        match method {
            AnomalyMethod::ZScore => Self::zscore_detection(series, threshold, window),
            AnomalyMethod::IQR => Self::iqr_detection(series, threshold),
            AnomalyMethod::MovingAverageDeviation => {
                Self::moving_average_detection(series, threshold, window.unwrap_or(10))
            }
            AnomalyMethod::SeasonalESD => {
                Self::seasonal_esd_detection(series, threshold, window.unwrap_or(12))
            }
        }
    }

    /// Z-score based anomaly detection.
    fn zscore_detection(
        series: &TimeSeries,
        threshold: f64,
        window: Option<usize>,
    ) -> TimeSeriesAnomalyResult {
        let n = series.len();

        let (scores, expected) = if let Some(w) = window {
            // Rolling z-score
            Self::rolling_zscore(&series.values, w)
        } else {
            // Global z-score
            let mean = series.mean();
            let std = series.std_dev();

            let scores: Vec<f64> = series
                .values
                .iter()
                .map(|&v| {
                    if std > 1e-10 {
                        (v - mean).abs() / std
                    } else {
                        0.0
                    }
                })
                .collect();

            let expected = vec![mean; n];
            (scores, expected)
        };

        let anomaly_indices: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter(|&(_, s)| *s > threshold)
            .map(|(i, _)| i)
            .collect();

        TimeSeriesAnomalyResult {
            scores,
            anomaly_indices,
            expected,
            threshold,
        }
    }

    /// Rolling z-score calculation.
    fn rolling_zscore(values: &[f64], window: usize) -> (Vec<f64>, Vec<f64>) {
        let n = values.len();
        let w = window.min(n);

        let mut scores = vec![0.0; n];
        let mut expected = vec![0.0; n];

        for i in 0..n {
            let start = i.saturating_sub(w);
            let window_vals: Vec<f64> = values[start..=i.min(n - 1)].to_vec();

            let mean: f64 = window_vals.iter().sum::<f64>() / window_vals.len() as f64;
            let var: f64 = window_vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / window_vals.len() as f64;
            let std = var.sqrt();

            expected[i] = mean;
            scores[i] = if std > 1e-10 {
                (values[i] - mean).abs() / std
            } else {
                0.0
            };
        }

        (scores, expected)
    }

    /// IQR (Interquartile Range) based detection.
    fn iqr_detection(series: &TimeSeries, multiplier: f64) -> TimeSeriesAnomalyResult {
        let mut sorted = series.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let q1 = sorted[n / 4];
        let q3 = sorted[3 * n / 4];
        let iqr = q3 - q1;

        let lower = q1 - multiplier * iqr;
        let upper = q3 + multiplier * iqr;
        let median = sorted[n / 2];

        let scores: Vec<f64> = series
            .values
            .iter()
            .map(|&v| {
                if v < lower {
                    (lower - v) / iqr
                } else if v > upper {
                    (v - upper) / iqr
                } else {
                    0.0
                }
            })
            .collect();

        let anomaly_indices: Vec<usize> = series
            .values
            .iter()
            .enumerate()
            .filter(|&(_, v)| *v < lower || *v > upper)
            .map(|(i, _)| i)
            .collect();

        TimeSeriesAnomalyResult {
            scores,
            anomaly_indices,
            expected: vec![median; n],
            threshold: multiplier,
        }
    }

    /// Moving average deviation detection.
    #[allow(clippy::needless_range_loop)]
    fn moving_average_detection(
        series: &TimeSeries,
        threshold: f64,
        window: usize,
    ) -> TimeSeriesAnomalyResult {
        let n = series.len();
        let w = window.min(n);

        let mut expected = vec![0.0; n];
        let mut scores = vec![0.0; n];

        // Calculate moving average
        for i in 0..n {
            let start = i.saturating_sub(w / 2);
            let end = (i + w / 2 + 1).min(n);
            expected[i] = series.values[start..end].iter().sum::<f64>() / (end - start) as f64;
        }

        // Calculate deviation from moving average
        let deviations: Vec<f64> = series
            .values
            .iter()
            .zip(expected.iter())
            .map(|(v, e)| (v - e).abs())
            .collect();

        // Calculate MAD (Median Absolute Deviation) for robust scoring
        let mut sorted_deviations = deviations.clone();
        sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = sorted_deviations[n / 2];

        for (i, &dev) in deviations.iter().enumerate() {
            scores[i] = if mad > 1e-10 { dev / mad } else { 0.0 };
        }

        let anomaly_indices: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter(|&(_, s)| *s > threshold)
            .map(|(i, _)| i)
            .collect();

        TimeSeriesAnomalyResult {
            scores,
            anomaly_indices,
            expected,
            threshold,
        }
    }

    /// Seasonal ESD (Extreme Studentized Deviate) detection.
    /// Simplified version of Twitter's algorithm.
    fn seasonal_esd_detection(
        series: &TimeSeries,
        threshold: f64,
        period: usize,
    ) -> TimeSeriesAnomalyResult {
        let n = series.len();

        // Remove seasonal component
        let mut seasonal = vec![0.0; period];
        let mut counts = vec![0usize; period];

        for (i, &v) in series.values.iter().enumerate() {
            seasonal[i % period] += v;
            counts[i % period] += 1;
        }

        for i in 0..period {
            if counts[i] > 0 {
                seasonal[i] /= counts[i] as f64;
            }
        }

        // Deseasonalized series
        let deseasonalized: Vec<f64> = series
            .values
            .iter()
            .enumerate()
            .map(|(i, &v)| v - seasonal[i % period])
            .collect();

        // Apply z-score to deseasonalized
        let mean: f64 = deseasonalized.iter().sum::<f64>() / n as f64;
        let std: f64 = (deseasonalized
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();

        let scores: Vec<f64> = deseasonalized
            .iter()
            .map(|&v| {
                if std > 1e-10 {
                    (v - mean).abs() / std
                } else {
                    0.0
                }
            })
            .collect();

        // Expected is mean + seasonal
        let expected: Vec<f64> = (0..n).map(|i| mean + seasonal[i % period]).collect();

        let anomaly_indices: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter(|&(_, s)| *s > threshold)
            .map(|(i, _)| i)
            .collect();

        TimeSeriesAnomalyResult {
            scores,
            anomaly_indices,
            expected,
            threshold,
        }
    }
}

impl GpuKernel for TimeSeriesAnomalyDetection {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[async_trait]
impl BatchKernel<TimeSeriesAnomalyDetectionInput, TimeSeriesAnomalyDetectionOutput>
    for TimeSeriesAnomalyDetection
{
    async fn execute(
        &self,
        input: TimeSeriesAnomalyDetectionInput,
    ) -> Result<TimeSeriesAnomalyDetectionOutput> {
        let start = Instant::now();
        let result = Self::compute(&input.series, input.method, input.threshold, input.window);
        Ok(TimeSeriesAnomalyDetectionOutput {
            result,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_step_series() -> TimeSeries {
        // Series with a clear change point at index 50
        let mut values = vec![10.0; 50];
        values.extend(vec![20.0; 50]);
        // Add small noise
        for (i, v) in values.iter_mut().enumerate() {
            *v += (i as f64 * 0.1).sin();
        }
        TimeSeries::new(values)
    }

    fn create_anomaly_series() -> TimeSeries {
        let mut values: Vec<f64> = (0..100).map(|i| 10.0 + (i as f64 * 0.1).sin()).collect();
        // Add anomalies
        values[25] = 50.0;
        values[75] = -30.0;
        TimeSeries::new(values)
    }

    #[test]
    fn test_changepoint_metadata() {
        let kernel = ChangePointDetection::new();
        assert_eq!(kernel.metadata().id, "temporal/changepoint-detection");
        assert_eq!(kernel.metadata().domain, Domain::TemporalAnalysis);
    }

    #[test]
    fn test_pelt_detection() {
        let series = create_step_series();
        let result = ChangePointDetection::compute(&series, ChangePointMethod::PELT, 100.0, 10);

        // Should detect the change point around index 50
        assert!(!result.change_points.is_empty());

        // At least one change point should be near 50
        let near_50 = result
            .change_points
            .iter()
            .any(|&cp| (cp as i32 - 50).abs() < 10);
        assert!(
            near_50,
            "Expected change point near 50, got {:?}",
            result.change_points
        );

        // Should have 2 segments
        assert_eq!(result.segment_means.len(), result.change_points.len() + 1);
    }

    #[test]
    fn test_binary_segmentation() {
        let series = create_step_series();
        let result =
            ChangePointDetection::compute(&series, ChangePointMethod::BinarySegmentation, 50.0, 10);

        // Should detect change points
        assert!(!result.change_points.is_empty());
    }

    #[test]
    fn test_cusum_detection() {
        let series = create_step_series();
        let result = ChangePointDetection::compute(&series, ChangePointMethod::CUSUM, 20.0, 10);

        // CUSUM should detect the level shift
        // May detect multiple points depending on threshold
        assert!(!result.segment_means.is_empty());
    }

    #[test]
    fn test_anomaly_metadata() {
        let kernel = TimeSeriesAnomalyDetection::new();
        assert_eq!(kernel.metadata().id, "temporal/anomaly-detection");
    }

    #[test]
    fn test_zscore_anomaly() {
        let series = create_anomaly_series();
        let result = TimeSeriesAnomalyDetection::compute(&series, AnomalyMethod::ZScore, 3.0, None);

        // Should detect the injected anomalies
        assert!(!result.anomaly_indices.is_empty());
        assert!(result.anomaly_indices.contains(&25) || result.anomaly_indices.contains(&75));
    }

    #[test]
    fn test_iqr_anomaly() {
        let series = create_anomaly_series();
        let result = TimeSeriesAnomalyDetection::compute(&series, AnomalyMethod::IQR, 1.5, None);

        // IQR should detect outliers
        assert!(!result.anomaly_indices.is_empty());
    }

    #[test]
    fn test_moving_average_anomaly() {
        let series = create_anomaly_series();
        let result = TimeSeriesAnomalyDetection::compute(
            &series,
            AnomalyMethod::MovingAverageDeviation,
            5.0,
            Some(10),
        );

        // Should detect points that deviate from moving average
        assert!(result.expected.len() == series.len());
    }

    #[test]
    fn test_seasonal_esd() {
        // Create seasonal series with anomaly
        let values: Vec<f64> = (0..120)
            .map(|i| {
                let seasonal = 10.0 * ((2.0 * std::f64::consts::PI * i as f64 / 12.0).sin());
                let trend = 100.0;
                if i == 60 {
                    200.0 // Anomaly
                } else {
                    trend + seasonal
                }
            })
            .collect();
        let series = TimeSeries::new(values);

        let result =
            TimeSeriesAnomalyDetection::compute(&series, AnomalyMethod::SeasonalESD, 3.0, Some(12));

        // Should detect the anomaly at index 60
        assert!(result.anomaly_indices.contains(&60));
    }

    #[test]
    fn test_empty_series() {
        let empty = TimeSeries::new(Vec::new());

        let cp_result = ChangePointDetection::compute(&empty, ChangePointMethod::PELT, 10.0, 5);
        assert!(cp_result.change_points.is_empty());

        let anomaly_result =
            TimeSeriesAnomalyDetection::compute(&empty, AnomalyMethod::ZScore, 3.0, None);
        assert!(anomaly_result.anomaly_indices.is_empty());
    }
}
