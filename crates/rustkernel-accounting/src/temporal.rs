//! Temporal correlation kernel.
//!
//! This module provides temporal correlation analysis for accounting:
//! - Calculate correlations between account time series
//! - Detect anomalies based on expected correlations
//! - Identify pattern changes

use crate::types::{
    AccountCorrelation, AnomalyType, CorrelationAnomaly, CorrelationResult, CorrelationStats,
    CorrelationType, AccountTimeSeries, TimeSeriesPoint,
};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// Temporal Correlation Kernel
// ============================================================================

/// Temporal correlation kernel.
///
/// Analyzes correlations between account time series.
#[derive(Debug, Clone)]
pub struct TemporalCorrelation {
    metadata: KernelMetadata,
}

impl Default for TemporalCorrelation {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalCorrelation {
    /// Create a new temporal correlation kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("accounting/temporal-correlation", Domain::Accounting)
                .with_description("Account time series correlation analysis")
                .with_throughput(5_000)
                .with_latency_us(500.0),
        }
    }

    /// Calculate correlations between account time series.
    pub fn correlate(
        time_series: &[AccountTimeSeries],
        config: &CorrelationConfig,
    ) -> CorrelationResult {
        let mut correlations = Vec::new();
        let mut anomalies = Vec::new();

        // Calculate pairwise correlations
        for i in 0..time_series.len() {
            for j in (i + 1)..time_series.len() {
                let ts_a = &time_series[i];
                let ts_b = &time_series[j];

                if let Some(corr) = Self::calculate_correlation(ts_a, ts_b, config) {
                    correlations.push(corr);
                }
            }
        }

        // Detect anomalies based on expected correlations
        if let Some(ref expected) = config.expected_correlations {
            for (pair, expected_coef) in expected {
                let actual = correlations.iter().find(|c| {
                    (c.account_a == pair.0 && c.account_b == pair.1)
                        || (c.account_a == pair.1 && c.account_b == pair.0)
                });

                if let Some(actual_corr) = actual {
                    let diff = (actual_corr.coefficient - expected_coef).abs();
                    if diff > config.correlation_threshold {
                        // Find the corresponding time series
                        if let Some(ts) = time_series.iter().find(|t| t.account_code == pair.0) {
                            if let Some(last_point) = ts.data_points.last() {
                                anomalies.push(CorrelationAnomaly {
                                    account_code: pair.0.clone(),
                                    date: last_point.date,
                                    expected: *expected_coef,
                                    actual: actual_corr.coefficient,
                                    z_score: diff / 0.1, // Simplified z-score
                                    anomaly_type: AnomalyType::MissingCorrelation,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Detect point anomalies using correlation-based prediction
        for ts in time_series {
            let related: Vec<_> = correlations
                .iter()
                .filter(|c| c.account_a == ts.account_code || c.account_b == ts.account_code)
                .filter(|c| c.coefficient.abs() > config.significant_correlation)
                .collect();

            if !related.is_empty() {
                let ts_anomalies = Self::detect_point_anomalies(ts, &related, time_series, config);
                anomalies.extend(ts_anomalies);
            }
        }

        let significant_count = correlations
            .iter()
            .filter(|c| c.coefficient.abs() >= config.significant_correlation)
            .count();

        let avg_correlation = if !correlations.is_empty() {
            correlations.iter().map(|c| c.coefficient.abs()).sum::<f64>() / correlations.len() as f64
        } else {
            0.0
        };

        let anomaly_count = anomalies.len();

        CorrelationResult {
            correlations,
            anomalies,
            stats: CorrelationStats {
                accounts_analyzed: time_series.len(),
                significant_correlations: significant_count,
                anomaly_count,
                avg_correlation,
            },
        }
    }

    /// Calculate correlation between two time series.
    fn calculate_correlation(
        ts_a: &AccountTimeSeries,
        ts_b: &AccountTimeSeries,
        config: &CorrelationConfig,
    ) -> Option<AccountCorrelation> {
        // Align time series by date
        let (values_a, values_b) = Self::align_series(ts_a, ts_b);

        if values_a.len() < config.min_data_points {
            return None;
        }

        // Calculate Pearson correlation
        let n = values_a.len() as f64;
        let sum_a: f64 = values_a.iter().sum();
        let sum_b: f64 = values_b.iter().sum();
        let sum_ab: f64 = values_a.iter().zip(values_b.iter()).map(|(a, b)| a * b).sum();
        let sum_a2: f64 = values_a.iter().map(|a| a * a).sum();
        let sum_b2: f64 = values_b.iter().map(|b| b * b).sum();

        let numerator = n * sum_ab - sum_a * sum_b;
        let denominator = ((n * sum_a2 - sum_a * sum_a) * (n * sum_b2 - sum_b * sum_b)).sqrt();

        if denominator.abs() < 1e-10 {
            return None;
        }

        let coefficient = numerator / denominator;

        // Calculate p-value (simplified t-test approximation)
        let t_stat = coefficient * ((n - 2.0) / (1.0 - coefficient * coefficient)).sqrt();
        let p_value = Self::t_distribution_pvalue(t_stat.abs(), (n - 2.0) as u32);

        let correlation_type = if p_value > config.significance_level {
            CorrelationType::None
        } else if coefficient > 0.0 {
            CorrelationType::Positive
        } else {
            CorrelationType::Negative
        };

        Some(AccountCorrelation {
            account_a: ts_a.account_code.clone(),
            account_b: ts_b.account_code.clone(),
            coefficient,
            p_value,
            correlation_type,
        })
    }

    /// Align two time series by date.
    fn align_series(ts_a: &AccountTimeSeries, ts_b: &AccountTimeSeries) -> (Vec<f64>, Vec<f64>) {
        let dates_a: HashMap<u64, f64> = ts_a
            .data_points
            .iter()
            .map(|p| (p.date, p.balance))
            .collect();

        let dates_b: HashMap<u64, f64> = ts_b
            .data_points
            .iter()
            .map(|p| (p.date, p.balance))
            .collect();

        let common_dates: Vec<u64> = dates_a
            .keys()
            .filter(|d| dates_b.contains_key(d))
            .copied()
            .collect();

        let values_a: Vec<f64> = common_dates.iter().filter_map(|d| dates_a.get(d)).copied().collect();
        let values_b: Vec<f64> = common_dates.iter().filter_map(|d| dates_b.get(d)).copied().collect();

        (values_a, values_b)
    }

    /// Simplified p-value from t-distribution.
    fn t_distribution_pvalue(t: f64, df: u32) -> f64 {
        // Simplified approximation using normal distribution for large df
        if df > 30 {
            2.0 * (1.0 - Self::normal_cdf(t))
        } else {
            // Very rough approximation
            2.0 * (1.0 - Self::normal_cdf(t * (1.0 - 1.0 / (4.0 * df as f64))))
        }
    }

    /// Standard normal CDF approximation.
    fn normal_cdf(x: f64) -> f64 {
        // Approximation using error function
        0.5 * (1.0 + Self::erf(x / std::f64::consts::SQRT_2))
    }

    /// Error function approximation.
    fn erf(x: f64) -> f64 {
        // Horner form approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Detect point anomalies using correlation-based prediction.
    fn detect_point_anomalies(
        ts: &AccountTimeSeries,
        related_correlations: &[&AccountCorrelation],
        all_series: &[AccountTimeSeries],
        config: &CorrelationConfig,
    ) -> Vec<CorrelationAnomaly> {
        let mut anomalies = Vec::new();

        for point in &ts.data_points {
            // Predict value based on correlated accounts
            let mut predictions = Vec::new();

            for corr in related_correlations {
                let related_code = if corr.account_a == ts.account_code {
                    &corr.account_b
                } else {
                    &corr.account_a
                };

                if let Some(related_ts) = all_series.iter().find(|t| t.account_code == *related_code) {
                    if let Some(related_point) = related_ts.data_points.iter().find(|p| p.date == point.date) {
                        // Simple linear prediction
                        let predicted = related_point.balance * corr.coefficient;
                        predictions.push(predicted);
                    }
                }
            }

            if predictions.is_empty() {
                continue;
            }

            let avg_prediction = predictions.iter().sum::<f64>() / predictions.len() as f64;
            let std_dev = if predictions.len() > 1 {
                let variance = predictions.iter()
                    .map(|p| (p - avg_prediction).powi(2))
                    .sum::<f64>() / (predictions.len() - 1) as f64;
                variance.sqrt()
            } else {
                avg_prediction.abs() * 0.1 // Fallback
            };

            if std_dev > 0.0 {
                let z_score = (point.balance - avg_prediction) / std_dev;

                if z_score.abs() > config.anomaly_threshold {
                    let anomaly_type = if z_score > 0.0 {
                        AnomalyType::UnexpectedHigh
                    } else {
                        AnomalyType::UnexpectedLow
                    };

                    anomalies.push(CorrelationAnomaly {
                        account_code: ts.account_code.clone(),
                        date: point.date,
                        expected: avg_prediction,
                        actual: point.balance,
                        z_score,
                        anomaly_type,
                    });
                }
            }
        }

        anomalies
    }

    /// Calculate rolling correlation over time windows.
    pub fn rolling_correlation(
        ts_a: &AccountTimeSeries,
        ts_b: &AccountTimeSeries,
        window_size: usize,
    ) -> Vec<RollingCorrelation> {
        let mut results = Vec::new();
        let (values_a, values_b) = Self::align_series(ts_a, ts_b);

        if values_a.len() < window_size {
            return results;
        }

        // Get dates for the aligned series
        let dates_a: HashMap<u64, usize> = ts_a
            .data_points
            .iter()
            .enumerate()
            .map(|(i, p)| (p.date, i))
            .collect();

        let common_dates: Vec<u64> = ts_a
            .data_points
            .iter()
            .filter(|p| {
                ts_b.data_points.iter().any(|pb| pb.date == p.date)
            })
            .map(|p| p.date)
            .collect();

        for i in window_size..=values_a.len() {
            let window_a = &values_a[i - window_size..i];
            let window_b = &values_b[i - window_size..i];

            let correlation = Self::calculate_window_correlation(window_a, window_b);

            if i - 1 < common_dates.len() {
                results.push(RollingCorrelation {
                    end_date: common_dates[i - 1],
                    correlation,
                    window_size,
                });
            }
        }

        results
    }

    /// Calculate correlation for a window.
    fn calculate_window_correlation(values_a: &[f64], values_b: &[f64]) -> f64 {
        let n = values_a.len() as f64;
        let sum_a: f64 = values_a.iter().sum();
        let sum_b: f64 = values_b.iter().sum();
        let sum_ab: f64 = values_a.iter().zip(values_b.iter()).map(|(a, b)| a * b).sum();
        let sum_a2: f64 = values_a.iter().map(|a| a * a).sum();
        let sum_b2: f64 = values_b.iter().map(|b| b * b).sum();

        let numerator = n * sum_ab - sum_a * sum_b;
        let denominator = ((n * sum_a2 - sum_a * sum_a) * (n * sum_b2 - sum_b * sum_b)).sqrt();

        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Detect structural breaks in correlation.
    pub fn detect_correlation_breaks(
        rolling: &[RollingCorrelation],
        threshold: f64,
    ) -> Vec<CorrelationBreak> {
        let mut breaks = Vec::new();

        for i in 1..rolling.len() {
            let change = rolling[i].correlation - rolling[i - 1].correlation;
            if change.abs() > threshold {
                breaks.push(CorrelationBreak {
                    date: rolling[i].end_date,
                    change,
                    before: rolling[i - 1].correlation,
                    after: rolling[i].correlation,
                });
            }
        }

        breaks
    }
}

impl GpuKernel for TemporalCorrelation {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Correlation configuration.
#[derive(Debug, Clone)]
pub struct CorrelationConfig {
    /// Minimum data points required.
    pub min_data_points: usize,
    /// Significance level for correlation.
    pub significance_level: f64,
    /// Threshold for significant correlation.
    pub significant_correlation: f64,
    /// Correlation change threshold.
    pub correlation_threshold: f64,
    /// Z-score threshold for anomalies.
    pub anomaly_threshold: f64,
    /// Expected correlations (account pair -> expected coefficient).
    pub expected_correlations: Option<HashMap<(String, String), f64>>,
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            min_data_points: 10,
            significance_level: 0.05,
            significant_correlation: 0.5,
            correlation_threshold: 0.3,
            anomaly_threshold: 2.0,
            expected_correlations: None,
        }
    }
}

/// Rolling correlation result.
#[derive(Debug, Clone)]
pub struct RollingCorrelation {
    /// End date of window.
    pub end_date: u64,
    /// Correlation value.
    pub correlation: f64,
    /// Window size used.
    pub window_size: usize,
}

/// Correlation break point.
#[derive(Debug, Clone)]
pub struct CorrelationBreak {
    /// Date of break.
    pub date: u64,
    /// Change in correlation.
    pub change: f64,
    /// Correlation before break.
    pub before: f64,
    /// Correlation after break.
    pub after: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TimeFrequency;

    fn create_correlated_series() -> (AccountTimeSeries, AccountTimeSeries) {
        let base_values = vec![100.0, 110.0, 105.0, 120.0, 115.0, 130.0, 125.0, 140.0, 135.0, 150.0];
        let correlated_values: Vec<f64> = base_values.iter().map(|v| v * 0.5 + 20.0).collect();

        let ts_a = AccountTimeSeries {
            account_code: "1000".to_string(),
            data_points: base_values
                .iter()
                .enumerate()
                .map(|(i, &v)| TimeSeriesPoint {
                    date: 1700000000 + (i as u64 * 86400),
                    balance: v,
                    period_change: if i > 0 { v - base_values[i - 1] } else { 0.0 },
                })
                .collect(),
            frequency: TimeFrequency::Daily,
        };

        let ts_b = AccountTimeSeries {
            account_code: "2000".to_string(),
            data_points: correlated_values
                .iter()
                .enumerate()
                .map(|(i, &v)| TimeSeriesPoint {
                    date: 1700000000 + (i as u64 * 86400),
                    balance: v,
                    period_change: if i > 0 { v - correlated_values[i - 1] } else { 0.0 },
                })
                .collect(),
            frequency: TimeFrequency::Daily,
        };

        (ts_a, ts_b)
    }

    #[test]
    fn test_temporal_metadata() {
        let kernel = TemporalCorrelation::new();
        assert_eq!(kernel.metadata().id, "accounting/temporal-correlation");
        assert_eq!(kernel.metadata().domain, Domain::Accounting);
    }

    #[test]
    fn test_positive_correlation() {
        let (ts_a, ts_b) = create_correlated_series();
        let time_series = vec![ts_a, ts_b];
        let config = CorrelationConfig::default();

        let result = TemporalCorrelation::correlate(&time_series, &config);

        assert_eq!(result.correlations.len(), 1);
        let corr = &result.correlations[0];
        assert!(corr.coefficient > 0.9); // Should be highly correlated
        assert_eq!(corr.correlation_type, CorrelationType::Positive);
    }

    #[test]
    fn test_negative_correlation() {
        let (ts_a, mut ts_b) = create_correlated_series();

        // Make ts_b negatively correlated
        for point in &mut ts_b.data_points {
            point.balance = 200.0 - point.balance;
        }

        let time_series = vec![ts_a, ts_b];
        let config = CorrelationConfig::default();

        let result = TemporalCorrelation::correlate(&time_series, &config);

        assert_eq!(result.correlations.len(), 1);
        let corr = &result.correlations[0];
        assert!(corr.coefficient < -0.9); // Should be highly negatively correlated
        assert_eq!(corr.correlation_type, CorrelationType::Negative);
    }

    #[test]
    fn test_no_correlation() {
        let ts_a = AccountTimeSeries {
            account_code: "1000".to_string(),
            data_points: (0..10)
                .map(|i| TimeSeriesPoint {
                    date: 1700000000 + (i as u64 * 86400),
                    balance: 100.0 + (i as f64),
                    period_change: 1.0,
                })
                .collect(),
            frequency: TimeFrequency::Daily,
        };

        // Random-ish uncorrelated data
        let ts_b = AccountTimeSeries {
            account_code: "2000".to_string(),
            data_points: (0..10)
                .map(|i| TimeSeriesPoint {
                    date: 1700000000 + (i as u64 * 86400),
                    balance: [50.0, 80.0, 45.0, 90.0, 40.0, 85.0, 35.0, 95.0, 30.0, 100.0][i],
                    period_change: 0.0,
                })
                .collect(),
            frequency: TimeFrequency::Daily,
        };

        let time_series = vec![ts_a, ts_b];
        let config = CorrelationConfig::default();

        let result = TemporalCorrelation::correlate(&time_series, &config);

        assert_eq!(result.correlations.len(), 1);
        // Correlation should be weak
        assert!(result.correlations[0].coefficient.abs() < 0.5);
    }

    #[test]
    fn test_insufficient_data() {
        let ts_a = AccountTimeSeries {
            account_code: "1000".to_string(),
            data_points: vec![
                TimeSeriesPoint { date: 1700000000, balance: 100.0, period_change: 0.0 },
            ],
            frequency: TimeFrequency::Daily,
        };

        let ts_b = AccountTimeSeries {
            account_code: "2000".to_string(),
            data_points: vec![
                TimeSeriesPoint { date: 1700000000, balance: 50.0, period_change: 0.0 },
            ],
            frequency: TimeFrequency::Daily,
        };

        let time_series = vec![ts_a, ts_b];
        let config = CorrelationConfig {
            min_data_points: 10,
            ..Default::default()
        };

        let result = TemporalCorrelation::correlate(&time_series, &config);

        assert!(result.correlations.is_empty());
    }

    #[test]
    fn test_rolling_correlation() {
        let (ts_a, ts_b) = create_correlated_series();

        let rolling = TemporalCorrelation::rolling_correlation(&ts_a, &ts_b, 5);

        assert!(!rolling.is_empty());
        // All rolling correlations should be high for perfectly correlated series
        assert!(rolling.iter().all(|r| r.correlation > 0.9));
    }

    #[test]
    fn test_correlation_break_detection() {
        let rolling = vec![
            RollingCorrelation { end_date: 1700000000, correlation: 0.9, window_size: 5 },
            RollingCorrelation { end_date: 1700086400, correlation: 0.85, window_size: 5 },
            RollingCorrelation { end_date: 1700172800, correlation: 0.2, window_size: 5 }, // Break!
            RollingCorrelation { end_date: 1700259200, correlation: 0.25, window_size: 5 },
        ];

        let breaks = TemporalCorrelation::detect_correlation_breaks(&rolling, 0.5);

        assert_eq!(breaks.len(), 1);
        assert!((breaks[0].change - (-0.65)).abs() < 0.01);
    }

    #[test]
    fn test_correlation_stats() {
        let (ts_a, ts_b) = create_correlated_series();
        let time_series = vec![ts_a, ts_b];
        let config = CorrelationConfig::default();

        let result = TemporalCorrelation::correlate(&time_series, &config);

        assert_eq!(result.stats.accounts_analyzed, 2);
        assert_eq!(result.stats.significant_correlations, 1);
    }

    #[test]
    fn test_expected_correlation_anomaly() {
        let (ts_a, mut ts_b) = create_correlated_series();

        // Make ts_b uncorrelated
        for (i, point) in ts_b.data_points.iter_mut().enumerate() {
            point.balance = [50.0, 80.0, 45.0, 90.0, 40.0, 85.0, 35.0, 95.0, 30.0, 100.0][i];
        }

        let mut expected = HashMap::new();
        expected.insert(("1000".to_string(), "2000".to_string()), 0.9);

        let time_series = vec![ts_a, ts_b];
        let config = CorrelationConfig {
            expected_correlations: Some(expected),
            correlation_threshold: 0.3,
            ..Default::default()
        };

        let result = TemporalCorrelation::correlate(&time_series, &config);

        // Should detect anomaly because correlation doesn't match expected
        assert!(result.anomalies.iter().any(|a| a.anomaly_type == AnomalyType::MissingCorrelation));
    }
}
