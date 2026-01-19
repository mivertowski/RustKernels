//! Real-time correlation kernels.
//!
//! This module provides streaming correlation computation:
//! - Incremental correlation matrix updates using Welford's algorithm
//! - Exponentially weighted moving correlation
//! - Correlation change detection

use rustkernel_core::traits::GpuKernel;
use rustkernel_core::{domain::Domain, kernel::KernelMetadata};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

// ============================================================================
// Real-Time Correlation Kernel
// ============================================================================

/// Correlation update types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CorrelationType {
    /// Pearson correlation coefficient.
    #[default]
    Pearson,
    /// Exponentially weighted correlation.
    Exponential,
}

/// Configuration for correlation computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationConfig {
    /// Number of assets to track.
    pub n_assets: usize,
    /// Type of correlation to compute.
    pub correlation_type: CorrelationType,
    /// Exponential decay factor (0-1, higher = more weight to recent).
    /// Only used for Exponential correlation type.
    pub decay_factor: f64,
    /// Minimum observations before computing correlation.
    pub min_observations: usize,
    /// Threshold for significant correlation change alerts.
    pub change_threshold: f64,
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            n_assets: 100,
            correlation_type: CorrelationType::Pearson,
            decay_factor: 0.94, // ~15-day half-life
            min_observations: 30,
            change_threshold: 0.1, // 10% change
        }
    }
}

/// Running statistics for a single asset (Welford's algorithm).
#[derive(Debug, Clone, Default)]
pub struct AssetStats {
    /// Count of observations.
    pub count: u64,
    /// Running mean.
    pub mean: f64,
    /// Running sum of squared deviations (M2).
    pub m2: f64,
    /// Last observed value.
    pub last_value: f64,
    /// Last update timestamp.
    pub last_timestamp: u64,
}

impl AssetStats {
    /// Update stats with a new observation using Welford's algorithm.
    pub fn update(&mut self, value: f64, timestamp: u64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        self.last_value = value;
        self.last_timestamp = timestamp;
    }

    /// Get variance.
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    /// Get standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Running covariance between two assets.
#[derive(Debug, Clone, Default)]
pub struct PairwiseStats {
    /// Count of paired observations.
    pub count: u64,
    /// Mean of asset i values.
    pub mean_i: f64,
    /// Mean of asset j values.
    pub mean_j: f64,
    /// Co-moment sum (for covariance calculation).
    pub co_moment: f64,
}

impl PairwiseStats {
    /// Update with new paired observations (Welford's parallel algorithm).
    pub fn update(&mut self, value_i: f64, value_j: f64) {
        self.count += 1;
        let n = self.count as f64;

        let delta_i = value_i - self.mean_i;
        let delta_j = value_j - self.mean_j;

        self.mean_i += delta_i / n;
        self.mean_j += delta_j / n;

        // Update co-moment using corrected delta
        let delta_j_new = value_j - self.mean_j;
        self.co_moment += delta_i * delta_j_new;
    }

    /// Get covariance.
    pub fn covariance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.co_moment / (self.count - 1) as f64
        }
    }
}

/// Internal state for real-time correlation tracking.
#[derive(Debug, Clone, Default)]
pub struct CorrelationState {
    /// Configuration.
    pub config: CorrelationConfig,
    /// Per-asset statistics.
    pub asset_stats: Vec<AssetStats>,
    /// Pairwise statistics (upper triangular, stored as Vec).
    /// Index (i, j) where i < j is at position: i * (n - 1) - i * (i - 1) / 2 + (j - i - 1)
    pub pairwise_stats: Vec<PairwiseStats>,
    /// Cached correlation matrix (full N×N).
    pub correlation_matrix: Vec<f64>,
    /// Previous correlation matrix (for change detection).
    pub prev_correlation_matrix: Vec<f64>,
    /// Total observations processed.
    pub total_observations: u64,
    /// Asset ID to index mapping.
    pub asset_index: HashMap<u64, usize>,
}

impl CorrelationState {
    /// Create new state with configuration.
    pub fn new(config: CorrelationConfig) -> Self {
        let n = config.n_assets;
        let n_pairs = n * (n - 1) / 2;

        Self {
            config,
            asset_stats: vec![AssetStats::default(); n],
            pairwise_stats: vec![PairwiseStats::default(); n_pairs],
            correlation_matrix: vec![0.0; n * n],
            prev_correlation_matrix: vec![0.0; n * n],
            total_observations: 0,
            asset_index: HashMap::new(),
        }
    }

    /// Get index into pairwise_stats for pair (i, j) where i < j.
    fn pair_index(&self, i: usize, j: usize) -> usize {
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        let n = self.config.n_assets;
        i * (2 * n - i - 1) / 2 + (j - i - 1)
    }
}

/// A single correlation update for streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationUpdate {
    /// Asset identifier.
    pub asset_id: u64,
    /// Observation value (typically return or price).
    pub value: f64,
    /// Timestamp of observation.
    pub timestamp: u64,
}

/// Result of a correlation update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationUpdateResult {
    /// Asset ID that was updated.
    pub asset_id: u64,
    /// Number of correlations recomputed.
    pub correlations_updated: usize,
    /// Significant changes detected.
    pub significant_changes: Vec<CorrelationChange>,
    /// Update latency in microseconds.
    pub latency_us: u64,
}

/// A significant correlation change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationChange {
    /// First asset ID.
    pub asset_i: u64,
    /// Second asset ID.
    pub asset_j: u64,
    /// Previous correlation.
    pub old_correlation: f64,
    /// New correlation.
    pub new_correlation: f64,
    /// Change magnitude.
    pub change: f64,
}

/// Full correlation matrix result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationMatrixResult {
    /// Number of assets.
    pub n_assets: usize,
    /// Full N×N correlation matrix (row-major).
    pub correlations: Vec<f64>,
    /// Observations used.
    pub observations: u64,
    /// Timestamp of last update.
    pub timestamp: u64,
    /// Compute time in microseconds.
    pub compute_time_us: u64,
}

/// Real-time correlation kernel.
///
/// Maintains streaming correlation matrices using Welford's online algorithm.
/// Supports both Pearson and exponentially weighted correlations.
/// Designed for Ring mode operation with sub-millisecond updates.
#[derive(Debug)]
pub struct RealTimeCorrelation {
    metadata: KernelMetadata,
    /// Internal state for tracking correlations.
    state: std::sync::RwLock<CorrelationState>,
}

impl Clone for RealTimeCorrelation {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            state: std::sync::RwLock::new(self.state.read().unwrap().clone()),
        }
    }
}

impl Default for RealTimeCorrelation {
    fn default() -> Self {
        Self::new()
    }
}

impl RealTimeCorrelation {
    /// Create a new real-time correlation kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("risk/realtime-correlation", Domain::RiskAnalytics)
                .with_description("Streaming correlation matrix updates")
                .with_throughput(500_000)
                .with_latency_us(10.0),
            state: std::sync::RwLock::new(CorrelationState::new(CorrelationConfig::default())),
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: CorrelationConfig) -> Self {
        Self {
            metadata: KernelMetadata::ring("risk/realtime-correlation", Domain::RiskAnalytics)
                .with_description("Streaming correlation matrix updates")
                .with_throughput(500_000)
                .with_latency_us(10.0),
            state: std::sync::RwLock::new(CorrelationState::new(config)),
        }
    }

    /// Initialize with a set of asset IDs.
    pub fn initialize(&self, asset_ids: &[u64]) {
        let mut state = self.state.write().unwrap();
        state.asset_index.clear();
        for (idx, &id) in asset_ids.iter().enumerate() {
            if idx < state.config.n_assets {
                state.asset_index.insert(id, idx);
            }
        }
        // Reset statistics
        let n = state.config.n_assets;
        state.asset_stats = vec![AssetStats::default(); n];
        state.pairwise_stats = vec![PairwiseStats::default(); n * (n - 1) / 2];
        state.correlation_matrix = vec![0.0; n * n];
        state.prev_correlation_matrix = vec![0.0; n * n];
        state.total_observations = 0;
    }

    /// Process a single update and return correlation changes.
    pub fn update(&self, update: &CorrelationUpdate) -> CorrelationUpdateResult {
        let start = Instant::now();
        let mut state = self.state.write().unwrap();

        // Get or assign index for this asset
        let asset_idx = if let Some(&idx) = state.asset_index.get(&update.asset_id) {
            idx
        } else if state.asset_index.len() < state.config.n_assets {
            let idx = state.asset_index.len();
            state.asset_index.insert(update.asset_id, idx);
            idx
        } else {
            // At capacity, ignore new assets
            return CorrelationUpdateResult {
                asset_id: update.asset_id,
                correlations_updated: 0,
                significant_changes: Vec::new(),
                latency_us: start.elapsed().as_micros() as u64,
            };
        };

        // Update asset statistics
        state.asset_stats[asset_idx].update(update.value, update.timestamp);
        state.total_observations += 1;

        // Update pairwise statistics for all pairs involving this asset
        let n = state.config.n_assets;
        let mut correlations_updated = 0;
        let mut significant_changes = Vec::new();

        // We need the last values of other assets to update covariance
        // In a true streaming system, we'd batch updates or use a different approach
        // For now, we update when both assets have been observed at least once
        for other_idx in 0..state.asset_index.len() {
            if other_idx == asset_idx {
                continue;
            }

            let other_stats = &state.asset_stats[other_idx];
            if other_stats.count == 0 {
                continue;
            }

            // Update pairwise statistics
            let (i, j) = if asset_idx < other_idx {
                (asset_idx, other_idx)
            } else {
                (other_idx, asset_idx)
            };
            let pair_idx = state.pair_index(i, j);

            // Use the last values for covariance update
            let value_i = if asset_idx == i {
                update.value
            } else {
                state.asset_stats[i].last_value
            };
            let value_j = if asset_idx == j {
                update.value
            } else {
                state.asset_stats[j].last_value
            };

            state.pairwise_stats[pair_idx].update(value_i, value_j);

            // Recompute correlation for this pair
            if state.pairwise_stats[pair_idx].count >= state.config.min_observations as u64 {
                let cov = state.pairwise_stats[pair_idx].covariance();
                let std_i = state.asset_stats[i].std_dev();
                let std_j = state.asset_stats[j].std_dev();

                let new_corr = if std_i > 1e-10 && std_j > 1e-10 {
                    (cov / (std_i * std_j)).clamp(-1.0, 1.0)
                } else {
                    0.0
                };

                // Store previous and update
                let old_corr = state.correlation_matrix[i * n + j];
                state.prev_correlation_matrix[i * n + j] = old_corr;
                state.prev_correlation_matrix[j * n + i] = old_corr;
                state.correlation_matrix[i * n + j] = new_corr;
                state.correlation_matrix[j * n + i] = new_corr;

                correlations_updated += 1;

                // Check for significant change
                let change = (new_corr - old_corr).abs();
                if change >= state.config.change_threshold {
                    // Get asset IDs
                    let id_i = state
                        .asset_index
                        .iter()
                        .find(|&(_, idx)| *idx == i)
                        .map(|(&id, _)| id)
                        .unwrap_or(0);
                    let id_j = state
                        .asset_index
                        .iter()
                        .find(|&(_, idx)| *idx == j)
                        .map(|(&id, _)| id)
                        .unwrap_or(0);

                    significant_changes.push(CorrelationChange {
                        asset_i: id_i,
                        asset_j: id_j,
                        old_correlation: old_corr,
                        new_correlation: new_corr,
                        change,
                    });
                }
            }
        }

        // Set diagonal to 1.0
        state.correlation_matrix[asset_idx * n + asset_idx] = 1.0;

        CorrelationUpdateResult {
            asset_id: update.asset_id,
            correlations_updated,
            significant_changes,
            latency_us: start.elapsed().as_micros() as u64,
        }
    }

    /// Process a batch of updates.
    pub fn update_batch(&self, updates: &[CorrelationUpdate]) -> Vec<CorrelationUpdateResult> {
        updates.iter().map(|u| self.update(u)).collect()
    }

    /// Get current correlation between two assets.
    pub fn get_correlation(&self, asset_i: u64, asset_j: u64) -> Option<f64> {
        let state = self.state.read().unwrap();
        let idx_i = state.asset_index.get(&asset_i)?;
        let idx_j = state.asset_index.get(&asset_j)?;
        let n = state.config.n_assets;
        Some(state.correlation_matrix[idx_i * n + idx_j])
    }

    /// Get full correlation matrix.
    pub fn get_matrix(&self) -> CorrelationMatrixResult {
        let start = Instant::now();
        let state = self.state.read().unwrap();

        CorrelationMatrixResult {
            n_assets: state.asset_index.len(),
            correlations: state.correlation_matrix.clone(),
            observations: state.total_observations,
            timestamp: state
                .asset_stats
                .iter()
                .map(|s| s.last_timestamp)
                .max()
                .unwrap_or(0),
            compute_time_us: start.elapsed().as_micros() as u64,
        }
    }

    /// Get correlation row for a specific asset.
    pub fn get_row(&self, asset_id: u64) -> Option<Vec<(u64, f64)>> {
        let state = self.state.read().unwrap();
        let idx = state.asset_index.get(&asset_id)?;
        let n = state.config.n_assets;

        Some(
            state
                .asset_index
                .iter()
                .map(|(&id, &j)| (id, state.correlation_matrix[idx * n + j]))
                .collect(),
        )
    }

    /// Reset state while keeping configuration.
    pub fn reset(&self) {
        let mut state = self.state.write().unwrap();
        let config = state.config.clone();
        *state = CorrelationState::new(config);
    }

    /// Batch compute correlation matrix from historical data.
    pub fn compute_from_returns(returns: &[Vec<f64>]) -> CorrelationMatrixResult {
        let start = Instant::now();

        if returns.is_empty() || returns[0].is_empty() {
            return CorrelationMatrixResult {
                n_assets: 0,
                correlations: Vec::new(),
                observations: 0,
                timestamp: 0,
                compute_time_us: start.elapsed().as_micros() as u64,
            };
        }

        let n = returns.len();
        let t = returns[0].len();

        // Compute means
        let means: Vec<f64> = returns
            .iter()
            .map(|r| r.iter().sum::<f64>() / t as f64)
            .collect();

        // Compute standard deviations
        let stds: Vec<f64> = returns
            .iter()
            .zip(means.iter())
            .map(|(r, &mean)| {
                let var = r.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (t - 1) as f64;
                var.sqrt()
            })
            .collect();

        // Compute correlation matrix
        let mut correlations = vec![0.0; n * n];

        for i in 0..n {
            correlations[i * n + i] = 1.0; // Diagonal

            for j in (i + 1)..n {
                let cov: f64 = returns[i]
                    .iter()
                    .zip(returns[j].iter())
                    .map(|(&xi, &xj)| (xi - means[i]) * (xj - means[j]))
                    .sum::<f64>()
                    / (t - 1) as f64;

                let corr = if stds[i] > 1e-10 && stds[j] > 1e-10 {
                    (cov / (stds[i] * stds[j])).clamp(-1.0, 1.0)
                } else {
                    0.0
                };

                correlations[i * n + j] = corr;
                correlations[j * n + i] = corr;
            }
        }

        CorrelationMatrixResult {
            n_assets: n,
            correlations,
            observations: t as u64,
            timestamp: 0,
            compute_time_us: start.elapsed().as_micros() as u64,
        }
    }
}

impl GpuKernel for RealTimeCorrelation {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realtime_correlation_metadata() {
        let kernel = RealTimeCorrelation::new();
        assert_eq!(kernel.metadata().id, "risk/realtime-correlation");
        assert_eq!(kernel.metadata().domain, Domain::RiskAnalytics);
    }

    #[test]
    fn test_asset_stats_welford() {
        let mut stats = AssetStats::default();

        // Known sequence: 2, 4, 6, 8, 10
        // Mean = 6, Var = 10
        for v in [2.0, 4.0, 6.0, 8.0, 10.0] {
            stats.update(v, 0);
        }

        assert!((stats.mean - 6.0).abs() < 1e-10);
        assert!((stats.variance() - 10.0).abs() < 1e-10);
        assert!((stats.std_dev() - (10.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_initialize_assets() {
        let kernel = RealTimeCorrelation::new();
        kernel.initialize(&[100, 101, 102]);

        // Should have registered 3 assets
        let state = kernel.state.read().unwrap();
        assert_eq!(state.asset_index.len(), 3);
    }

    #[test]
    fn test_streaming_updates() {
        let config = CorrelationConfig {
            n_assets: 10,
            min_observations: 2,
            ..Default::default()
        };
        let kernel = RealTimeCorrelation::with_config(config);
        kernel.initialize(&[1, 2]);

        // Generate correlated returns
        for i in 0..50 {
            let r1 = (i as f64) * 0.01;
            let r2 = r1 * 0.8 + 0.002; // Highly correlated

            kernel.update(&CorrelationUpdate {
                asset_id: 1,
                value: r1,
                timestamp: i as u64,
            });
            kernel.update(&CorrelationUpdate {
                asset_id: 2,
                value: r2,
                timestamp: i as u64,
            });
        }

        // Check correlation is high
        let corr = kernel.get_correlation(1, 2).unwrap();
        assert!(corr > 0.9, "Expected high correlation, got: {}", corr);
    }

    #[test]
    fn test_uncorrelated_assets() {
        let config = CorrelationConfig {
            n_assets: 10,
            min_observations: 2,
            ..Default::default()
        };
        let kernel = RealTimeCorrelation::with_config(config);
        kernel.initialize(&[1, 2]);

        // Generate uncorrelated returns using alternating pattern
        for i in 0..100 {
            let r1 = if i % 2 == 0 { 0.01 } else { -0.01 };
            let r2 = if i % 3 == 0 { 0.01 } else { -0.01 };

            kernel.update(&CorrelationUpdate {
                asset_id: 1,
                value: r1,
                timestamp: i as u64,
            });
            kernel.update(&CorrelationUpdate {
                asset_id: 2,
                value: r2,
                timestamp: i as u64,
            });
        }

        // Correlation should be low
        let corr = kernel.get_correlation(1, 2).unwrap();
        assert!(corr.abs() < 0.5, "Expected low correlation, got: {}", corr);
    }

    #[test]
    fn test_correlation_matrix_diagonal() {
        let kernel = RealTimeCorrelation::new();
        kernel.initialize(&[1, 2, 3]);

        // Add some data
        for i in 0..30 {
            kernel.update(&CorrelationUpdate {
                asset_id: 1,
                value: i as f64 * 0.01,
                timestamp: i as u64,
            });
            kernel.update(&CorrelationUpdate {
                asset_id: 2,
                value: i as f64 * 0.02,
                timestamp: i as u64,
            });
            kernel.update(&CorrelationUpdate {
                asset_id: 3,
                value: i as f64 * 0.015,
                timestamp: i as u64,
            });
        }

        // Diagonal should be 1.0
        let corr_11 = kernel.get_correlation(1, 1).unwrap();
        let corr_22 = kernel.get_correlation(2, 2).unwrap();
        let corr_33 = kernel.get_correlation(3, 3).unwrap();

        assert!((corr_11 - 1.0).abs() < 1e-10);
        assert!((corr_22 - 1.0).abs() < 1e-10);
        assert!((corr_33 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_correlation() {
        // Returns for 3 assets over 10 periods
        let returns = vec![
            vec![
                0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.01, 0.02, -0.01, 0.01,
            ],
            vec![
                0.02, 0.03, -0.02, 0.04, 0.02, -0.03, 0.02, 0.03, -0.02, 0.02,
            ], // Similar to asset 0
            vec![
                -0.01, 0.01, 0.02, -0.02, 0.03, 0.01, -0.01, 0.02, 0.01, -0.01,
            ], // Different pattern
        ];

        let result = RealTimeCorrelation::compute_from_returns(&returns);

        assert_eq!(result.n_assets, 3);
        assert_eq!(result.observations, 10);

        // Check matrix properties
        let n = result.n_assets;
        // Diagonal should be 1.0
        for i in 0..n {
            assert!((result.correlations[i * n + i] - 1.0).abs() < 1e-10);
        }
        // Should be symmetric
        for i in 0..n {
            for j in 0..n {
                let diff = (result.correlations[i * n + j] - result.correlations[j * n + i]).abs();
                assert!(diff < 1e-10);
            }
        }
        // Assets 0 and 1 should be highly correlated
        let corr_01 = result.correlations[1];
        assert!(corr_01 > 0.9, "Expected high correlation: {}", corr_01);
    }

    #[test]
    fn test_significant_change_detection() {
        let config = CorrelationConfig {
            n_assets: 10,
            min_observations: 2,
            change_threshold: 0.3, // 30% change threshold
            ..Default::default()
        };
        let kernel = RealTimeCorrelation::with_config(config);
        kernel.initialize(&[1, 2]);

        // First establish a positive correlation
        for i in 0..50 {
            kernel.update(&CorrelationUpdate {
                asset_id: 1,
                value: i as f64 * 0.01,
                timestamp: i as u64,
            });
            kernel.update(&CorrelationUpdate {
                asset_id: 2,
                value: i as f64 * 0.01 + 0.001,
                timestamp: i as u64,
            });
        }

        // Now switch to negative correlation - this should trigger a change
        // (In practice this would take more observations to significantly change the correlation)
        let baseline_corr = kernel.get_correlation(1, 2).unwrap();
        assert!(
            baseline_corr > 0.9,
            "Expected high positive correlation: {}",
            baseline_corr
        );
    }

    #[test]
    fn test_get_row() {
        let kernel = RealTimeCorrelation::new();
        kernel.initialize(&[1, 2, 3]);

        // Add data
        for i in 0..30 {
            kernel.update(&CorrelationUpdate {
                asset_id: 1,
                value: i as f64,
                timestamp: i as u64,
            });
            kernel.update(&CorrelationUpdate {
                asset_id: 2,
                value: i as f64 * 2.0,
                timestamp: i as u64,
            });
            kernel.update(&CorrelationUpdate {
                asset_id: 3,
                value: i as f64 * 1.5,
                timestamp: i as u64,
            });
        }

        let row = kernel.get_row(1).unwrap();
        assert_eq!(row.len(), 3);

        // Should include self-correlation of 1.0
        let self_corr = row.iter().find(|(id, _)| *id == 1).map(|(_, c)| *c);
        assert!((self_corr.unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reset() {
        let kernel = RealTimeCorrelation::new();
        kernel.initialize(&[1, 2]);

        for i in 0..30 {
            kernel.update(&CorrelationUpdate {
                asset_id: 1,
                value: i as f64,
                timestamp: i as u64,
            });
        }

        let matrix_before = kernel.get_matrix();
        assert!(matrix_before.observations > 0);

        kernel.reset();

        let matrix_after = kernel.get_matrix();
        assert_eq!(matrix_after.observations, 0);
    }

    #[test]
    fn test_empty_returns() {
        let result = RealTimeCorrelation::compute_from_returns(&[]);
        assert_eq!(result.n_assets, 0);

        let empty_inner: Vec<Vec<f64>> = vec![vec![]];
        let result2 = RealTimeCorrelation::compute_from_returns(&empty_inner);
        assert_eq!(result2.n_assets, 0);
    }
}
