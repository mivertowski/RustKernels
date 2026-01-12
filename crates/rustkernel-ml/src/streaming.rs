//! Streaming anomaly detection kernels.
//!
//! This module provides online/streaming anomaly detection algorithms:
//! - StreamingIsolationForest - Online anomaly detection with sliding window
//! - AdaptiveThreshold - Self-adjusting anomaly thresholds

use crate::types::{AnomalyResult, DataMatrix};
use rand::prelude::*;
use rand::{rng, Rng};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ============================================================================
// Streaming Isolation Forest Kernel
// ============================================================================

/// Configuration for streaming isolation forest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Number of trees in the forest.
    pub n_trees: usize,
    /// Maximum samples to use per tree.
    pub sample_size: usize,
    /// Window size for sliding window mode.
    pub window_size: usize,
    /// How often to rebuild trees (every N samples).
    pub rebuild_interval: usize,
    /// Expected proportion of anomalies.
    pub contamination: f64,
    /// Whether to use sliding window (vs growing window).
    pub use_sliding_window: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            n_trees: 100,
            sample_size: 256,
            window_size: 10000,
            rebuild_interval: 1000,
            contamination: 0.1,
            use_sliding_window: true,
        }
    }
}

/// State maintained by streaming isolation forest.
#[derive(Debug, Clone)]
pub struct StreamingState {
    /// Sliding window of recent samples.
    window: VecDeque<Vec<f64>>,
    /// Number of features per sample.
    n_features: usize,
    /// Current isolation trees.
    trees: Vec<StreamingITree>,
    /// Samples processed since last rebuild.
    samples_since_rebuild: usize,
    /// Total samples processed.
    total_samples: usize,
    /// Running statistics for threshold estimation.
    score_stats: OnlineStats,
    /// Current anomaly threshold.
    threshold: f64,
}

impl StreamingState {
    /// Create new streaming state.
    pub fn new(n_features: usize) -> Self {
        Self {
            window: VecDeque::new(),
            n_features,
            trees: Vec::new(),
            samples_since_rebuild: 0,
            total_samples: 0,
            score_stats: OnlineStats::new(),
            threshold: 0.5,
        }
    }

    /// Get current window size.
    pub fn window_size(&self) -> usize {
        self.window.len()
    }

    /// Get total samples processed.
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Get current threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

/// Online statistics tracker.
#[derive(Debug, Clone, Default)]
struct OnlineStats {
    count: u64,
    mean: f64,
    m2: f64, // Sum of squares of differences from mean
    min: f64,
    max: f64,
}

impl OnlineStats {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    /// Update with new value using Welford's algorithm.
    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Streaming isolation tree node.
#[derive(Debug, Clone)]
enum StreamingINode {
    Internal {
        split_feature: usize,
        split_value: f64,
        left: Box<StreamingINode>,
        right: Box<StreamingINode>,
    },
    External {
        size: usize,
    },
}

/// Streaming isolation tree.
#[derive(Debug, Clone)]
struct StreamingITree {
    root: StreamingINode,
    max_depth: usize,
}

impl StreamingITree {
    /// Build a tree from samples.
    fn build(samples: &[Vec<f64>], max_depth: usize) -> Self {
        let root = Self::build_node(samples, 0, max_depth);
        Self { root, max_depth }
    }

    fn build_node(samples: &[Vec<f64>], depth: usize, max_depth: usize) -> StreamingINode {
        if samples.is_empty() || depth >= max_depth || samples.len() <= 1 {
            return StreamingINode::External { size: samples.len() };
        }

        let n_features = samples[0].len();
        if n_features == 0 {
            return StreamingINode::External { size: samples.len() };
        }

        let mut rng = rng();
        let feature = rng.random_range(0..n_features);

        // Find min/max for this feature
        let values: Vec<f64> = samples.iter().map(|s| s[feature]).collect();
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < 1e-10 {
            return StreamingINode::External { size: samples.len() };
        }

        let split_value = rng.random_range(min_val..max_val);

        let (left_samples, right_samples): (Vec<_>, Vec<_>) = samples
            .iter()
            .cloned()
            .partition(|s| s[feature] < split_value);

        StreamingINode::Internal {
            split_feature: feature,
            split_value,
            left: Box::new(Self::build_node(&left_samples, depth + 1, max_depth)),
            right: Box::new(Self::build_node(&right_samples, depth + 1, max_depth)),
        }
    }

    /// Compute path length for a point.
    fn path_length(&self, point: &[f64]) -> f64 {
        self.path_length_node(&self.root, point, 0)
    }

    fn path_length_node(&self, node: &StreamingINode, point: &[f64], depth: usize) -> f64 {
        match node {
            StreamingINode::External { size } => {
                depth as f64 + Self::c_factor(*size)
            }
            StreamingINode::Internal {
                split_feature,
                split_value,
                left,
                right,
            } => {
                if point[*split_feature] < *split_value {
                    self.path_length_node(left, point, depth + 1)
                } else {
                    self.path_length_node(right, point, depth + 1)
                }
            }
        }
    }

    /// Average path length correction factor.
    fn c_factor(n: usize) -> f64 {
        if n <= 1 {
            0.0
        } else if n == 2 {
            1.0
        } else {
            let n_f = n as f64;
            // Euler's constant approximation for harmonic number
            2.0 * ((n_f - 1.0).ln() + 0.5772156649) - 2.0 * (n_f - 1.0) / n_f
        }
    }
}

/// Streaming Isolation Forest kernel.
///
/// Online anomaly detection that maintains a sliding window of samples
/// and incrementally updates the isolation forest. Suitable for
/// real-time streaming data where batch retraining is impractical.
#[derive(Debug, Clone)]
pub struct StreamingIsolationForest {
    metadata: KernelMetadata,
}

impl Default for StreamingIsolationForest {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingIsolationForest {
    /// Create a new Streaming Isolation Forest kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/streaming-isolation-forest", Domain::StatisticalML)
                .with_description("Online streaming anomaly detection with sliding window")
                .with_throughput(50_000)
                .with_latency_us(20.0),
        }
    }

    /// Initialize streaming state.
    pub fn init(n_features: usize) -> StreamingState {
        StreamingState::new(n_features)
    }

    /// Process a single sample and return anomaly score.
    ///
    /// Returns (score, is_anomaly).
    pub fn process_sample(
        state: &mut StreamingState,
        sample: Vec<f64>,
        config: &StreamingConfig,
    ) -> (f64, bool) {
        if sample.len() != state.n_features && state.n_features > 0 {
            return (0.0, false); // Feature mismatch
        }

        if state.n_features == 0 {
            state.n_features = sample.len();
        }

        // Add to window
        state.window.push_back(sample.clone());
        if config.use_sliding_window && state.window.len() > config.window_size {
            state.window.pop_front();
        }

        state.total_samples += 1;
        state.samples_since_rebuild += 1;

        // Rebuild trees if needed
        if state.trees.is_empty()
            || (state.samples_since_rebuild >= config.rebuild_interval
                && state.window.len() >= config.sample_size)
        {
            Self::rebuild_forest(state, config);
            state.samples_since_rebuild = 0;
        }

        // Compute anomaly score
        let score = if state.trees.is_empty() {
            0.5 // Default score when no trees
        } else {
            Self::compute_score(&state.trees, &sample, config.sample_size)
        };

        // Update running statistics
        state.score_stats.update(score);

        // Update threshold based on statistics
        if state.score_stats.count > 100 {
            // Threshold at mean + k * std_dev where k is derived from contamination
            // For contamination c, we use approximately the (1-c) percentile
            let k = Self::contamination_to_k(config.contamination);
            state.threshold = state.score_stats.mean + k * state.score_stats.std_dev();
            state.threshold = state.threshold.clamp(0.0, 1.0);
        }

        let is_anomaly = score >= state.threshold;
        (score, is_anomaly)
    }

    /// Process a batch of samples.
    pub fn process_batch(
        state: &mut StreamingState,
        samples: &DataMatrix,
        config: &StreamingConfig,
    ) -> AnomalyResult {
        let mut scores = Vec::with_capacity(samples.n_samples);
        let mut labels = Vec::with_capacity(samples.n_samples);

        for i in 0..samples.n_samples {
            let sample = samples.row(i).to_vec();
            let (score, is_anomaly) = Self::process_sample(state, sample, config);
            scores.push(score);
            labels.push(if is_anomaly { -1 } else { 1 });
        }

        AnomalyResult {
            scores,
            labels,
            threshold: state.threshold,
        }
    }

    /// Rebuild the forest from current window.
    fn rebuild_forest(state: &mut StreamingState, config: &StreamingConfig) {
        if state.window.is_empty() {
            return;
        }

        let samples: Vec<Vec<f64>> = state.window.iter().cloned().collect();
        let sample_size = config.sample_size.min(samples.len());
        let max_depth = (sample_size as f64).log2().ceil() as usize;

        let mut rng = rng();
        state.trees = (0..config.n_trees)
            .map(|_| {
                let subset: Vec<Vec<f64>> = samples
                    .choose_multiple(&mut rng, sample_size)
                    .cloned()
                    .collect();
                StreamingITree::build(&subset, max_depth)
            })
            .collect();
    }

    /// Compute anomaly score for a point.
    fn compute_score(trees: &[StreamingITree], point: &[f64], sample_size: usize) -> f64 {
        if trees.is_empty() {
            return 0.5;
        }

        let avg_path_length: f64 =
            trees.iter().map(|tree| tree.path_length(point)).sum::<f64>() / trees.len() as f64;

        let c_n = StreamingITree::c_factor(sample_size);
        if c_n.abs() < 1e-10 {
            return 0.5;
        }

        (2.0_f64).powf(-avg_path_length / c_n)
    }

    /// Convert contamination rate to number of standard deviations.
    fn contamination_to_k(contamination: f64) -> f64 {
        // Approximate inverse normal CDF for (1 - contamination)
        // Using a simple approximation
        if contamination <= 0.01 {
            2.33
        } else if contamination <= 0.05 {
            1.65
        } else if contamination <= 0.10 {
            1.28
        } else if contamination <= 0.20 {
            0.84
        } else {
            0.5
        }
    }
}

impl GpuKernel for StreamingIsolationForest {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Adaptive Threshold Kernel
// ============================================================================

/// Configuration for adaptive threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveThresholdConfig {
    /// Initial threshold value.
    pub initial_threshold: f64,
    /// Window size for statistics.
    pub window_size: usize,
    /// Target false positive rate.
    pub target_fpr: f64,
    /// Learning rate for threshold adjustment.
    pub learning_rate: f64,
    /// Minimum threshold.
    pub min_threshold: f64,
    /// Maximum threshold.
    pub max_threshold: f64,
    /// Enable drift detection.
    pub detect_drift: bool,
    /// Drift detection sensitivity.
    pub drift_sensitivity: f64,
}

impl Default for AdaptiveThresholdConfig {
    fn default() -> Self {
        Self {
            initial_threshold: 0.5,
            window_size: 1000,
            target_fpr: 0.05,
            learning_rate: 0.01,
            min_threshold: 0.1,
            max_threshold: 0.9,
            detect_drift: true,
            drift_sensitivity: 2.0,
        }
    }
}

/// State for adaptive threshold tracking.
#[derive(Debug, Clone)]
pub struct AdaptiveThresholdState {
    /// Current threshold.
    threshold: f64,
    /// Score history window.
    score_window: VecDeque<f64>,
    /// Label history (ground truth when available).
    label_window: VecDeque<Option<bool>>,
    /// Running statistics for scores.
    stats: OnlineStats,
    /// Statistics from previous window (for drift detection).
    prev_window_stats: Option<WindowStats>,
    /// Current window statistics.
    curr_window_stats: WindowStats,
    /// Total samples processed.
    total_samples: usize,
    /// Drift detected flag.
    drift_detected: bool,
    /// Number of drift events.
    drift_count: usize,
}

/// Statistics for a window of data.
#[derive(Debug, Clone, Default)]
struct WindowStats {
    mean: f64,
    variance: f64,
    count: usize,
}

impl AdaptiveThresholdState {
    /// Create new adaptive threshold state.
    pub fn new(config: &AdaptiveThresholdConfig) -> Self {
        Self {
            threshold: config.initial_threshold,
            score_window: VecDeque::new(),
            label_window: VecDeque::new(),
            stats: OnlineStats::new(),
            prev_window_stats: None,
            curr_window_stats: WindowStats::default(),
            total_samples: 0,
            drift_detected: false,
            drift_count: 0,
        }
    }

    /// Get current threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Get total samples processed.
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Check if drift was detected.
    pub fn drift_detected(&self) -> bool {
        self.drift_detected
    }

    /// Get drift count.
    pub fn drift_count(&self) -> usize {
        self.drift_count
    }
}

/// Result of threshold evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdResult {
    /// Current threshold value.
    pub threshold: f64,
    /// Whether current score is above threshold.
    pub is_anomaly: bool,
    /// Estimated false positive rate.
    pub estimated_fpr: f64,
    /// Whether concept drift was detected.
    pub drift_detected: bool,
    /// Confidence in the threshold (0-1).
    pub confidence: f64,
}

/// Adaptive Threshold kernel.
///
/// Self-adjusting anomaly threshold that adapts to changing data distributions.
/// Uses exponential moving statistics and optional feedback from ground truth
/// labels to maintain a target false positive rate.
#[derive(Debug, Clone)]
pub struct AdaptiveThreshold {
    metadata: KernelMetadata,
}

impl Default for AdaptiveThreshold {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveThreshold {
    /// Create a new Adaptive Threshold kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/adaptive-threshold", Domain::StatisticalML)
                .with_description("Self-adjusting anomaly thresholds with drift detection")
                .with_throughput(100_000)
                .with_latency_us(5.0),
        }
    }

    /// Initialize state.
    pub fn init(config: &AdaptiveThresholdConfig) -> AdaptiveThresholdState {
        AdaptiveThresholdState::new(config)
    }

    /// Process a score and get threshold result.
    pub fn process_score(
        state: &mut AdaptiveThresholdState,
        score: f64,
        ground_truth: Option<bool>,
        config: &AdaptiveThresholdConfig,
    ) -> ThresholdResult {
        // Update statistics
        state.stats.update(score);
        state.total_samples += 1;

        // Update windows
        state.score_window.push_back(score);
        state.label_window.push_back(ground_truth);

        if state.score_window.len() > config.window_size {
            state.score_window.pop_front();
            state.label_window.pop_front();
        }

        // Update window statistics
        state.curr_window_stats = Self::compute_window_stats(&state.score_window);

        // Check for drift
        state.drift_detected = false;
        if config.detect_drift {
            if let Some(prev) = &state.prev_window_stats {
                let drift = Self::detect_drift(prev, &state.curr_window_stats, config);
                if drift {
                    state.drift_detected = true;
                    state.drift_count += 1;
                    // Reset threshold on drift
                    state.threshold = Self::estimate_threshold_from_window(
                        &state.score_window,
                        config.target_fpr,
                    );
                }
            }
        }

        // Update threshold based on feedback
        if let Some(is_anomaly) = ground_truth {
            Self::update_threshold_with_feedback(state, score, is_anomaly, config);
        } else {
            // No feedback - use quantile-based threshold
            Self::update_threshold_quantile(state, config);
        }

        // Store window stats for next drift detection
        // Only update baseline when: no previous baseline, or drift was detected (reset)
        if state.score_window.len() == config.window_size {
            if state.prev_window_stats.is_none() || state.drift_detected {
                state.prev_window_stats = Some(state.curr_window_stats.clone());
            }
        }

        let is_anomaly = score >= state.threshold;
        let estimated_fpr = Self::estimate_fpr(state, config);
        let confidence = Self::compute_confidence(state, config);

        ThresholdResult {
            threshold: state.threshold,
            is_anomaly,
            estimated_fpr,
            drift_detected: state.drift_detected,
            confidence,
        }
    }

    /// Compute window statistics.
    fn compute_window_stats(window: &VecDeque<f64>) -> WindowStats {
        if window.is_empty() {
            return WindowStats::default();
        }

        let count = window.len();
        let mean: f64 = window.iter().sum::<f64>() / count as f64;
        let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;

        WindowStats {
            mean,
            variance,
            count,
        }
    }

    /// Detect concept drift between windows.
    fn detect_drift(
        prev: &WindowStats,
        curr: &WindowStats,
        config: &AdaptiveThresholdConfig,
    ) -> bool {
        if prev.count < 10 || curr.count < 10 {
            return false;
        }

        // Use Welch's t-test approximation
        let se = ((prev.variance / prev.count as f64) + (curr.variance / curr.count as f64)).sqrt();
        if se.abs() < 1e-10 {
            return false;
        }

        let t_stat = (curr.mean - prev.mean).abs() / se;
        t_stat > config.drift_sensitivity
    }

    /// Estimate threshold from window using quantile.
    fn estimate_threshold_from_window(window: &VecDeque<f64>, target_fpr: f64) -> f64 {
        if window.is_empty() {
            return 0.5;
        }

        let mut sorted: Vec<f64> = window.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((1.0 - target_fpr) * sorted.len() as f64) as usize;
        let idx = idx.min(sorted.len() - 1);
        sorted[idx]
    }

    /// Update threshold with ground truth feedback.
    fn update_threshold_with_feedback(
        state: &mut AdaptiveThresholdState,
        score: f64,
        is_anomaly: bool,
        config: &AdaptiveThresholdConfig,
    ) {
        // If false positive (predicted anomaly but actually normal)
        if score >= state.threshold && !is_anomaly {
            // Increase threshold
            state.threshold += config.learning_rate * (score - state.threshold);
        }
        // If false negative (predicted normal but actually anomaly)
        else if score < state.threshold && is_anomaly {
            // Decrease threshold
            state.threshold -= config.learning_rate * (state.threshold - score);
        }

        state.threshold = state.threshold.clamp(config.min_threshold, config.max_threshold);
    }

    /// Update threshold using quantile estimation.
    fn update_threshold_quantile(
        state: &mut AdaptiveThresholdState,
        config: &AdaptiveThresholdConfig,
    ) {
        if state.score_window.len() < 10 {
            return;
        }

        let target = Self::estimate_threshold_from_window(&state.score_window, config.target_fpr);

        // Smooth update
        state.threshold =
            state.threshold * (1.0 - config.learning_rate) + target * config.learning_rate;
        state.threshold = state.threshold.clamp(config.min_threshold, config.max_threshold);
    }

    /// Estimate current false positive rate.
    fn estimate_fpr(state: &AdaptiveThresholdState, _config: &AdaptiveThresholdConfig) -> f64 {
        if state.score_window.is_empty() {
            return 0.0;
        }

        let above_threshold = state
            .score_window
            .iter()
            .filter(|&&s| s >= state.threshold)
            .count();

        above_threshold as f64 / state.score_window.len() as f64
    }

    /// Compute confidence in current threshold.
    fn compute_confidence(state: &AdaptiveThresholdState, config: &AdaptiveThresholdConfig) -> f64 {
        // Confidence based on sample size and stability
        let sample_factor = (state.score_window.len() as f64 / config.window_size as f64).min(1.0);

        // Lower confidence if recent drift
        let drift_factor = if state.drift_detected { 0.5 } else { 1.0 };

        // Lower confidence if threshold is at bounds
        let bound_factor = if (state.threshold - config.min_threshold).abs() < 0.01
            || (state.threshold - config.max_threshold).abs() < 0.01
        {
            0.7
        } else {
            1.0
        };

        sample_factor * drift_factor * bound_factor
    }

    /// Batch process multiple scores.
    pub fn process_batch(
        state: &mut AdaptiveThresholdState,
        scores: &[f64],
        ground_truth: Option<&[bool]>,
        config: &AdaptiveThresholdConfig,
    ) -> Vec<ThresholdResult> {
        scores
            .iter()
            .enumerate()
            .map(|(i, &score)| {
                let gt = ground_truth.map(|gt| gt[i]);
                Self::process_score(state, score, gt, config)
            })
            .collect()
    }
}

impl GpuKernel for AdaptiveThreshold {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_isolation_forest_metadata() {
        let kernel = StreamingIsolationForest::new();
        assert_eq!(kernel.metadata().id, "ml/streaming-isolation-forest");
    }

    #[test]
    fn test_streaming_isolation_forest_basic() {
        let config = StreamingConfig {
            n_trees: 10,
            sample_size: 50,
            window_size: 100,
            rebuild_interval: 20,
            contamination: 0.1,
            use_sliding_window: true,
        };

        let mut state = StreamingIsolationForest::init(2);

        // Add normal samples
        for _ in 0..50 {
            let sample = vec![rng().random_range(0.0..1.0), rng().random_range(0.0..1.0)];
            StreamingIsolationForest::process_sample(&mut state, sample, &config);
        }

        assert!(state.window_size() > 0);
        assert_eq!(state.total_samples(), 50);

        // Add an anomaly
        let (score, _is_anomaly) =
            StreamingIsolationForest::process_sample(&mut state, vec![100.0, 100.0], &config);
        assert!(score > 0.0);
    }

    #[test]
    fn test_streaming_sliding_window() {
        let config = StreamingConfig {
            window_size: 10,
            use_sliding_window: true,
            ..Default::default()
        };

        let mut state = StreamingIsolationForest::init(1);

        // Add more samples than window size
        for i in 0..20 {
            StreamingIsolationForest::process_sample(&mut state, vec![i as f64], &config);
        }

        // Window should be capped
        assert_eq!(state.window_size(), 10);
        assert_eq!(state.total_samples(), 20);
    }

    #[test]
    fn test_adaptive_threshold_metadata() {
        let kernel = AdaptiveThreshold::new();
        assert_eq!(kernel.metadata().id, "ml/adaptive-threshold");
    }

    #[test]
    fn test_adaptive_threshold_basic() {
        let config = AdaptiveThresholdConfig {
            initial_threshold: 0.5,
            window_size: 100,
            target_fpr: 0.1,
            learning_rate: 0.1,
            ..Default::default()
        };

        let mut state = AdaptiveThreshold::init(&config);

        // Process some normal scores
        for _ in 0..50 {
            let score = rng().random_range(0.0..0.4);
            AdaptiveThreshold::process_score(&mut state, score, None, &config);
        }

        // Process an anomaly
        let result = AdaptiveThreshold::process_score(&mut state, 0.9, None, &config);
        assert!(result.is_anomaly);
    }

    #[test]
    fn test_adaptive_threshold_feedback() {
        let config = AdaptiveThresholdConfig {
            initial_threshold: 0.5,
            learning_rate: 0.2,
            ..Default::default()
        };

        let mut state = AdaptiveThreshold::init(&config);

        // False positive feedback - should increase threshold
        let initial_threshold = state.threshold();
        AdaptiveThreshold::process_score(&mut state, 0.6, Some(false), &config);
        assert!(state.threshold() > initial_threshold);

        // False negative feedback - should decrease threshold
        let prev_threshold = state.threshold();
        AdaptiveThreshold::process_score(&mut state, 0.3, Some(true), &config);
        assert!(state.threshold() < prev_threshold);
    }

    #[test]
    fn test_drift_detection() {
        let config = AdaptiveThresholdConfig {
            window_size: 10,
            detect_drift: true,
            drift_sensitivity: 1.5, // Lower sensitivity for easier drift detection
            ..Default::default()
        };

        let mut state = AdaptiveThreshold::init(&config);

        // Fill first window with consistent low scores
        for _ in 0..10 {
            AdaptiveThreshold::process_score(&mut state, 0.15, None, &config);
        }

        // Add enough high scores to completely replace the window and trigger drift
        let mut drift_found = false;
        for _ in 0..15 {
            let result = AdaptiveThreshold::process_score(&mut state, 0.85, None, &config);
            if result.drift_detected {
                drift_found = true;
            }
        }

        // Should have detected drift at some point
        assert!(drift_found || state.drift_count() > 0, "Should detect drift between 0.15 and 0.85 score ranges");
    }

    #[test]
    fn test_batch_processing() {
        let config = StreamingConfig::default();
        let mut state = StreamingIsolationForest::init(2);

        let data = DataMatrix::new(
            vec![
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 10.0, 10.0, // anomaly
            ],
            4,
            2,
        );

        let result = StreamingIsolationForest::process_batch(&mut state, &data, &config);
        assert_eq!(result.scores.len(), 4);
        assert_eq!(result.labels.len(), 4);
    }

    #[test]
    fn test_online_stats() {
        let mut stats = OnlineStats::new();

        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            stats.update(v);
        }

        assert!((stats.mean - 5.0).abs() < 0.01);
        assert!((stats.variance() - 4.57).abs() < 0.1);
        assert_eq!(stats.min, 2.0);
        assert_eq!(stats.max, 9.0);
    }
}
