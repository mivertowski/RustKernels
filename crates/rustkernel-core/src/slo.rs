//! Service Level Objective (SLO) validation.
//!
//! This module provides runtime validation of throughput and latency SLOs
//! for kernels, ensuring they meet their performance targets.

use crate::kernel::KernelMetadata;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// SLO validation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SLOResult {
    /// SLO is met.
    Met {
        /// Actual value.
        actual: f64,
        /// Target value.
        target: f64,
        /// Headroom percentage.
        headroom_pct: f64,
    },
    /// SLO is at risk (within 10% of target).
    AtRisk {
        /// Actual value.
        actual: f64,
        /// Target value.
        target: f64,
        /// Percentage of target used.
        usage_pct: f64,
    },
    /// SLO is violated.
    Violated {
        /// Actual value.
        actual: f64,
        /// Target value.
        target: f64,
        /// Percentage over target.
        overage_pct: f64,
    },
}

impl SLOResult {
    /// Returns true if the SLO is met.
    #[must_use]
    pub fn is_met(&self) -> bool {
        matches!(self, SLOResult::Met { .. })
    }

    /// Returns true if the SLO is at risk.
    #[must_use]
    pub fn is_at_risk(&self) -> bool {
        matches!(self, SLOResult::AtRisk { .. })
    }

    /// Returns true if the SLO is violated.
    #[must_use]
    pub fn is_violated(&self) -> bool {
        matches!(self, SLOResult::Violated { .. })
    }
}

/// SLO validator for kernel performance.
#[derive(Debug, Default)]
pub struct SLOValidator {
    /// Kernel-specific overrides.
    overrides: HashMap<String, SLOOverride>,
    /// Whether to enable strict mode (fail on violation).
    strict_mode: bool,
}

/// SLO override for a specific kernel.
#[derive(Debug, Clone)]
pub struct SLOOverride {
    /// Override throughput target (ops/sec).
    pub throughput: Option<u64>,
    /// Override latency target (microseconds).
    pub latency_us: Option<f64>,
    /// Tolerance percentage (default 10%).
    pub tolerance_pct: f64,
}

impl Default for SLOOverride {
    fn default() -> Self {
        Self {
            throughput: None,
            latency_us: None,
            tolerance_pct: 10.0,
        }
    }
}

impl SLOValidator {
    /// Create a new SLO validator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable strict mode (fail on any violation).
    #[must_use]
    pub fn with_strict_mode(mut self) -> Self {
        self.strict_mode = true;
        self
    }

    /// Add an SLO override for a specific kernel.
    pub fn with_override(mut self, kernel_id: impl Into<String>, override_: SLOOverride) -> Self {
        self.overrides.insert(kernel_id.into(), override_);
        self
    }

    /// Validate throughput against target.
    #[must_use]
    pub fn validate_throughput(&self, metadata: &KernelMetadata, actual_ops_per_sec: u64) -> SLOResult {
        let target = self
            .overrides
            .get(&metadata.id)
            .and_then(|o| o.throughput)
            .unwrap_or(metadata.expected_throughput);

        let tolerance_pct = self
            .overrides
            .get(&metadata.id)
            .map(|o| o.tolerance_pct)
            .unwrap_or(10.0);

        let actual = actual_ops_per_sec as f64;
        let target_f64 = target as f64;

        // For throughput, we want actual >= target
        if actual >= target_f64 {
            let headroom = ((actual - target_f64) / target_f64) * 100.0;
            SLOResult::Met {
                actual,
                target: target_f64,
                headroom_pct: headroom,
            }
        } else {
            let usage = (actual / target_f64) * 100.0;
            if usage >= (100.0 - tolerance_pct) {
                SLOResult::AtRisk {
                    actual,
                    target: target_f64,
                    usage_pct: usage,
                }
            } else {
                let overage = ((target_f64 - actual) / target_f64) * 100.0;
                SLOResult::Violated {
                    actual,
                    target: target_f64,
                    overage_pct: overage,
                }
            }
        }
    }

    /// Validate latency against target.
    #[must_use]
    pub fn validate_latency(&self, metadata: &KernelMetadata, actual_latency_us: f64) -> SLOResult {
        let target = self
            .overrides
            .get(&metadata.id)
            .and_then(|o| o.latency_us)
            .unwrap_or(metadata.target_latency_us);

        let tolerance_pct = self
            .overrides
            .get(&metadata.id)
            .map(|o| o.tolerance_pct)
            .unwrap_or(10.0);

        // For latency, we want actual <= target
        if actual_latency_us <= target {
            let headroom = ((target - actual_latency_us) / target) * 100.0;
            SLOResult::Met {
                actual: actual_latency_us,
                target,
                headroom_pct: headroom,
            }
        } else {
            let usage = (actual_latency_us / target) * 100.0;
            if usage <= (100.0 + tolerance_pct) {
                SLOResult::AtRisk {
                    actual: actual_latency_us,
                    target,
                    usage_pct: usage,
                }
            } else {
                let overage = ((actual_latency_us - target) / target) * 100.0;
                SLOResult::Violated {
                    actual: actual_latency_us,
                    target,
                    overage_pct: overage,
                }
            }
        }
    }

    /// Check if strict mode is enabled.
    #[must_use]
    pub fn is_strict(&self) -> bool {
        self.strict_mode
    }
}

/// Performance metrics for a kernel.
#[derive(Debug, Clone, Default)]
pub struct KernelMetrics {
    /// Total operations completed.
    pub operations: u64,
    /// Total processing time.
    pub total_time: Duration,
    /// Minimum latency observed.
    pub min_latency: Option<Duration>,
    /// Maximum latency observed.
    pub max_latency: Option<Duration>,
    /// Sum of latencies for average calculation.
    pub latency_sum: Duration,
    /// Number of latency samples.
    pub latency_count: u64,
}

impl KernelMetrics {
    /// Create new metrics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an operation.
    pub fn record(&mut self, latency: Duration) {
        self.operations += 1;
        self.latency_count += 1;
        self.latency_sum += latency;

        match self.min_latency {
            Some(min) if latency < min => self.min_latency = Some(latency),
            None => self.min_latency = Some(latency),
            _ => {}
        }

        match self.max_latency {
            Some(max) if latency > max => self.max_latency = Some(latency),
            None => self.max_latency = Some(latency),
            _ => {}
        }
    }

    /// Calculate average latency.
    #[must_use]
    pub fn avg_latency(&self) -> Option<Duration> {
        if self.latency_count > 0 {
            Some(self.latency_sum / self.latency_count as u32)
        } else {
            None
        }
    }

    /// Calculate throughput in operations per second.
    #[must_use]
    pub fn throughput(&self) -> f64 {
        if self.total_time.is_zero() {
            0.0
        } else {
            self.operations as f64 / self.total_time.as_secs_f64()
        }
    }

    /// Reset metrics.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Metrics collector for all kernels.
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    metrics: Arc<RwLock<HashMap<String, KernelMetrics>>>,
}

impl MetricsCollector {
    /// Create a new metrics collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Record an operation for a kernel.
    pub fn record(&self, kernel_id: &str, latency: Duration) {
        let mut metrics = self.metrics.write().unwrap();
        metrics
            .entry(kernel_id.to_string())
            .or_default()
            .record(latency);
    }

    /// Get metrics for a kernel.
    #[must_use]
    pub fn get(&self, kernel_id: &str) -> Option<KernelMetrics> {
        let metrics = self.metrics.read().unwrap();
        metrics.get(kernel_id).cloned()
    }

    /// Get all metrics.
    #[must_use]
    pub fn all(&self) -> HashMap<String, KernelMetrics> {
        self.metrics.read().unwrap().clone()
    }

    /// Reset metrics for a kernel.
    pub fn reset(&self, kernel_id: &str) {
        let mut metrics = self.metrics.write().unwrap();
        if let Some(m) = metrics.get_mut(kernel_id) {
            m.reset();
        }
    }

    /// Reset all metrics.
    pub fn reset_all(&self) {
        let mut metrics = self.metrics.write().unwrap();
        metrics.clear();
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard for timing operations.
pub struct TimingGuard<'a> {
    collector: &'a MetricsCollector,
    kernel_id: String,
    start: Instant,
}

impl<'a> TimingGuard<'a> {
    /// Create a new timing guard.
    #[must_use]
    pub fn new(collector: &'a MetricsCollector, kernel_id: impl Into<String>) -> Self {
        Self {
            collector,
            kernel_id: kernel_id.into(),
            start: Instant::now(),
        }
    }
}

impl<'a> Drop for TimingGuard<'a> {
    fn drop(&mut self) {
        let latency = self.start.elapsed();
        self.collector.record(&self.kernel_id, latency);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Domain;
    use crate::kernel::KernelMetadata;

    fn test_metadata() -> KernelMetadata {
        KernelMetadata::ring("test-kernel", Domain::Core)
            .with_throughput(100_000)
            .with_latency_us(1.0)
    }

    #[test]
    fn test_throughput_met() {
        let validator = SLOValidator::new();
        let metadata = test_metadata();

        let result = validator.validate_throughput(&metadata, 120_000);
        assert!(result.is_met());

        if let SLOResult::Met { headroom_pct, .. } = result {
            assert!((headroom_pct - 20.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_throughput_at_risk() {
        let validator = SLOValidator::new();
        let metadata = test_metadata();

        let result = validator.validate_throughput(&metadata, 95_000);
        assert!(result.is_at_risk());
    }

    #[test]
    fn test_throughput_violated() {
        let validator = SLOValidator::new();
        let metadata = test_metadata();

        let result = validator.validate_throughput(&metadata, 50_000);
        assert!(result.is_violated());
    }

    #[test]
    fn test_latency_met() {
        let validator = SLOValidator::new();
        let metadata = test_metadata();

        let result = validator.validate_latency(&metadata, 0.5);
        assert!(result.is_met());
    }

    #[test]
    fn test_latency_at_risk() {
        let validator = SLOValidator::new();
        let metadata = test_metadata();

        let result = validator.validate_latency(&metadata, 1.05);
        assert!(result.is_at_risk());
    }

    #[test]
    fn test_latency_violated() {
        let validator = SLOValidator::new();
        let metadata = test_metadata();

        let result = validator.validate_latency(&metadata, 2.0);
        assert!(result.is_violated());
    }

    #[test]
    fn test_metrics_recording() {
        let collector = MetricsCollector::new();

        collector.record("test", Duration::from_micros(100));
        collector.record("test", Duration::from_micros(200));
        collector.record("test", Duration::from_micros(150));

        let metrics = collector.get("test").unwrap();
        assert_eq!(metrics.operations, 3);
        assert_eq!(metrics.min_latency, Some(Duration::from_micros(100)));
        assert_eq!(metrics.max_latency, Some(Duration::from_micros(200)));
        assert_eq!(metrics.avg_latency(), Some(Duration::from_micros(150)));
    }

    #[test]
    fn test_slo_override() {
        let validator = SLOValidator::new().with_override(
            "test-kernel",
            SLOOverride {
                throughput: Some(50_000),
                latency_us: None,
                tolerance_pct: 5.0,
            },
        );

        let metadata = test_metadata();

        // With override, 60K should be met (target is now 50K)
        let result = validator.validate_throughput(&metadata, 60_000);
        assert!(result.is_met());
    }
}
