//! Health Checking
//!
//! Provides health checking infrastructure for kernels and the runtime.
//!
//! # Features
//!
//! - Liveness probes (is the kernel alive?)
//! - Readiness probes (is the kernel ready to serve?)
//! - Health aggregation for kernel groups
//! - Degradation mode support
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_core::resilience::health::{HealthCheck, HealthProbe};
//!
//! let probe = HealthProbe::new("graph/pagerank")
//!     .with_interval(Duration::from_secs(10))
//!     .with_timeout(Duration::from_secs(5));
//!
//! let result = probe.check(&kernel).await;
//! println!("Health: {:?}", result.status);
//! ```

use crate::traits::HealthStatus;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Health check result
#[derive(Debug, Clone, Serialize)]
pub struct HealthCheckResult {
    /// Overall health status
    pub status: HealthStatus,
    /// Kernel ID (if applicable)
    pub kernel_id: Option<String>,
    /// Check timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Check duration
    pub duration: Duration,
    /// Additional details
    pub details: Option<HealthDetails>,
    /// Error message (if unhealthy)
    pub error: Option<String>,
}

impl HealthCheckResult {
    /// Create a healthy result
    pub fn healthy() -> Self {
        Self {
            status: HealthStatus::Healthy,
            kernel_id: None,
            timestamp: chrono::Utc::now(),
            duration: Duration::ZERO,
            details: None,
            error: None,
        }
    }

    /// Create an unhealthy result
    pub fn unhealthy(error: impl Into<String>) -> Self {
        Self {
            status: HealthStatus::Unhealthy,
            kernel_id: None,
            timestamp: chrono::Utc::now(),
            duration: Duration::ZERO,
            details: None,
            error: Some(error.into()),
        }
    }

    /// Create a degraded result
    pub fn degraded(reason: impl Into<String>) -> Self {
        Self {
            status: HealthStatus::Degraded,
            kernel_id: None,
            timestamp: chrono::Utc::now(),
            duration: Duration::ZERO,
            details: None,
            error: Some(reason.into()),
        }
    }

    /// Set kernel ID
    pub fn with_kernel_id(mut self, id: impl Into<String>) -> Self {
        self.kernel_id = Some(id.into());
        self
    }

    /// Set duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    /// Set details
    pub fn with_details(mut self, details: HealthDetails) -> Self {
        self.details = Some(details);
        self
    }

    /// Check if healthy
    pub fn is_healthy(&self) -> bool {
        self.status == HealthStatus::Healthy
    }

    /// Check if degraded
    pub fn is_degraded(&self) -> bool {
        self.status == HealthStatus::Degraded
    }

    /// Check if unhealthy
    pub fn is_unhealthy(&self) -> bool {
        self.status == HealthStatus::Unhealthy
    }
}

/// Additional health check details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthDetails {
    /// Queue depth
    pub queue_depth: Option<u64>,
    /// Messages processed
    pub messages_processed: Option<u64>,
    /// Error rate
    pub error_rate: Option<f64>,
    /// Average latency in microseconds
    pub avg_latency_us: Option<f64>,
    /// GPU memory usage in bytes
    pub gpu_memory_bytes: Option<u64>,
    /// Custom metrics
    pub custom: std::collections::HashMap<String, serde_json::Value>,
}

impl Default for HealthDetails {
    fn default() -> Self {
        Self {
            queue_depth: None,
            messages_processed: None,
            error_rate: None,
            avg_latency_us: None,
            gpu_memory_bytes: None,
            custom: std::collections::HashMap::new(),
        }
    }
}

impl HealthDetails {
    /// Create new health details
    pub fn new() -> Self {
        Self::default()
    }

    /// Set queue depth
    pub fn with_queue_depth(mut self, depth: u64) -> Self {
        self.queue_depth = Some(depth);
        self
    }

    /// Set messages processed
    pub fn with_messages(mut self, count: u64) -> Self {
        self.messages_processed = Some(count);
        self
    }

    /// Set error rate
    pub fn with_error_rate(mut self, rate: f64) -> Self {
        self.error_rate = Some(rate);
        self
    }

    /// Set average latency
    pub fn with_latency(mut self, latency_us: f64) -> Self {
        self.avg_latency_us = Some(latency_us);
        self
    }

    /// Set GPU memory
    pub fn with_gpu_memory(mut self, bytes: u64) -> Self {
        self.gpu_memory_bytes = Some(bytes);
        self
    }

    /// Add custom metric
    pub fn with_custom(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.custom.insert(key.into(), json_value);
        }
        self
    }
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Check interval
    pub interval: Duration,
    /// Check timeout
    pub timeout: Duration,
    /// Number of consecutive failures before unhealthy
    pub failure_threshold: u32,
    /// Number of consecutive successes before healthy
    pub success_threshold: u32,
    /// Enable liveness checks
    pub liveness_enabled: bool,
    /// Enable readiness checks
    pub readiness_enabled: bool,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(10),
            timeout: Duration::from_secs(5),
            failure_threshold: 3,
            success_threshold: 1,
            liveness_enabled: true,
            readiness_enabled: true,
        }
    }
}

/// Health probe for a kernel
pub struct HealthProbe {
    /// Kernel ID
    kernel_id: String,
    /// Configuration
    config: HealthCheckConfig,
    /// Last check result
    last_result: Option<HealthCheckResult>,
    /// Consecutive failures
    consecutive_failures: u32,
    /// Consecutive successes
    consecutive_successes: u32,
}

impl HealthProbe {
    /// Create a new health probe
    pub fn new(kernel_id: impl Into<String>) -> Self {
        Self {
            kernel_id: kernel_id.into(),
            config: HealthCheckConfig::default(),
            last_result: None,
            consecutive_failures: 0,
            consecutive_successes: 0,
        }
    }

    /// Set check interval
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.config.interval = interval;
        self
    }

    /// Set check timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Set failure threshold
    pub fn with_failure_threshold(mut self, threshold: u32) -> Self {
        self.config.failure_threshold = threshold;
        self
    }

    /// Get the kernel ID
    pub fn kernel_id(&self) -> &str {
        &self.kernel_id
    }

    /// Get last check result
    pub fn last_result(&self) -> Option<&HealthCheckResult> {
        self.last_result.as_ref()
    }

    /// Check kernel health
    pub async fn check<K: crate::traits::GpuKernel>(&mut self, kernel: &K) -> HealthCheckResult {
        let start = Instant::now();

        // Perform health check with timeout
        let status = match tokio::time::timeout(self.config.timeout, async {
            kernel.health_check()
        })
        .await
        {
            Ok(status) => status,
            Err(_) => {
                // Timeout - treat as unhealthy
                self.record_failure();
                let result = HealthCheckResult::unhealthy("Health check timed out")
                    .with_kernel_id(&self.kernel_id)
                    .with_duration(start.elapsed());
                self.last_result = Some(result.clone());
                return result;
            }
        };

        let result = match status {
            HealthStatus::Healthy => {
                self.record_success();
                HealthCheckResult::healthy()
            }
            HealthStatus::Degraded => {
                self.record_failure();
                HealthCheckResult::degraded("Kernel reported degraded status")
            }
            HealthStatus::Unhealthy => {
                self.record_failure();
                HealthCheckResult::unhealthy("Kernel reported unhealthy status")
            }
            HealthStatus::Unknown => {
                self.record_failure();
                HealthCheckResult::unhealthy("Kernel health unknown")
            }
        };

        let result = result
            .with_kernel_id(&self.kernel_id)
            .with_duration(start.elapsed());

        self.last_result = Some(result.clone());
        result
    }

    fn record_success(&mut self) {
        self.consecutive_successes += 1;
        self.consecutive_failures = 0;
    }

    fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        self.consecutive_successes = 0;
    }

    /// Check if kernel should be considered unhealthy
    pub fn is_unhealthy(&self) -> bool {
        self.consecutive_failures >= self.config.failure_threshold
    }

    /// Check if kernel should be considered healthy
    pub fn is_healthy(&self) -> bool {
        self.consecutive_successes >= self.config.success_threshold
    }
}

/// Health check trait for components
pub trait HealthCheck {
    /// Perform a health check
    fn check_health(&self) -> HealthCheckResult;

    /// Check if component is alive (liveness)
    fn is_alive(&self) -> bool {
        self.check_health().status != HealthStatus::Unhealthy
    }

    /// Check if component is ready (readiness)
    fn is_ready(&self) -> bool {
        self.check_health().status == HealthStatus::Healthy
    }
}

/// Aggregate health from multiple checks
pub fn aggregate_health(results: &[HealthCheckResult]) -> HealthCheckResult {
    if results.is_empty() {
        return HealthCheckResult::healthy();
    }

    let mut unhealthy_count = 0;
    let mut degraded_count = 0;
    let mut errors = Vec::new();

    for result in results {
        match result.status {
            HealthStatus::Unhealthy => {
                unhealthy_count += 1;
                if let Some(ref error) = result.error {
                    errors.push(error.clone());
                }
            }
            HealthStatus::Degraded => {
                degraded_count += 1;
            }
            _ => {}
        }
    }

    if unhealthy_count > 0 {
        HealthCheckResult::unhealthy(format!(
            "{} unhealthy: {}",
            unhealthy_count,
            errors.join(", ")
        ))
    } else if degraded_count > 0 {
        HealthCheckResult::degraded(format!("{} degraded", degraded_count))
    } else {
        HealthCheckResult::healthy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_check_result() {
        let healthy = HealthCheckResult::healthy();
        assert!(healthy.is_healthy());

        let unhealthy = HealthCheckResult::unhealthy("test error");
        assert!(unhealthy.is_unhealthy());
        assert_eq!(unhealthy.error.as_deref(), Some("test error"));

        let degraded = HealthCheckResult::degraded("test degradation");
        assert!(degraded.is_degraded());
    }

    #[test]
    fn test_health_details() {
        let details = HealthDetails::new()
            .with_queue_depth(100)
            .with_error_rate(0.01)
            .with_latency(150.0);

        assert_eq!(details.queue_depth, Some(100));
        assert_eq!(details.error_rate, Some(0.01));
        assert_eq!(details.avg_latency_us, Some(150.0));
    }

    #[test]
    fn test_aggregate_health_all_healthy() {
        let results = vec![
            HealthCheckResult::healthy(),
            HealthCheckResult::healthy(),
            HealthCheckResult::healthy(),
        ];

        let aggregate = aggregate_health(&results);
        assert!(aggregate.is_healthy());
    }

    #[test]
    fn test_aggregate_health_some_unhealthy() {
        let results = vec![
            HealthCheckResult::healthy(),
            HealthCheckResult::unhealthy("kernel1 failed"),
            HealthCheckResult::healthy(),
        ];

        let aggregate = aggregate_health(&results);
        assert!(aggregate.is_unhealthy());
    }

    #[test]
    fn test_aggregate_health_some_degraded() {
        let results = vec![
            HealthCheckResult::healthy(),
            HealthCheckResult::degraded("kernel1 slow"),
            HealthCheckResult::healthy(),
        ];

        let aggregate = aggregate_health(&results);
        assert!(aggregate.is_degraded());
    }

    #[test]
    fn test_health_probe_creation() {
        let probe = HealthProbe::new("graph/pagerank")
            .with_interval(Duration::from_secs(30))
            .with_timeout(Duration::from_secs(10));

        assert_eq!(probe.kernel_id(), "graph/pagerank");
        assert_eq!(probe.config.interval, Duration::from_secs(30));
        assert_eq!(probe.config.timeout, Duration::from_secs(10));
    }
}
