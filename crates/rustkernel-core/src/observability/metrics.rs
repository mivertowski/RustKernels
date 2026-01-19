//! Kernel Metrics
//!
//! Prometheus-compatible metrics for kernel performance monitoring.
//!
//! # Metrics
//!
//! - `rustkernel_messages_total` - Total messages processed
//! - `rustkernel_message_latency_seconds` - Message processing latency histogram
//! - `rustkernel_kernel_health` - Kernel health gauge (1=healthy, 0=unhealthy)
//! - `rustkernel_gpu_memory_bytes` - GPU memory usage
//! - `rustkernel_queue_depth` - Message queue depth per kernel
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_core::observability::metrics::{KernelMetrics, MetricsConfig};
//!
//! // Initialize metrics
//! let config = MetricsConfig::prometheus(9090);
//! config.init().await?;
//!
//! // Record kernel execution
//! KernelMetrics::record_execution("graph/pagerank", Duration::from_micros(150), true);
//! ```

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enabled: bool,
    /// Prometheus endpoint port
    pub prometheus_port: Option<u16>,
    /// Metrics push gateway URL
    pub push_gateway: Option<String>,
    /// Push interval for gateway
    pub push_interval: Duration,
    /// Include default process metrics
    pub include_process_metrics: bool,
    /// Include default runtime metrics
    pub include_runtime_metrics: bool,
    /// Histogram buckets for latency (in seconds)
    pub latency_buckets: Vec<f64>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prometheus_port: Some(9090),
            push_gateway: None,
            push_interval: Duration::from_secs(15),
            include_process_metrics: true,
            include_runtime_metrics: true,
            latency_buckets: vec![
                0.000_1,  // 100μs
                0.000_5,  // 500μs
                0.001,    // 1ms
                0.005,    // 5ms
                0.01,     // 10ms
                0.05,     // 50ms
                0.1,      // 100ms
                0.5,      // 500ms
                1.0,      // 1s
                5.0,      // 5s
            ],
        }
    }
}

impl MetricsConfig {
    /// Create config for Prometheus scraping
    pub fn prometheus(port: u16) -> Self {
        Self {
            prometheus_port: Some(port),
            ..Default::default()
        }
    }

    /// Create config for push gateway
    pub fn push_gateway(url: impl Into<String>) -> Self {
        Self {
            prometheus_port: None,
            push_gateway: Some(url.into()),
            ..Default::default()
        }
    }

    /// Initialize metrics collection
    #[cfg(feature = "metrics")]
    pub async fn init(&self) -> crate::error::Result<()> {
        use ::metrics_exporter_prometheus::PrometheusBuilder;

        if !self.enabled {
            return Ok(());
        }

        if let Some(port) = self.prometheus_port {
            let builder = PrometheusBuilder::new();
            builder
                .with_http_listener(([0, 0, 0, 0], port))
                .install()
                .map_err(|e| crate::error::KernelError::ConfigError(e.to_string()))?;
        }

        Ok(())
    }

    /// No-op init when metrics feature is disabled
    #[cfg(not(feature = "metrics"))]
    pub async fn init(&self) -> crate::error::Result<()> {
        Ok(())
    }
}

/// Metrics exporter handle
pub struct MetricsExporter {
    config: MetricsConfig,
}

impl MetricsExporter {
    /// Create a new metrics exporter
    pub fn new(config: MetricsConfig) -> Self {
        Self { config }
    }

    /// Start the metrics server
    #[cfg(feature = "metrics")]
    pub async fn start(&self) -> crate::error::Result<()> {
        self.config.init().await
    }

    /// No-op start when metrics feature is disabled
    #[cfg(not(feature = "metrics"))]
    pub async fn start(&self) -> crate::error::Result<()> {
        Ok(())
    }
}

/// Kernel-specific metrics
#[derive(Debug)]
pub struct KernelMetrics {
    /// Kernel ID
    pub kernel_id: String,
    /// Domain
    pub domain: String,
    /// Total messages processed
    pub messages_total: AtomicU64,
    /// Successful messages
    pub messages_success: AtomicU64,
    /// Failed messages
    pub messages_failed: AtomicU64,
    /// Total processing time in nanoseconds
    pub processing_time_ns: AtomicU64,
    /// Current queue depth
    pub queue_depth: AtomicU64,
}

impl KernelMetrics {
    /// Create new metrics for a kernel
    pub fn new(kernel_id: impl Into<String>, domain: impl Into<String>) -> Self {
        Self {
            kernel_id: kernel_id.into(),
            domain: domain.into(),
            messages_total: AtomicU64::new(0),
            messages_success: AtomicU64::new(0),
            messages_failed: AtomicU64::new(0),
            processing_time_ns: AtomicU64::new(0),
            queue_depth: AtomicU64::new(0),
        }
    }

    /// Record a message execution
    pub fn record_execution(&self, latency: Duration, success: bool) {
        self.messages_total.fetch_add(1, Ordering::Relaxed);

        if success {
            self.messages_success.fetch_add(1, Ordering::Relaxed);
        } else {
            self.messages_failed.fetch_add(1, Ordering::Relaxed);
        }

        self.processing_time_ns
            .fetch_add(latency.as_nanos() as u64, Ordering::Relaxed);

        // Record to global metrics if available
        #[cfg(feature = "metrics")]
        {
            use ::metrics::{counter, histogram};

            counter!("rustkernel_messages_total",
                "kernel_id" => self.kernel_id.clone(),
                "domain" => self.domain.clone(),
                "status" => if success { "success" } else { "error" }
            )
            .increment(1);

            histogram!("rustkernel_message_latency_seconds",
                "kernel_id" => self.kernel_id.clone(),
                "domain" => self.domain.clone()
            )
            .record(latency.as_secs_f64());
        }
    }

    /// Update queue depth
    pub fn set_queue_depth(&self, depth: u64) {
        self.queue_depth.store(depth, Ordering::Relaxed);

        #[cfg(feature = "metrics")]
        {
            use ::metrics::gauge;
            gauge!("rustkernel_queue_depth",
                "kernel_id" => self.kernel_id.clone(),
                "domain" => self.domain.clone()
            )
            .set(depth as f64);
        }
    }

    /// Get average latency in microseconds
    pub fn avg_latency_us(&self) -> f64 {
        let total = self.messages_total.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let time_ns = self.processing_time_ns.load(Ordering::Relaxed);
        (time_ns as f64 / total as f64) / 1000.0
    }

    /// Get success rate (0.0 - 1.0)
    pub fn success_rate(&self) -> f64 {
        let total = self.messages_total.load(Ordering::Relaxed);
        if total == 0 {
            return 1.0;
        }
        let success = self.messages_success.load(Ordering::Relaxed);
        success as f64 / total as f64
    }

    /// Get throughput (messages per second) over a given duration
    pub fn throughput(&self, duration: Duration) -> f64 {
        let total = self.messages_total.load(Ordering::Relaxed);
        total as f64 / duration.as_secs_f64()
    }
}

/// Global metrics for the entire runtime
pub struct RuntimeMetrics {
    inner: Arc<RuntimeMetricsInner>,
}

struct RuntimeMetricsInner {
    /// Total kernels registered
    pub kernels_registered: AtomicU64,
    /// Active kernel instances
    pub kernels_active: AtomicU64,
    /// Total messages processed across all kernels
    pub messages_total: AtomicU64,
    /// GPU memory usage in bytes
    pub gpu_memory_bytes: AtomicU64,
    /// Peak GPU memory usage
    pub gpu_memory_peak_bytes: AtomicU64,
}

impl RuntimeMetrics {
    /// Create new runtime metrics
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RuntimeMetricsInner {
                kernels_registered: AtomicU64::new(0),
                kernels_active: AtomicU64::new(0),
                messages_total: AtomicU64::new(0),
                gpu_memory_bytes: AtomicU64::new(0),
                gpu_memory_peak_bytes: AtomicU64::new(0),
            }),
        }
    }

    /// Record kernel registration
    pub fn record_kernel_registered(&self) {
        self.inner.kernels_registered.fetch_add(1, Ordering::Relaxed);

        #[cfg(feature = "metrics")]
        {
            use ::metrics::gauge;
            gauge!("rustkernel_kernels_registered").set(
                self.inner.kernels_registered.load(Ordering::Relaxed) as f64,
            );
        }
    }

    /// Record kernel activation
    pub fn record_kernel_activated(&self) {
        self.inner.kernels_active.fetch_add(1, Ordering::Relaxed);

        #[cfg(feature = "metrics")]
        {
            use ::metrics::gauge;
            gauge!("rustkernel_kernels_active").set(
                self.inner.kernels_active.load(Ordering::Relaxed) as f64,
            );
        }
    }

    /// Record kernel deactivation
    pub fn record_kernel_deactivated(&self) {
        self.inner.kernels_active.fetch_sub(1, Ordering::Relaxed);

        #[cfg(feature = "metrics")]
        {
            use ::metrics::gauge;
            gauge!("rustkernel_kernels_active").set(
                self.inner.kernels_active.load(Ordering::Relaxed) as f64,
            );
        }
    }

    /// Record message processed
    pub fn record_message(&self) {
        self.inner.messages_total.fetch_add(1, Ordering::Relaxed);

        #[cfg(feature = "metrics")]
        {
            use ::metrics::counter;
            counter!("rustkernel_messages_total_global").increment(1);
        }
    }

    /// Update GPU memory usage
    pub fn set_gpu_memory(&self, bytes: u64) {
        self.inner.gpu_memory_bytes.store(bytes, Ordering::Relaxed);

        let current_peak = self.inner.gpu_memory_peak_bytes.load(Ordering::Relaxed);
        if bytes > current_peak {
            self.inner.gpu_memory_peak_bytes.store(bytes, Ordering::Relaxed);
        }

        #[cfg(feature = "metrics")]
        {
            use ::metrics::gauge;
            gauge!("rustkernel_gpu_memory_bytes").set(bytes as f64);
            gauge!("rustkernel_gpu_memory_peak_bytes").set(
                self.inner.gpu_memory_peak_bytes.load(Ordering::Relaxed) as f64,
            );
        }
    }

    /// Get current GPU memory usage
    pub fn gpu_memory(&self) -> u64 {
        self.inner.gpu_memory_bytes.load(Ordering::Relaxed)
    }

    /// Get peak GPU memory usage
    pub fn gpu_memory_peak(&self) -> u64 {
        self.inner.gpu_memory_peak_bytes.load(Ordering::Relaxed)
    }
}

impl Default for RuntimeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for RuntimeMetrics {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_metrics() {
        let metrics = KernelMetrics::new("graph/pagerank", "GraphAnalytics");

        metrics.record_execution(Duration::from_micros(100), true);
        metrics.record_execution(Duration::from_micros(200), true);
        metrics.record_execution(Duration::from_micros(300), false);

        assert_eq!(metrics.messages_total.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.messages_success.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.messages_failed.load(Ordering::Relaxed), 1);

        assert!((metrics.avg_latency_us() - 200.0).abs() < 1.0);
        assert!((metrics.success_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_runtime_metrics() {
        let metrics = RuntimeMetrics::new();

        metrics.record_kernel_registered();
        metrics.record_kernel_registered();
        metrics.record_kernel_activated();

        assert_eq!(metrics.inner.kernels_registered.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.inner.kernels_active.load(Ordering::Relaxed), 1);

        metrics.set_gpu_memory(1024 * 1024);
        assert_eq!(metrics.gpu_memory(), 1024 * 1024);
    }

    #[test]
    fn test_metrics_config() {
        let config = MetricsConfig::prometheus(9090);
        assert_eq!(config.prometheus_port, Some(9090));
        assert!(config.enabled);

        let push_config = MetricsConfig::push_gateway("http://localhost:9091");
        assert!(push_config.push_gateway.is_some());
        assert!(push_config.prometheus_port.is_none());
    }
}
