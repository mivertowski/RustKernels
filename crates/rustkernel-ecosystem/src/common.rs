//! Common types and utilities for ecosystem integrations

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Service configuration shared across integrations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    /// Service name
    pub name: String,
    /// Service version
    pub version: String,
    /// Listen address
    pub address: String,
    /// Port
    pub port: u16,
    /// Enable request logging
    pub request_logging: bool,
    /// Enable metrics
    pub metrics_enabled: bool,
    /// Default timeout
    pub default_timeout: Duration,
    /// Max request body size
    pub max_body_size: usize,
    /// Enable CORS
    pub cors_enabled: bool,
    /// CORS allowed origins
    pub cors_origins: Vec<String>,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            name: "rustkernels".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            address: "0.0.0.0".to_string(),
            port: 8080,
            request_logging: true,
            metrics_enabled: true,
            default_timeout: Duration::from_secs(30),
            max_body_size: 10 * 1024 * 1024, // 10MB
            cors_enabled: true,
            cors_origins: vec!["*".to_string()],
        }
    }
}

impl ServiceConfig {
    /// Create development configuration
    pub fn development() -> Self {
        Self {
            name: "rustkernels-dev".to_string(),
            address: "127.0.0.1".to_string(),
            request_logging: true,
            ..Default::default()
        }
    }

    /// Create production configuration
    pub fn production() -> Self {
        Self {
            request_logging: false, // Use structured logging
            cors_origins: vec![],   // Configure explicitly
            ..Default::default()
        }
    }

    /// Get the full bind address
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.address, self.port)
    }
}

/// Request context passed through the service stack
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Unique request ID
    pub request_id: String,
    /// Start time
    pub start_time: Instant,
    /// Trace ID for distributed tracing
    pub trace_id: Option<String>,
    /// Span ID
    pub span_id: Option<String>,
    /// Tenant ID
    pub tenant_id: Option<String>,
    /// User ID (if authenticated)
    pub user_id: Option<String>,
    /// Request path
    pub path: String,
    /// Request method
    pub method: String,
}

impl RequestContext {
    /// Create a new request context
    pub fn new(path: impl Into<String>, method: impl Into<String>) -> Self {
        Self {
            request_id: uuid::Uuid::new_v4().to_string(),
            start_time: Instant::now(),
            trace_id: None,
            span_id: None,
            tenant_id: None,
            user_id: None,
            path: path.into(),
            method: method.into(),
        }
    }

    /// Set trace ID from header
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    /// Set tenant ID
    pub fn with_tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// Set user ID
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Get elapsed time since request start
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get elapsed time in microseconds
    pub fn elapsed_us(&self) -> u64 {
        self.start_time.elapsed().as_micros() as u64
    }
}

/// Rate limiter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    pub enabled: bool,
    /// Requests per second
    pub requests_per_second: u32,
    /// Burst size
    pub burst_size: u32,
    /// Per-tenant rate limiting
    pub per_tenant: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            requests_per_second: 100,
            burst_size: 200,
            per_tenant: true,
        }
    }
}

/// Metrics collector for service endpoints
pub struct ServiceMetrics {
    /// Total requests
    total_requests: std::sync::atomic::AtomicU64,
    /// Total errors
    total_errors: std::sync::atomic::AtomicU64,
    /// Total latency (microseconds)
    total_latency_us: std::sync::atomic::AtomicU64,
    /// Min latency (microseconds)
    min_latency_us: std::sync::atomic::AtomicU64,
    /// Max latency (microseconds)
    max_latency_us: std::sync::atomic::AtomicU64,
}

impl ServiceMetrics {
    /// Create new metrics
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            total_requests: std::sync::atomic::AtomicU64::new(0),
            total_errors: std::sync::atomic::AtomicU64::new(0),
            total_latency_us: std::sync::atomic::AtomicU64::new(0),
            min_latency_us: std::sync::atomic::AtomicU64::new(u64::MAX),
            max_latency_us: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Record a request
    pub fn record_request(&self, latency_us: u64, is_error: bool) {
        use std::sync::atomic::Ordering;
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_latency_us
            .fetch_add(latency_us, Ordering::Relaxed);
        if is_error {
            self.total_errors.fetch_add(1, Ordering::Relaxed);
        }
        // Update min latency
        self.min_latency_us.fetch_min(latency_us, Ordering::Relaxed);
        // Update max latency
        self.max_latency_us.fetch_max(latency_us, Ordering::Relaxed);
    }

    /// Get request count
    pub fn request_count(&self) -> u64 {
        self.total_requests
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get error count
    pub fn error_count(&self) -> u64 {
        self.total_errors.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get average latency in microseconds
    pub fn avg_latency_us(&self) -> f64 {
        use std::sync::atomic::Ordering;
        let total = self.total_latency_us.load(Ordering::Relaxed) as f64;
        let count = self.total_requests.load(Ordering::Relaxed) as f64;
        if count > 0.0 { total / count } else { 0.0 }
    }

    /// Get minimum latency in microseconds (returns 0 if no requests)
    pub fn min_latency_us(&self) -> u64 {
        let val = self
            .min_latency_us
            .load(std::sync::atomic::Ordering::Relaxed);
        if val == u64::MAX { 0 } else { val }
    }

    /// Get maximum latency in microseconds
    pub fn max_latency_us(&self) -> u64 {
        self.max_latency_us
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Default for ServiceMetrics {
    fn default() -> Self {
        Self {
            total_requests: std::sync::atomic::AtomicU64::new(0),
            total_errors: std::sync::atomic::AtomicU64::new(0),
            total_latency_us: std::sync::atomic::AtomicU64::new(0),
            min_latency_us: std::sync::atomic::AtomicU64::new(u64::MAX),
            max_latency_us: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

/// Standard API paths
pub mod paths {
    /// Health check endpoint
    pub const HEALTH: &str = "/health";
    /// Liveness probe
    pub const LIVENESS: &str = "/health/live";
    /// Readiness probe
    pub const READINESS: &str = "/health/ready";
    /// Metrics endpoint
    pub const METRICS: &str = "/metrics";
    /// Kernel execution endpoint
    pub const KERNEL_EXECUTE: &str = "/api/v1/kernels/:kernel_id/execute";
    /// List kernels endpoint
    pub const KERNEL_LIST: &str = "/api/v1/kernels";
    /// Get kernel info endpoint
    pub const KERNEL_INFO: &str = "/api/v1/kernels/:kernel_id";
}

/// Standard HTTP headers
pub mod headers {
    /// Request ID header
    pub const X_REQUEST_ID: &str = "X-Request-ID";
    /// Trace ID header (W3C)
    pub const TRACEPARENT: &str = "traceparent";
    /// Tenant ID header
    pub const X_TENANT_ID: &str = "X-Tenant-ID";
    /// API key header
    pub const X_API_KEY: &str = "X-API-Key";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_config() {
        let config = ServiceConfig::default();
        assert_eq!(config.bind_address(), "0.0.0.0:8080");
    }

    #[test]
    fn test_request_context() {
        let ctx = RequestContext::new("/api/v1/kernels", "POST")
            .with_tenant_id("tenant-123")
            .with_user_id("user-456");

        assert!(!ctx.request_id.is_empty());
        assert_eq!(ctx.tenant_id, Some("tenant-123".to_string()));
        assert_eq!(ctx.user_id, Some("user-456".to_string()));
    }

    #[test]
    fn test_service_metrics() {
        let metrics = ServiceMetrics::new();

        // No requests yet
        assert_eq!(metrics.min_latency_us(), 0);
        assert_eq!(metrics.max_latency_us(), 0);

        metrics.record_request(1000, false);
        metrics.record_request(2000, false);
        metrics.record_request(3000, true);

        assert_eq!(metrics.request_count(), 3);
        assert_eq!(metrics.error_count(), 1);
        assert!((metrics.avg_latency_us() - 2000.0).abs() < 0.1);
        assert_eq!(metrics.min_latency_us(), 1000);
        assert_eq!(metrics.max_latency_us(), 3000);
    }
}
