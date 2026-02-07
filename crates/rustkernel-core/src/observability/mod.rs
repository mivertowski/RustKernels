//! Observability Infrastructure
//!
//! This module provides production-grade observability for RustKernels:
//!
//! - **Metrics**: Prometheus-compatible metrics for kernel performance
//! - **Tracing**: Distributed tracing with OpenTelemetry support
//! - **Logging**: Structured logging with context propagation
//! - **Alerting**: Alert rules and routing
//!
//! # Feature Flags
//!
//! - `metrics`: Enable Prometheus metrics export
//! - `otlp`: Enable OpenTelemetry Protocol export
//! - `structured-logging`: Enable JSON structured logging
//! - `alerting`: Enable alert rule engine
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_core::observability::{MetricsConfig, ObservabilityConfig};
//!
//! let config = ObservabilityConfig::builder()
//!     .with_metrics(MetricsConfig::prometheus(9090))
//!     .with_tracing(TracingConfig::otlp("http://jaeger:4317"))
//!     .build();
//!
//! // Initialize observability
//! config.init().await?;
//! ```

pub mod alerting;
pub mod logging;
pub mod metrics;
pub mod tracing;

pub use alerting::{AlertConfig, AlertRule, AlertSeverity, AlertState};
pub use logging::{LogConfig, LogLevel, StructuredLogger};
pub use metrics::{KernelMetrics, MetricsConfig, MetricsExporter};
pub use tracing::{KernelSpan, SpanContext, TracingConfig};

// Re-export ringkernel-core 0.4.2 observability primitives for deep integration.
pub use ringkernel_core::observability as ring_observability;
pub use ringkernel_core::telemetry as ring_telemetry;
pub use ringkernel_core::telemetry_pipeline as ring_telemetry_pipeline;
pub use ringkernel_core::alerting as ring_alerting;
pub use ringkernel_core::logging as ring_logging;

use serde::{Deserialize, Serialize};

/// Unified observability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Metrics configuration
    pub metrics: Option<MetricsConfig>,
    /// Tracing configuration
    pub tracing: Option<TracingConfig>,
    /// Logging configuration
    pub logging: LogConfig,
    /// Alerting configuration
    pub alerting: Option<AlertConfig>,
    /// Service name for all telemetry
    pub service_name: String,
    /// Service version
    pub service_version: String,
    /// Environment (dev, staging, prod)
    pub environment: String,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            metrics: None,
            tracing: None,
            logging: LogConfig::default(),
            alerting: None,
            service_name: "rustkernels".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            environment: "development".to_string(),
        }
    }
}

impl ObservabilityConfig {
    /// Create a new builder
    pub fn builder() -> ObservabilityConfigBuilder {
        ObservabilityConfigBuilder::default()
    }

    /// Development configuration - verbose logging, no external exports
    pub fn development() -> Self {
        Self {
            logging: LogConfig {
                level: LogLevel::Debug,
                structured: false,
                ..Default::default()
            },
            environment: "development".to_string(),
            ..Default::default()
        }
    }

    /// Production configuration - structured logging, metrics enabled
    pub fn production() -> Self {
        Self {
            metrics: Some(MetricsConfig::default()),
            tracing: Some(TracingConfig::default()),
            logging: LogConfig {
                level: LogLevel::Info,
                structured: true,
                ..Default::default()
            },
            alerting: Some(AlertConfig::default()),
            environment: "production".to_string(),
            ..Default::default()
        }
    }

    /// Initialize all observability components
    #[cfg(feature = "metrics")]
    pub async fn init(&self) -> crate::error::Result<()> {
        // Initialize logging
        self.logging.init()?;

        // Initialize metrics
        if let Some(ref metrics) = self.metrics {
            metrics.init().await?;
        }

        // Initialize tracing
        if let Some(ref tracing) = self.tracing {
            tracing.init().await?;
        }

        Ok(())
    }

    /// Initialize without metrics feature
    #[cfg(not(feature = "metrics"))]
    pub async fn init(&self) -> crate::error::Result<()> {
        self.logging.init()?;
        Ok(())
    }
}

/// Builder for observability configuration
#[derive(Default)]
pub struct ObservabilityConfigBuilder {
    config: ObservabilityConfig,
}

impl ObservabilityConfigBuilder {
    /// Set metrics configuration
    pub fn with_metrics(mut self, config: MetricsConfig) -> Self {
        self.config.metrics = Some(config);
        self
    }

    /// Set tracing configuration
    pub fn with_tracing(mut self, config: TracingConfig) -> Self {
        self.config.tracing = Some(config);
        self
    }

    /// Set logging configuration
    pub fn with_logging(mut self, config: LogConfig) -> Self {
        self.config.logging = config;
        self
    }

    /// Set alerting configuration
    pub fn with_alerting(mut self, config: AlertConfig) -> Self {
        self.config.alerting = Some(config);
        self
    }

    /// Set service name
    pub fn service_name(mut self, name: impl Into<String>) -> Self {
        self.config.service_name = name.into();
        self
    }

    /// Set service version
    pub fn service_version(mut self, version: impl Into<String>) -> Self {
        self.config.service_version = version.into();
        self
    }

    /// Set environment
    pub fn environment(mut self, env: impl Into<String>) -> Self {
        self.config.environment = env.into();
        self
    }

    /// Build the configuration
    pub fn build(self) -> ObservabilityConfig {
        self.config
    }
}

/// Common labels for all metrics
#[derive(Debug, Clone, Default)]
pub struct MetricLabels {
    /// Kernel ID
    pub kernel_id: Option<String>,
    /// Domain
    pub domain: Option<String>,
    /// Tenant ID
    pub tenant_id: Option<String>,
    /// Additional labels
    pub extra: std::collections::HashMap<String, String>,
}

impl MetricLabels {
    /// Create new metric labels
    pub fn new() -> Self {
        Self::default()
    }

    /// Set kernel ID
    pub fn with_kernel(mut self, id: impl Into<String>) -> Self {
        self.kernel_id = Some(id.into());
        self
    }

    /// Set domain
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Set tenant ID
    pub fn with_tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant.into());
        self
    }

    /// Add extra label
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    /// Convert to label pairs
    pub fn to_pairs(&self) -> Vec<(&str, String)> {
        let mut pairs = Vec::new();
        if let Some(ref id) = self.kernel_id {
            pairs.push(("kernel_id", id.clone()));
        }
        if let Some(ref domain) = self.domain {
            pairs.push(("domain", domain.clone()));
        }
        if let Some(ref tenant) = self.tenant_id {
            pairs.push(("tenant_id", tenant.clone()));
        }
        for (k, v) in &self.extra {
            pairs.push((k.as_str(), v.clone()));
        }
        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ObservabilityConfig::default();
        assert_eq!(config.service_name, "rustkernels");
        assert_eq!(config.environment, "development");
        assert!(config.metrics.is_none());
    }

    #[test]
    fn test_production_config() {
        let config = ObservabilityConfig::production();
        assert_eq!(config.environment, "production");
        assert!(config.metrics.is_some());
        assert!(config.tracing.is_some());
    }

    #[test]
    fn test_builder() {
        let config = ObservabilityConfig::builder()
            .service_name("test-service")
            .environment("testing")
            .build();

        assert_eq!(config.service_name, "test-service");
        assert_eq!(config.environment, "testing");
    }

    #[test]
    fn test_metric_labels() {
        let labels = MetricLabels::new()
            .with_kernel("graph/pagerank")
            .with_domain("GraphAnalytics")
            .with_tenant("tenant-123");

        let pairs = labels.to_pairs();
        assert_eq!(pairs.len(), 3);
    }
}
