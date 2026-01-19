//! Distributed Tracing
//!
//! OpenTelemetry-compatible distributed tracing for kernel execution.
//!
//! # Features
//!
//! - Span creation for kernel execution
//! - Trace context propagation in K2K messages
//! - OTLP export to Jaeger, Zipkin, etc.
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_core::observability::tracing::{KernelSpan, TracingConfig};
//!
//! let config = TracingConfig::otlp("http://jaeger:4317");
//! config.init().await?;
//!
//! let span = KernelSpan::start("graph/pagerank", "execute");
//! // ... kernel execution ...
//! span.end();
//! ```

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Tracing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    /// Enable tracing
    pub enabled: bool,
    /// OTLP endpoint URL
    pub otlp_endpoint: Option<String>,
    /// Sampling rate (0.0 - 1.0)
    pub sampling_rate: f64,
    /// Service name
    pub service_name: String,
    /// Include span events
    pub include_events: bool,
    /// Max attributes per span
    pub max_attributes: u32,
    /// Batch export settings
    pub batch_size: usize,
    /// Export timeout
    pub export_timeout: Duration,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            otlp_endpoint: None,
            sampling_rate: 1.0, // 100% sampling in dev
            service_name: "rustkernels".to_string(),
            include_events: true,
            max_attributes: 128,
            batch_size: 512,
            export_timeout: Duration::from_secs(30),
        }
    }
}

impl TracingConfig {
    /// Create config for OTLP export
    pub fn otlp(endpoint: impl Into<String>) -> Self {
        Self {
            otlp_endpoint: Some(endpoint.into()),
            ..Default::default()
        }
    }

    /// Set sampling rate
    pub fn with_sampling(mut self, rate: f64) -> Self {
        self.sampling_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Initialize tracing
    #[cfg(feature = "otlp")]
    pub async fn init(&self) -> crate::error::Result<()> {
        use opentelemetry_otlp::WithExportConfig;
        use opentelemetry_sdk::trace::SdkTracerProvider;

        if !self.enabled {
            return Ok(());
        }

        if let Some(ref endpoint) = self.otlp_endpoint {
            let exporter = opentelemetry_otlp::SpanExporter::builder()
                .with_tonic()
                .with_endpoint(endpoint)
                .with_timeout(self.export_timeout)
                .build()
                .map_err(|e| crate::error::KernelError::ConfigError(e.to_string()))?;

            let provider = SdkTracerProvider::builder()
                .with_batch_exporter(exporter)
                .build();

            opentelemetry::global::set_tracer_provider(provider);
        }

        Ok(())
    }

    /// No-op init when OTLP feature is disabled
    #[cfg(not(feature = "otlp"))]
    pub async fn init(&self) -> crate::error::Result<()> {
        Ok(())
    }
}

/// Span context for trace propagation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanContext {
    /// Trace ID (128-bit hex)
    pub trace_id: String,
    /// Span ID (64-bit hex)
    pub span_id: String,
    /// Trace flags
    pub trace_flags: u8,
    /// Trace state
    pub trace_state: Option<String>,
}

impl SpanContext {
    /// Create a new span context
    pub fn new(trace_id: impl Into<String>, span_id: impl Into<String>) -> Self {
        Self {
            trace_id: trace_id.into(),
            span_id: span_id.into(),
            trace_flags: 0x01, // Sampled
            trace_state: None,
        }
    }

    /// Generate a new trace ID
    pub fn generate_trace_id() -> String {
        format!("{:032x}", rand::random::<u128>())
    }

    /// Generate a new span ID
    pub fn generate_span_id() -> String {
        format!("{:016x}", rand::random::<u64>())
    }

    /// Create a new root span context
    pub fn new_root() -> Self {
        Self::new(Self::generate_trace_id(), Self::generate_span_id())
    }

    /// Create a child span context
    pub fn new_child(&self) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            span_id: Self::generate_span_id(),
            trace_flags: self.trace_flags,
            trace_state: self.trace_state.clone(),
        }
    }

    /// Convert to W3C trace context header value
    pub fn to_traceparent(&self) -> String {
        format!(
            "00-{}-{}-{:02x}",
            self.trace_id, self.span_id, self.trace_flags
        )
    }

    /// Parse from W3C trace context header
    pub fn from_traceparent(header: &str) -> Option<Self> {
        let parts: Vec<&str> = header.split('-').collect();
        if parts.len() != 4 {
            return None;
        }

        Some(Self {
            trace_id: parts[1].to_string(),
            span_id: parts[2].to_string(),
            trace_flags: u8::from_str_radix(parts[3], 16).ok()?,
            trace_state: None,
        })
    }
}

/// A kernel execution span
pub struct KernelSpan {
    /// Kernel ID
    pub kernel_id: String,
    /// Operation name
    pub operation: String,
    /// Span context
    pub context: SpanContext,
    /// Start time
    pub start: Instant,
    /// Attributes
    pub attributes: std::collections::HashMap<String, String>,
    /// Events
    pub events: Vec<SpanEvent>,
}

impl KernelSpan {
    /// Start a new span
    pub fn start(kernel_id: impl Into<String>, operation: impl Into<String>) -> Self {
        Self {
            kernel_id: kernel_id.into(),
            operation: operation.into(),
            context: SpanContext::new_root(),
            start: Instant::now(),
            attributes: std::collections::HashMap::new(),
            events: Vec::new(),
        }
    }

    /// Start a child span
    pub fn start_child(
        parent: &SpanContext,
        kernel_id: impl Into<String>,
        operation: impl Into<String>,
    ) -> Self {
        Self {
            kernel_id: kernel_id.into(),
            operation: operation.into(),
            context: parent.new_child(),
            start: Instant::now(),
            attributes: std::collections::HashMap::new(),
            events: Vec::new(),
        }
    }

    /// Add an attribute
    pub fn set_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.insert(key.into(), value.into());
    }

    /// Add an event
    pub fn add_event(&mut self, name: impl Into<String>) {
        self.events.push(SpanEvent {
            name: name.into(),
            timestamp: Instant::now(),
            attributes: std::collections::HashMap::new(),
        });
    }

    /// Add an event with attributes
    pub fn add_event_with_attributes(
        &mut self,
        name: impl Into<String>,
        attributes: std::collections::HashMap<String, String>,
    ) {
        self.events.push(SpanEvent {
            name: name.into(),
            timestamp: Instant::now(),
            attributes,
        });
    }

    /// Record an error
    pub fn record_error(&mut self, error: &dyn std::error::Error) {
        self.set_attribute("error", "true");
        self.set_attribute("error.message", error.to_string());
        self.add_event("exception");
    }

    /// End the span
    pub fn end(self) -> Duration {
        let duration = self.start.elapsed();

        #[cfg(feature = "otlp")]
        {
            use tracing::info_span;
            // Record span to tracing
            let span = info_span!(
                "kernel_execution",
                kernel_id = %self.kernel_id,
                operation = %self.operation,
                trace_id = %self.context.trace_id,
                span_id = %self.context.span_id,
                duration_us = duration.as_micros() as u64,
            );
            span.in_scope(|| {
                for (key, value) in &self.attributes {
                    tracing::info!(key = %key, value = %value, "span attribute");
                }
            });
        }

        duration
    }

    /// Get duration so far
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

/// A span event
#[derive(Debug, Clone)]
pub struct SpanEvent {
    /// Event name
    pub name: String,
    /// Event timestamp
    pub timestamp: Instant,
    /// Event attributes
    pub attributes: std::collections::HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_context() {
        let ctx = SpanContext::new_root();
        assert_eq!(ctx.trace_id.len(), 32);
        assert_eq!(ctx.span_id.len(), 16);

        let child = ctx.new_child();
        assert_eq!(child.trace_id, ctx.trace_id);
        assert_ne!(child.span_id, ctx.span_id);
    }

    #[test]
    fn test_traceparent() {
        let ctx = SpanContext::new("0af7651916cd43dd8448eb211c80319c", "b7ad6b7169203331");
        let header = ctx.to_traceparent();
        assert!(header.starts_with("00-"));

        let parsed = SpanContext::from_traceparent(&header).unwrap();
        assert_eq!(parsed.trace_id, ctx.trace_id);
        assert_eq!(parsed.span_id, ctx.span_id);
    }

    #[test]
    fn test_kernel_span() {
        let mut span = KernelSpan::start("graph/pagerank", "execute");
        span.set_attribute("input_size", "1000");
        span.add_event("started");

        std::thread::sleep(std::time::Duration::from_millis(10));

        let duration = span.end();
        assert!(duration >= std::time::Duration::from_millis(10));
    }

    #[test]
    fn test_tracing_config() {
        let config = TracingConfig::otlp("http://jaeger:4317").with_sampling(0.5);

        assert_eq!(config.otlp_endpoint, Some("http://jaeger:4317".to_string()));
        assert_eq!(config.sampling_rate, 0.5);
    }
}
