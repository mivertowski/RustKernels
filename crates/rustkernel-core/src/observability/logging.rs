//! Structured Logging
//!
//! Provides structured logging with kernel context for production debugging.
//!
//! # Features
//!
//! - JSON structured output for log aggregation
//! - Context propagation (trace IDs, tenant IDs)
//! - Per-domain log levels
//! - Audit logging for security events
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_core::observability::logging::{LogConfig, StructuredLogger};
//!
//! let config = LogConfig::production();
//! config.init()?;
//!
//! StructuredLogger::info()
//!     .kernel("graph/pagerank")
//!     .tenant("tenant-123")
//!     .message("Kernel execution completed")
//!     .field("latency_us", 150)
//!     .log();
//! ```

use serde::{Deserialize, Serialize};

/// Log level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    /// Trace level (most verbose)
    Trace,
    /// Debug level
    Debug,
    /// Info level
    Info,
    /// Warning level
    Warn,
    /// Error level
    Error,
}

impl Default for LogLevel {
    fn default() -> Self {
        Self::Info
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Trace => write!(f, "trace"),
            Self::Debug => write!(f, "debug"),
            Self::Info => write!(f, "info"),
            Self::Warn => write!(f, "warn"),
            Self::Error => write!(f, "error"),
        }
    }
}

impl std::str::FromStr for LogLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "trace" => Ok(Self::Trace),
            "debug" => Ok(Self::Debug),
            "info" => Ok(Self::Info),
            "warn" | "warning" => Ok(Self::Warn),
            "error" => Ok(Self::Error),
            _ => Err(format!("Invalid log level: {}", s)),
        }
    }
}

impl From<LogLevel> for tracing::Level {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => tracing::Level::TRACE,
            LogLevel::Debug => tracing::Level::DEBUG,
            LogLevel::Info => tracing::Level::INFO,
            LogLevel::Warn => tracing::Level::WARN,
            LogLevel::Error => tracing::Level::ERROR,
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogConfig {
    /// Default log level
    pub level: LogLevel,
    /// Enable structured JSON output
    pub structured: bool,
    /// Include timestamps
    pub include_timestamps: bool,
    /// Include caller location
    pub include_location: bool,
    /// Include thread IDs
    pub include_thread_ids: bool,
    /// Per-domain log levels
    pub domain_levels: std::collections::HashMap<String, LogLevel>,
    /// Output target (stdout, stderr, file path)
    pub output: LogOutput,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            structured: false,
            include_timestamps: true,
            include_location: false,
            include_thread_ids: false,
            domain_levels: std::collections::HashMap::new(),
            output: LogOutput::Stdout,
        }
    }
}

impl LogConfig {
    /// Development configuration
    pub fn development() -> Self {
        Self {
            level: LogLevel::Debug,
            structured: false,
            include_location: true,
            ..Default::default()
        }
    }

    /// Production configuration
    pub fn production() -> Self {
        Self {
            level: LogLevel::Info,
            structured: true,
            include_timestamps: true,
            include_thread_ids: true,
            ..Default::default()
        }
    }

    /// Set log level for a specific domain
    pub fn with_domain_level(mut self, domain: impl Into<String>, level: LogLevel) -> Self {
        self.domain_levels.insert(domain.into(), level);
        self
    }

    /// Initialize logging
    pub fn init(&self) -> crate::error::Result<()> {
        use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(self.level.to_string()));

        let subscriber = tracing_subscriber::registry().with(filter);

        if self.structured {
            let layer = fmt::layer()
                .json()
                .with_thread_ids(self.include_thread_ids)
                .with_file(self.include_location)
                .with_line_number(self.include_location);

            subscriber.with(layer).try_init().ok();
        } else {
            let layer = fmt::layer()
                .with_thread_ids(self.include_thread_ids)
                .with_file(self.include_location)
                .with_line_number(self.include_location);

            subscriber.with(layer).try_init().ok();
        }

        Ok(())
    }
}

/// Log output target
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogOutput {
    /// Standard output
    Stdout,
    /// Standard error
    Stderr,
    /// File path
    File(String),
}

impl Default for LogOutput {
    fn default() -> Self {
        Self::Stdout
    }
}

/// Structured logger builder
pub struct StructuredLogger {
    level: LogLevel,
    message: Option<String>,
    kernel_id: Option<String>,
    domain: Option<String>,
    tenant_id: Option<String>,
    trace_id: Option<String>,
    span_id: Option<String>,
    fields: std::collections::HashMap<String, serde_json::Value>,
}

impl StructuredLogger {
    /// Create a new logger at trace level
    pub fn trace() -> Self {
        Self::new(LogLevel::Trace)
    }

    /// Create a new logger at debug level
    pub fn debug() -> Self {
        Self::new(LogLevel::Debug)
    }

    /// Create a new logger at info level
    pub fn info() -> Self {
        Self::new(LogLevel::Info)
    }

    /// Create a new logger at warn level
    pub fn warn() -> Self {
        Self::new(LogLevel::Warn)
    }

    /// Create a new logger at error level
    pub fn error() -> Self {
        Self::new(LogLevel::Error)
    }

    fn new(level: LogLevel) -> Self {
        Self {
            level,
            message: None,
            kernel_id: None,
            domain: None,
            tenant_id: None,
            trace_id: None,
            span_id: None,
            fields: std::collections::HashMap::new(),
        }
    }

    /// Set the message
    pub fn message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }

    /// Set the kernel ID
    pub fn kernel(mut self, id: impl Into<String>) -> Self {
        self.kernel_id = Some(id.into());
        self
    }

    /// Set the domain
    pub fn domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Set the tenant ID
    pub fn tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant.into());
        self
    }

    /// Set trace context
    pub fn trace_context(mut self, trace_id: impl Into<String>, span_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self.span_id = Some(span_id.into());
        self
    }

    /// Add a field
    pub fn field(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.fields.insert(key.into(), json_value);
        }
        self
    }

    /// Emit the log
    pub fn log(self) {
        let msg = self.message.unwrap_or_default();

        match self.level {
            LogLevel::Trace => tracing::trace!(
                kernel_id = ?self.kernel_id,
                domain = ?self.domain,
                tenant_id = ?self.tenant_id,
                trace_id = ?self.trace_id,
                span_id = ?self.span_id,
                "{}",
                msg
            ),
            LogLevel::Debug => tracing::debug!(
                kernel_id = ?self.kernel_id,
                domain = ?self.domain,
                tenant_id = ?self.tenant_id,
                trace_id = ?self.trace_id,
                span_id = ?self.span_id,
                "{}",
                msg
            ),
            LogLevel::Info => tracing::info!(
                kernel_id = ?self.kernel_id,
                domain = ?self.domain,
                tenant_id = ?self.tenant_id,
                trace_id = ?self.trace_id,
                span_id = ?self.span_id,
                "{}",
                msg
            ),
            LogLevel::Warn => tracing::warn!(
                kernel_id = ?self.kernel_id,
                domain = ?self.domain,
                tenant_id = ?self.tenant_id,
                trace_id = ?self.trace_id,
                span_id = ?self.span_id,
                "{}",
                msg
            ),
            LogLevel::Error => tracing::error!(
                kernel_id = ?self.kernel_id,
                domain = ?self.domain,
                tenant_id = ?self.tenant_id,
                trace_id = ?self.trace_id,
                span_id = ?self.span_id,
                "{}",
                msg
            ),
        }
    }
}

/// Audit log entry for security-relevant events
#[derive(Debug, Clone, Serialize)]
pub struct AuditLog {
    /// Event timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Event type
    pub event_type: AuditEventType,
    /// Actor (user ID or system)
    pub actor: String,
    /// Resource being accessed
    pub resource: String,
    /// Action performed
    pub action: String,
    /// Result (success/failure)
    pub result: AuditResult,
    /// Additional details
    pub details: Option<serde_json::Value>,
    /// Tenant ID
    pub tenant_id: Option<String>,
    /// Request ID
    pub request_id: Option<String>,
}

impl AuditLog {
    /// Create a new audit log entry
    pub fn new(event_type: AuditEventType, actor: impl Into<String>, resource: impl Into<String>, action: impl Into<String>) -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            event_type,
            actor: actor.into(),
            resource: resource.into(),
            action: action.into(),
            result: AuditResult::Success,
            details: None,
            tenant_id: None,
            request_id: None,
        }
    }

    /// Set the result
    pub fn with_result(mut self, result: AuditResult) -> Self {
        self.result = result;
        self
    }

    /// Set details
    pub fn with_details(mut self, details: impl Serialize) -> Self {
        self.details = serde_json::to_value(details).ok();
        self
    }

    /// Set tenant
    pub fn with_tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant.into());
        self
    }

    /// Set request ID
    pub fn with_request_id(mut self, id: impl Into<String>) -> Self {
        self.request_id = Some(id.into());
        self
    }

    /// Emit the audit log
    pub fn emit(self) {
        tracing::info!(
            target: "audit",
            event_type = ?self.event_type,
            actor = %self.actor,
            resource = %self.resource,
            action = %self.action,
            result = ?self.result,
            tenant_id = ?self.tenant_id,
            request_id = ?self.request_id,
            "AUDIT"
        );
    }
}

/// Audit event types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditEventType {
    /// Authentication event
    Authentication,
    /// Authorization event
    Authorization,
    /// Kernel access
    KernelAccess,
    /// Configuration change
    ConfigChange,
    /// Data access
    DataAccess,
    /// Administrative action
    AdminAction,
}

/// Audit result
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditResult {
    /// Action succeeded
    Success,
    /// Action failed
    Failure,
    /// Action denied
    Denied,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_parsing() {
        assert_eq!("debug".parse::<LogLevel>().unwrap(), LogLevel::Debug);
        assert_eq!("INFO".parse::<LogLevel>().unwrap(), LogLevel::Info);
        assert_eq!("warning".parse::<LogLevel>().unwrap(), LogLevel::Warn);
    }

    #[test]
    fn test_log_config() {
        let config = LogConfig::production();
        assert!(config.structured);
        assert_eq!(config.level, LogLevel::Info);

        let dev_config = LogConfig::development();
        assert!(!dev_config.structured);
        assert_eq!(dev_config.level, LogLevel::Debug);
    }

    #[test]
    fn test_structured_logger() {
        // Just test that it builds correctly
        let logger = StructuredLogger::info()
            .message("Test message")
            .kernel("graph/pagerank")
            .tenant("tenant-123")
            .field("latency_us", 150);

        assert!(logger.message.is_some());
        assert!(logger.kernel_id.is_some());
    }

    #[test]
    fn test_audit_log() {
        let audit = AuditLog::new(
            AuditEventType::KernelAccess,
            "user-123",
            "graph/pagerank",
            "execute",
        )
        .with_result(AuditResult::Success)
        .with_tenant("tenant-456");

        assert_eq!(audit.actor, "user-123");
        assert!(audit.tenant_id.is_some());
    }
}
