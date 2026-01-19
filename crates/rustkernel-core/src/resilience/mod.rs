//! Resilience Patterns
//!
//! This module provides production-grade resilience patterns for RustKernels:
//!
//! - **Circuit Breaker**: Prevent cascade failures by detecting unhealthy kernels
//! - **Timeout**: Deadline propagation and timeout enforcement
//! - **Recovery**: Automatic recovery from transient failures
//! - **Health**: Health checking for liveness/readiness probes
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_core::resilience::{CircuitBreaker, CircuitBreakerConfig};
//!
//! let config = CircuitBreakerConfig::default()
//!     .failure_threshold(5)
//!     .reset_timeout(Duration::from_secs(30));
//!
//! let cb = CircuitBreaker::new("graph/pagerank", config);
//!
//! // Execute with circuit breaker protection
//! cb.execute(|| async {
//!     kernel.execute(input).await
//! }).await?;
//! ```

pub mod circuit_breaker;
pub mod health;
pub mod recovery;
pub mod timeout;

pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState};
pub use health::{HealthCheck, HealthCheckResult, HealthProbe};
pub use recovery::{RecoveryPolicy, RecoveryStrategy, RetryConfig};
pub use timeout::{DeadlineContext, TimeoutConfig, TimeoutError};

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Unified resilience configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResilienceConfig {
    /// Circuit breaker configuration
    pub circuit_breaker: Option<CircuitBreakerConfig>,
    /// Timeout configuration
    pub timeout: Option<TimeoutConfig>,
    /// Recovery policy
    pub recovery: Option<RecoveryPolicy>,
    /// Health check configuration
    pub health_check_interval: Duration,
}

impl Default for ResilienceConfig {
    fn default() -> Self {
        Self {
            circuit_breaker: Some(CircuitBreakerConfig::default()),
            timeout: Some(TimeoutConfig::default()),
            recovery: Some(RecoveryPolicy::default()),
            health_check_interval: Duration::from_secs(10),
        }
    }
}

impl ResilienceConfig {
    /// Create a new resilience config
    pub fn new() -> Self {
        Self::default()
    }

    /// Disable all resilience features
    pub fn disabled() -> Self {
        Self {
            circuit_breaker: None,
            timeout: None,
            recovery: None,
            health_check_interval: Duration::from_secs(60),
        }
    }

    /// Production configuration with conservative settings
    pub fn production() -> Self {
        Self {
            circuit_breaker: Some(CircuitBreakerConfig::production()),
            timeout: Some(TimeoutConfig::production()),
            recovery: Some(RecoveryPolicy::production()),
            health_check_interval: Duration::from_secs(10),
        }
    }

    /// Development configuration with relaxed settings
    pub fn development() -> Self {
        Self {
            circuit_breaker: Some(CircuitBreakerConfig::default()),
            timeout: Some(TimeoutConfig::development()),
            recovery: Some(RecoveryPolicy::development()),
            health_check_interval: Duration::from_secs(30),
        }
    }

    /// Set circuit breaker config
    pub fn with_circuit_breaker(mut self, config: CircuitBreakerConfig) -> Self {
        self.circuit_breaker = Some(config);
        self
    }

    /// Set timeout config
    pub fn with_timeout(mut self, config: TimeoutConfig) -> Self {
        self.timeout = Some(config);
        self
    }

    /// Set recovery policy
    pub fn with_recovery(mut self, policy: RecoveryPolicy) -> Self {
        self.recovery = Some(policy);
        self
    }

    /// Set health check interval
    pub fn with_health_check_interval(mut self, interval: Duration) -> Self {
        self.health_check_interval = interval;
        self
    }
}

/// Result type for resilience operations
pub type ResilienceResult<T> = std::result::Result<T, ResilienceError>;

/// Errors from resilience patterns
#[derive(Debug, thiserror::Error)]
pub enum ResilienceError {
    /// Circuit breaker is open
    #[error("Circuit breaker is open for {kernel_id}")]
    CircuitOpen { kernel_id: String },

    /// Request timed out
    #[error("Request timed out after {timeout:?}")]
    Timeout { timeout: Duration },

    /// Deadline exceeded
    #[error("Deadline exceeded")]
    DeadlineExceeded,

    /// Max retries exceeded
    #[error("Max retries ({retries}) exceeded")]
    MaxRetriesExceeded { retries: u32 },

    /// Health check failed
    #[error("Health check failed: {reason}")]
    HealthCheckFailed { reason: String },

    /// Kernel error during execution
    #[error("Kernel error: {0}")]
    KernelError(#[from] crate::error::KernelError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ResilienceConfig::default();
        assert!(config.circuit_breaker.is_some());
        assert!(config.timeout.is_some());
        assert!(config.recovery.is_some());
    }

    #[test]
    fn test_disabled_config() {
        let config = ResilienceConfig::disabled();
        assert!(config.circuit_breaker.is_none());
        assert!(config.timeout.is_none());
        assert!(config.recovery.is_none());
    }

    #[test]
    fn test_production_config() {
        let config = ResilienceConfig::production();
        assert!(config.circuit_breaker.is_some());
        assert!(config.timeout.is_some());
    }
}
