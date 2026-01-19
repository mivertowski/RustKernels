//! Circuit Breaker Pattern
//!
//! Prevents cascade failures by detecting and isolating unhealthy kernels.
//!
//! # States
//!
//! - **Closed**: Normal operation, requests pass through
//! - **Open**: Failures exceeded threshold, requests fail fast
//! - **HalfOpen**: Testing if service has recovered
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_core::resilience::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
//!
//! let config = CircuitBreakerConfig::default()
//!     .failure_threshold(5)
//!     .reset_timeout(Duration::from_secs(30));
//!
//! let cb = CircuitBreaker::new("graph/pagerank", config);
//!
//! match cb.execute(|| async { /* kernel execution */ }).await {
//!     Ok(result) => println!("Success: {:?}", result),
//!     Err(ResilienceError::CircuitOpen { .. }) => println!("Circuit is open"),
//!     Err(e) => println!("Error: {:?}", e),
//! }
//! ```

use super::{ResilienceError, ResilienceResult};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CircuitState {
    /// Circuit is closed, requests pass through
    #[default]
    Closed,
    /// Circuit is open, requests fail fast
    Open,
    /// Circuit is half-open, testing recovery
    HalfOpen,
}

impl std::fmt::Display for CircuitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => write!(f, "closed"),
            Self::Open => write!(f, "open"),
            Self::HalfOpen => write!(f, "half-open"),
        }
    }
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening circuit
    pub failure_threshold: u32,
    /// Number of successes to close circuit from half-open
    pub success_threshold: u32,
    /// Time to wait before transitioning from open to half-open
    pub reset_timeout: Duration,
    /// Sliding window size for tracking failures
    pub window_size: Duration,
    /// Maximum concurrent requests in half-open state
    pub half_open_max_requests: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            reset_timeout: Duration::from_secs(30),
            window_size: Duration::from_secs(60),
            half_open_max_requests: 3,
        }
    }
}

impl CircuitBreakerConfig {
    /// Production configuration with conservative settings
    pub fn production() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            reset_timeout: Duration::from_secs(60),
            window_size: Duration::from_secs(120),
            half_open_max_requests: 5,
        }
    }

    /// Set failure threshold
    pub fn failure_threshold(mut self, threshold: u32) -> Self {
        self.failure_threshold = threshold;
        self
    }

    /// Set success threshold for closing
    pub fn success_threshold(mut self, threshold: u32) -> Self {
        self.success_threshold = threshold;
        self
    }

    /// Set reset timeout
    pub fn reset_timeout(mut self, timeout: Duration) -> Self {
        self.reset_timeout = timeout;
        self
    }

    /// Set sliding window size
    pub fn window_size(mut self, size: Duration) -> Self {
        self.window_size = size;
        self
    }

    /// Set max requests in half-open state
    pub fn half_open_max_requests(mut self, max: u32) -> Self {
        self.half_open_max_requests = max;
        self
    }
}

/// Circuit breaker for a kernel
pub struct CircuitBreaker {
    /// Kernel ID this circuit breaker protects
    kernel_id: String,
    /// Configuration
    config: CircuitBreakerConfig,
    /// Inner state
    inner: Arc<CircuitBreakerInner>,
}

struct CircuitBreakerInner {
    state: RwLock<CircuitState>,
    failure_count: AtomicU32,
    success_count: AtomicU32,
    last_failure_time: RwLock<Option<Instant>>,
    half_open_requests: AtomicU32,
    total_requests: AtomicU64,
    total_failures: AtomicU64,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(kernel_id: impl Into<String>, config: CircuitBreakerConfig) -> Self {
        Self {
            kernel_id: kernel_id.into(),
            config,
            inner: Arc::new(CircuitBreakerInner {
                state: RwLock::new(CircuitState::Closed),
                failure_count: AtomicU32::new(0),
                success_count: AtomicU32::new(0),
                last_failure_time: RwLock::new(None),
                half_open_requests: AtomicU32::new(0),
                total_requests: AtomicU64::new(0),
                total_failures: AtomicU64::new(0),
            }),
        }
    }

    /// Get current state
    pub async fn state(&self) -> CircuitState {
        let state = *self.inner.state.read().await;

        // Check if we should transition from Open to HalfOpen
        if state == CircuitState::Open {
            if let Some(last_failure) = *self.inner.last_failure_time.read().await {
                if last_failure.elapsed() >= self.config.reset_timeout {
                    return self.try_transition_to_half_open().await;
                }
            }
        }

        state
    }

    /// Get the kernel ID
    pub fn kernel_id(&self) -> &str {
        &self.kernel_id
    }

    /// Check if requests are allowed
    pub async fn is_allowed(&self) -> bool {
        match self.state().await {
            CircuitState::Closed => true,
            CircuitState::Open => false,
            CircuitState::HalfOpen => {
                self.inner.half_open_requests.load(Ordering::Relaxed)
                    < self.config.half_open_max_requests
            }
        }
    }

    /// Execute a function with circuit breaker protection
    pub async fn execute<F, Fut, T, E>(&self, f: F) -> ResilienceResult<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: Into<crate::error::KernelError>,
    {
        self.inner.total_requests.fetch_add(1, Ordering::Relaxed);

        // Check if circuit allows the request
        let state = self.state().await;
        match state {
            CircuitState::Open => {
                return Err(ResilienceError::CircuitOpen {
                    kernel_id: self.kernel_id.clone(),
                });
            }
            CircuitState::HalfOpen => {
                // Limit concurrent requests in half-open state
                let current = self
                    .inner
                    .half_open_requests
                    .fetch_add(1, Ordering::Relaxed);
                if current >= self.config.half_open_max_requests {
                    self.inner
                        .half_open_requests
                        .fetch_sub(1, Ordering::Relaxed);
                    return Err(ResilienceError::CircuitOpen {
                        kernel_id: self.kernel_id.clone(),
                    });
                }
            }
            CircuitState::Closed => {}
        }

        // Execute the function
        let result = f().await;

        // Record the result
        match &result {
            Ok(_) => self.record_success().await,
            Err(_) => self.record_failure().await,
        }

        // If we were in half-open, decrement the counter
        if state == CircuitState::HalfOpen {
            self.inner
                .half_open_requests
                .fetch_sub(1, Ordering::Relaxed);
        }

        result.map_err(|e| ResilienceError::KernelError(e.into()))
    }

    /// Manually record a success
    pub async fn record_success(&self) {
        let state = *self.inner.state.read().await;

        match state {
            CircuitState::Closed => {
                // Reset failure count on success
                self.inner.failure_count.store(0, Ordering::Relaxed);
            }
            CircuitState::HalfOpen => {
                let successes = self.inner.success_count.fetch_add(1, Ordering::Relaxed) + 1;
                if successes >= self.config.success_threshold {
                    self.transition_to_closed().await;
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Manually record a failure
    pub async fn record_failure(&self) {
        self.inner.total_failures.fetch_add(1, Ordering::Relaxed);
        *self.inner.last_failure_time.write().await = Some(Instant::now());

        let state = *self.inner.state.read().await;

        match state {
            CircuitState::Closed => {
                let failures = self.inner.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
                if failures >= self.config.failure_threshold {
                    self.transition_to_open().await;
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open goes back to open
                self.transition_to_open().await;
            }
            CircuitState::Open => {}
        }
    }

    /// Manually reset the circuit breaker
    pub async fn reset(&self) {
        *self.inner.state.write().await = CircuitState::Closed;
        self.inner.failure_count.store(0, Ordering::Relaxed);
        self.inner.success_count.store(0, Ordering::Relaxed);
        self.inner.half_open_requests.store(0, Ordering::Relaxed);
        *self.inner.last_failure_time.write().await = None;
    }

    /// Get statistics
    pub fn stats(&self) -> CircuitBreakerStats {
        CircuitBreakerStats {
            total_requests: self.inner.total_requests.load(Ordering::Relaxed),
            total_failures: self.inner.total_failures.load(Ordering::Relaxed),
            current_failures: self.inner.failure_count.load(Ordering::Relaxed),
        }
    }

    // Private transition methods

    async fn transition_to_open(&self) {
        *self.inner.state.write().await = CircuitState::Open;
        self.inner.success_count.store(0, Ordering::Relaxed);
        tracing::warn!(
            kernel_id = %self.kernel_id,
            "Circuit breaker opened"
        );
    }

    async fn transition_to_closed(&self) {
        *self.inner.state.write().await = CircuitState::Closed;
        self.inner.failure_count.store(0, Ordering::Relaxed);
        self.inner.success_count.store(0, Ordering::Relaxed);
        tracing::info!(
            kernel_id = %self.kernel_id,
            "Circuit breaker closed"
        );
    }

    async fn try_transition_to_half_open(&self) -> CircuitState {
        let mut state = self.inner.state.write().await;
        if *state == CircuitState::Open {
            *state = CircuitState::HalfOpen;
            self.inner.success_count.store(0, Ordering::Relaxed);
            self.inner.half_open_requests.store(0, Ordering::Relaxed);
            tracing::info!(
                kernel_id = %self.kernel_id,
                "Circuit breaker half-open"
            );
        }
        *state
    }
}

impl Clone for CircuitBreaker {
    fn clone(&self) -> Self {
        Self {
            kernel_id: self.kernel_id.clone(),
            config: self.config.clone(),
            inner: self.inner.clone(),
        }
    }
}

/// Circuit breaker statistics
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    /// Total requests through this breaker
    pub total_requests: u64,
    /// Total failures recorded
    pub total_failures: u64,
    /// Current failure count in window
    pub current_failures: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_circuit_breaker_starts_closed() {
        let cb = CircuitBreaker::new("test", CircuitBreakerConfig::default());
        assert_eq!(cb.state().await, CircuitState::Closed);
        assert!(cb.is_allowed().await);
    }

    #[tokio::test]
    async fn test_circuit_opens_after_failures() {
        let config = CircuitBreakerConfig::default().failure_threshold(3);
        let cb = CircuitBreaker::new("test", config);

        // Record failures
        for _ in 0..3 {
            cb.record_failure().await;
        }

        assert_eq!(cb.state().await, CircuitState::Open);
        assert!(!cb.is_allowed().await);
    }

    #[tokio::test]
    async fn test_circuit_resets_on_success() {
        let config = CircuitBreakerConfig::default().failure_threshold(3);
        let cb = CircuitBreaker::new("test", config);

        // Record some failures
        cb.record_failure().await;
        cb.record_failure().await;

        // Success should reset
        cb.record_success().await;

        assert_eq!(cb.inner.failure_count.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_manual_reset() {
        let config = CircuitBreakerConfig::default().failure_threshold(3);
        let cb = CircuitBreaker::new("test", config);

        // Open the circuit
        for _ in 0..3 {
            cb.record_failure().await;
        }
        assert_eq!(cb.state().await, CircuitState::Open);

        // Manual reset
        cb.reset().await;
        assert_eq!(cb.state().await, CircuitState::Closed);
    }

    #[test]
    fn test_config_builder() {
        let config = CircuitBreakerConfig::default()
            .failure_threshold(10)
            .reset_timeout(Duration::from_secs(60))
            .success_threshold(5);

        assert_eq!(config.failure_threshold, 10);
        assert_eq!(config.reset_timeout, Duration::from_secs(60));
        assert_eq!(config.success_threshold, 5);
    }
}
