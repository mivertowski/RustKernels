//! Timeout and Deadline Management
//!
//! Provides timeout enforcement and deadline propagation for kernel execution.
//!
//! # Features
//!
//! - Per-kernel timeout configuration
//! - Deadline propagation in K2K chains
//! - Cancellation token support
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_core::resilience::timeout::{TimeoutConfig, DeadlineContext};
//!
//! let config = TimeoutConfig::default()
//!     .default_timeout(Duration::from_secs(30))
//!     .max_timeout(Duration::from_secs(300));
//!
//! let deadline = DeadlineContext::new(Duration::from_secs(10));
//! let result = deadline.execute(|| async {
//!     kernel.execute(input).await
//! }).await?;
//! ```

use super::{ResilienceError, ResilienceResult};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Default timeout for kernel execution
    pub default_timeout: Duration,
    /// Maximum allowed timeout
    pub max_timeout: Duration,
    /// Enable deadline propagation in K2K chains
    pub propagate_deadline: bool,
    /// Include queue wait time in timeout
    pub include_queue_time: bool,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(30),
            max_timeout: Duration::from_secs(300),
            propagate_deadline: true,
            include_queue_time: false,
        }
    }
}

impl TimeoutConfig {
    /// Production configuration
    pub fn production() -> Self {
        Self {
            default_timeout: Duration::from_secs(60),
            max_timeout: Duration::from_secs(600),
            propagate_deadline: true,
            include_queue_time: true,
        }
    }

    /// Development configuration
    pub fn development() -> Self {
        Self {
            default_timeout: Duration::from_secs(300),
            max_timeout: Duration::from_secs(3600),
            propagate_deadline: false,
            include_queue_time: false,
        }
    }

    /// Set default timeout
    pub fn default_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Set max timeout
    pub fn max_timeout(mut self, timeout: Duration) -> Self {
        self.max_timeout = timeout;
        self
    }

    /// Enable deadline propagation
    pub fn propagate_deadline(mut self, propagate: bool) -> Self {
        self.propagate_deadline = propagate;
        self
    }

    /// Include queue time in timeout
    pub fn include_queue_time(mut self, include: bool) -> Self {
        self.include_queue_time = include;
        self
    }

    /// Clamp a timeout to the configured maximum
    pub fn clamp(&self, timeout: Duration) -> Duration {
        timeout.min(self.max_timeout)
    }
}

/// Timeout error
#[derive(Debug, thiserror::Error)]
pub enum TimeoutError {
    /// Operation timed out
    #[error("Operation timed out after {timeout:?}")]
    Timeout { timeout: Duration },

    /// Deadline exceeded
    #[error("Deadline exceeded (remaining: {remaining:?})")]
    DeadlineExceeded { remaining: Duration },

    /// Invalid timeout value
    #[error("Invalid timeout: {reason}")]
    Invalid { reason: String },
}

/// Deadline context for propagating deadlines
#[derive(Debug, Clone)]
pub struct DeadlineContext {
    /// Absolute deadline
    deadline: Instant,
    /// Original timeout
    original_timeout: Duration,
    /// When the context was created
    created_at: Instant,
}

impl DeadlineContext {
    /// Create a new deadline context
    pub fn new(timeout: Duration) -> Self {
        let now = Instant::now();
        Self {
            deadline: now + timeout,
            original_timeout: timeout,
            created_at: now,
        }
    }

    /// Create from an absolute deadline
    pub fn from_deadline(deadline: Instant) -> Self {
        let now = Instant::now();
        let remaining = deadline.saturating_duration_since(now);
        Self {
            deadline,
            original_timeout: remaining,
            created_at: now,
        }
    }

    /// Get remaining time until deadline
    pub fn remaining(&self) -> Duration {
        self.deadline.saturating_duration_since(Instant::now())
    }

    /// Check if deadline has passed
    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.deadline
    }

    /// Get the original timeout
    pub fn original_timeout(&self) -> Duration {
        self.original_timeout
    }

    /// Get elapsed time since context creation
    pub fn elapsed(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Create a child context with the same deadline
    pub fn child(&self) -> Self {
        Self {
            deadline: self.deadline,
            original_timeout: self.remaining(),
            created_at: Instant::now(),
        }
    }

    /// Create a child context with a reduced timeout
    pub fn child_with_timeout(&self, max_timeout: Duration) -> Self {
        let remaining = self.remaining();
        let timeout = remaining.min(max_timeout);
        Self {
            deadline: Instant::now() + timeout,
            original_timeout: timeout,
            created_at: Instant::now(),
        }
    }

    /// Execute a future with this deadline
    pub async fn execute<F, Fut, T, E>(&self, f: F) -> ResilienceResult<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: Into<crate::error::KernelError>,
    {
        if self.is_expired() {
            return Err(ResilienceError::DeadlineExceeded);
        }

        let remaining = self.remaining();
        match tokio::time::timeout(remaining, f()).await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(e)) => Err(ResilienceError::KernelError(e.into())),
            Err(_elapsed) => Err(ResilienceError::Timeout { timeout: remaining }),
        }
    }

    /// Check deadline and return error if exceeded
    pub fn check(&self) -> ResilienceResult<()> {
        if self.is_expired() {
            Err(ResilienceError::DeadlineExceeded)
        } else {
            Ok(())
        }
    }
}

/// Timeout guard for tracking execution time
pub struct TimeoutGuard {
    start: Instant,
    timeout: Duration,
    name: String,
}

impl TimeoutGuard {
    /// Create a new timeout guard
    pub fn new(name: impl Into<String>, timeout: Duration) -> Self {
        Self {
            start: Instant::now(),
            timeout,
            name: name.into(),
        }
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Check if timeout is exceeded
    pub fn is_exceeded(&self) -> bool {
        self.elapsed() > self.timeout
    }

    /// Get remaining time
    pub fn remaining(&self) -> Duration {
        self.timeout.saturating_sub(self.elapsed())
    }

    /// Check and log if exceeded
    pub fn check(&self) -> ResilienceResult<()> {
        if self.is_exceeded() {
            tracing::warn!(
                name = %self.name,
                elapsed = ?self.elapsed(),
                timeout = ?self.timeout,
                "Timeout exceeded"
            );
            Err(ResilienceError::Timeout { timeout: self.timeout })
        } else {
            Ok(())
        }
    }
}

impl Drop for TimeoutGuard {
    fn drop(&mut self) {
        let elapsed = self.elapsed();
        if elapsed > self.timeout {
            tracing::warn!(
                name = %self.name,
                elapsed = ?elapsed,
                timeout = ?self.timeout,
                "Operation exceeded timeout"
            );
        } else if elapsed > self.timeout / 2 {
            tracing::debug!(
                name = %self.name,
                elapsed = ?elapsed,
                timeout = ?self.timeout,
                "Operation took >50% of timeout"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeout_config() {
        let config = TimeoutConfig::default()
            .default_timeout(Duration::from_secs(60))
            .max_timeout(Duration::from_secs(120));

        assert_eq!(config.default_timeout, Duration::from_secs(60));
        assert_eq!(config.max_timeout, Duration::from_secs(120));

        // Test clamping
        assert_eq!(
            config.clamp(Duration::from_secs(300)),
            Duration::from_secs(120)
        );
    }

    #[test]
    fn test_deadline_context() {
        let ctx = DeadlineContext::new(Duration::from_secs(10));
        assert!(!ctx.is_expired());
        assert!(ctx.remaining() <= Duration::from_secs(10));
    }

    #[test]
    fn test_deadline_child() {
        let parent = DeadlineContext::new(Duration::from_secs(10));
        let child = parent.child();

        // Child inherits the same deadline as parent, so remaining times should be
        // approximately equal (with small tolerance for execution time)
        let parent_remaining = parent.remaining();
        let child_remaining = child.remaining();

        // Allow 100ms tolerance for test execution time
        let tolerance = Duration::from_millis(100);
        assert!(
            child_remaining <= parent_remaining + tolerance,
            "Child remaining {:?} should be <= parent {:?} + tolerance {:?}",
            child_remaining,
            parent_remaining,
            tolerance
        );
    }

    #[test]
    fn test_timeout_guard() {
        let guard = TimeoutGuard::new("test", Duration::from_secs(10));
        assert!(!guard.is_exceeded());
        assert!(guard.remaining() <= Duration::from_secs(10));
    }

    #[tokio::test]
    async fn test_deadline_expired() {
        let ctx = DeadlineContext::new(Duration::from_nanos(1));
        std::thread::sleep(Duration::from_millis(1));

        assert!(ctx.is_expired());
        assert!(ctx.check().is_err());
    }
}
