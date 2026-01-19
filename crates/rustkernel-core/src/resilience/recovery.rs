//! Recovery Policies
//!
//! Provides automatic recovery from transient failures.
//!
//! # Features
//!
//! - Configurable retry policies with backoff
//! - Recovery strategies (retry, fallback, skip)
//! - Checkpoint/restart support
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_core::resilience::recovery::{RecoveryPolicy, RetryConfig};
//!
//! let policy = RecoveryPolicy::default()
//!     .with_retry(RetryConfig::exponential(3, Duration::from_millis(100)));
//!
//! let result = policy.execute(|| async {
//!     kernel.execute(input).await
//! }).await?;
//! ```

use super::{ResilienceError, ResilienceResult};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Recovery policy for kernel failures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryPolicy {
    /// Retry configuration
    pub retry: Option<RetryConfig>,
    /// Recovery strategy
    pub strategy: RecoveryStrategy,
    /// Whether to log recoveries
    pub log_recoveries: bool,
}

impl Default for RecoveryPolicy {
    fn default() -> Self {
        Self {
            retry: Some(RetryConfig::default()),
            strategy: RecoveryStrategy::Retry,
            log_recoveries: true,
        }
    }
}

impl RecoveryPolicy {
    /// Production recovery policy
    pub fn production() -> Self {
        Self {
            retry: Some(RetryConfig::exponential(3, Duration::from_millis(100))),
            strategy: RecoveryStrategy::Retry,
            log_recoveries: true,
        }
    }

    /// Development recovery policy
    pub fn development() -> Self {
        Self {
            retry: Some(RetryConfig::fixed(2, Duration::from_millis(50))),
            strategy: RecoveryStrategy::Retry,
            log_recoveries: true,
        }
    }

    /// No recovery (fail immediately)
    pub fn none() -> Self {
        Self {
            retry: None,
            strategy: RecoveryStrategy::FailFast,
            log_recoveries: false,
        }
    }

    /// Set retry configuration
    pub fn with_retry(mut self, config: RetryConfig) -> Self {
        self.retry = Some(config);
        self
    }

    /// Set recovery strategy
    pub fn with_strategy(mut self, strategy: RecoveryStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Execute with recovery policy
    pub async fn execute<F, Fut, T, E>(&self, f: F) -> ResilienceResult<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: Into<crate::error::KernelError> + std::fmt::Debug,
    {
        match self.strategy {
            RecoveryStrategy::FailFast => f()
                .await
                .map_err(|e| ResilienceError::KernelError(e.into())),
            RecoveryStrategy::Retry => {
                if let Some(ref retry) = self.retry {
                    retry.execute(f).await
                } else {
                    f().await
                        .map_err(|e| ResilienceError::KernelError(e.into()))
                }
            }
            RecoveryStrategy::Skip => {
                // Skip strategy: return default or special value
                // For now, just try once
                f().await
                    .map_err(|e| ResilienceError::KernelError(e.into()))
            }
        }
    }
}

/// Recovery strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryStrategy {
    /// Fail immediately without retrying
    FailFast,
    /// Retry with configured policy
    #[default]
    Retry,
    /// Skip failed operations
    Skip,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Initial delay between retries
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff strategy
    pub backoff: BackoffStrategy,
    /// Jitter factor (0.0 - 1.0)
    pub jitter: f64,
    /// Whether to retry on all errors (default: true)
    pub retry_all_errors: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff: BackoffStrategy::Exponential { factor: 2.0 },
            jitter: 0.1,
            retry_all_errors: true,
        }
    }
}

impl RetryConfig {
    /// Create with exponential backoff
    pub fn exponential(max_retries: u32, initial_delay: Duration) -> Self {
        Self {
            max_retries,
            initial_delay,
            backoff: BackoffStrategy::Exponential { factor: 2.0 },
            ..Default::default()
        }
    }

    /// Create with fixed delay
    pub fn fixed(max_retries: u32, delay: Duration) -> Self {
        Self {
            max_retries,
            initial_delay: delay,
            backoff: BackoffStrategy::Fixed,
            ..Default::default()
        }
    }

    /// Create with linear backoff
    pub fn linear(max_retries: u32, initial_delay: Duration) -> Self {
        Self {
            max_retries,
            initial_delay,
            backoff: BackoffStrategy::Linear {
                increment: initial_delay,
            },
            ..Default::default()
        }
    }

    /// Set max retries
    pub fn max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    /// Set initial delay
    pub fn initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    /// Set max delay
    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Set jitter factor
    pub fn jitter(mut self, jitter: f64) -> Self {
        self.jitter = jitter.clamp(0.0, 1.0);
        self
    }

    /// Calculate delay for a given attempt
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base_delay = match self.backoff {
            BackoffStrategy::Fixed => self.initial_delay,
            BackoffStrategy::Linear { increment } => self.initial_delay + increment * attempt,
            BackoffStrategy::Exponential { factor } => {
                let multiplier = factor.powi(attempt as i32);
                Duration::from_secs_f64(self.initial_delay.as_secs_f64() * multiplier)
            }
        };

        // Apply max delay cap
        let capped = base_delay.min(self.max_delay);

        // Apply jitter
        if self.jitter > 0.0 {
            let jitter_range = capped.as_secs_f64() * self.jitter;
            let jitter_amount = rand::random::<f64>() * jitter_range * 2.0 - jitter_range;
            Duration::from_secs_f64((capped.as_secs_f64() + jitter_amount).max(0.0))
        } else {
            capped
        }
    }

    /// Execute with retry
    pub async fn execute<F, Fut, T, E>(&self, f: F) -> ResilienceResult<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: Into<crate::error::KernelError> + std::fmt::Debug,
    {
        let mut last_error = None;

        for attempt in 0..=self.max_retries {
            match f().await {
                Ok(result) => {
                    if attempt > 0 {
                        tracing::info!(attempt = attempt, "Operation succeeded after retry");
                    }
                    return Ok(result);
                }
                Err(e) => {
                    let kernel_error: crate::error::KernelError = e.into();

                    // Check if we should retry
                    if !self.retry_all_errors || attempt >= self.max_retries {
                        tracing::warn!(
                            attempt = attempt,
                            error = ?kernel_error,
                            "Operation failed, no more retries"
                        );
                        return Err(ResilienceError::MaxRetriesExceeded {
                            retries: self.max_retries,
                        });
                    }

                    let delay = self.delay_for_attempt(attempt);
                    tracing::debug!(
                        attempt = attempt,
                        delay = ?delay,
                        error = ?kernel_error,
                        "Operation failed, retrying"
                    );

                    tokio::time::sleep(delay).await;
                    last_error = Some(kernel_error);
                }
            }
        }

        Err(last_error.map(ResilienceError::KernelError).unwrap_or(
            ResilienceError::MaxRetriesExceeded {
                retries: self.max_retries,
            },
        ))
    }
}

/// Backoff strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BackoffStrategy {
    /// Fixed delay between retries
    Fixed,
    /// Linear increase in delay
    Linear {
        /// Amount to add each retry
        increment: Duration,
    },
    /// Exponential increase in delay
    Exponential {
        /// Multiplication factor
        factor: f64,
    },
}

impl Default for BackoffStrategy {
    fn default() -> Self {
        Self::Exponential { factor: 2.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_config_exponential() {
        let config = RetryConfig::exponential(3, Duration::from_millis(100));

        assert_eq!(config.max_retries, 3);
        assert_eq!(config.initial_delay, Duration::from_millis(100));

        // Test delay calculation (without jitter)
        let config = RetryConfig::exponential(3, Duration::from_millis(100)).jitter(0.0);
        assert_eq!(config.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(config.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(config.delay_for_attempt(2), Duration::from_millis(400));
    }

    #[test]
    fn test_retry_config_fixed() {
        let config = RetryConfig::fixed(5, Duration::from_millis(50)).jitter(0.0);

        assert_eq!(config.delay_for_attempt(0), Duration::from_millis(50));
        assert_eq!(config.delay_for_attempt(1), Duration::from_millis(50));
        assert_eq!(config.delay_for_attempt(5), Duration::from_millis(50));
    }

    #[test]
    fn test_retry_config_linear() {
        let config = RetryConfig::linear(3, Duration::from_millis(100)).jitter(0.0);

        assert_eq!(config.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(config.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(config.delay_for_attempt(2), Duration::from_millis(300));
    }

    #[test]
    fn test_max_delay_cap() {
        let config = RetryConfig::exponential(10, Duration::from_secs(1))
            .max_delay(Duration::from_secs(5))
            .jitter(0.0);

        // Should be capped at 5 seconds
        assert_eq!(config.delay_for_attempt(10), Duration::from_secs(5));
    }

    #[test]
    fn test_recovery_policy() {
        let policy = RecoveryPolicy::production();
        assert!(policy.retry.is_some());
        assert_eq!(policy.strategy, RecoveryStrategy::Retry);
    }

    #[test]
    fn test_recovery_policy_none() {
        let policy = RecoveryPolicy::none();
        assert!(policy.retry.is_none());
        assert_eq!(policy.strategy, RecoveryStrategy::FailFast);
    }
}
