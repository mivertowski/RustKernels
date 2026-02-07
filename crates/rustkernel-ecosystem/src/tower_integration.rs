//! Tower Service Integration
//!
//! Provides Tower middleware and services for kernel execution.
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_ecosystem::tower::{KernelService, KernelLayer};
//! use tower::ServiceBuilder;
//!
//! let service = ServiceBuilder::new()
//!     .layer(KernelLayer::new(registry))
//!     .service(inner_service);
//! ```

use crate::{EcosystemError, KernelRequest, KernelResponse, ResponseMetadata};
use rustkernel_core::registry::KernelRegistry;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

/// Kernel service for Tower
pub struct KernelService {
    registry: Arc<KernelRegistry>,
    default_timeout: std::time::Duration,
}

impl KernelService {
    /// Create a new kernel service
    pub fn new(registry: Arc<KernelRegistry>) -> Self {
        Self {
            registry,
            default_timeout: std::time::Duration::from_secs(30),
        }
    }

    /// Set the default execution timeout
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Execute a kernel request.
    ///
    /// Looks up the batch kernel in the registry, creates an instance, and executes it
    /// with the provided JSON input. Ring kernels cannot be executed through this interface.
    pub async fn execute(&self, request: KernelRequest) -> Result<KernelResponse, EcosystemError> {
        let start = Instant::now();
        let request_id = uuid::Uuid::new_v4().to_string();

        // Try batch kernel execution
        if let Some(entry) = self.registry.get_batch(&request.kernel_id) {
            let kernel = entry.create();

            let input_bytes = serde_json::to_vec(&request.input)
                .map_err(|e| EcosystemError::InvalidRequest(format!("Invalid input: {}", e)))?;

            // Apply timeout from request metadata or default
            let timeout_ms = request
                .metadata
                .timeout_ms
                .unwrap_or(self.default_timeout.as_millis() as u64);
            let timeout = std::time::Duration::from_millis(timeout_ms);

            let result = tokio::time::timeout(timeout, kernel.execute_dyn(&input_bytes)).await;

            match result {
                Ok(Ok(output_bytes)) => {
                    let output: serde_json::Value = serde_json::from_slice(&output_bytes)
                        .map_err(|e| {
                            EcosystemError::InternalError(format!(
                                "Output deserialization: {}",
                                e
                            ))
                        })?;

                    let duration_us = start.elapsed().as_micros() as u64;

                    Ok(KernelResponse {
                        request_id,
                        kernel_id: request.kernel_id,
                        output,
                        metadata: ResponseMetadata {
                            duration_us,
                            backend: entry.metadata.mode.as_str().to_uppercase(),
                            gpu_memory_bytes: None,
                            trace_id: request.metadata.trace_id,
                        },
                    })
                }
                Ok(Err(e)) => Err(EcosystemError::ExecutionFailed(e.to_string())),
                Err(_) => Err(EcosystemError::ServiceUnavailable(format!(
                    "Kernel execution timed out after {}ms",
                    timeout_ms
                ))),
            }
        } else if self.registry.get(&request.kernel_id).is_some() {
            Err(EcosystemError::InvalidRequest(format!(
                "Kernel '{}' is a Ring kernel and cannot be executed via this interface. \
                 Use the Ring protocol or gRPC streaming API.",
                request.kernel_id
            )))
        } else {
            Err(EcosystemError::KernelNotFound(request.kernel_id))
        }
    }
}

impl Clone for KernelService {
    fn clone(&self) -> Self {
        Self {
            registry: self.registry.clone(),
            default_timeout: self.default_timeout,
        }
    }
}

/// Tower Service implementation for KernelService
impl tower::Service<KernelRequest> for KernelService {
    type Response = KernelResponse;
    type Error = EcosystemError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: KernelRequest) -> Self::Future {
        let service = self.clone();
        Box::pin(async move { service.execute(req).await })
    }
}

/// Layer for adding kernel service to a service stack
pub struct KernelLayer {
    registry: Arc<KernelRegistry>,
}

impl KernelLayer {
    /// Create a new kernel layer
    pub fn new(registry: Arc<KernelRegistry>) -> Self {
        Self { registry }
    }
}

impl<S> tower::Layer<S> for KernelLayer {
    type Service = KernelMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        KernelMiddleware {
            inner,
            registry: self.registry.clone(),
        }
    }
}

/// Middleware that adds kernel execution capabilities
pub struct KernelMiddleware<S> {
    inner: S,
    registry: Arc<KernelRegistry>,
}

impl<S> KernelMiddleware<S> {
    /// Get the kernel registry
    pub fn registry(&self) -> &Arc<KernelRegistry> {
        &self.registry
    }
}

impl<S: Clone> Clone for KernelMiddleware<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            registry: self.registry.clone(),
        }
    }
}

/// Request timeout layer
pub struct TimeoutLayer {
    timeout: std::time::Duration,
}

impl TimeoutLayer {
    /// Create a new timeout layer
    pub fn new(timeout: std::time::Duration) -> Self {
        Self { timeout }
    }
}

impl<S> tower::Layer<S> for TimeoutLayer {
    type Service = TimeoutService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        TimeoutService {
            inner,
            timeout: self.timeout,
        }
    }
}

/// Service that applies a timeout to requests
pub struct TimeoutService<S> {
    inner: S,
    timeout: std::time::Duration,
}

impl<S: Clone> Clone for TimeoutService<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            timeout: self.timeout,
        }
    }
}

impl<S, Request> tower::Service<Request> for TimeoutService<S>
where
    S: tower::Service<Request> + Clone + Send + 'static,
    S::Future: Send,
    Request: Send + 'static,
{
    type Response = S::Response;
    type Error = TimeoutError<S::Error>;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx).map_err(TimeoutError::Inner)
    }

    fn call(&mut self, req: Request) -> Self::Future {
        let mut inner = self.inner.clone();
        let timeout = self.timeout;

        Box::pin(async move {
            tokio::time::timeout(timeout, inner.call(req))
                .await
                .map_err(|_| TimeoutError::Timeout)?
                .map_err(TimeoutError::Inner)
        })
    }
}

/// Timeout error
#[derive(Debug, thiserror::Error)]
pub enum TimeoutError<E> {
    /// Request timed out
    #[error("Request timed out")]
    Timeout,
    /// Inner service error
    #[error("Service error: {0}")]
    Inner(E),
}

/// Rate limiter layer using a token bucket algorithm.
///
/// Allows up to `burst_size` requests immediately, then refills at
/// `requests_per_second` rate.
pub struct RateLimiterLayer {
    requests_per_second: u32,
    burst_size: u32,
}

impl RateLimiterLayer {
    /// Create a new rate limiter layer
    pub fn new(requests_per_second: u32, burst_size: u32) -> Self {
        Self {
            requests_per_second,
            burst_size,
        }
    }
}

impl<S> tower::Layer<S> for RateLimiterLayer {
    type Service = RateLimiterService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RateLimiterService {
            inner,
            requests_per_second: self.requests_per_second as f64,
            burst_size: self.burst_size as f64,
            state: Arc::new(tokio::sync::Mutex::new(TokenBucketState {
                tokens: self.burst_size as f64,
                last_refill: std::time::Instant::now(),
            })),
        }
    }
}

/// Internal state for the token bucket
struct TokenBucketState {
    tokens: f64,
    last_refill: std::time::Instant,
}

/// Rate limiter service using a token bucket algorithm
pub struct RateLimiterService<S> {
    inner: S,
    requests_per_second: f64,
    burst_size: f64,
    state: Arc<tokio::sync::Mutex<TokenBucketState>>,
}

impl<S: Clone> Clone for RateLimiterService<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            requests_per_second: self.requests_per_second,
            burst_size: self.burst_size,
            state: self.state.clone(),
        }
    }
}

impl<S, Request> tower::Service<Request> for RateLimiterService<S>
where
    S: tower::Service<Request> + Clone + Send + 'static,
    S::Future: Send,
    Request: Send + 'static,
{
    type Response = S::Response;
    type Error = RateLimitError<S::Error>;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx).map_err(RateLimitError::Inner)
    }

    fn call(&mut self, req: Request) -> Self::Future {
        let mut inner = self.inner.clone();
        let rps = self.requests_per_second;
        let burst = self.burst_size;
        let state = self.state.clone();

        Box::pin(async move {
            let mut bucket = state.lock().await;

            // Refill tokens based on elapsed time
            let elapsed = bucket.last_refill.elapsed().as_secs_f64();
            bucket.tokens = (bucket.tokens + elapsed * rps).min(burst);
            bucket.last_refill = std::time::Instant::now();

            // Check if we have a token available
            if bucket.tokens < 1.0 {
                return Err(RateLimitError::RateLimitExceeded);
            }

            bucket.tokens -= 1.0;
            drop(bucket);

            inner.call(req).await.map_err(RateLimitError::Inner)
        })
    }
}

/// Rate limit error
#[derive(Debug, thiserror::Error)]
pub enum RateLimitError<E> {
    /// Rate limit exceeded
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
    /// Inner service error
    #[error("Service error: {0}")]
    Inner(E),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequestMetadata;

    #[tokio::test]
    async fn test_kernel_service() {
        let registry = Arc::new(KernelRegistry::new());
        let service = KernelService::new(registry);

        let request = KernelRequest {
            kernel_id: "nonexistent".to_string(),
            input: serde_json::json!({}),
            metadata: RequestMetadata::default(),
        };

        let result = service.execute(request).await;
        assert!(matches!(result, Err(EcosystemError::KernelNotFound(_))));
    }

    #[tokio::test]
    async fn test_timeout_layer() {
        use std::time::Duration;

        let layer = TimeoutLayer::new(Duration::from_millis(100));
        assert_eq!(layer.timeout, Duration::from_millis(100));
    }

    #[test]
    fn test_rate_limiter_layer() {
        let layer = RateLimiterLayer::new(100, 200);
        assert_eq!(layer.requests_per_second, 100);
    }
}
