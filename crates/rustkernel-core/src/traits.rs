//! Core kernel traits.
//!
//! This module defines the fundamental traits that all kernels implement:
//! - `GpuKernel`: Base trait for all GPU kernels
//! - `BatchKernel`: Trait for batch (CPU-orchestrated) kernels
//! - `RingKernelHandler`: Trait for ring (persistent actor) kernels
//! - `CheckpointableKernel`: Trait for kernels that support checkpoint/restore (0.3.1)
//!
//! ## Enterprise Features (0.3.1)
//!
//! - Health checking for liveness/readiness probes
//! - Execution context with auth, tenant, and tracing
//! - Secure message handling with authentication
//! - Checkpoint/restore for recovery

use crate::error::{KernelError, Result};
use crate::kernel::KernelMetadata;
use async_trait::async_trait;
use ringkernel_core::{RingContext, RingMessage};
use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::time::Duration;
use uuid::Uuid;

// ============================================================================
// Health & Status Types
// ============================================================================

/// Health status for kernel health checks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Kernel is healthy and operational
    #[default]
    Healthy,
    /// Kernel is degraded but still operational
    Degraded,
    /// Kernel is unhealthy and should not receive traffic
    Unhealthy,
    /// Health status is unknown (check failed)
    Unknown,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded => write!(f, "degraded"),
            Self::Unhealthy => write!(f, "unhealthy"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

// ============================================================================
// Execution Context Types
// ============================================================================

/// Execution context for kernel invocations.
///
/// Provides authentication, tenant isolation, and distributed tracing context
/// for kernel execution.
#[derive(Debug, Clone, Default)]
pub struct ExecutionContext {
    /// Request ID for tracing
    pub request_id: Option<Uuid>,
    /// Trace ID for distributed tracing
    pub trace_id: Option<String>,
    /// Span ID for distributed tracing
    pub span_id: Option<String>,
    /// Authenticated user ID (if any)
    pub user_id: Option<String>,
    /// Tenant ID for multi-tenancy
    pub tenant_id: Option<String>,
    /// Request timeout (if specified)
    pub timeout: Option<Duration>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl ExecutionContext {
    /// Create a new execution context
    pub fn new() -> Self {
        Self {
            request_id: Some(Uuid::new_v4()),
            ..Default::default()
        }
    }

    /// Create context with request ID
    pub fn with_request_id(mut self, id: Uuid) -> Self {
        self.request_id = Some(id);
        self
    }

    /// Set trace context
    pub fn with_trace(mut self, trace_id: impl Into<String>, span_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self.span_id = Some(span_id.into());
        self
    }

    /// Set authenticated user
    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Set tenant
    pub fn with_tenant(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Secure ring context with authentication.
///
/// Wraps `RingContext` with security context for authenticated message handling.
pub struct SecureRingContext<'ctx, 'ring> {
    /// The underlying ring context
    pub ring_ctx: &'ctx mut RingContext<'ring>,
    /// Execution context with auth info
    pub exec_ctx: &'ctx ExecutionContext,
}

impl<'ctx, 'ring> SecureRingContext<'ctx, 'ring> {
    /// Create a new secure context
    pub fn new(ring_ctx: &'ctx mut RingContext<'ring>, exec_ctx: &'ctx ExecutionContext) -> Self {
        Self { ring_ctx, exec_ctx }
    }

    /// Get the authenticated user ID
    pub fn user_id(&self) -> Option<&str> {
        self.exec_ctx.user_id.as_deref()
    }

    /// Get the tenant ID
    pub fn tenant_id(&self) -> Option<&str> {
        self.exec_ctx.tenant_id.as_deref()
    }

    /// Check if request is authenticated
    pub fn is_authenticated(&self) -> bool {
        self.exec_ctx.user_id.is_some()
    }
}

// ============================================================================
// Kernel Configuration
// ============================================================================

/// Runtime configuration for a kernel instance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KernelConfig {
    /// Maximum queue depth
    pub max_queue_depth: Option<usize>,
    /// Execution timeout
    pub timeout: Option<Duration>,
    /// Enable tracing
    pub tracing_enabled: bool,
    /// Enable metrics collection
    pub metrics_enabled: bool,
    /// Custom configuration values
    pub custom: std::collections::HashMap<String, serde_json::Value>,
}

impl KernelConfig {
    /// Create a new kernel config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set queue depth
    pub fn with_queue_depth(mut self, depth: usize) -> Self {
        self.max_queue_depth = Some(depth);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Enable tracing
    pub fn with_tracing(mut self, enabled: bool) -> Self {
        self.tracing_enabled = enabled;
        self
    }

    /// Enable metrics
    pub fn with_metrics(mut self, enabled: bool) -> Self {
        self.metrics_enabled = enabled;
        self
    }

    /// Set custom value
    pub fn with_custom(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.custom.insert(key.into(), value);
        self
    }
}

// ============================================================================
// Core Kernel Traits
// ============================================================================

/// Base trait for all GPU kernels.
///
/// Provides access to kernel metadata, health checking, and lifecycle management.
///
/// ## Enterprise Features (0.3.1)
///
/// - `health_check()` - Report kernel health for liveness/readiness probes
/// - `shutdown()` - Graceful shutdown with resource cleanup
/// - `refresh_config()` - Hot configuration reload
pub trait GpuKernel: Send + Sync + Debug {
    /// Returns the kernel metadata.
    fn metadata(&self) -> &KernelMetadata;

    /// Validate kernel configuration.
    ///
    /// Called before kernel launch to ensure configuration is valid.
    fn validate(&self) -> Result<()> {
        Ok(())
    }

    /// Returns the kernel ID.
    fn id(&self) -> &str {
        &self.metadata().id
    }

    /// Returns true if this kernel requires GPU-native execution.
    fn requires_gpu_native(&self) -> bool {
        self.metadata().requires_gpu_native
    }

    // ========================================================================
    // Enterprise Features (0.3.1)
    // ========================================================================

    /// Perform a health check on this kernel.
    ///
    /// Used by liveness and readiness probes. Override to implement
    /// custom health checking logic (e.g., checking GPU memory, connections).
    ///
    /// # Returns
    ///
    /// The current health status of the kernel.
    fn health_check(&self) -> HealthStatus {
        HealthStatus::Healthy
    }

    /// Graceful shutdown of the kernel.
    ///
    /// Called during runtime shutdown to release resources. Override to
    /// implement custom cleanup (e.g., flushing buffers, closing connections).
    ///
    /// Default implementation does nothing.
    fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    /// Refresh kernel configuration at runtime.
    ///
    /// Called when configuration is hot-reloaded. Only safe-to-reload
    /// configuration values should be applied.
    ///
    /// # Arguments
    ///
    /// * `config` - The new configuration to apply
    ///
    /// # Returns
    ///
    /// Ok if configuration was applied, Err if configuration is invalid.
    fn refresh_config(&mut self, _config: &KernelConfig) -> Result<()> {
        Ok(())
    }
}

/// Trait for batch (CPU-orchestrated) kernels.
///
/// Batch kernels are launched on-demand with CPU orchestration.
/// They have 10-50μs launch overhead and state resides in CPU memory.
///
/// ## Enterprise Features (0.3.1)
///
/// - `execute_with_context()` - Execute with auth, tenant, and tracing context
/// - `execute_with_timeout()` - Execute with deadline enforcement
///
/// # Type Parameters
///
/// - `I`: Input type
/// - `O`: Output type
#[async_trait]
pub trait BatchKernel<I, O>: GpuKernel
where
    I: Send + Sync,
    O: Send + Sync,
{
    /// Execute the kernel with the given input.
    ///
    /// # Arguments
    ///
    /// * `input` - The input data for the kernel
    ///
    /// # Returns
    ///
    /// The kernel output or an error.
    async fn execute(&self, input: I) -> Result<O>;

    /// Validate the input before execution.
    ///
    /// Override to provide custom input validation.
    fn validate_input(&self, _input: &I) -> Result<()> {
        Ok(())
    }

    // ========================================================================
    // Enterprise Features (0.3.1)
    // ========================================================================

    /// Execute the kernel with execution context.
    ///
    /// Provides authentication, tenant isolation, and distributed tracing
    /// context for the kernel execution.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The execution context with auth, tenant, and tracing info
    /// * `input` - The input data for the kernel
    ///
    /// # Returns
    ///
    /// The kernel output or an error.
    ///
    /// # Default Implementation
    ///
    /// Delegates to `execute()` ignoring the context. Override to use context.
    async fn execute_with_context(&self, ctx: &ExecutionContext, input: I) -> Result<O>
    where
        I: 'async_trait,
    {
        // Default: ignore context, just execute
        let _ = ctx;
        self.execute(input).await
    }

    /// Execute the kernel with a timeout.
    ///
    /// # Arguments
    ///
    /// * `input` - The input data for the kernel
    /// * `timeout` - Maximum execution time
    ///
    /// # Returns
    ///
    /// The kernel output or a timeout error.
    async fn execute_with_timeout(&self, input: I, timeout: Duration) -> Result<O>
    where
        I: 'async_trait,
    {
        match tokio::time::timeout(timeout, self.execute(input)).await {
            Ok(result) => result,
            Err(_elapsed) => Err(crate::error::KernelError::Timeout(timeout)),
        }
    }
}

/// Trait for ring (persistent actor) kernels.
///
/// Ring kernels are persistent GPU actors with 100-500ns message latency.
/// State resides permanently in GPU memory.
///
/// ## Enterprise Features (0.3.1)
///
/// - `handle_secure()` - Handle messages with security context
///
/// # Type Parameters
///
/// - `M`: Request message type
/// - `R`: Response message type
#[async_trait]
pub trait RingKernelHandler<M, R>: GpuKernel
where
    M: RingMessage + Send + Sync,
    R: RingMessage + Send + Sync,
{
    /// Handle an incoming message.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The ring kernel context with GPU intrinsics
    /// * `msg` - The incoming message
    ///
    /// # Returns
    ///
    /// The response message or an error.
    async fn handle(&self, ctx: &mut RingContext, msg: M) -> Result<R>;

    /// Initialize the kernel state.
    ///
    /// Called once when the kernel is first activated.
    async fn initialize(&self, _ctx: &mut RingContext) -> Result<()> {
        Ok(())
    }

    /// Called when the kernel is being shut down.
    ///
    /// Use this to clean up resources.
    async fn ring_shutdown(&self, _ctx: &mut RingContext) -> Result<()> {
        Ok(())
    }

    // ========================================================================
    // Enterprise Features (0.3.1)
    // ========================================================================

    /// Handle a message with security context.
    ///
    /// Provides authentication and tenant isolation for message handling.
    /// Use this for operations that require authorization checks.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Secure ring context with auth info
    /// * `msg` - The incoming message
    ///
    /// # Returns
    ///
    /// The response message or an error.
    ///
    /// # Default Implementation
    ///
    /// Delegates to `handle()` ignoring security context. Override to
    /// implement authorization checks.
    async fn handle_secure(&self, ctx: &mut SecureRingContext<'_, '_>, msg: M) -> Result<R>
    where
        M: 'async_trait,
        R: 'async_trait,
    {
        // Default: ignore security context, delegate to handle
        self.handle(ctx.ring_ctx, msg).await
    }
}

/// Trait for iterative (multi-pass) kernels.
///
/// Provides support for algorithms that require multiple iterations
/// to converge (e.g., PageRank, K-Means).
///
/// # Type Parameters
///
/// - `S`: State type
/// - `I`: Input type
/// - `O`: Output type
#[async_trait]
pub trait IterativeKernel<S, I, O>: GpuKernel
where
    S: Send + Sync + 'static,
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    /// Create the initial state.
    fn initial_state(&self, input: &I) -> S;

    /// Perform one iteration.
    ///
    /// # Arguments
    ///
    /// * `state` - The current state (mutable)
    /// * `input` - The input data
    ///
    /// # Returns
    ///
    /// The iteration result.
    async fn iterate(&self, state: &mut S, input: &I) -> Result<IterationResult<O>>;

    /// Check if the algorithm has converged.
    ///
    /// # Arguments
    ///
    /// * `state` - The current state
    /// * `threshold` - The convergence threshold
    ///
    /// # Returns
    ///
    /// `true` if converged, `false` otherwise.
    fn converged(&self, state: &S, threshold: f64) -> bool;

    /// Maximum number of iterations.
    fn max_iterations(&self) -> usize {
        100
    }

    /// Default convergence threshold.
    fn default_threshold(&self) -> f64 {
        1e-6
    }

    /// Run the iterative algorithm to convergence.
    async fn run_to_convergence(&self, input: I) -> Result<O> {
        self.run_to_convergence_with_threshold(input, self.default_threshold())
            .await
    }

    /// Run the iterative algorithm with a custom threshold.
    async fn run_to_convergence_with_threshold(&self, input: I, threshold: f64) -> Result<O> {
        let mut state = self.initial_state(&input);
        let max_iter = self.max_iterations();

        for _ in 0..max_iter {
            let result = self.iterate(&mut state, &input).await?;

            if let IterationResult::Converged(output) = result {
                return Ok(output);
            }

            if self.converged(&state, threshold) {
                if let IterationResult::Continue(output) = result {
                    return Ok(output);
                }
            }
        }

        // Return final state even if not converged
        match self.iterate(&mut state, &input).await? {
            IterationResult::Converged(output) | IterationResult::Continue(output) => Ok(output),
        }
    }
}

/// Result of a single iteration.
#[derive(Debug, Clone)]
pub enum IterationResult<O> {
    /// Algorithm has converged with final output.
    Converged(O),
    /// Algorithm should continue; current intermediate output.
    Continue(O),
}

impl<O> IterationResult<O> {
    /// Returns true if converged.
    #[must_use]
    pub fn is_converged(&self) -> bool {
        matches!(self, IterationResult::Converged(_))
    }

    /// Extract the output.
    #[must_use]
    pub fn into_output(self) -> O {
        match self {
            IterationResult::Converged(o) | IterationResult::Continue(o) => o,
        }
    }
}

/// Type-erased batch kernel for registry storage.
#[async_trait]
pub trait BatchKernelDyn: GpuKernel {
    /// Execute with type-erased input/output.
    async fn execute_dyn(&self, input: &[u8]) -> Result<Vec<u8>>;
}

/// Type-erased ring kernel for registry storage.
#[async_trait]
pub trait RingKernelDyn: GpuKernel {
    /// Handle with type-erased messages.
    async fn handle_dyn(&self, ctx: &mut RingContext, msg: &[u8]) -> Result<Vec<u8>>;
}

// ============================================================================
// Type-Erased Kernel Adapters
// ============================================================================

/// Type-erased wrapper for batch kernels enabling dynamic dispatch.
///
/// Wraps any `BatchKernel<I, O>` implementation and provides the
/// `BatchKernelDyn` interface for type-erased execution through
/// JSON serialization/deserialization.
///
/// This enables batch kernels to be stored in the registry and invoked
/// via REST, gRPC, and other service interfaces without compile-time
/// knowledge of the kernel's input/output types.
///
/// # Example
///
/// ```ignore
/// use rustkernel_core::traits::TypeErasedBatchKernel;
///
/// let kernel = TypeErasedBatchKernel::new(MyKernel::new());
/// let output = kernel.execute_dyn(b"{\"field\": 42}").await?;
/// ```
pub struct TypeErasedBatchKernel<K, I, O> {
    inner: K,
    // fn(I) -> O is always Send + Sync regardless of I/O bounds
    _phantom: PhantomData<fn(I) -> O>,
}

impl<K: Debug, I, O> Debug for TypeErasedBatchKernel<K, I, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TypeErasedBatchKernel")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<K, I, O> TypeErasedBatchKernel<K, I, O> {
    /// Wrap a typed batch kernel for type-erased execution.
    pub fn new(kernel: K) -> Self {
        Self {
            inner: kernel,
            _phantom: PhantomData,
        }
    }

    /// Access the inner kernel.
    pub fn inner(&self) -> &K {
        &self.inner
    }
}

impl<K, I, O> GpuKernel for TypeErasedBatchKernel<K, I, O>
where
    K: GpuKernel,
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    fn metadata(&self) -> &KernelMetadata {
        self.inner.metadata()
    }

    fn validate(&self) -> Result<()> {
        self.inner.validate()
    }

    fn health_check(&self) -> HealthStatus {
        self.inner.health_check()
    }

    fn shutdown(&self) -> Result<()> {
        self.inner.shutdown()
    }

    fn refresh_config(&mut self, config: &KernelConfig) -> Result<()> {
        self.inner.refresh_config(config)
    }
}

#[async_trait]
impl<K, I, O> BatchKernelDyn for TypeErasedBatchKernel<K, I, O>
where
    K: BatchKernel<I, O> + 'static,
    I: serde::de::DeserializeOwned + Send + Sync + 'static,
    O: serde::Serialize + Send + Sync + 'static,
{
    async fn execute_dyn(&self, input: &[u8]) -> Result<Vec<u8>> {
        let typed_input: I = serde_json::from_slice(input)
            .map_err(|e| KernelError::DeserializationError(e.to_string()))?;
        let output = self.inner.execute(typed_input).await?;
        serde_json::to_vec(&output)
            .map_err(|e| KernelError::SerializationError(e.to_string()))
    }
}

/// Type-erased wrapper for ring kernels enabling dynamic dispatch.
///
/// Similar to [`TypeErasedBatchKernel`] but for ring kernels that handle
/// messages through the RingKernel persistent actor model.
pub struct TypeErasedRingKernel<K, M, R> {
    inner: K,
    _phantom: PhantomData<fn(M) -> R>,
}

impl<K: Debug, M, R> Debug for TypeErasedRingKernel<K, M, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TypeErasedRingKernel")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<K, M, R> TypeErasedRingKernel<K, M, R> {
    /// Wrap a typed ring kernel for type-erased message handling.
    pub fn new(kernel: K) -> Self {
        Self {
            inner: kernel,
            _phantom: PhantomData,
        }
    }
}

impl<K, M, R> GpuKernel for TypeErasedRingKernel<K, M, R>
where
    K: GpuKernel,
    M: Send + Sync + 'static,
    R: Send + Sync + 'static,
{
    fn metadata(&self) -> &KernelMetadata {
        self.inner.metadata()
    }

    fn validate(&self) -> Result<()> {
        self.inner.validate()
    }

    fn health_check(&self) -> HealthStatus {
        self.inner.health_check()
    }

    fn shutdown(&self) -> Result<()> {
        self.inner.shutdown()
    }

    fn refresh_config(&mut self, config: &KernelConfig) -> Result<()> {
        self.inner.refresh_config(config)
    }
}

#[async_trait]
impl<K, M, R> RingKernelDyn for TypeErasedRingKernel<K, M, R>
where
    K: RingKernelHandler<M, R> + 'static,
    M: RingMessage + serde::de::DeserializeOwned + Send + Sync + 'static,
    R: RingMessage + serde::Serialize + Send + Sync + 'static,
{
    async fn handle_dyn(&self, ctx: &mut RingContext, msg: &[u8]) -> Result<Vec<u8>> {
        let typed_msg: M = serde_json::from_slice(msg)
            .map_err(|e| KernelError::DeserializationError(e.to_string()))?;
        let response = self.inner.handle(ctx, typed_msg).await?;
        serde_json::to_vec(&response)
            .map_err(|e| KernelError::SerializationError(e.to_string()))
    }
}

// ============================================================================
// Enterprise Traits (0.3.1)
// ============================================================================

/// Trait for kernels that support checkpoint/restore.
///
/// Enables recovery from failures by saving and restoring kernel state.
/// Useful for long-running or stateful kernels.
///
/// # Type Parameters
///
/// - `C`: Checkpoint type (must be serializable)
#[async_trait]
pub trait CheckpointableKernel: GpuKernel {
    /// The checkpoint state type
    type Checkpoint: Serialize + serde::de::DeserializeOwned + Send + Sync;

    /// Create a checkpoint of current kernel state.
    ///
    /// # Returns
    ///
    /// A serializable checkpoint that can be used to restore state.
    async fn checkpoint(&self) -> Result<Self::Checkpoint>;

    /// Restore kernel state from a checkpoint.
    ///
    /// # Arguments
    ///
    /// * `checkpoint` - Previously saved checkpoint state
    ///
    /// # Returns
    ///
    /// Ok if state was restored, Err if checkpoint is invalid.
    async fn restore(&mut self, checkpoint: Self::Checkpoint) -> Result<()>;

    /// Check if checkpointing is currently safe.
    ///
    /// Returns false if the kernel is in the middle of an operation
    /// that cannot be interrupted.
    fn can_checkpoint(&self) -> bool {
        true
    }

    /// Get the size of the checkpoint in bytes (estimate).
    ///
    /// Useful for monitoring and capacity planning.
    fn checkpoint_size_estimate(&self) -> usize {
        0
    }
}

/// Trait for kernels that support graceful degradation.
///
/// When resources are constrained, these kernels can operate in
/// a reduced-functionality mode rather than failing completely.
pub trait DegradableKernel: GpuKernel {
    /// Enter degraded mode.
    ///
    /// Called when resources are constrained. The kernel should
    /// reduce functionality while remaining operational.
    fn enter_degraded_mode(&mut self) -> Result<()>;

    /// Exit degraded mode.
    ///
    /// Called when resources are restored. The kernel should
    /// resume full functionality.
    fn exit_degraded_mode(&mut self) -> Result<()>;

    /// Check if kernel is in degraded mode.
    fn is_degraded(&self) -> bool;

    /// Get description of current degradation.
    fn degradation_info(&self) -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iteration_result() {
        let converged: IterationResult<i32> = IterationResult::Converged(42);
        assert!(converged.is_converged());
        assert_eq!(converged.into_output(), 42);

        let continuing: IterationResult<i32> = IterationResult::Continue(0);
        assert!(!continuing.is_converged());
    }

    #[test]
    fn test_health_status() {
        assert_eq!(HealthStatus::default(), HealthStatus::Healthy);
        assert_eq!(format!("{}", HealthStatus::Healthy), "healthy");
        assert_eq!(format!("{}", HealthStatus::Degraded), "degraded");
    }

    #[test]
    fn test_execution_context() {
        let ctx = ExecutionContext::new()
            .with_user("user123")
            .with_tenant("tenant456")
            .with_timeout(Duration::from_secs(30));

        assert!(ctx.request_id.is_some());
        assert_eq!(ctx.user_id.as_deref(), Some("user123"));
        assert_eq!(ctx.tenant_id.as_deref(), Some("tenant456"));
        assert_eq!(ctx.timeout, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_kernel_config() {
        let config = KernelConfig::new()
            .with_queue_depth(1000)
            .with_timeout(Duration::from_secs(60))
            .with_tracing(true)
            .with_metrics(true);

        assert_eq!(config.max_queue_depth, Some(1000));
        assert_eq!(config.timeout, Some(Duration::from_secs(60)));
        assert!(config.tracing_enabled);
        assert!(config.metrics_enabled);
    }
}
