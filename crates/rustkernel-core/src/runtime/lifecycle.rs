//! Runtime Lifecycle Management
//!
//! Manages the lifecycle of the RustKernels runtime including:
//! - State machine transitions
//! - Graceful shutdown with drain periods
//! - Health check coordination
//! - Event callbacks

use super::config::RuntimeConfig;
use super::{RuntimeEvent, RuntimeEventCallback, RuntimeStats, ShutdownSignal};
use crate::error::{KernelError, Result};
use crate::registry::KernelRegistry;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{watch, RwLock};
use tracing::{debug, info, warn};

/// Runtime lifecycle states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleState {
    /// Runtime has been created but not started
    Created,
    /// Runtime is starting up
    Starting,
    /// Runtime is fully operational
    Running,
    /// Runtime is draining (accepting no new work, completing existing)
    Draining,
    /// Runtime has stopped
    Stopped,
    /// Runtime encountered a fatal error
    Failed,
}

impl std::fmt::Display for LifecycleState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Created => write!(f, "Created"),
            Self::Starting => write!(f, "Starting"),
            Self::Running => write!(f, "Running"),
            Self::Draining => write!(f, "Draining"),
            Self::Stopped => write!(f, "Stopped"),
            Self::Failed => write!(f, "Failed"),
        }
    }
}

/// Handle to a running runtime for status queries and control
#[derive(Clone)]
pub struct RuntimeHandle {
    state: Arc<RwLock<LifecycleState>>,
    shutdown_tx: watch::Sender<bool>,
    stats: Arc<RuntimeStatsInner>,
    start_time: Instant,
}

impl RuntimeHandle {
    /// Get current lifecycle state
    pub async fn state(&self) -> LifecycleState {
        *self.state.read().await
    }

    /// Check if runtime is running
    pub async fn is_running(&self) -> bool {
        *self.state.read().await == LifecycleState::Running
    }

    /// Request graceful shutdown
    pub fn request_shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }

    /// Get runtime statistics
    pub fn stats(&self) -> RuntimeStats {
        RuntimeStats {
            kernels_registered: self.stats.kernels_registered.load(Ordering::Relaxed) as usize,
            kernels_active: self.stats.kernels_active.load(Ordering::Relaxed) as usize,
            messages_processed: self.stats.messages_processed.load(Ordering::Relaxed),
            messages_in_flight: self.stats.messages_in_flight.load(Ordering::Relaxed),
            gpu_memory_bytes: self.stats.gpu_memory_bytes.load(Ordering::Relaxed),
            gpu_memory_peak_bytes: self.stats.gpu_memory_peak_bytes.load(Ordering::Relaxed),
            uptime_secs: self.start_time.elapsed().as_secs(),
        }
    }
}

#[derive(Debug, Default)]
struct RuntimeStatsInner {
    kernels_registered: AtomicU64,
    kernels_active: AtomicU64,
    messages_processed: AtomicU64,
    messages_in_flight: AtomicU64,
    gpu_memory_bytes: AtomicU64,
    gpu_memory_peak_bytes: AtomicU64,
}

/// The main RustKernels runtime
pub struct KernelRuntime {
    config: RuntimeConfig,
    state: Arc<RwLock<LifecycleState>>,
    registry: Arc<KernelRegistry>,
    shutdown_tx: watch::Sender<bool>,
    shutdown_rx: watch::Receiver<bool>,
    stats: Arc<RuntimeStatsInner>,
    start_time: Option<Instant>,
    event_callbacks: Vec<RuntimeEventCallback>,
}

impl KernelRuntime {
    /// Create a new runtime with the given configuration
    pub fn new(config: RuntimeConfig) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        Self {
            config,
            state: Arc::new(RwLock::new(LifecycleState::Created)),
            registry: Arc::new(KernelRegistry::new()),
            shutdown_tx,
            shutdown_rx,
            stats: Arc::new(RuntimeStatsInner::default()),
            start_time: None,
            event_callbacks: Vec::new(),
        }
    }

    /// Create a runtime builder
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }

    /// Get the kernel registry
    pub fn registry(&self) -> &Arc<KernelRegistry> {
        &self.registry
    }

    /// Get the current configuration
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Get current lifecycle state
    pub async fn state(&self) -> LifecycleState {
        *self.state.read().await
    }

    /// Add an event callback
    pub fn on_event(&mut self, callback: RuntimeEventCallback) {
        self.event_callbacks.push(callback);
    }

    /// Start the runtime
    pub async fn start(&mut self) -> Result<RuntimeHandle> {
        let current_state = *self.state.read().await;
        if current_state != LifecycleState::Created {
            return Err(KernelError::ConfigError(format!(
                "Cannot start runtime in state: {}",
                current_state
            )));
        }

        // Transition to Starting
        *self.state.write().await = LifecycleState::Starting;
        self.emit_event(RuntimeEvent::Starting);
        info!("Runtime starting with config: {:?}", self.config);

        // Initialize backend
        if self.config.gpu_enabled {
            self.initialize_gpu_backend().await?;
        }

        // Mark as running
        *self.state.write().await = LifecycleState::Running;
        let start_time = Instant::now();
        self.start_time = Some(start_time);
        self.emit_event(RuntimeEvent::Started);
        info!("Runtime started successfully");

        // Start background tasks
        self.start_health_check_task();

        Ok(RuntimeHandle {
            state: self.state.clone(),
            shutdown_tx: self.shutdown_tx.clone(),
            stats: self.stats.clone(),
            start_time,
        })
    }

    /// Initiate graceful shutdown
    pub async fn shutdown(&mut self) -> Result<()> {
        let current_state = *self.state.read().await;
        if current_state != LifecycleState::Running {
            return Err(KernelError::ConfigError(format!(
                "Cannot shutdown runtime in state: {}",
                current_state
            )));
        }

        // Transition to Draining
        *self.state.write().await = LifecycleState::Draining;
        self.emit_event(RuntimeEvent::Draining);
        info!(
            "Runtime draining, timeout: {:?}",
            self.config.drain_timeout
        );

        // Signal shutdown to all tasks
        let _ = self.shutdown_tx.send(true);

        // Wait for drain period or completion
        let drain_start = Instant::now();
        while drain_start.elapsed() < self.config.drain_timeout {
            let in_flight = self.stats.messages_in_flight.load(Ordering::Relaxed);
            if in_flight == 0 {
                debug!("All in-flight messages completed");
                break;
            }
            debug!("Waiting for {} in-flight messages", in_flight);
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Force stop remaining work
        let remaining = self.stats.messages_in_flight.load(Ordering::Relaxed);
        if remaining > 0 {
            warn!("Drain timeout reached with {} messages still in flight", remaining);
        }

        // Cleanup GPU resources
        if self.config.gpu_enabled {
            self.cleanup_gpu_backend().await?;
        }

        // Transition to Stopped
        *self.state.write().await = LifecycleState::Stopped;
        self.emit_event(RuntimeEvent::Stopped);
        info!("Runtime stopped");

        Ok(())
    }

    /// Force immediate shutdown (not graceful)
    pub async fn force_shutdown(&mut self) -> Result<()> {
        warn!("Force shutdown initiated");

        let _ = self.shutdown_tx.send(true);

        if self.config.gpu_enabled {
            self.cleanup_gpu_backend().await?;
        }

        *self.state.write().await = LifecycleState::Stopped;
        self.emit_event(RuntimeEvent::Stopped);

        Ok(())
    }

    /// Reload configuration (if hot reload enabled)
    pub async fn reload_config(&mut self, new_config: RuntimeConfig) -> Result<()> {
        if !self.config.hot_reload_enabled {
            return Err(KernelError::ConfigError(
                "Hot reload not enabled".to_string(),
            ));
        }

        new_config
            .validate()
            .map_err(|e| KernelError::ConfigError(e.to_string()))?;

        // Only update safe-to-reload fields
        self.config.log_level = new_config.log_level;
        self.config.metrics_interval = new_config.metrics_interval;
        self.config.health_check_interval = new_config.health_check_interval;
        self.config.max_queue_depth = new_config.max_queue_depth;

        self.emit_event(RuntimeEvent::ConfigReloaded);
        info!("Configuration reloaded");

        Ok(())
    }

    /// Get a shutdown signal receiver
    pub fn shutdown_signal(&self) -> ShutdownSignal {
        self.shutdown_rx.clone()
    }

    /// Get runtime statistics
    pub fn stats(&self) -> RuntimeStats {
        RuntimeStats {
            kernels_registered: self.stats.kernels_registered.load(Ordering::Relaxed) as usize,
            kernels_active: self.stats.kernels_active.load(Ordering::Relaxed) as usize,
            messages_processed: self.stats.messages_processed.load(Ordering::Relaxed),
            messages_in_flight: self.stats.messages_in_flight.load(Ordering::Relaxed),
            gpu_memory_bytes: self.stats.gpu_memory_bytes.load(Ordering::Relaxed),
            gpu_memory_peak_bytes: self.stats.gpu_memory_peak_bytes.load(Ordering::Relaxed),
            uptime_secs: self.start_time.map(|t| t.elapsed().as_secs()).unwrap_or(0),
        }
    }

    /// Record a message being processed
    pub fn record_message_start(&self) {
        self.stats.messages_in_flight.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a message completed
    pub fn record_message_complete(&self) {
        self.stats.messages_in_flight.fetch_sub(1, Ordering::Relaxed);
        self.stats.messages_processed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record kernel registration
    pub fn record_kernel_registered(&self, id: &str) {
        self.stats.kernels_registered.fetch_add(1, Ordering::Relaxed);
        self.emit_event(RuntimeEvent::KernelRegistered { id: id.to_string() });
    }

    /// Record kernel activation
    pub fn record_kernel_activated(&self) {
        self.stats.kernels_active.fetch_add(1, Ordering::Relaxed);
    }

    /// Record kernel deactivation
    pub fn record_kernel_deactivated(&self, id: &str) {
        self.stats.kernels_active.fetch_sub(1, Ordering::Relaxed);
        self.emit_event(RuntimeEvent::KernelUnregistered { id: id.to_string() });
    }

    // Private methods

    async fn initialize_gpu_backend(&self) -> Result<()> {
        info!(
            "Initializing GPU backend: {}",
            self.config.primary_backend
        );
        // GPU backend initialization will be implemented with ringkernel 0.3.1
        // For now, this is a placeholder that logs the intent
        Ok(())
    }

    async fn cleanup_gpu_backend(&self) -> Result<()> {
        info!("Cleaning up GPU backend");
        // GPU cleanup will be implemented with ringkernel 0.3.1
        Ok(())
    }

    fn start_health_check_task(&self) {
        let state = self.state.clone();
        let mut shutdown_rx = self.shutdown_rx.clone();
        let interval = self.config.health_check_interval;
        let callbacks = self.event_callbacks.clone();

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                tokio::select! {
                    _ = interval_timer.tick() => {
                        let current_state = *state.read().await;
                        if current_state != LifecycleState::Running {
                            break;
                        }

                        // Perform health check
                        let healthy = true; // TODO: Real health check

                        for callback in &callbacks {
                            callback(RuntimeEvent::HealthCheckCompleted { healthy });
                        }
                    }
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            break;
                        }
                    }
                }
            }

            debug!("Health check task stopped");
        });
    }

    fn emit_event(&self, event: RuntimeEvent) {
        for callback in &self.event_callbacks {
            callback(event.clone());
        }
    }
}

impl std::fmt::Debug for KernelRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelRuntime")
            .field("config", &self.config)
            .field("start_time", &self.start_time)
            .finish()
    }
}

/// Builder for KernelRuntime
#[derive(Default)]
pub struct RuntimeBuilder {
    config: Option<RuntimeConfig>,
    event_callbacks: Vec<RuntimeEventCallback>,
}

impl RuntimeBuilder {
    /// Use development configuration
    pub fn development(mut self) -> Self {
        self.config = Some(RuntimeConfig::development());
        self
    }

    /// Use production configuration
    pub fn production(mut self) -> Self {
        self.config = Some(RuntimeConfig::production());
        self
    }

    /// Use high-performance configuration
    pub fn high_performance(mut self) -> Self {
        self.config = Some(RuntimeConfig::high_performance());
        self
    }

    /// Use custom configuration
    pub fn with_config(mut self, config: RuntimeConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Load configuration from environment
    pub fn from_env(mut self) -> Self {
        self.config = Some(RuntimeConfig::from_env());
        self
    }

    /// Load configuration from file
    pub fn from_file(mut self, path: &std::path::Path) -> Result<Self> {
        let config =
            RuntimeConfig::from_file(path).map_err(|e| KernelError::ConfigError(e.to_string()))?;
        self.config = Some(config);
        Ok(self)
    }

    /// Set drain timeout
    pub fn with_drain_timeout(mut self, timeout: Duration) -> Self {
        if let Some(ref mut config) = self.config {
            config.drain_timeout = timeout;
        }
        self
    }

    /// Set max kernel instances
    pub fn with_max_instances(mut self, count: usize) -> Self {
        if let Some(ref mut config) = self.config {
            config.max_kernel_instances = count;
        }
        self
    }

    /// Add event callback
    pub fn on_event(mut self, callback: RuntimeEventCallback) -> Self {
        self.event_callbacks.push(callback);
        self
    }

    /// Build the runtime
    pub fn build(self) -> Result<KernelRuntime> {
        let config = self.config.unwrap_or_default();
        config
            .validate()
            .map_err(|e| KernelError::ConfigError(e.to_string()))?;

        let mut runtime = KernelRuntime::new(config);
        runtime.event_callbacks = self.event_callbacks;

        Ok(runtime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_runtime_lifecycle() {
        let mut runtime = KernelRuntime::builder()
            .development()
            .with_drain_timeout(Duration::from_millis(100))
            .build()
            .unwrap();

        assert_eq!(runtime.state().await, LifecycleState::Created);

        let handle = runtime.start().await.unwrap();
        assert_eq!(handle.state().await, LifecycleState::Running);

        runtime.shutdown().await.unwrap();
        assert_eq!(runtime.state().await, LifecycleState::Stopped);
    }

    #[tokio::test]
    async fn test_runtime_stats() {
        let runtime = KernelRuntime::new(RuntimeConfig::testing());
        let stats = runtime.stats();

        assert_eq!(stats.kernels_registered, 0);
        assert_eq!(stats.messages_processed, 0);
    }

    #[tokio::test]
    async fn test_shutdown_signal() {
        let runtime = KernelRuntime::new(RuntimeConfig::testing());
        let mut signal = runtime.shutdown_signal();

        assert!(!*signal.borrow());

        runtime.shutdown_tx.send(true).unwrap();
        signal.changed().await.unwrap();
        assert!(*signal.borrow());
    }
}
