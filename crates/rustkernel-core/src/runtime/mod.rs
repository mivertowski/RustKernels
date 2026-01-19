//! Runtime Lifecycle Management
//!
//! This module provides runtime lifecycle management for RustKernels, including:
//! - Runtime configuration and initialization
//! - Graceful shutdown with drain periods
//! - Hot configuration reload
//! - Runtime state machine management
//!
//! # Architecture
//!
//! The runtime is built on ringkernel 0.3.1's `RingKernelRuntime` with additional
//! enterprise features for production deployments:
//!
//! - **Lifecycle States**: `Created` → `Starting` → `Running` → `Draining` → `Stopped`
//! - **Configuration**: Environment variables, files, and programmatic configuration
//! - **Graceful Shutdown**: Configurable drain period to complete in-flight requests
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_core::runtime::{KernelRuntime, RuntimeConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create runtime with production configuration
//!     let runtime = KernelRuntime::builder()
//!         .production()
//!         .with_drain_timeout(Duration::from_secs(30))
//!         .build()
//!         .await?;
//!
//!     // Start the runtime
//!     runtime.start().await?;
//!
//!     // ... use runtime ...
//!
//!     // Graceful shutdown
//!     runtime.shutdown().await?;
//!     Ok(())
//! }
//! ```

pub mod config;
pub mod lifecycle;

pub use config::{BackendConfig, RuntimeConfig, RuntimeConfigBuilder};
pub use lifecycle::{KernelRuntime, LifecycleState, RuntimeBuilder, RuntimeHandle};

use std::sync::Arc;
use std::time::Duration;

/// Default drain timeout for graceful shutdown
pub const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

/// Default health check interval
pub const DEFAULT_HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(10);

/// Runtime presets for common deployment scenarios
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimePreset {
    /// Development mode - minimal overhead, verbose logging
    Development,
    /// Production mode - optimized settings, structured logging
    Production,
    /// High-performance mode - maximum throughput, minimal overhead
    HighPerformance,
    /// Testing mode - deterministic behavior, mock backends
    Testing,
}

impl RuntimePreset {
    /// Convert preset to configuration
    pub fn to_config(&self) -> RuntimeConfig {
        match self {
            Self::Development => RuntimeConfig::development(),
            Self::Production => RuntimeConfig::production(),
            Self::HighPerformance => RuntimeConfig::high_performance(),
            Self::Testing => RuntimeConfig::testing(),
        }
    }
}

/// Runtime statistics
#[derive(Debug, Clone, Default)]
pub struct RuntimeStats {
    /// Total kernels registered
    pub kernels_registered: usize,
    /// Active kernel instances
    pub kernels_active: usize,
    /// Total messages processed
    pub messages_processed: u64,
    /// Messages currently in flight
    pub messages_in_flight: u64,
    /// Current GPU memory usage in bytes
    pub gpu_memory_bytes: u64,
    /// Peak GPU memory usage in bytes
    pub gpu_memory_peak_bytes: u64,
    /// Runtime uptime in seconds
    pub uptime_secs: u64,
}

/// Shutdown signal receiver
pub type ShutdownSignal = tokio::sync::watch::Receiver<bool>;

/// Create a shutdown signal channel
pub fn shutdown_channel() -> (tokio::sync::watch::Sender<bool>, ShutdownSignal) {
    tokio::sync::watch::channel(false)
}

/// Runtime event types for lifecycle hooks
#[derive(Debug, Clone)]
pub enum RuntimeEvent {
    /// Runtime is starting
    Starting,
    /// Runtime has started successfully
    Started,
    /// Runtime is beginning to drain (graceful shutdown)
    Draining,
    /// Runtime has stopped
    Stopped,
    /// A kernel was registered
    KernelRegistered {
        /// The ID of the registered kernel
        id: String,
    },
    /// A kernel was unregistered
    KernelUnregistered {
        /// The ID of the unregistered kernel
        id: String,
    },
    /// Configuration was reloaded
    ConfigReloaded,
    /// Health check completed
    HealthCheckCompleted {
        /// Whether the health check passed
        healthy: bool,
    },
}

/// Callback for runtime events
pub type RuntimeEventCallback = Arc<dyn Fn(RuntimeEvent) + Send + Sync>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_preset_to_config() {
        let dev_config = RuntimePreset::Development.to_config();
        assert!(!dev_config.gpu_enabled);

        let prod_config = RuntimePreset::Production.to_config();
        assert!(prod_config.gpu_enabled);
    }

    #[test]
    fn test_shutdown_channel() {
        let (tx, rx) = shutdown_channel();
        assert!(!*rx.borrow());

        tx.send(true).unwrap();
        assert!(*rx.borrow());
    }
}
