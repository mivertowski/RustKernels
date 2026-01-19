//! Runtime Configuration
//!
//! Configuration types for the RustKernels runtime, supporting:
//! - Programmatic configuration via builders
//! - Environment variable overrides
//! - File-based configuration (TOML/JSON)

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Enable GPU backends
    pub gpu_enabled: bool,
    /// Primary GPU backend (cuda, wgpu, metal, cpu)
    pub primary_backend: String,
    /// Fallback backend if primary unavailable
    pub fallback_backend: Option<String>,
    /// Maximum concurrent kernel instances
    pub max_kernel_instances: usize,
    /// Maximum message queue depth per kernel
    pub max_queue_depth: usize,
    /// Drain timeout for graceful shutdown
    pub drain_timeout: Duration,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Enable hot configuration reload
    pub hot_reload_enabled: bool,
    /// Enable structured JSON logging
    pub structured_logging: bool,
    /// Log level (trace, debug, info, warn, error)
    pub log_level: String,
    /// Metrics export interval
    pub metrics_interval: Duration,
    /// Backend-specific configuration
    pub backend: BackendConfig,
    /// Worker thread count (0 = auto-detect)
    pub worker_threads: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::development()
    }
}

impl RuntimeConfig {
    /// Development configuration - minimal overhead, verbose logging
    pub fn development() -> Self {
        Self {
            gpu_enabled: false,
            primary_backend: "cpu".to_string(),
            fallback_backend: None,
            max_kernel_instances: 100,
            max_queue_depth: 1000,
            drain_timeout: Duration::from_secs(5),
            health_check_interval: Duration::from_secs(30),
            hot_reload_enabled: true,
            structured_logging: false,
            log_level: "debug".to_string(),
            metrics_interval: Duration::from_secs(60),
            backend: BackendConfig::default(),
            worker_threads: 0,
        }
    }

    /// Production configuration - optimized settings
    pub fn production() -> Self {
        Self {
            gpu_enabled: true,
            primary_backend: "cuda".to_string(),
            fallback_backend: Some("cpu".to_string()),
            max_kernel_instances: 1000,
            max_queue_depth: 10_000,
            drain_timeout: Duration::from_secs(30),
            health_check_interval: Duration::from_secs(10),
            hot_reload_enabled: false,
            structured_logging: true,
            log_level: "info".to_string(),
            metrics_interval: Duration::from_secs(15),
            backend: BackendConfig::production(),
            worker_threads: 0,
        }
    }

    /// High-performance configuration - maximum throughput
    pub fn high_performance() -> Self {
        Self {
            gpu_enabled: true,
            primary_backend: "cuda".to_string(),
            fallback_backend: None,
            max_kernel_instances: 10_000,
            max_queue_depth: 100_000,
            drain_timeout: Duration::from_secs(10),
            health_check_interval: Duration::from_secs(5),
            hot_reload_enabled: false,
            structured_logging: true,
            log_level: "warn".to_string(),
            metrics_interval: Duration::from_secs(5),
            backend: BackendConfig::high_performance(),
            worker_threads: 0,
        }
    }

    /// Testing configuration - deterministic behavior
    pub fn testing() -> Self {
        Self {
            gpu_enabled: false,
            primary_backend: "cpu".to_string(),
            fallback_backend: None,
            max_kernel_instances: 10,
            max_queue_depth: 100,
            drain_timeout: Duration::from_millis(100),
            health_check_interval: Duration::from_secs(1),
            hot_reload_enabled: false,
            structured_logging: false,
            log_level: "trace".to_string(),
            metrics_interval: Duration::from_secs(1),
            backend: BackendConfig::testing(),
            worker_threads: 1,
        }
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(val) = std::env::var("RUSTKERNEL_GPU_ENABLED") {
            config.gpu_enabled = val.parse().unwrap_or(config.gpu_enabled);
        }
        if let Ok(val) = std::env::var("RUSTKERNEL_BACKEND") {
            config.primary_backend = val;
        }
        if let Ok(val) = std::env::var("RUSTKERNEL_FALLBACK_BACKEND") {
            config.fallback_backend = Some(val);
        }
        if let Ok(val) = std::env::var("RUSTKERNEL_MAX_INSTANCES") {
            config.max_kernel_instances = val.parse().unwrap_or(config.max_kernel_instances);
        }
        if let Ok(val) = std::env::var("RUSTKERNEL_QUEUE_DEPTH") {
            config.max_queue_depth = val.parse().unwrap_or(config.max_queue_depth);
        }
        if let Ok(val) = std::env::var("RUSTKERNEL_DRAIN_TIMEOUT_SECS") {
            config.drain_timeout =
                Duration::from_secs(val.parse().unwrap_or(config.drain_timeout.as_secs()));
        }
        if let Ok(val) = std::env::var("RUSTKERNEL_LOG_LEVEL") {
            config.log_level = val;
        }
        if let Ok(val) = std::env::var("RUSTKERNEL_STRUCTURED_LOGGING") {
            config.structured_logging = val.parse().unwrap_or(config.structured_logging);
        }
        if let Ok(val) = std::env::var("RUSTKERNEL_WORKER_THREADS") {
            config.worker_threads = val.parse().unwrap_or(config.worker_threads);
        }

        config
    }

    /// Load configuration from file
    pub fn from_file(path: &std::path::Path) -> Result<Self, ConfigError> {
        let contents = std::fs::read_to_string(path).map_err(ConfigError::IoError)?;

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("toml");

        match ext {
            "json" => serde_json::from_str(&contents).map_err(ConfigError::JsonError),
            "toml" => toml::from_str(&contents).map_err(ConfigError::TomlError),
            _ => Err(ConfigError::UnsupportedFormat(ext.to_string())),
        }
    }

    /// Create a builder for this configuration
    pub fn builder() -> RuntimeConfigBuilder {
        RuntimeConfigBuilder::default()
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.max_kernel_instances == 0 {
            return Err(ConfigError::InvalidValue(
                "max_kernel_instances must be > 0".to_string(),
            ));
        }
        if self.max_queue_depth == 0 {
            return Err(ConfigError::InvalidValue(
                "max_queue_depth must be > 0".to_string(),
            ));
        }
        if self.drain_timeout.is_zero() {
            return Err(ConfigError::InvalidValue(
                "drain_timeout must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Backend-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// CUDA device index (for cuda backend)
    pub cuda_device: i32,
    /// Enable CUDA unified memory
    pub cuda_unified_memory: bool,
    /// GPU memory limit in bytes (0 = unlimited)
    pub memory_limit_bytes: u64,
    /// Enable memory pooling
    pub memory_pooling: bool,
    /// Memory pool initial size in bytes
    pub pool_initial_bytes: u64,
    /// Memory pool growth factor
    pub pool_growth_factor: f32,
    /// Enable async memory operations
    pub async_memory: bool,
    /// Enable kernel fusion optimizations
    pub kernel_fusion: bool,
    /// Maximum batch size for batched execution
    pub max_batch_size: usize,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            cuda_device: 0,
            cuda_unified_memory: false,
            memory_limit_bytes: 0,
            memory_pooling: true,
            pool_initial_bytes: 64 * 1024 * 1024, // 64 MB
            pool_growth_factor: 2.0,
            async_memory: true,
            kernel_fusion: false,
            max_batch_size: 1024,
        }
    }
}

impl BackendConfig {
    /// Production backend configuration
    pub fn production() -> Self {
        Self {
            cuda_device: 0,
            cuda_unified_memory: false,
            memory_limit_bytes: 0,
            memory_pooling: true,
            pool_initial_bytes: 256 * 1024 * 1024, // 256 MB
            pool_growth_factor: 1.5,
            async_memory: true,
            kernel_fusion: true,
            max_batch_size: 4096,
        }
    }

    /// High-performance backend configuration
    pub fn high_performance() -> Self {
        Self {
            cuda_device: 0,
            cuda_unified_memory: false,
            memory_limit_bytes: 0,
            memory_pooling: true,
            pool_initial_bytes: 1024 * 1024 * 1024, // 1 GB
            pool_growth_factor: 1.25,
            async_memory: true,
            kernel_fusion: true,
            max_batch_size: 16384,
        }
    }

    /// Testing backend configuration
    pub fn testing() -> Self {
        Self {
            cuda_device: 0,
            cuda_unified_memory: false,
            memory_limit_bytes: 16 * 1024 * 1024, // 16 MB limit
            memory_pooling: false,
            pool_initial_bytes: 1024 * 1024, // 1 MB
            pool_growth_factor: 2.0,
            async_memory: false,
            kernel_fusion: false,
            max_batch_size: 64,
        }
    }
}

/// Builder for RuntimeConfig
#[derive(Debug, Clone, Default)]
pub struct RuntimeConfigBuilder {
    config: RuntimeConfig,
}

impl RuntimeConfigBuilder {
    /// Create builder with development defaults
    pub fn development() -> Self {
        Self {
            config: RuntimeConfig::development(),
        }
    }

    /// Create builder with production defaults
    pub fn production() -> Self {
        Self {
            config: RuntimeConfig::production(),
        }
    }

    /// Create builder with high-performance defaults
    pub fn high_performance() -> Self {
        Self {
            config: RuntimeConfig::high_performance(),
        }
    }

    /// Enable or disable GPU
    pub fn gpu_enabled(mut self, enabled: bool) -> Self {
        self.config.gpu_enabled = enabled;
        self
    }

    /// Set primary backend
    pub fn primary_backend(mut self, backend: impl Into<String>) -> Self {
        self.config.primary_backend = backend.into();
        self
    }

    /// Set fallback backend
    pub fn fallback_backend(mut self, backend: impl Into<String>) -> Self {
        self.config.fallback_backend = Some(backend.into());
        self
    }

    /// Set max kernel instances
    pub fn max_kernel_instances(mut self, count: usize) -> Self {
        self.config.max_kernel_instances = count;
        self
    }

    /// Set max queue depth
    pub fn max_queue_depth(mut self, depth: usize) -> Self {
        self.config.max_queue_depth = depth;
        self
    }

    /// Set drain timeout
    pub fn drain_timeout(mut self, timeout: Duration) -> Self {
        self.config.drain_timeout = timeout;
        self
    }

    /// Set health check interval
    pub fn health_check_interval(mut self, interval: Duration) -> Self {
        self.config.health_check_interval = interval;
        self
    }

    /// Enable hot reload
    pub fn hot_reload(mut self, enabled: bool) -> Self {
        self.config.hot_reload_enabled = enabled;
        self
    }

    /// Enable structured logging
    pub fn structured_logging(mut self, enabled: bool) -> Self {
        self.config.structured_logging = enabled;
        self
    }

    /// Set log level
    pub fn log_level(mut self, level: impl Into<String>) -> Self {
        self.config.log_level = level.into();
        self
    }

    /// Set metrics interval
    pub fn metrics_interval(mut self, interval: Duration) -> Self {
        self.config.metrics_interval = interval;
        self
    }

    /// Set backend configuration
    pub fn backend_config(mut self, config: BackendConfig) -> Self {
        self.config.backend = config;
        self
    }

    /// Set worker thread count
    pub fn worker_threads(mut self, count: usize) -> Self {
        self.config.worker_threads = count;
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<RuntimeConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Build without validation
    pub fn build_unchecked(self) -> RuntimeConfig {
        self.config
    }
}

/// Configuration error types
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// IO error reading config file
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON parsing error
    #[error("JSON parse error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// TOML parsing error
    #[error("TOML parse error: {0}")]
    TomlError(#[from] toml::de::Error),

    /// Unsupported config format
    #[error("Unsupported config format: {0}")]
    UnsupportedFormat(String),

    /// Invalid configuration value
    #[error("Invalid config value: {0}")]
    InvalidValue(String),

    /// Missing required field
    #[error("Missing required field: {0}")]
    MissingField(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RuntimeConfig::default();
        assert!(!config.gpu_enabled);
        assert_eq!(config.primary_backend, "cpu");
    }

    #[test]
    fn test_production_config() {
        let config = RuntimeConfig::production();
        assert!(config.gpu_enabled);
        assert_eq!(config.primary_backend, "cuda");
        assert!(config.structured_logging);
    }

    #[test]
    fn test_config_validation() {
        let config = RuntimeConfig {
            max_kernel_instances: 0,
            ..RuntimeConfig::default()
        };
        assert!(config.validate().is_err());

        let config = RuntimeConfig {
            max_kernel_instances: 100,
            max_queue_depth: 0,
            ..RuntimeConfig::default()
        };
        assert!(config.validate().is_err());

        let config = RuntimeConfig {
            max_kernel_instances: 100,
            max_queue_depth: 1000,
            ..RuntimeConfig::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let config = RuntimeConfigBuilder::production()
            .gpu_enabled(false)
            .max_kernel_instances(500)
            .build()
            .unwrap();

        assert!(!config.gpu_enabled);
        assert_eq!(config.max_kernel_instances, 500);
    }
}
