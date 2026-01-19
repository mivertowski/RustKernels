//! Production Configuration Management
//!
//! Provides unified configuration for production deployments including:
//! - Security settings
//! - Observability configuration
//! - Resilience patterns
//! - Runtime parameters
//! - Memory management
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_core::config::ProductionConfig;
//!
//! // Load from environment
//! let config = ProductionConfig::from_env()?;
//!
//! // Or load from file
//! let config = ProductionConfig::from_file("config/production.toml")?;
//!
//! // Apply configuration
//! config.apply().await?;
//! ```

use crate::error::{KernelError, Result};
use crate::memory::MemoryConfig;
use crate::observability::ObservabilityConfig;
use crate::resilience::ResilienceConfig;
use crate::runtime::RuntimeConfig;
use crate::security::SecurityConfig;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Unified production configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionConfig {
    /// Security configuration
    pub security: SecurityConfig,
    /// Observability configuration
    pub observability: ObservabilityConfig,
    /// Resilience configuration
    pub resilience: ResilienceConfig,
    /// Runtime configuration
    pub runtime: RuntimeConfig,
    /// Memory configuration
    pub memory: MemoryConfig,
    /// Environment name
    pub environment: String,
    /// Service name
    pub service_name: String,
    /// Service version
    pub service_version: String,
}

impl Default for ProductionConfig {
    fn default() -> Self {
        Self {
            security: SecurityConfig::default(),
            observability: ObservabilityConfig::default(),
            resilience: ResilienceConfig::default(),
            runtime: RuntimeConfig::default(),
            memory: MemoryConfig::default(),
            environment: "development".to_string(),
            service_name: "rustkernels".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

impl ProductionConfig {
    /// Create development configuration
    pub fn development() -> Self {
        Self {
            security: SecurityConfig::development(),
            observability: ObservabilityConfig::development(),
            resilience: ResilienceConfig::development(),
            runtime: RuntimeConfig::development(),
            memory: MemoryConfig::development(),
            environment: "development".to_string(),
            ..Default::default()
        }
    }

    /// Create production configuration
    pub fn production() -> Self {
        Self {
            security: SecurityConfig::production(),
            observability: ObservabilityConfig::production(),
            resilience: ResilienceConfig::production(),
            runtime: RuntimeConfig::production(),
            memory: MemoryConfig::production(),
            environment: "production".to_string(),
            ..Default::default()
        }
    }

    /// Create high-performance configuration
    pub fn high_performance() -> Self {
        Self {
            security: SecurityConfig::default(),
            observability: ObservabilityConfig::default(),
            resilience: ResilienceConfig::production(), // Use production resilience for high-perf
            runtime: RuntimeConfig::high_performance(),
            memory: MemoryConfig::high_performance(),
            environment: "high-performance".to_string(),
            ..Default::default()
        }
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let mut config = match std::env::var("RUSTKERNEL_ENV")
            .as_deref()
            .unwrap_or("development")
        {
            "production" | "prod" => Self::production(),
            "high-performance" | "hp" => Self::high_performance(),
            _ => Self::development(),
        };

        // Override with specific environment variables
        if let Ok(name) = std::env::var("RUSTKERNEL_SERVICE_NAME") {
            config.service_name = name;
        }

        if let Ok(version) = std::env::var("RUSTKERNEL_SERVICE_VERSION") {
            config.service_version = version;
        }

        // Security overrides
        if std::env::var("RUSTKERNEL_AUTH_ENABLED").is_ok() {
            config.security.rbac_enabled = true;
        }

        if std::env::var("RUSTKERNEL_MULTI_TENANT").is_ok() {
            config.security.multi_tenancy_enabled = true;
        }

        // Runtime overrides
        if let Ok(val) = std::env::var("RUSTKERNEL_GPU_ENABLED") {
            config.runtime.gpu_enabled = val.parse().unwrap_or(true);
        }

        if let Ok(val) = std::env::var("RUSTKERNEL_MAX_INSTANCES") {
            if let Ok(n) = val.parse() {
                config.runtime.max_kernel_instances = n;
            }
        }

        // Memory overrides
        if let Ok(val) = std::env::var("RUSTKERNEL_MAX_GPU_MEMORY_GB") {
            if let Ok(gb) = val.parse::<u64>() {
                config.memory.max_gpu_memory = gb * 1024 * 1024 * 1024;
            }
        }

        Ok(config)
    }

    /// Load configuration from a TOML file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| KernelError::ConfigError(format!("Failed to read config: {}", e)))?;

        toml::from_str(&content)
            .map_err(|e| KernelError::ConfigError(format!("Failed to parse config: {}", e)))
    }

    /// Save configuration to a TOML file
    pub fn to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| KernelError::ConfigError(format!("Failed to serialize config: {}", e)))?;

        std::fs::write(path.as_ref(), content)
            .map_err(|e| KernelError::ConfigError(format!("Failed to write config: {}", e)))?;

        Ok(())
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        self.runtime
            .validate()
            .map_err(|e| KernelError::ConfigError(e.to_string()))?;

        // Additional validation
        if self.environment == "production" {
            // Production should have security enabled
            if !self.security.rbac_enabled && self.security.auth.is_none() {
                tracing::warn!(
                    "Production environment without authentication or RBAC enabled"
                );
            }
        }

        Ok(())
    }

    /// Set environment
    pub fn with_environment(mut self, env: impl Into<String>) -> Self {
        self.environment = env.into();
        self
    }

    /// Set service name
    pub fn with_service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }

    /// Set service version
    pub fn with_service_version(mut self, version: impl Into<String>) -> Self {
        self.service_version = version.into();
        self
    }

    /// Set security configuration
    pub fn with_security(mut self, config: SecurityConfig) -> Self {
        self.security = config;
        self
    }

    /// Set observability configuration
    pub fn with_observability(mut self, config: ObservabilityConfig) -> Self {
        self.observability = config;
        self
    }

    /// Set resilience configuration
    pub fn with_resilience(mut self, config: ResilienceConfig) -> Self {
        self.resilience = config;
        self
    }

    /// Set runtime configuration
    pub fn with_runtime(mut self, config: RuntimeConfig) -> Self {
        self.runtime = config;
        self
    }

    /// Set memory configuration
    pub fn with_memory(mut self, config: MemoryConfig) -> Self {
        self.memory = config;
        self
    }
}

/// Configuration builder
#[derive(Default)]
pub struct ProductionConfigBuilder {
    config: ProductionConfig,
}

impl ProductionConfigBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Start from production preset
    pub fn production() -> Self {
        Self {
            config: ProductionConfig::production(),
        }
    }

    /// Start from development preset
    pub fn development() -> Self {
        Self {
            config: ProductionConfig::development(),
        }
    }

    /// Start from high-performance preset
    pub fn high_performance() -> Self {
        Self {
            config: ProductionConfig::high_performance(),
        }
    }

    /// Set environment
    pub fn environment(mut self, env: impl Into<String>) -> Self {
        self.config.environment = env.into();
        self
    }

    /// Set service name
    pub fn service_name(mut self, name: impl Into<String>) -> Self {
        self.config.service_name = name.into();
        self
    }

    /// Configure security
    pub fn security(mut self, f: impl FnOnce(SecurityConfig) -> SecurityConfig) -> Self {
        self.config.security = f(self.config.security);
        self
    }

    /// Configure observability
    pub fn observability(
        mut self,
        f: impl FnOnce(ObservabilityConfig) -> ObservabilityConfig,
    ) -> Self {
        self.config.observability = f(self.config.observability);
        self
    }

    /// Configure resilience
    pub fn resilience(mut self, f: impl FnOnce(ResilienceConfig) -> ResilienceConfig) -> Self {
        self.config.resilience = f(self.config.resilience);
        self
    }

    /// Configure runtime
    pub fn runtime(mut self, f: impl FnOnce(RuntimeConfig) -> RuntimeConfig) -> Self {
        self.config.runtime = f(self.config.runtime);
        self
    }

    /// Configure memory
    pub fn memory(mut self, f: impl FnOnce(MemoryConfig) -> MemoryConfig) -> Self {
        self.config.memory = f(self.config.memory);
        self
    }

    /// Build and validate the configuration
    pub fn build(self) -> Result<ProductionConfig> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Build without validation
    pub fn build_unchecked(self) -> ProductionConfig {
        self.config
    }
}

/// Health endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthEndpointConfig {
    /// Enable health endpoints
    pub enabled: bool,
    /// Liveness endpoint path
    pub liveness_path: String,
    /// Readiness endpoint path
    pub readiness_path: String,
    /// Startup endpoint path
    pub startup_path: String,
    /// Include detailed health info
    pub detailed: bool,
}

impl Default for HealthEndpointConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            liveness_path: "/health/live".to_string(),
            readiness_path: "/health/ready".to_string(),
            startup_path: "/health/startup".to_string(),
            detailed: false,
        }
    }
}

/// Metrics endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsEndpointConfig {
    /// Enable metrics endpoint
    pub enabled: bool,
    /// Metrics endpoint path
    pub path: String,
    /// Include detailed histogram buckets
    pub detailed_histograms: bool,
}

impl Default for MetricsEndpointConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: "/metrics".to_string(),
            detailed_histograms: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ProductionConfig::default();
        assert_eq!(config.environment, "development");
        assert_eq!(config.service_name, "rustkernels");
    }

    #[test]
    fn test_production_config() {
        let config = ProductionConfig::production();
        assert_eq!(config.environment, "production");
        assert!(config.security.rbac_enabled);
        assert!(config.security.audit_logging);
    }

    #[test]
    fn test_development_config() {
        let config = ProductionConfig::development();
        assert_eq!(config.environment, "development");
    }

    #[test]
    fn test_builder() {
        let config = ProductionConfigBuilder::production()
            .service_name("test-service")
            .environment("staging")
            .build_unchecked();

        assert_eq!(config.service_name, "test-service");
        assert_eq!(config.environment, "staging");
    }

    #[test]
    fn test_config_validation() {
        let config = ProductionConfig::development();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_with_methods() {
        let config = ProductionConfig::default()
            .with_environment("staging")
            .with_service_name("my-service");

        assert_eq!(config.environment, "staging");
        assert_eq!(config.service_name, "my-service");
    }
}
