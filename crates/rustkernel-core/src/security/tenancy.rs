//! Multi-Tenancy Support
//!
//! Provides tenant isolation and resource quotas.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tenant identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TenantId(String);

impl TenantId {
    /// Create a new tenant ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the ID as a string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TenantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for TenantId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for TenantId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Tenant configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantConfig {
    /// Tenant ID
    pub id: TenantId,
    /// Tenant name
    pub name: String,
    /// Resource quotas
    pub quotas: ResourceQuota,
    /// Enabled features
    pub features: Vec<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
    /// Whether tenant is active
    pub active: bool,
}

impl TenantConfig {
    /// Create a new tenant config
    pub fn new(id: impl Into<TenantId>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            quotas: ResourceQuota::default(),
            features: Vec::new(),
            metadata: HashMap::new(),
            active: true,
        }
    }

    /// Set resource quotas
    pub fn with_quotas(mut self, quotas: ResourceQuota) -> Self {
        self.quotas = quotas;
        self
    }

    /// Enable a feature
    pub fn with_feature(mut self, feature: impl Into<String>) -> Self {
        self.features.push(feature.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if a feature is enabled
    pub fn has_feature(&self, feature: &str) -> bool {
        self.features.iter().any(|f| f == feature)
    }
}

/// Tenant information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tenant {
    /// Tenant configuration
    pub config: TenantConfig,
    /// Current resource usage
    pub usage: ResourceUsage,
}

impl Tenant {
    /// Create a new tenant
    pub fn new(config: TenantConfig) -> Self {
        Self {
            config,
            usage: ResourceUsage::default(),
        }
    }

    /// Get tenant ID
    pub fn id(&self) -> &TenantId {
        &self.config.id
    }

    /// Check if a resource quota is exceeded
    pub fn is_quota_exceeded(&self, resource: &str) -> bool {
        match resource {
            "kernels" => {
                self.config.quotas.max_kernels > 0
                    && self.usage.active_kernels >= self.config.quotas.max_kernels
            }
            "messages" => {
                self.config.quotas.max_messages_per_second > 0
                    && self.usage.messages_per_second >= self.config.quotas.max_messages_per_second
            }
            "memory" => {
                self.config.quotas.max_memory_bytes > 0
                    && self.usage.memory_bytes >= self.config.quotas.max_memory_bytes
            }
            _ => false,
        }
    }
}

/// Resource quotas for a tenant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuota {
    /// Maximum active kernel instances
    pub max_kernels: u64,
    /// Maximum messages per second
    pub max_messages_per_second: u64,
    /// Maximum GPU memory in bytes
    pub max_memory_bytes: u64,
    /// Maximum CPU cores
    pub max_cpu_cores: u64,
    /// Maximum concurrent requests
    pub max_concurrent_requests: u64,
    /// Maximum storage in bytes
    pub max_storage_bytes: u64,
}

impl Default for ResourceQuota {
    fn default() -> Self {
        Self {
            max_kernels: 100,
            max_messages_per_second: 10_000,
            max_memory_bytes: 1024 * 1024 * 1024, // 1 GB
            max_cpu_cores: 4,
            max_concurrent_requests: 100,
            max_storage_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
        }
    }
}

impl ResourceQuota {
    /// Create unlimited quotas
    pub fn unlimited() -> Self {
        Self {
            max_kernels: 0,
            max_messages_per_second: 0,
            max_memory_bytes: 0,
            max_cpu_cores: 0,
            max_concurrent_requests: 0,
            max_storage_bytes: 0,
        }
    }

    /// Set max kernels
    pub fn with_max_kernels(mut self, max: u64) -> Self {
        self.max_kernels = max;
        self
    }

    /// Set max messages per second
    pub fn with_max_messages(mut self, max: u64) -> Self {
        self.max_messages_per_second = max;
        self
    }

    /// Set max memory
    pub fn with_max_memory(mut self, bytes: u64) -> Self {
        self.max_memory_bytes = bytes;
        self
    }
}

/// Current resource usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Active kernel instances
    pub active_kernels: u64,
    /// Messages per second (current rate)
    pub messages_per_second: u64,
    /// GPU memory usage in bytes
    pub memory_bytes: u64,
    /// CPU cores in use
    pub cpu_cores: f64,
    /// Concurrent requests
    pub concurrent_requests: u64,
    /// Storage used in bytes
    pub storage_bytes: u64,
}

impl ResourceUsage {
    /// Record kernel activation
    pub fn record_kernel_activated(&mut self) {
        self.active_kernels += 1;
    }

    /// Record kernel deactivation
    pub fn record_kernel_deactivated(&mut self) {
        self.active_kernels = self.active_kernels.saturating_sub(1);
    }

    /// Update memory usage
    pub fn set_memory_usage(&mut self, bytes: u64) {
        self.memory_bytes = bytes;
    }

    /// Update message rate
    pub fn set_message_rate(&mut self, rate: u64) {
        self.messages_per_second = rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tenant_id() {
        let id = TenantId::new("tenant-123");
        assert_eq!(id.as_str(), "tenant-123");
        assert_eq!(format!("{}", id), "tenant-123");
    }

    #[test]
    fn test_tenant_config() {
        let config = TenantConfig::new("tenant-123", "Test Tenant")
            .with_feature("gpu-kernels")
            .with_metadata("plan", "enterprise");

        assert_eq!(config.id.as_str(), "tenant-123");
        assert!(config.has_feature("gpu-kernels"));
        assert!(!config.has_feature("unknown-feature"));
    }

    #[test]
    fn test_resource_quotas() {
        let quotas = ResourceQuota::default()
            .with_max_kernels(50)
            .with_max_memory(2 * 1024 * 1024 * 1024);

        assert_eq!(quotas.max_kernels, 50);
        assert_eq!(quotas.max_memory_bytes, 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_quota_exceeded() {
        let mut tenant = Tenant::new(TenantConfig::new("test", "Test"));
        tenant.config.quotas.max_kernels = 10;
        tenant.usage.active_kernels = 10;

        assert!(tenant.is_quota_exceeded("kernels"));
        assert!(!tenant.is_quota_exceeded("memory"));
    }
}
