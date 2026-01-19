//! Secure Secrets Management
//!
//! Provides secure storage and access to sensitive credentials.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Reference to a secret (for configuration without exposing values)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretRef {
    /// Secret name
    pub name: String,
    /// Key within the secret (optional)
    pub key: Option<String>,
    /// Namespace or scope
    pub namespace: Option<String>,
}

impl SecretRef {
    /// Create a new secret reference
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            key: None,
            namespace: None,
        }
    }

    /// Specify a key within the secret
    pub fn key(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }

    /// Specify namespace
    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Get the full path to the secret
    pub fn path(&self) -> String {
        let mut path = String::new();
        if let Some(ref ns) = self.namespace {
            path.push_str(ns);
            path.push('/');
        }
        path.push_str(&self.name);
        if let Some(ref key) = self.key {
            path.push('/');
            path.push_str(key);
        }
        path
    }
}

/// Secret value (zeroized on drop when crypto feature is enabled)
#[derive(Clone)]
pub struct SecretValue {
    value: Vec<u8>,
}

impl SecretValue {
    /// Create a new secret value
    pub fn new(value: impl Into<Vec<u8>>) -> Self {
        Self {
            value: value.into(),
        }
    }

    /// Create from a string
    pub fn from_string(s: impl Into<String>) -> Self {
        Self {
            value: s.into().into_bytes(),
        }
    }

    /// Get the value as bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.value
    }

    /// Get the value as a string (if valid UTF-8)
    pub fn as_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.value).ok()
    }

    /// Get the length
    pub fn len(&self) -> usize {
        self.value.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.value.is_empty()
    }
}

impl std::fmt::Debug for SecretValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SecretValue([REDACTED, {} bytes])", self.value.len())
    }
}

impl Drop for SecretValue {
    fn drop(&mut self) {
        // Zero out the secret value
        for byte in &mut self.value {
            *byte = 0;
        }
    }
}

/// Secret store trait
pub trait SecretStore: Send + Sync {
    /// Get a secret by name
    fn get(&self, secret_ref: &SecretRef) -> Result<SecretValue, super::SecurityError>;

    /// Set a secret
    fn set(&self, secret_ref: &SecretRef, value: SecretValue) -> Result<(), super::SecurityError>;

    /// Delete a secret
    fn delete(&self, secret_ref: &SecretRef) -> Result<(), super::SecurityError>;

    /// List secret names
    fn list(&self, namespace: Option<&str>) -> Result<Vec<String>, super::SecurityError>;
}

/// In-memory secret store (for development/testing)
#[derive(Default)]
pub struct InMemorySecretStore {
    secrets: Arc<RwLock<HashMap<String, SecretValue>>>,
}

impl InMemorySecretStore {
    /// Create a new in-memory secret store
    pub fn new() -> Self {
        Self {
            secrets: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get a secret (async version)
    pub async fn get_async(
        &self,
        secret_ref: &SecretRef,
    ) -> Result<SecretValue, super::SecurityError> {
        let secrets = self.secrets.read().await;
        secrets.get(&secret_ref.path()).cloned().ok_or_else(|| {
            super::SecurityError::SecretNotFound {
                name: secret_ref.path(),
            }
        })
    }

    /// Set a secret (async version)
    pub async fn set_async(
        &self,
        secret_ref: &SecretRef,
        value: SecretValue,
    ) -> Result<(), super::SecurityError> {
        let mut secrets = self.secrets.write().await;
        secrets.insert(secret_ref.path(), value);
        Ok(())
    }

    /// Delete a secret (async version)
    pub async fn delete_async(&self, secret_ref: &SecretRef) -> Result<(), super::SecurityError> {
        let mut secrets = self.secrets.write().await;
        secrets.remove(&secret_ref.path());
        Ok(())
    }

    /// List secrets (async version)
    pub async fn list_async(
        &self,
        namespace: Option<&str>,
    ) -> Result<Vec<String>, super::SecurityError> {
        let secrets = self.secrets.read().await;
        let names: Vec<String> = secrets
            .keys()
            .filter(|k| {
                namespace
                    .map(|ns| k.starts_with(&format!("{}/", ns)))
                    .unwrap_or(true)
            })
            .cloned()
            .collect();
        Ok(names)
    }
}

/// Environment variable secret store
pub struct EnvSecretStore {
    prefix: Option<String>,
}

impl EnvSecretStore {
    /// Create a new env secret store
    pub fn new() -> Self {
        Self { prefix: None }
    }

    /// Set a prefix for environment variables
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Get the env var name for a secret
    fn env_name(&self, secret_ref: &SecretRef) -> String {
        let name = secret_ref.name.to_uppercase().replace(['-', '/'], "_");

        match &self.prefix {
            Some(prefix) => format!("{}_{}", prefix.to_uppercase(), name),
            None => name,
        }
    }

    /// Get a secret
    pub fn get(&self, secret_ref: &SecretRef) -> Result<SecretValue, super::SecurityError> {
        let env_name = self.env_name(secret_ref);
        std::env::var(&env_name)
            .map(SecretValue::from_string)
            .map_err(|_| super::SecurityError::SecretNotFound {
                name: secret_ref.path(),
            })
    }
}

impl Default for EnvSecretStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secret_ref() {
        let secret_ref = SecretRef::new("database-password")
            .namespace("prod")
            .key("password");

        assert_eq!(secret_ref.path(), "prod/database-password/password");
    }

    #[test]
    fn test_secret_value() {
        let secret = SecretValue::from_string("super-secret");
        assert_eq!(secret.as_str(), Some("super-secret"));
        assert_eq!(secret.len(), 12);
    }

    #[test]
    fn test_secret_value_debug() {
        let secret = SecretValue::from_string("super-secret");
        let debug = format!("{:?}", secret);
        assert!(!debug.contains("super-secret"));
        assert!(debug.contains("REDACTED"));
    }

    #[tokio::test]
    async fn test_in_memory_store() {
        let store = InMemorySecretStore::new();
        let secret_ref = SecretRef::new("test-secret");
        let value = SecretValue::from_string("test-value");

        store.set_async(&secret_ref, value).await.unwrap();

        let retrieved = store.get_async(&secret_ref).await.unwrap();
        assert_eq!(retrieved.as_str(), Some("test-value"));

        store.delete_async(&secret_ref).await.unwrap();
        assert!(store.get_async(&secret_ref).await.is_err());
    }

    #[test]
    fn test_env_secret_store_name() {
        let store = EnvSecretStore::new().with_prefix("RUSTKERNEL");
        let secret_ref = SecretRef::new("database-password");

        assert_eq!(store.env_name(&secret_ref), "RUSTKERNEL_DATABASE_PASSWORD");
    }
}
