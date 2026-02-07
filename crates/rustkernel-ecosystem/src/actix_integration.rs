//! Actix Actor Integration
//!
//! Provides GPU-persistent actors for the Actix framework.
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_ecosystem::actix::{KernelActor, KernelActorConfig};
//! use actix::prelude::*;
//!
//! let actor = KernelActor::new(registry, config);
//! let addr = actor.start();
//!
//! // Send kernel request
//! let result = addr.send(ExecuteKernel {
//!     kernel_id: "graph/pagerank".to_string(),
//!     input: serde_json::json!({ "edges": [...] }),
//! }).await?;
//! ```

use crate::{EcosystemError, RequestMetadata, ResponseMetadata};
use actix::prelude::*;
use rustkernel_core::registry::KernelRegistry;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Actor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelActorConfig {
    /// Actor name
    pub name: String,
    /// Mailbox capacity
    pub mailbox_capacity: usize,
    /// Default timeout for requests
    pub default_timeout: Duration,
    /// Enable message batching
    pub batching_enabled: bool,
    /// Batch size
    pub batch_size: usize,
    /// Batch timeout
    pub batch_timeout: Duration,
}

impl Default for KernelActorConfig {
    fn default() -> Self {
        Self {
            name: "kernel-actor".to_string(),
            mailbox_capacity: 1000,
            default_timeout: Duration::from_secs(30),
            batching_enabled: false,
            batch_size: 100,
            batch_timeout: Duration::from_millis(10),
        }
    }
}

/// GPU-persistent kernel actor
pub struct KernelActor {
    registry: Arc<KernelRegistry>,
    config: KernelActorConfig,
    start_time: Instant,
    messages_processed: u64,
}

impl KernelActor {
    /// Create a new kernel actor
    pub fn new(registry: Arc<KernelRegistry>, config: KernelActorConfig) -> Self {
        Self {
            registry,
            config,
            start_time: Instant::now(),
            messages_processed: 0,
        }
    }

    /// Create with default configuration
    pub fn with_registry(registry: Arc<KernelRegistry>) -> Self {
        Self::new(registry, KernelActorConfig::default())
    }

    /// Get actor statistics
    pub fn stats(&self) -> ActorStats {
        ActorStats {
            uptime_secs: self.start_time.elapsed().as_secs(),
            messages_processed: self.messages_processed,
            kernels_available: self.registry.stats().total,
        }
    }
}

impl Actor for KernelActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        tracing::info!("KernelActor started: {}", self.config.name);
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        tracing::info!(
            "KernelActor stopped: {} (processed {} messages)",
            self.config.name,
            self.messages_processed
        );
    }
}

/// Execute kernel message
#[derive(Debug, Clone, Serialize, Deserialize, Message)]
#[rtype(result = "Result<ExecuteResult, ActorError>")]
pub struct ExecuteKernel {
    /// Kernel ID
    pub kernel_id: String,
    /// Input data
    pub input: serde_json::Value,
    /// Request metadata
    #[serde(default)]
    pub metadata: RequestMetadata,
}

/// Execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteResult {
    /// Request ID
    pub request_id: String,
    /// Kernel ID
    pub kernel_id: String,
    /// Output data
    pub output: serde_json::Value,
    /// Execution metadata
    pub metadata: ResponseMetadata,
}

impl Handler<ExecuteKernel> for KernelActor {
    type Result = Result<ExecuteResult, ActorError>;

    fn handle(&mut self, msg: ExecuteKernel, _ctx: &mut Context<Self>) -> Self::Result {
        let start = Instant::now();
        self.messages_processed += 1;

        let request_id = uuid::Uuid::new_v4().to_string();

        // Try batch kernel execution
        if let Some(entry) = self.registry.get_batch(&msg.kernel_id) {
            let kernel = entry.create();

            let input_bytes = serde_json::to_vec(&msg.input)
                .map_err(|e| ActorError::InvalidInput(format!("Invalid input: {}", e)))?;

            // Execute synchronously by blocking on the async operation.
            // Actix actor handlers are synchronous; bridge to async via block_in_place.
            let output_bytes = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(kernel.execute_dyn(&input_bytes))
            })
            .map_err(|e| ActorError::ExecutionFailed(e.to_string()))?;

            let output: serde_json::Value = serde_json::from_slice(&output_bytes)
                .map_err(|e| {
                    ActorError::ExecutionFailed(format!("Output deserialization: {}", e))
                })?;

            let duration_us = start.elapsed().as_micros() as u64;

            Ok(ExecuteResult {
                request_id,
                kernel_id: msg.kernel_id,
                output,
                metadata: ResponseMetadata {
                    duration_us,
                    backend: entry.metadata.mode.as_str().to_uppercase(),
                    gpu_memory_bytes: None,
                    trace_id: msg.metadata.trace_id,
                },
            })
        } else if self.registry.get(&msg.kernel_id).is_some() {
            Err(ActorError::InvalidInput(format!(
                "Kernel '{}' is a Ring kernel and cannot be executed via actor message. \
                 Use the Ring protocol for persistent kernel dispatch.",
                msg.kernel_id
            )))
        } else {
            Err(ActorError::KernelNotFound(msg.kernel_id))
        }
    }
}

/// Get kernel info message
#[derive(Debug, Clone, Message)]
#[rtype(result = "Result<KernelInfo, ActorError>")]
pub struct GetKernelInfo {
    /// Kernel ID
    pub kernel_id: String,
}

/// Kernel info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelInfo {
    /// Kernel ID
    pub id: String,
    /// Domain
    pub domain: String,
    /// Mode
    pub mode: String,
    /// Description
    pub description: String,
}

impl Handler<GetKernelInfo> for KernelActor {
    type Result = Result<KernelInfo, ActorError>;

    fn handle(&mut self, msg: GetKernelInfo, _ctx: &mut Context<Self>) -> Self::Result {
        let kernel_meta = self
            .registry
            .get(&msg.kernel_id)
            .ok_or_else(|| ActorError::KernelNotFound(msg.kernel_id.clone()))?;

        Ok(KernelInfo {
            id: kernel_meta.id.clone(),
            domain: format!("{:?}", kernel_meta.domain),
            mode: format!("{:?}", kernel_meta.mode),
            description: kernel_meta.description.clone(),
        })
    }
}

/// List kernels message
#[derive(Debug, Clone, Default, Message)]
#[rtype(result = "Result<Vec<KernelInfo>, ActorError>")]
pub struct ListKernels {
    /// Filter by domain
    pub domain: Option<String>,
}

impl Handler<ListKernels> for KernelActor {
    type Result = Result<Vec<KernelInfo>, ActorError>;

    fn handle(&mut self, msg: ListKernels, _ctx: &mut Context<Self>) -> Self::Result {
        let kernels: Vec<KernelInfo> = self
            .registry
            .all_kernel_ids()
            .iter()
            .filter_map(|id| self.registry.get(id))
            .filter(|k| {
                if let Some(ref domain) = msg.domain {
                    format!("{:?}", k.domain).to_lowercase() == domain.to_lowercase()
                } else {
                    true
                }
            })
            .map(|k| KernelInfo {
                id: k.id.clone(),
                domain: format!("{:?}", k.domain),
                mode: format!("{:?}", k.mode),
                description: k.description.clone(),
            })
            .collect();

        Ok(kernels)
    }
}

/// Get stats message
#[derive(Debug, Clone, Message)]
#[rtype(result = "Result<ActorStats, ActorError>")]
pub struct GetStats;

impl Handler<GetStats> for KernelActor {
    type Result = Result<ActorStats, ActorError>;

    fn handle(&mut self, _msg: GetStats, _ctx: &mut Context<Self>) -> Self::Result {
        Ok(self.stats())
    }
}

/// Actor statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorStats {
    /// Uptime in seconds
    pub uptime_secs: u64,
    /// Messages processed
    pub messages_processed: u64,
    /// Available kernels
    pub kernels_available: usize,
}

/// Actor errors
#[derive(Debug, thiserror::Error)]
pub enum ActorError {
    /// Kernel not found
    #[error("Kernel not found: {0}")]
    KernelNotFound(String),

    /// Execution failed
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Timeout
    #[error("Request timed out")]
    Timeout,

    /// Mailbox full
    #[error("Mailbox full")]
    MailboxFull,

    /// Actor stopped
    #[error("Actor stopped")]
    ActorStopped,
}

impl From<EcosystemError> for ActorError {
    fn from(err: EcosystemError) -> Self {
        match err {
            EcosystemError::KernelNotFound(id) => ActorError::KernelNotFound(id),
            EcosystemError::ExecutionFailed(msg) => ActorError::ExecutionFailed(msg),
            EcosystemError::InvalidRequest(msg) => ActorError::InvalidInput(msg),
            _ => ActorError::ExecutionFailed(err.to_string()),
        }
    }
}

/// Supervisor for managing kernel actors
pub struct KernelActorSupervisor {
    registry: Arc<KernelRegistry>,
    actors: Vec<Addr<KernelActor>>,
}

impl KernelActorSupervisor {
    /// Create a new supervisor
    pub fn new(registry: Arc<KernelRegistry>) -> Self {
        Self {
            registry,
            actors: Vec::new(),
        }
    }

    /// Spawn a new actor
    pub fn spawn(&mut self, config: KernelActorConfig) -> Addr<KernelActor> {
        let actor = KernelActor::new(self.registry.clone(), config);
        let addr = actor.start();
        self.actors.push(addr.clone());
        addr
    }

    /// Get all actor addresses
    pub fn actors(&self) -> &[Addr<KernelActor>] {
        &self.actors
    }

    /// Get number of actors
    pub fn actor_count(&self) -> usize {
        self.actors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_config() {
        let config = KernelActorConfig::default();
        assert_eq!(config.mailbox_capacity, 1000);
        assert!(!config.batching_enabled);
    }

    #[actix_rt::test]
    async fn test_kernel_actor() {
        let registry = Arc::new(KernelRegistry::new());
        let actor = KernelActor::with_registry(registry);
        let addr = actor.start();

        // Try to execute nonexistent kernel
        let result = addr
            .send(ExecuteKernel {
                kernel_id: "nonexistent".to_string(),
                input: serde_json::json!({}),
                metadata: RequestMetadata::default(),
            })
            .await
            .unwrap();

        assert!(matches!(result, Err(ActorError::KernelNotFound(_))));
    }

    #[actix_rt::test]
    async fn test_get_stats() {
        let registry = Arc::new(KernelRegistry::new());
        let actor = KernelActor::with_registry(registry);
        let addr = actor.start();

        let stats = addr.send(GetStats).await.unwrap().unwrap();
        assert_eq!(stats.messages_processed, 0);
    }

    #[test]
    fn test_supervisor() {
        let registry = Arc::new(KernelRegistry::new());
        let supervisor = KernelActorSupervisor::new(registry);

        assert_eq!(supervisor.actor_count(), 0);
    }
}
