//! gRPC Integration
//!
//! Provides gRPC services for kernel execution using Tonic.
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_ecosystem::grpc::{KernelGrpcServer, GrpcConfig};
//! use tonic::transport::Server;
//!
//! let server = KernelGrpcServer::new(registry);
//!
//! Server::builder()
//!     .add_service(server.into_service())
//!     .serve("[::1]:50051".parse()?)
//!     .await?;
//! ```

use crate::EcosystemError;
use rustkernel_core::registry::KernelRegistry;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

/// gRPC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcConfig {
    /// Listen address
    pub address: String,
    /// Enable reflection
    pub reflection: bool,
    /// Enable health service
    pub health_service: bool,
    /// Max message size (bytes)
    pub max_message_size: usize,
    /// Connection timeout
    pub connect_timeout_ms: u64,
    /// Request timeout
    pub request_timeout_ms: u64,
}

impl Default for GrpcConfig {
    fn default() -> Self {
        Self {
            address: "[::1]:50051".to_string(),
            reflection: true,
            health_service: true,
            max_message_size: 4 * 1024 * 1024, // 4MB
            connect_timeout_ms: 5000,
            request_timeout_ms: 30000,
        }
    }
}

/// Kernel execution request (gRPC)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcKernelRequest {
    /// Kernel ID
    pub kernel_id: String,
    /// Input data (JSON-encoded)
    pub input_json: String,
    /// Trace ID
    pub trace_id: Option<String>,
    /// Tenant ID
    pub tenant_id: Option<String>,
    /// Priority (0-10)
    pub priority: Option<i32>,
    /// Timeout in milliseconds
    pub timeout_ms: Option<u64>,
}

/// Kernel execution response (gRPC)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcKernelResponse {
    /// Request ID
    pub request_id: String,
    /// Kernel ID
    pub kernel_id: String,
    /// Output data (JSON-encoded)
    pub output_json: String,
    /// Duration in microseconds
    pub duration_us: u64,
    /// Backend used
    pub backend: String,
    /// GPU memory used (bytes)
    pub gpu_memory_bytes: Option<u64>,
    /// Trace ID
    pub trace_id: Option<String>,
}

/// Kernel info request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetKernelRequest {
    /// Kernel ID
    pub kernel_id: String,
}

/// Kernel info response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelInfo {
    /// Kernel ID
    pub id: String,
    /// Domain
    pub domain: String,
    /// Execution mode
    pub mode: String,
    /// Description
    pub description: String,
    /// Expected throughput (ops/sec)
    pub expected_throughput: u64,
    /// Target latency in microseconds
    pub target_latency_us: f64,
}

/// List kernels request
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ListKernelsRequest {
    /// Filter by domain
    pub domain: Option<String>,
    /// Filter by mode
    pub mode: Option<String>,
    /// Page size
    pub page_size: Option<i32>,
    /// Page token
    pub page_token: Option<String>,
}

/// List kernels response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListKernelsResponse {
    /// Kernels
    pub kernels: Vec<KernelInfo>,
    /// Next page token
    pub next_page_token: Option<String>,
    /// Total count
    pub total_count: i32,
}

/// gRPC error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcError {
    /// Error code
    pub code: i32,
    /// Error message
    pub message: String,
    /// Details
    pub details: Option<String>,
}

impl From<EcosystemError> for GrpcError {
    fn from(err: EcosystemError) -> Self {
        let (code, message) = match &err {
            EcosystemError::KernelNotFound(_) => (5, err.to_string()), // NOT_FOUND
            EcosystemError::InvalidRequest(_) => (3, err.to_string()), // INVALID_ARGUMENT
            EcosystemError::AuthenticationRequired => (16, err.to_string()), // UNAUTHENTICATED
            EcosystemError::PermissionDenied(_) => (7, err.to_string()), // PERMISSION_DENIED
            EcosystemError::RateLimitExceeded => (8, err.to_string()), // RESOURCE_EXHAUSTED
            EcosystemError::ServiceUnavailable(_) => (14, err.to_string()), // UNAVAILABLE
            _ => (13, err.to_string()),                                // INTERNAL
        };

        Self {
            code,
            message,
            details: None,
        }
    }
}

/// Kernel gRPC server implementation
pub struct KernelGrpcServer {
    registry: Arc<KernelRegistry>,
    config: GrpcConfig,
}

impl KernelGrpcServer {
    /// Create a new gRPC server
    pub fn new(registry: Arc<KernelRegistry>) -> Self {
        Self {
            registry,
            config: GrpcConfig::default(),
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: GrpcConfig) -> Self {
        self.config = config;
        self
    }

    /// Execute a kernel.
    ///
    /// Looks up the batch kernel in the registry, creates an instance, and executes it
    /// with the provided JSON input. Ring kernels cannot be executed through this unary RPC.
    pub async fn execute_kernel(
        &self,
        request: GrpcKernelRequest,
    ) -> Result<GrpcKernelResponse, GrpcError> {
        let start = Instant::now();
        let request_id = request
            .trace_id
            .as_deref()
            .map(|s| s.to_string())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        // Try batch kernel execution
        if let Some(entry) = self.registry.get_batch(&request.kernel_id) {
            let kernel = entry.create();

            let input_bytes = request.input_json.as_bytes();

            // Validate input is valid JSON before passing to kernel
            if serde_json::from_slice::<serde_json::Value>(input_bytes).is_err() {
                return Err(GrpcError::from(EcosystemError::InvalidRequest(
                    "Input must be valid JSON".to_string(),
                )));
            }

            // Apply timeout if specified
            let timeout_ms = request.timeout_ms.unwrap_or(self.config.request_timeout_ms);
            let timeout = std::time::Duration::from_millis(timeout_ms);

            let result = tokio::time::timeout(timeout, kernel.execute_dyn(input_bytes)).await;

            match result {
                Ok(Ok(output_bytes)) => {
                    let duration_us = start.elapsed().as_micros() as u64;
                    let output_json =
                        String::from_utf8(output_bytes).unwrap_or_else(|_| "{}".to_string());
                    Ok(GrpcKernelResponse {
                        request_id,
                        kernel_id: request.kernel_id,
                        output_json,
                        duration_us,
                        backend: entry.metadata.mode.as_str().to_uppercase(),
                        gpu_memory_bytes: None,
                        trace_id: request.trace_id,
                    })
                }
                Ok(Err(e)) => Err(GrpcError::from(EcosystemError::ExecutionFailed(
                    e.to_string(),
                ))),
                Err(_) => Err(GrpcError {
                    code: 4, // DEADLINE_EXCEEDED
                    message: format!("Kernel execution timed out after {}ms", timeout_ms),
                    details: None,
                }),
            }
        } else if self.registry.get(&request.kernel_id).is_some() {
            Err(GrpcError::from(EcosystemError::InvalidRequest(format!(
                "Kernel '{}' is a Ring kernel. Use bidirectional streaming RPC for Ring kernel dispatch.",
                request.kernel_id
            ))))
        } else {
            Err(GrpcError::from(EcosystemError::KernelNotFound(
                request.kernel_id,
            )))
        }
    }

    /// Get kernel info
    pub async fn get_kernel(&self, request: GetKernelRequest) -> Result<KernelInfo, GrpcError> {
        let kernel_meta = self.registry.get(&request.kernel_id).ok_or_else(|| {
            GrpcError::from(EcosystemError::KernelNotFound(request.kernel_id.clone()))
        })?;

        Ok(KernelInfo {
            id: kernel_meta.id.clone(),
            domain: format!("{:?}", kernel_meta.domain),
            mode: format!("{:?}", kernel_meta.mode),
            description: kernel_meta.description.clone(),
            expected_throughput: kernel_meta.expected_throughput,
            target_latency_us: kernel_meta.target_latency_us,
        })
    }

    /// List kernels with pagination support.
    ///
    /// The `page_token` is the kernel ID to start after (exclusive).
    /// Results are sorted by kernel ID for deterministic pagination.
    pub async fn list_kernels(
        &self,
        request: ListKernelsRequest,
    ) -> Result<ListKernelsResponse, GrpcError> {
        let page_size = request.page_size.unwrap_or(100).max(1) as usize;

        // Get all metadata sorted by ID for deterministic pagination
        let all_metadata = self.registry.all_metadata();

        // Apply domain and mode filters
        let filtered: Vec<_> = all_metadata
            .iter()
            .filter(|k| {
                if let Some(ref domain) = request.domain {
                    format!("{:?}", k.domain).to_lowercase() == domain.to_lowercase()
                } else {
                    true
                }
            })
            .filter(|k| {
                if let Some(ref mode) = request.mode {
                    format!("{:?}", k.mode).to_lowercase() == mode.to_lowercase()
                } else {
                    true
                }
            })
            .collect();

        let total_count = filtered.len() as i32;

        // Apply page_token: skip past the token ID
        let start_idx = if let Some(ref token) = request.page_token {
            filtered
                .iter()
                .position(|k| k.id == *token)
                .map(|pos| pos + 1)
                .unwrap_or(0)
        } else {
            0
        };

        let page: Vec<KernelInfo> = filtered
            .iter()
            .skip(start_idx)
            .take(page_size)
            .map(|k| KernelInfo {
                id: k.id.clone(),
                domain: format!("{:?}", k.domain),
                mode: format!("{:?}", k.mode),
                description: k.description.clone(),
                expected_throughput: k.expected_throughput,
                target_latency_us: k.target_latency_us,
            })
            .collect();

        // Set next_page_token if there are more results
        let next_page_token = if start_idx + page_size < filtered.len() {
            page.last().map(|k| k.id.clone())
        } else {
            None
        };

        Ok(ListKernelsResponse {
            total_count,
            kernels: page,
            next_page_token,
        })
    }

    /// Get server configuration
    pub fn config(&self) -> &GrpcConfig {
        &self.config
    }
}

impl Clone for KernelGrpcServer {
    fn clone(&self) -> Self {
        Self {
            registry: self.registry.clone(),
            config: self.config.clone(),
        }
    }
}

/// Health check service
pub struct HealthService {
    registry: Arc<KernelRegistry>,
}

impl HealthService {
    /// Create a new health service
    pub fn new(registry: Arc<KernelRegistry>) -> Self {
        Self { registry }
    }

    /// Check service health
    pub fn check(&self) -> HealthStatus {
        if self.registry.stats().total > 0 {
            HealthStatus::Serving
        } else {
            HealthStatus::NotServing
        }
    }
}

/// Health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Unknown status
    Unknown,
    /// Service is serving
    Serving,
    /// Service is not serving
    NotServing,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grpc_config() {
        let config = GrpcConfig::default();
        assert_eq!(config.address, "[::1]:50051");
        assert!(config.reflection);
    }

    #[tokio::test]
    async fn test_kernel_grpc_server() {
        let registry = Arc::new(KernelRegistry::new());
        let server = KernelGrpcServer::new(registry);

        let request = GrpcKernelRequest {
            kernel_id: "nonexistent".to_string(),
            input_json: "{}".to_string(),
            trace_id: None,
            tenant_id: None,
            priority: None,
            timeout_ms: None,
        };

        let result = server.execute_kernel(request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, 5); // NOT_FOUND
    }

    #[tokio::test]
    async fn test_list_kernels() {
        let registry = Arc::new(KernelRegistry::new());
        let server = KernelGrpcServer::new(registry);

        let request = ListKernelsRequest::default();
        let response = server.list_kernels(request).await.unwrap();

        assert_eq!(response.total_count, 0);
    }

    #[test]
    fn test_health_service() {
        let registry = Arc::new(KernelRegistry::new());
        let health = HealthService::new(registry);

        // Empty registry should be NotServing
        assert_eq!(health.check(), HealthStatus::NotServing);
    }
}
