//! # RustKernels Ecosystem
//!
//! Web framework integrations for RustKernels, providing REST APIs, gRPC services,
//! and middleware for integrating GPU kernels into production applications.
//!
//! # Features
//!
//! - **Axum Integration**: REST API endpoints for kernel invocation
//! - **Tower Middleware**: Service layer for kernel execution
//! - **gRPC Support**: Protobuf-based kernel RPC
//! - **Actix Actors**: GPU-persistent actors for Actix framework
//!
//! # Feature Flags
//!
//! - `axum`: Enable Axum REST API support
//! - `tower`: Enable Tower middleware
//! - `grpc`: Enable gRPC/Tonic support
//! - `actix`: Enable Actix actor support
//! - `full`: Enable all integrations
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_ecosystem::axum::{KernelRouter, RouterConfig};
//!
//! let router = KernelRouter::new(registry)
//!     .with_config(RouterConfig::default())
//!     .build();
//!
//! axum::serve(listener, router).await?;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

#[cfg(feature = "axum")]
pub mod axum_integration;
#[cfg(feature = "axum")]
pub use axum_integration as axum;

#[cfg(feature = "tower")]
pub mod tower_integration;
#[cfg(feature = "tower")]
pub use tower_integration as tower;

#[cfg(feature = "grpc")]
pub mod grpc_integration;
#[cfg(feature = "grpc")]
pub use grpc_integration as grpc;

#[cfg(feature = "actix")]
pub mod actix_integration;
#[cfg(feature = "actix")]
pub use actix_integration as actix;

// Common types used across integrations
mod common;
pub use common::*;

use serde::{Deserialize, Serialize};

/// Error types for ecosystem integrations
#[derive(Debug, thiserror::Error)]
pub enum EcosystemError {
    /// Kernel not found
    #[error("Kernel not found: {0}")]
    KernelNotFound(String),

    /// Kernel execution failed
    #[error("Kernel execution failed: {0}")]
    ExecutionFailed(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Invalid request
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Authentication required
    #[error("Authentication required")]
    AuthenticationRequired,

    /// Permission denied
    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    /// Service unavailable
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    /// Internal error
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl From<rustkernel_core::error::KernelError> for EcosystemError {
    fn from(err: rustkernel_core::error::KernelError) -> Self {
        match err {
            rustkernel_core::error::KernelError::KernelNotFound(id) => {
                EcosystemError::KernelNotFound(id)
            }
            rustkernel_core::error::KernelError::Unauthorized(msg) => {
                EcosystemError::PermissionDenied(msg)
            }
            rustkernel_core::error::KernelError::ServiceUnavailable(msg) => {
                EcosystemError::ServiceUnavailable(msg)
            }
            _ => EcosystemError::ExecutionFailed(err.to_string()),
        }
    }
}

/// Kernel invocation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelRequest {
    /// Kernel ID to invoke
    pub kernel_id: String,
    /// Input data (JSON)
    pub input: serde_json::Value,
    /// Request metadata
    #[serde(default)]
    pub metadata: RequestMetadata,
}

/// Request metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestMetadata {
    /// Trace ID for distributed tracing
    pub trace_id: Option<String>,
    /// Span ID
    pub span_id: Option<String>,
    /// Tenant ID
    pub tenant_id: Option<String>,
    /// Priority (0-10, higher is more important)
    pub priority: Option<u8>,
    /// Timeout in milliseconds
    pub timeout_ms: Option<u64>,
}

/// Kernel invocation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelResponse {
    /// Request ID
    pub request_id: String,
    /// Kernel ID
    pub kernel_id: String,
    /// Output data (JSON)
    pub output: serde_json::Value,
    /// Execution metadata
    pub metadata: ResponseMetadata,
}

/// Response metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResponseMetadata {
    /// Execution duration in microseconds
    pub duration_us: u64,
    /// Backend used (CUDA, CPU, etc.)
    pub backend: String,
    /// GPU memory used (bytes)
    pub gpu_memory_bytes: Option<u64>,
    /// Trace ID
    pub trace_id: Option<String>,
}

/// Error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Request ID (if available)
    pub request_id: Option<String>,
    /// Additional details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl ErrorResponse {
    /// Create from EcosystemError
    pub fn from_error(err: &EcosystemError, request_id: Option<String>) -> Self {
        let (code, message) = match err {
            EcosystemError::KernelNotFound(id) => {
                ("KERNEL_NOT_FOUND", format!("Kernel not found: {}", id))
            }
            EcosystemError::ExecutionFailed(msg) => ("EXECUTION_FAILED", msg.clone()),
            EcosystemError::SerializationError(msg) => ("SERIALIZATION_ERROR", msg.clone()),
            EcosystemError::InvalidRequest(msg) => ("INVALID_REQUEST", msg.clone()),
            EcosystemError::AuthenticationRequired => (
                "AUTHENTICATION_REQUIRED",
                "Authentication required".to_string(),
            ),
            EcosystemError::PermissionDenied(msg) => ("PERMISSION_DENIED", msg.clone()),
            EcosystemError::RateLimitExceeded => {
                ("RATE_LIMIT_EXCEEDED", "Rate limit exceeded".to_string())
            }
            EcosystemError::ServiceUnavailable(msg) => ("SERVICE_UNAVAILABLE", msg.clone()),
            EcosystemError::InternalError(msg) => ("INTERNAL_ERROR", msg.clone()),
        };

        Self {
            code: code.to_string(),
            message,
            request_id,
            details: None,
        }
    }
}

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Overall status
    pub status: HealthStatus,
    /// Service version
    pub version: String,
    /// Uptime in seconds
    pub uptime_secs: u64,
    /// Component health checks
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub components: Vec<ComponentHealth>,
}

/// Health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    /// Service is healthy
    Healthy,
    /// Service is degraded
    Degraded,
    /// Service is unhealthy
    Unhealthy,
}

/// Component health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component name
    pub name: String,
    /// Component status
    pub status: HealthStatus,
    /// Optional message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_response() {
        let err = EcosystemError::KernelNotFound("test-kernel".to_string());
        let response = ErrorResponse::from_error(&err, Some("req-123".to_string()));

        assert_eq!(response.code, "KERNEL_NOT_FOUND");
        assert!(response.message.contains("test-kernel"));
        assert_eq!(response.request_id, Some("req-123".to_string()));
    }

    #[test]
    fn test_health_response() {
        let response = HealthResponse {
            status: HealthStatus::Healthy,
            version: "0.1.0".to_string(),
            uptime_secs: 3600,
            components: vec![],
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("healthy"));
    }
}
