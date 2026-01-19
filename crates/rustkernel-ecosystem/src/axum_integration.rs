//! Axum REST API Integration
//!
//! Provides REST endpoints for kernel invocation, health checks, and metrics.
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_ecosystem::axum::{KernelRouter, RouterConfig};
//! use rustkernel_core::registry::KernelRegistry;
//!
//! let registry = KernelRegistry::new();
//! let router = KernelRouter::new(registry)
//!     .with_health_endpoints()
//!     .with_metrics()
//!     .build();
//!
//! let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
//! axum::serve(listener, router).await?;
//! ```

use crate::{
    ErrorResponse, HealthResponse, HealthStatus, KernelResponse, RequestMetadata, ResponseMetadata,
    common::{ServiceConfig, ServiceMetrics, headers, paths},
};
use axum::{
    Router,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::Json,
    routing::{get, post},
};
use rustkernel_core::registry::KernelRegistry;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    /// Kernel registry
    pub registry: Arc<KernelRegistry>,
    /// Service configuration
    pub config: ServiceConfig,
    /// Service metrics
    pub metrics: Arc<ServiceMetrics>,
    /// Start time
    pub start_time: Instant,
}

impl AppState {
    /// Create new app state
    pub fn new(registry: Arc<KernelRegistry>, config: ServiceConfig) -> Self {
        Self {
            registry,
            config,
            metrics: ServiceMetrics::new(),
            start_time: Instant::now(),
        }
    }
}

/// Router configuration
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Enable health endpoints
    pub health_endpoints: bool,
    /// Enable metrics endpoint
    pub metrics_endpoint: bool,
    /// Enable CORS
    pub cors_enabled: bool,
    /// API prefix
    pub api_prefix: String,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            health_endpoints: true,
            metrics_endpoint: true,
            cors_enabled: true,
            api_prefix: "/api/v1".to_string(),
        }
    }
}

/// Kernel router builder
pub struct KernelRouter {
    registry: Arc<KernelRegistry>,
    config: RouterConfig,
    service_config: ServiceConfig,
}

impl KernelRouter {
    /// Create a new router builder
    pub fn new(registry: Arc<KernelRegistry>) -> Self {
        Self {
            registry,
            config: RouterConfig::default(),
            service_config: ServiceConfig::default(),
        }
    }

    /// Set router configuration
    pub fn with_config(mut self, config: RouterConfig) -> Self {
        self.config = config;
        self
    }

    /// Set service configuration
    pub fn with_service_config(mut self, config: ServiceConfig) -> Self {
        self.service_config = config;
        self
    }

    /// Enable health endpoints
    pub fn with_health_endpoints(mut self) -> Self {
        self.config.health_endpoints = true;
        self
    }

    /// Enable metrics endpoint
    pub fn with_metrics(mut self) -> Self {
        self.config.metrics_endpoint = true;
        self
    }

    /// Build the router
    pub fn build(self) -> Router {
        let state = AppState::new(self.registry, self.service_config);

        let mut router = Router::new();

        // API routes
        let api_routes = Router::new()
            .route("/kernels", get(list_kernels))
            .route("/kernels/:kernel_id", get(get_kernel_info))
            .route("/kernels/:kernel_id/execute", post(execute_kernel));

        router = router.nest(&self.config.api_prefix, api_routes);

        // Health endpoints
        if self.config.health_endpoints {
            router = router
                .route(paths::HEALTH, get(health_check))
                .route(paths::LIVENESS, get(liveness_check))
                .route(paths::READINESS, get(readiness_check));
        }

        // Metrics endpoint
        if self.config.metrics_endpoint {
            router = router.route(paths::METRICS, get(metrics_endpoint));
        }

        router.with_state(state)
    }
}

// Handler implementations

/// Health check handler
async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let uptime = state.start_time.elapsed().as_secs();

    Json(HealthResponse {
        status: HealthStatus::Healthy,
        version: state.config.version.clone(),
        uptime_secs: uptime,
        components: vec![],
    })
}

/// Liveness probe handler
async fn liveness_check() -> StatusCode {
    StatusCode::OK
}

/// Readiness probe handler
async fn readiness_check(State(state): State<AppState>) -> StatusCode {
    // Check if registry has kernels
    if state.registry.stats().total > 0 {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

/// Metrics endpoint handler
async fn metrics_endpoint(State(state): State<AppState>) -> String {
    let metrics = &state.metrics;
    let uptime = state.start_time.elapsed().as_secs();

    format!(
        "# HELP rustkernels_requests_total Total number of requests\n\
         # TYPE rustkernels_requests_total counter\n\
         rustkernels_requests_total {}\n\
         # HELP rustkernels_errors_total Total number of errors\n\
         # TYPE rustkernels_errors_total counter\n\
         rustkernels_errors_total {}\n\
         # HELP rustkernels_avg_latency_us Average request latency in microseconds\n\
         # TYPE rustkernels_avg_latency_us gauge\n\
         rustkernels_avg_latency_us {:.2}\n\
         # HELP rustkernels_uptime_seconds Service uptime in seconds\n\
         # TYPE rustkernels_uptime_seconds gauge\n\
         rustkernels_uptime_seconds {}\n",
        metrics.request_count(),
        metrics.error_count(),
        metrics.avg_latency_us(),
        uptime
    )
}

/// List available kernels
async fn list_kernels(State(state): State<AppState>) -> Json<KernelListResponse> {
    let stats = state.registry.stats();

    // Get kernel summaries by iterating all kernel IDs
    let kernels: Vec<KernelSummary> = state
        .registry
        .all_kernel_ids()
        .iter()
        .filter_map(|id| state.registry.get(id))
        .map(|meta| KernelSummary {
            id: meta.id.clone(),
            domain: format!("{:?}", meta.domain),
            mode: format!("{:?}", meta.mode),
            description: meta.description.clone(),
        })
        .collect();

    Json(KernelListResponse {
        total: stats.total,
        kernels,
    })
}

/// Get kernel info
async fn get_kernel_info(
    State(state): State<AppState>,
    Path(kernel_id): Path<String>,
) -> Result<Json<KernelInfoResponse>, (StatusCode, Json<ErrorResponse>)> {
    match state.registry.get(&kernel_id) {
        Some(meta) => Ok(Json(KernelInfoResponse {
            id: meta.id.clone(),
            domain: format!("{:?}", meta.domain),
            mode: format!("{:?}", meta.mode),
            description: meta.description.clone(),
            expected_throughput: meta.expected_throughput,
            target_latency_us: meta.target_latency_us,
        })),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                code: "KERNEL_NOT_FOUND".to_string(),
                message: format!("Kernel not found: {}", kernel_id),
                request_id: None,
                details: None,
            }),
        )),
    }
}

/// Execute a kernel
async fn execute_kernel(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(kernel_id): Path<String>,
    Json(request): Json<ExecuteRequest>,
) -> Result<Json<KernelResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();
    let request_id = extract_request_id(&headers);

    // Check if kernel exists
    let _kernel_meta = state.registry.get(&kernel_id).ok_or_else(|| {
        state
            .metrics
            .record_request(start.elapsed().as_micros() as u64, true);
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                code: "KERNEL_NOT_FOUND".to_string(),
                message: format!("Kernel not found: {}", kernel_id),
                request_id: Some(request_id.clone()),
                details: None,
            }),
        )
    })?;

    // For now, return a mock response
    // Actual kernel execution will be implemented with the runtime
    let duration_us = start.elapsed().as_micros() as u64;
    state.metrics.record_request(duration_us, false);

    Ok(Json(KernelResponse {
        request_id,
        kernel_id: kernel_id.clone(),
        output: serde_json::json!({
            "status": "executed",
            "kernel": kernel_id,
            "input_size": request.input.to_string().len()
        }),
        metadata: ResponseMetadata {
            duration_us,
            backend: "CPU".to_string(),
            gpu_memory_bytes: None,
            trace_id: extract_trace_id(&headers),
        },
    }))
}

// Helper functions

fn extract_request_id(headers: &HeaderMap) -> String {
    headers
        .get(headers::X_REQUEST_ID)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
}

fn extract_trace_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get(headers::TRACEPARENT)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

// Request/Response types

/// Execute request body
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteRequest {
    /// Input data
    pub input: serde_json::Value,
    /// Optional metadata
    #[serde(default)]
    pub metadata: RequestMetadata,
}

/// Kernel list response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelListResponse {
    /// Total kernel count
    pub total: usize,
    /// Kernel summaries
    pub kernels: Vec<KernelSummary>,
}

/// Kernel summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSummary {
    /// Kernel ID
    pub id: String,
    /// Domain
    pub domain: String,
    /// Execution mode
    pub mode: String,
    /// Description
    pub description: String,
}

/// Kernel info response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelInfoResponse {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_config() {
        let config = RouterConfig::default();
        assert!(config.health_endpoints);
        assert!(config.metrics_endpoint);
        assert_eq!(config.api_prefix, "/api/v1");
    }

    #[test]
    fn test_app_state() {
        let registry = Arc::new(KernelRegistry::new());
        let state = AppState::new(registry, ServiceConfig::default());

        assert_eq!(state.metrics.request_count(), 0);
    }
}
