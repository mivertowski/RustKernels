# Service Deployment

RustKernels 0.2.0 includes the `rustkernel-ecosystem` crate for deploying kernels as production services.

## Overview

| Integration | Description |
|-------------|-------------|
| **Axum** | REST API endpoints |
| **Tower** | Middleware services |
| **Tonic** | gRPC server |
| **Actix** | Actor-based integration |

## Installation

```toml
[dependencies]
rustkernel-ecosystem = { version = "0.2.0", features = ["axum", "grpc"] }
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `axum` | Axum REST API router |
| `tower` | Tower middleware |
| `grpc` | Tonic gRPC server |
| `actix` | Actix actor integration |
| `full` | All integrations |

## Axum REST API

### Quick Start

```rust
use rustkernel_ecosystem::axum::{KernelRouter, RouterConfig};
use rustkernel_core::registry::KernelRegistry;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // Create registry with kernels
    let registry = Arc::new(KernelRegistry::new());

    // Build router
    let router = KernelRouter::new(registry, RouterConfig::default());
    let app = router.into_router();

    // Serve
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/kernels` | List available kernels |
| GET | `/kernels/{id}` | Get kernel info |
| POST | `/execute` | Execute a kernel |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

### Execute Request

```bash
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{
    "kernel_id": "graph/pagerank",
    "input": {
      "num_nodes": 1000,
      "edges": [[0,1], [1,2], [2,0]],
      "damping_factor": 0.85
    }
  }'
```

### Response

```json
{
  "request_id": "req-123",
  "kernel_id": "graph/pagerank",
  "output": {
    "scores": [0.33, 0.33, 0.33],
    "iterations": 10
  },
  "metadata": {
    "duration_us": 1500,
    "backend": "CUDA",
    "trace_id": "abc123"
  }
}
```

### Custom Configuration

```rust
let config = RouterConfig {
    prefix: "/api/v1".to_string(),
    enable_metrics: true,
    enable_health: true,
    cors_enabled: true,
    max_request_size: 10 * 1024 * 1024, // 10MB
};
```

## Tower Middleware

### Kernel Service

```rust
use rustkernel_ecosystem::tower::KernelService;
use tower::ServiceExt;

let service = KernelService::new(registry);

// Use as Tower service
let response = service
    .ready()
    .await?
    .call(request)
    .await?;
```

### Timeout Layer

```rust
use rustkernel_ecosystem::tower::TimeoutLayer;

let layer = TimeoutLayer::new(Duration::from_secs(30));

let service = ServiceBuilder::new()
    .layer(layer)
    .service(kernel_service);
```

### Rate Limiting

```rust
use rustkernel_ecosystem::tower::RateLimiterLayer;

let layer = RateLimiterLayer::new(
    100,  // requests per second
    Duration::from_secs(1),
);
```

### Middleware Stack

```rust
use tower::ServiceBuilder;

let service = ServiceBuilder::new()
    .layer(TimeoutLayer::new(Duration::from_secs(30)))
    .layer(RateLimiterLayer::new(100, Duration::from_secs(1)))
    .layer(TracingLayer::new())
    .service(KernelService::new(registry));
```

## gRPC Server

### Server Setup

```rust
use rustkernel_ecosystem::grpc::KernelGrpcServer;
use tonic::transport::Server;

let server = KernelGrpcServer::new(registry);

Server::builder()
    .add_service(server.into_service())
    .serve("[::1]:50051".parse().unwrap())
    .await?;
```

### Client Usage

```rust
// Generated from proto
let mut client = KernelClient::connect("http://[::1]:50051").await?;

let request = tonic::Request::new(ExecuteRequest {
    kernel_id: "graph/pagerank".to_string(),
    input: serde_json::to_string(&input)?,
});

let response = client.execute(request).await?;
```

### Health Service

```rust
use rustkernel_ecosystem::grpc::HealthService;

// gRPC health checking protocol
Server::builder()
    .add_service(HealthService::new())
    .add_service(kernel_server.into_service())
    .serve(addr)
    .await?;
```

## Actix Actors

### Kernel Actor

```rust
use rustkernel_ecosystem::actix::{KernelActor, KernelActorConfig, ExecuteKernel};
use actix::prelude::*;

let config = KernelActorConfig {
    name: "kernel-worker".to_string(),
    mailbox_capacity: 1000,
    default_timeout: Duration::from_secs(30),
    ..Default::default()
};

let actor = KernelActor::new(registry, config);
let addr = actor.start();

// Send execution request
let result = addr.send(ExecuteKernel {
    kernel_id: "graph/pagerank".to_string(),
    input: serde_json::json!({ ... }),
    metadata: Default::default(),
}).await??;
```

### Actor Supervisor

```rust
use rustkernel_ecosystem::actix::KernelActorSupervisor;

let mut supervisor = KernelActorSupervisor::new(registry);

// Spawn worker pool
for i in 0..num_workers {
    supervisor.spawn(KernelActorConfig {
        name: format!("worker-{}", i),
        ..Default::default()
    });
}

// Get addresses for load balancing
let workers = supervisor.actors();
```

### Messages

| Message | Description |
|---------|-------------|
| `ExecuteKernel` | Execute a kernel computation |
| `GetKernelInfo` | Get kernel metadata |
| `ListKernels` | List available kernels |
| `GetStats` | Get actor statistics |

## Docker Deployment

### Dockerfile

```dockerfile
FROM rust:1.85 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --package rustkernel-ecosystem --features full

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/rustkernel-server /usr/local/bin/
EXPOSE 8080 50051
CMD ["rustkernel-server"]
```

### docker-compose.yml

```yaml
version: '3.8'
services:
  rustkernels:
    build: .
    ports:
      - "8080:8080"   # REST API
      - "50051:50051" # gRPC
    environment:
      - RUSTKERNEL_ENV=production
      - RUSTKERNEL_GPU_ENABLED=true
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rustkernels
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: rustkernels
        image: rustkernels:0.2.0
        ports:
        - containerPort: 8080
        - containerPort: 50051
        env:
        - name: RUSTKERNEL_ENV
          value: "production"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
        resources:
          limits:
            nvidia.com/gpu: 1
```

## Next Steps

- [Security](security.md) - Secure your deployment
- [Observability](observability.md) - Monitor service health
- [Runtime](runtime.md) - Configure for production
