# Service Deployment

RustKernels 0.4.0 includes the `rustkernel-ecosystem` crate for deploying kernels as production services. All service integrations perform **real kernel execution** — requests are routed through the `KernelRegistry`, dispatched to type-erased `BatchKernelDyn` implementations, and return actual computation results.

## Overview

| Integration | Description | Execution |
|-------------|-------------|-----------|
| **Axum** | REST API endpoints | Real batch kernel dispatch with timeout |
| **Tower** | Middleware services | Real batch kernel dispatch via Tower `Service` |
| **Tonic** | gRPC server | Real batch kernel dispatch with deadline support |
| **Actix** | Actor-based integration | Real batch kernel dispatch via actor messages |

Ring kernels are discoverable through metadata endpoints but require the RingKernel persistent actor runtime for execution. REST/gRPC endpoints return an informative error (HTTP 422 / gRPC `UNIMPLEMENTED`) with guidance to use the Ring protocol.

## Installation

```toml
[dependencies]
rustkernel-ecosystem = { version = "0.4.0", features = ["axum", "grpc"] }
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `axum` | Axum REST API router |
| `tower` | Tower middleware |
| `grpc` | Tonic gRPC server |
| `actix` | Actix actor integration |
| `full` | All integrations |

## How Execution Works

All four integrations follow the same execution path:

1. **Registry lookup** — Find the kernel by ID in the `KernelRegistry`
2. **Mode check** — Verify the kernel is a Batch kernel (Ring kernels return an error)
3. **Factory create** — Instantiate the kernel via its registered factory closure
4. **JSON serialize** — Serialize the request input to JSON bytes
5. **Type-erased dispatch** — Call `execute_dyn(&input_bytes)` on the `BatchKernelDyn` trait object
6. **Deserialize response** — Convert the output bytes back to a JSON response

The `TypeErasedBatchKernel<K, I, O>` wrapper bridges the typed `BatchKernel<I, O>` interface to the type-erased `BatchKernelDyn` trait using serde JSON serialization.

## Axum REST API

### Quick Start

```rust
use rustkernel_ecosystem::axum::{KernelRouter, RouterConfig};
use rustkernel_core::registry::KernelRegistry;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // Create and populate registry
    let registry = Arc::new(KernelRegistry::new());
    rustkernels::register_all(&registry).unwrap();

    // Build router — all endpoints perform real kernel execution
    let router = KernelRouter::new(registry, RouterConfig::default());
    let app = router.into_router();

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/kernels` | List available kernels with metadata |
| GET | `/kernels/{id}` | Get kernel info and capabilities |
| POST | `/execute` | Execute a batch kernel |
| GET | `/health` | Aggregated health check with component status |
| GET | `/metrics` | Prometheus-compatible metrics with per-domain breakdown |

### Execute Request

```bash
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{
    "kernel_id": "graph/betweenness-centrality",
    "input": {
      "num_nodes": 4,
      "edges": [[0, 1], [1, 2], [2, 3], [0, 3]],
      "normalized": true
    }
  }'
```

### Response

```json
{
  "request_id": "req-abc123",
  "kernel_id": "graph/betweenness-centrality",
  "output": {
    "scores": [0.3333, 0.6667, 0.6667, 0.3333]
  },
  "metadata": {
    "duration_us": 850,
    "backend": "CPU"
  }
}
```

### Health Check

The health endpoint aggregates component status:

```json
{
  "status": "healthy",
  "components": {
    "registry": { "status": "healthy", "kernel_count": 106 },
    "execution": { "status": "healthy", "error_rate": 0.0 }
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
    max_request_size: 10 * 1024 * 1024, // 10 MB
};
```

## Tower Middleware

### Kernel Service

```rust
use rustkernel_ecosystem::tower::KernelService;
use tower::ServiceExt;

let service = KernelService::new(registry);

// Execute via Tower Service trait — dispatches to real kernels
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

gRPC execution includes deadline support — if the client sets a gRPC deadline, the server applies it as a timeout around kernel execution. Exceeded deadlines return `DEADLINE_EXCEEDED`.

### Client Usage

```rust
let mut client = KernelClient::connect("http://[::1]:50051").await?;

let request = tonic::Request::new(ExecuteRequest {
    kernel_id: "graph/betweenness-centrality".to_string(),
    input: serde_json::to_string(&input)?,
});

let response = client.execute(request).await?;
```

### Health Service

```rust
use rustkernel_ecosystem::grpc::HealthService;

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

// Execute — dispatches to real batch kernel
let result = addr.send(ExecuteKernel {
    kernel_id: "graph/betweenness-centrality".to_string(),
    input: serde_json::json!({
        "num_nodes": 4,
        "edges": [[0, 1], [1, 2], [2, 3]],
        "normalized": true
    }),
    metadata: Default::default(),
}).await??;
```

### Actor Supervisor

```rust
use rustkernel_ecosystem::actix::KernelActorSupervisor;

let mut supervisor = KernelActorSupervisor::new(registry);

for i in 0..num_workers {
    supervisor.spawn(KernelActorConfig {
        name: format!("worker-{}", i),
        ..Default::default()
    });
}

let workers = supervisor.actors();
```

### Messages

| Message | Description |
|---------|-------------|
| `ExecuteKernel` | Execute a batch kernel computation |
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
        image: rustkernels:0.4.0
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

- [Security](security.md) — Secure your deployment
- [Observability](observability.md) — Monitor service health
- [Runtime](runtime.md) — Configure for production
