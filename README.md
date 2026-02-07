# RustKernels

**GPU-accelerated kernel library for financial services, compliance, and enterprise analytics.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)
[![Documentation](https://img.shields.io/badge/docs-online-green.svg)](https://mivertowski.github.io/RustKernels/)
[![Version](https://img.shields.io/badge/version-0.4.0-blue.svg)](CHANGELOG.md)
[![RingKernel](https://img.shields.io/badge/ringkernel-0.4.2-purple.svg)](https://crates.io/crates/ringkernel-core)

---

## Overview

RustKernels delivers **106 production-ready GPU kernels** across **14 domain-specific crates**, purpose-built for financial institutions, compliance operations, and enterprise analytics platforms. It is the Rust port of the DotCompute GPU kernel library, built on the [RingKernel 0.4.2](https://crates.io/crates/ringkernel-core) persistent actor runtime.

Version 0.4.0 provides full end-to-end kernel execution through REST, gRPC, Tower middleware, Actix actor interfaces, and native Python bindings — no stubs, no mocks.

### Key Capabilities

| Requirement | Solution |
|---|---|
| Diverse latency profiles | **Batch mode** (10–50 μs launch) and **Ring mode** (100–500 ns message latency) |
| Multi-kernel orchestration | Built-in K2K coordination: scatter-gather, fan-out, pipeline |
| Production deployment | REST (Axum), gRPC (Tonic), Tower middleware, Actix actors, Python bindings |
| Enterprise security | JWT/API key auth, RBAC, multi-tenancy, secrets management |
| Observability | Prometheus metrics, OTLP tracing, structured logging, SLO alerting |
| Fault tolerance | Circuit breakers, exponential retry, timeout propagation, health probes |
| GPU availability | Automatic CPU fallback when CUDA is unavailable |
| Regulatory explainability | SHAP values, feature importance, audit trail kernels |

---

## Performance

| Metric | Batch Mode | Ring Mode |
|---|---|---|
| Launch overhead | 10–50 μs | N/A (persistent) |
| Message latency | N/A | 100–500 ns |
| State location | CPU memory, transferred per invocation | GPU-resident, zero-copy ring buffers |
| Throughput (PageRank) | ~100K nodes/s | ~500K updates/s |

**Batch mode** is suited for scheduled, compute-heavy workloads: end-of-day risk aggregation, batch AML screening, model training, compliance reporting.

**Ring mode** targets high-frequency, latency-sensitive operations: order book matching, real-time fraud scoring, streaming anomaly detection, live transaction monitoring.

---

## Domain Coverage

| Domain | Crate | Kernels | Key Algorithms |
|---|---|---|---|
| Graph Analytics | `rustkernel-graph` | 28 | PageRank, Louvain, GNN inference, graph attention, cycle detection |
| Statistical ML | `rustkernel-ml` | 17 | K-Means, DBSCAN, isolation forest, federated learning, SHAP |
| Compliance | `rustkernel-compliance` | 11 | Circular flow detection, rapid movement, sanctions screening |
| Temporal Analysis | `rustkernel-temporal` | 7 | ARIMA, Prophet decomposition, change point detection |
| Risk Analytics | `rustkernel-risk` | 5 | Monte Carlo VaR, credit scoring, stress testing, correlation |
| Process Intelligence | `rustkernel-procint` | 7 | DFG construction, conformance checking, digital twin |
| Behavioral Analytics | `rustkernel-behavioral` | 6 | Profiling, forensic queries, causal graph analysis |
| Treasury | `rustkernel-treasury` | 5 | Liquidity optimization, FX hedging, NSFR calculation |
| Clearing | `rustkernel-clearing` | 5 | Multilateral netting, DVP matching, settlement |
| Accounting | `rustkernel-accounting` | 9 | Network generation, GL reconciliation, GAAP detection |
| Banking | `rustkernel-banking` | 1 | Fraud pattern matching (Aho-Corasick) |
| Order Matching | `rustkernel-orderbook` | 1 | Price-time priority order book engine |
| Payments | `rustkernel-payments` | 2 | Payment processing, flow analysis |
| Audit | `rustkernel-audit` | 2 | Feature extraction, hypergraph construction |

---

## Installation

### Rust

```toml
[dependencies]
rustkernels = "0.4.0"
```

### Python

```bash
pip install rustkernels
```

### Feature Flags

```toml
# Default features (graph, ml, compliance, temporal, risk)
rustkernels = "0.4.0"

# Selective domain inclusion
rustkernels = { version = "0.4.0", features = ["graph", "compliance", "procint"] }

# All domains
rustkernels = { version = "0.4.0", features = ["full"] }

# Enterprise ecosystem (REST/gRPC service)
rustkernel-ecosystem = { version = "0.4.0", features = ["axum", "grpc"] }
```

### Requirements

| Dependency | Version | Notes |
|---|---|---|
| Rust | 1.85+ | Edition 2024 |
| RingKernel | 0.4.2 | GPU-native persistent actor runtime (crates.io) |
| CUDA Toolkit | 12.0+ | Optional; CPU fallback when unavailable |

---

## Quick Start

```rust
use rustkernel::prelude::*;
use rustkernel::graph::centrality::BetweennessCentrality;
use rustkernel::graph::messages::CentralityInput;

#[tokio::main]
async fn main() -> Result<()> {
    // Create a registry and register kernels
    let registry = KernelRegistry::new();
    rustkernel::graph::register_all(&registry)?;

    // Instantiate a kernel
    let kernel = BetweennessCentrality::new();
    println!("{} ({})", kernel.metadata().id, kernel.metadata().domain);

    // Execute via the typed BatchKernel interface
    let input = CentralityInput {
        num_nodes: 4,
        edges: vec![(0, 1), (1, 2), (2, 3), (0, 3)],
    };
    let result = kernel.execute(input).await?;
    println!("Centrality scores: {:?}", result.scores);

    Ok(())
}
```

### Type-Erased Execution via REST

Kernels registered with `register_batch_typed()` are automatically available for execution through the ecosystem service layer:

```rust
use rustkernel_ecosystem::axum::{KernelRouter, RouterConfig};
use rustkernel_core::registry::KernelRegistry;
use std::sync::Arc;

let registry = Arc::new(KernelRegistry::new());
rustkernel::graph::register_all(&registry)?;
rustkernel::ml::register_all(&registry)?;

let router = KernelRouter::new(registry, RouterConfig::default());
let app = router.into_router();

// POST /api/v1/kernels/:kernel_id/execute
// GET  /api/v1/kernels
// GET  /health
// GET  /metrics
let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
axum::serve(listener, app).await?;
```

### Python Bindings

Native Python access via `pip install rustkernels` — no server required:

```python
import rustkernels

# Discover available kernels
reg = rustkernels.KernelRegistry()
print(f"{len(reg)} kernels available")

# Execute a batch kernel
result = reg.execute("graph/betweenness_centrality", {
    "num_nodes": 4,
    "edges": [[0, 1], [1, 2], [2, 3], [0, 3]],
    "normalized": True,
})

# Module-level convenience (cached default registry)
result = rustkernels.execute("graph/betweenness_centrality", {...})

# Catalog
rustkernels.list_domains()         # 14 domains
rustkernels.total_kernel_count()   # 106
rustkernels.enabled_domains()      # ["graph", "ml", ...]
```

See [`crates/rustkernel-python/README.md`](crates/rustkernel-python/README.md) for the full Python API reference.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           rustkernels                                │
│                      (facade crate, re-exports)                      │
├──────────────────────────────────────────────────────────────────────┤
│  rustkernel-core (0.4.0)       │  rustkernel-ecosystem (0.4.0)      │
│  ├── traits (Gpu/Batch/Ring)   │  ├── axum (REST API)               │
│  ├── registry (typed factory)  │  ├── tower (middleware)             │
│  ├── security (auth, RBAC)     │  ├── grpc (Tonic server)           │
│  ├── observability (metrics)   │  └── actix (actors)                │
│  ├── resilience (circuit)      ├─────────────────────────────────────┤
│  ├── runtime (lifecycle)       │  rustkernel-derive                  │
│  ├── memory (pooling)          │  ├── #[gpu_kernel] macro            │
│  ├── config (production)       │  └── #[derive(RingMessage)]         │
│  └── k2k (coordination)       │                                     │
├──────────────────────────────────────────────────────────────────────┤
│                       Domain Crates (14)                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  graph   │ │    ml    │ │compliance│ │ temporal │ │   risk   │  │
│  │   (28)   │ │   (17)   │ │   (11)   │ │    (7)  │ │    (5)   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ procint  │ │behavioral│ │ treasury │ │ clearing │ │accounting│  │
│  │    (7)   │ │    (6)   │ │    (5)   │ │    (5)   │ │    (9)   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│  │ banking  │ │orderbook │ │ payments │ │  audit   │               │
│  │    (1)   │ │    (1)   │ │    (2)   │ │    (2)   │               │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘               │
├──────────────────────────────────────────────────────────────────────┤
│  rustkernel-cli              │  rustkernel-python (PyO3 bindings)    │
│  (kernel management CLI)     │  pip install rustkernels              │
├──────────────────────────────┼───────────────────────────────────────┤
│                   RingKernel 0.4.2 (crates.io)                       │
│               (GPU-native persistent actor runtime)                  │
└──────────────────────────────────────────────────────────────────────┘
```

**20 crates total**: 1 facade, 1 core, 1 derive, 1 ecosystem, 1 CLI, 1 Python bindings, 14 domain crates.

### Execution Model

RustKernels supports two execution paths:

1. **Typed execution** — Call `BatchKernel<I, O>::execute(input)` directly with compile-time type safety.
2. **Type-erased execution** — Kernels registered via `register_batch_typed()` are wrapped in `TypeErasedBatchKernel`, enabling invocation through REST/gRPC with JSON serialization. The ecosystem layer handles `input → JSON bytes → execute_dyn() → JSON bytes → output` automatically.

Ring kernels require the RingKernel persistent actor runtime and are not callable through REST. They communicate through zero-copy ring buffers at sub-microsecond latency.

### K2K Coordination

Cross-kernel orchestration patterns built on RingKernel 0.4.2 K2K messaging:

- **IterativeState** — Track convergence across multi-pass algorithms (PageRank, K-Means)
- **ScatterGatherState** — Parallel worker fan-out with result aggregation
- **FanOutTracker** — Broadcast to multiple downstream kernels
- **PipelineTracker** — Multi-stage sequential processing

---

## Enterprise Features

### Security
```rust
use rustkernel_core::security::{SecurityContext, Role, KernelPermission};

let ctx = SecurityContext::new(user_id, tenant_id)
    .with_roles(vec![Role::KernelExecutor])
    .with_permissions(vec![KernelPermission::Execute]);
```

### Resilience
```rust
use rustkernel_core::resilience::{CircuitBreaker, CircuitBreakerConfig};

let cb = CircuitBreaker::new(CircuitBreakerConfig {
    failure_threshold: 5,
    success_threshold: 2,
    timeout: Duration::from_secs(30),
    ..Default::default()
});
```

### Production Configuration
```rust
use rustkernel_core::config::ProductionConfig;

// Load from environment or TOML
let config = ProductionConfig::from_env()?;
config.validate()?;
```

### Observability

- **Prometheus-compatible metrics** — request counts, latency, per-domain kernel counts, error rates
- **Distributed tracing** — OTLP export with kernel-level span instrumentation
- **Structured logging** — JSON-formatted, kernel-context-aware
- **SLO alerting** — configurable alert rules with multi-channel notification

---

## Building and Testing

```bash
# Build entire workspace
cargo build --workspace

# Run all 895 tests
cargo test --workspace

# Test a specific domain
cargo test --package rustkernel-graph
cargo test --package rustkernel-ml

# Lint (warnings as errors)
cargo clippy --all-targets --all-features -- -D warnings

# Format
cargo fmt --all

# Generate API documentation
cargo doc --workspace --no-deps --open

# Build mdBook documentation
cd docs && mdbook build
```

---

## Documentation

| Resource | Description |
|---|---|
| [Online Documentation](https://mivertowski.github.io/RustKernels/) | Guides, architecture, and API reference |
| [Kernel Catalogue](https://mivertowski.github.io/RustKernels/domains/) | All 106 kernels across 14 domains |
| [Architecture Guide](https://mivertowski.github.io/RustKernels/architecture/overview.html) | System design, execution modes, K2K patterns |
| [Enterprise Guide](https://mivertowski.github.io/RustKernels/enterprise/security.html) | Security, observability, resilience, runtime |
| [API Docs](https://docs.rs/rustkernels) | Auto-generated Rust API documentation |

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](docs/src/appendix/contributing.md) for development setup, code style guidelines, and the pull request process.

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

**Author**: Michael Ivertowski
**Version**: 0.4.0 — Deep integration with RingKernel 0.4.2
**Scope**: 106 kernels, 14 domains, 20 crates
