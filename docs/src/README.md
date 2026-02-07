# RustKernels

**GPU-accelerated kernel library for financial services, compliance, and enterprise analytics**

**Version 0.4.0** | RingKernel 0.4.2 | 106 kernels | 14 domains | 19 crates

---

## Overview

RustKernels provides **106 GPU-accelerated algorithms** across **14 domain-specific crates**, engineered for financial services, regulatory compliance, and enterprise analytics workloads. Built on [RingKernel 0.4.2](https://crates.io/crates/ringkernel-core), it delivers both CPU-orchestrated batch execution and GPU-persistent ring execution with sub-microsecond message latency.

<div class="warning">

RustKernels is a specialized compute library for financial and enterprise workloads. It is not a general-purpose GPU compute framework.

</div>

## Key Features

| Feature | Description |
|---------|-------------|
| **14 Domain Categories** | Graph analytics, ML, compliance, risk, treasury, and more |
| **106 Kernels** | Comprehensive coverage of financial and analytical algorithms |
| **Dual Execution Modes** | Batch (CPU-orchestrated) and Ring (GPU-persistent actor) |
| **Type-Erased Execution** | `TypeErasedBatchKernel` enables REST/gRPC dispatch without compile-time types |
| **Factory Registration** | `register_batch_typed()` with automatic type inference |
| **Enterprise Security** | JWT/API key auth, RBAC, multi-tenancy, secrets management |
| **Production Observability** | Prometheus metrics, OTLP tracing, structured logging, SLO alerting |
| **Resilience Patterns** | Circuit breakers, retry with backoff, deadline propagation, health probes |
| **Service Deployment** | REST (Axum), gRPC (Tonic), Tower middleware, Actix actors |
| **K2K Messaging** | Cross-kernel coordination: iterative, scatter-gather, fan-out, pipeline |
| **Fixed-Point Arithmetic** | GPU-compatible exact financial calculations |
| **RingKernel 0.4.2** | Deep integration with GPU-native persistent actor runtime |

## Execution Model

Kernels operate in one of two modes, selected based on latency and throughput requirements:

| Mode | Latency | Overhead | State Location | Best For |
|------|---------|----------|----------------|----------|
| **Batch** | 10–50 μs | Higher (CPU round-trip) | CPU memory | Heavy periodic computation |
| **Ring** | 100–500 ns | Minimal (lock-free) | GPU memory | High-frequency streaming |

Batch kernels implementing `BatchKernel<I, O>` can be executed directly via typed calls or through the type-erased `BatchKernelDyn` interface used by REST and gRPC endpoints. Ring kernels require the RingKernel persistent actor runtime.

## Domains at a Glance

| Domain | Crate | Kernels | Description |
|--------|-------|---------|-------------|
| Graph Analytics | `rustkernel-graph` | 28 | PageRank, community detection, GNN inference, graph attention |
| Statistical ML | `rustkernel-ml` | 17 | Clustering, NLP embeddings, federated learning, healthcare |
| Compliance | `rustkernel-compliance` | 11 | AML patterns, KYC scoring, sanctions screening |
| Temporal Analysis | `rustkernel-temporal` | 7 | ARIMA, Prophet decomposition, change-point detection |
| Risk Analytics | `rustkernel-risk` | 5 | Credit scoring, Monte Carlo VaR, stress testing, correlation |
| Banking | `rustkernel-banking` | 1 | Fraud pattern matching |
| Behavioral Analytics | `rustkernel-behavioral` | 6 | Profiling, forensics, event correlation |
| Order Matching | `rustkernel-orderbook` | 1 | Order book matching engine |
| Process Intelligence | `rustkernel-procint` | 7 | DFG, conformance, digital twin simulation |
| Clearing | `rustkernel-clearing` | 5 | Netting, settlement, DVP matching |
| Treasury | `rustkernel-treasury` | 5 | Cash flow, FX hedging, liquidity optimization |
| Accounting | `rustkernel-accounting` | 9 | Network generation, reconciliation |
| Payments | `rustkernel-payments` | 2 | Payment processing, flow analysis |
| Audit | `rustkernel-audit` | 2 | Feature extraction, hypergraph construction |

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernels = "0.4.0"
```

Basic usage:

```rust
use rustkernels::prelude::*;
use rustkernels::graph::centrality::{BetweennessCentrality, BetweennessCentralityInput};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let kernel = BetweennessCentrality::new();
    println!("Kernel: {}", kernel.metadata().id);

    let input = BetweennessCentralityInput {
        num_nodes: 4,
        edges: vec![(0, 1), (1, 2), (2, 3), (0, 3)],
        normalized: true,
    };

    let result = kernel.execute(input).await?;
    for (node, score) in result.scores.iter().enumerate() {
        println!("  Node {}: {:.4}", node, score);
    }
    Ok(())
}
```

### Feature Flags

```toml
# Default domains (graph, ml, compliance, temporal, risk)
rustkernels = "0.4.0"

# Selective compilation
rustkernels = { version = "0.4.0", features = ["graph", "accounting"] }

# All 14 domains
rustkernels = { version = "0.4.0", features = ["full"] }

# Service deployment
rustkernel-ecosystem = { version = "0.4.0", features = ["axum", "grpc"] }
```

## Enterprise Features

Version 0.4.0 provides production-ready enterprise capabilities with deep RingKernel 0.4.2 integration:

| Module | Features |
|--------|----------|
| **Security** | JWT/API key auth, RBAC, multi-tenancy, secrets management |
| **Observability** | Prometheus metrics, OTLP tracing, structured logging, SLO alerting |
| **Resilience** | Circuit breakers, retry with backoff, deadline propagation, health probes |
| **Runtime** | Lifecycle state machine, graceful shutdown, configuration presets |
| **Memory** | Size-stratified pooling, pressure handling, multi-phase reductions |
| **Ecosystem** | Axum REST with real execution, Tower middleware, Tonic gRPC, Actix actors |

See [Enterprise Features](enterprise/security.md) for detailed documentation.

## Requirements

- **Rust 1.85** or later
- **RingKernel 0.4.2** (from [crates.io](https://crates.io/crates/ringkernel-core))
- **CUDA toolkit** (optional; falls back to CPU execution)

## Project Structure

```
crates/
├── rustkernels/             # Facade crate — re-exports all domains
├── rustkernel-core/         # Core traits, registry, enterprise modules
│   ├── security/            # Auth, RBAC, multi-tenancy
│   ├── observability/       # Metrics, tracing, logging
│   ├── resilience/          # Circuit breaker, retry, health
│   ├── runtime/             # Lifecycle, configuration
│   ├── memory/              # Pooling, reductions
│   └── config/              # Production configuration
├── rustkernel-ecosystem/    # Service integrations (Axum, gRPC, Actix)
├── rustkernel-derive/       # Procedural macros
├── rustkernel-cli/          # Command-line interface
└── rustkernel-{domain}/     # 14 domain-specific crates
```

## Building

```bash
# Build entire workspace
cargo build --workspace

# Run all tests (895 tests)
cargo test --workspace

# Test a single domain
cargo test --package rustkernel-graph

# Lint with warnings as errors
cargo clippy --all-targets --all-features -- -D warnings

# Generate API documentation
cargo doc --workspace --no-deps --open
```

## License

Licensed under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). See LICENSE for details.

---

<div style="text-align: center; margin-top: 2rem;">

[Getting Started](getting-started/installation.md) | [Architecture](architecture/overview.md) | [Enterprise Features](enterprise/security.md) | [Kernel Catalogue](domains/README.md)

</div>
