# RustKernels

**High-performance GPU kernel library for financial services, compliance, and enterprise analytics.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)
[![Documentation](https://img.shields.io/badge/docs-online-green.svg)](https://mivertowski.github.io/RustKernels/)
[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](CHANGELOG.md)

---

## Why RustKernels?

Financial institutions face a common challenge: implementing high-performance analytics that scale from batch processing to real-time streaming while maintaining regulatory compliance. RustKernels solves this by providing:

- **Battle-tested algorithms** ported from production C# Orleans grains
- **Nanosecond-scale latency** for real-time fraud detection and order matching
- **Unified API** across 14 specialized domains
- **GPU acceleration** with automatic CPU fallback

```rust
// Detect AML patterns in transaction graphs - sub-millisecond response
let detector = CircularFlowRatio::new();
let risk_score = detector.compute(&transaction_graph)?;
if risk_score > 0.8 {
    alert_compliance_team(&transaction);
}
```

---

## Overview

RustKernels provides **106 GPU-accelerated algorithms** across **14 domain-specific crates**, purpose-built for financial institutions, compliance teams, and enterprise analytics platforms.

### What Makes RustKernels Different

| Challenge | RustKernels Solution |
|-----------|---------------------|
| Latency requirements vary widely | Dual execution modes: Batch (10-50μs) or Ring (100-500ns) |
| Complex multi-kernel workflows | Built-in K2K coordination patterns |
| Production reliability concerns | Ported from battle-tested C# implementations |
| GPU availability uncertainty | Automatic CPU fallback when CUDA unavailable |
| Regulatory explainability | SHAP values and feature importance kernels |
| Enterprise security requirements | Auth, RBAC, multi-tenancy, secrets management |
| Production observability | Metrics, tracing, logging, alerting |
| Fault tolerance | Circuit breakers, retry, health checks |
| Service deployment | REST, gRPC, Actix actor integrations |

---

## Performance Characteristics

| Metric | Batch Mode | Ring Mode |
|--------|------------|-----------|
| Launch overhead | 10-50μs | N/A (persistent) |
| Message latency | N/A | 100-500ns |
| State location | CPU → GPU transfer | GPU-resident |
| Throughput (PageRank) | ~100K nodes/sec | ~500K updates/sec |
| Memory efficiency | Standard | Optimized (persistent) |

### When to Use Each Mode

**Batch Mode** - Best for scheduled, heavy computation:
- End-of-day risk aggregation
- Batch AML screening (millions of transactions)
- Monthly compliance reporting
- Model training and backtesting

**Ring Mode** - Best for real-time, high-frequency operations:
- Order book matching (sub-millisecond)
- Real-time fraud scoring
- Streaming anomaly detection
- Live transaction monitoring

---

## Domain Coverage

| Domain | Crate | Kernels | Key Algorithms |
|--------|-------|---------|----------------|
| **Graph Analytics** | `rustkernel-graph` | 28 | PageRank, Louvain, GNN inference, graph attention, cycle detection |
| **Statistical ML** | `rustkernel-ml` | 17 | K-Means, DBSCAN, isolation forest, federated learning, SHAP |
| **Compliance** | `rustkernel-compliance` | 11 | Circular flow detection, rapid movement, sanctions screening |
| **Temporal Analysis** | `rustkernel-temporal` | 7 | ARIMA, Prophet decomposition, change point detection |
| **Risk Analytics** | `rustkernel-risk` | 5 | Monte Carlo VaR, credit scoring, stress testing |
| **Process Intelligence** | `rustkernel-procint` | 7 | DFG construction, conformance checking, digital twin |
| **Behavioral Analytics** | `rustkernel-behavioral` | 6 | Profiling, forensic queries, causal analysis |
| **Treasury** | `rustkernel-treasury` | 5 | Liquidity optimization, FX hedging, NSFR calculation |
| **Clearing** | `rustkernel-clearing` | 5 | Multilateral netting, DVP matching, settlement |
| **Accounting** | `rustkernel-accounting` | 9 | Network generation, GL reconciliation, GAAP detection |
| **Banking** | `rustkernel-banking` | 1 | Fraud pattern matching (Aho-Corasick) |
| **Order Matching** | `rustkernel-orderbook` | 1 | High-frequency order book engine |
| **Payments** | `rustkernel-payments` | 2 | Payment processing, flow analysis |
| **Audit** | `rustkernel-audit` | 2 | Feature extraction, hypergraph construction |

---

## Use Cases

### Anti-Money Laundering (AML)

Detect layering, structuring, and circular transaction patterns:

```rust
use rustkernel::graph::cycles::ShortCycleParticipation;
use rustkernel::compliance::circular_flow::CircularFlowRatio;

// Detect nodes participating in suspicious cycles
let cycle_detector = ShortCycleParticipation::new();
let results = cycle_detector.compute_all(&transaction_graph);

for result in results.iter().filter(|r| r.risk_level == CycleRiskLevel::Critical) {
    // Nodes in 4-cycles are high-priority for investigation
    flag_for_investigation(result.node_index);
}

// Compute circular flow ratios
let cfr = CircularFlowRatio::new();
let scores = cfr.compute_batch(&graph);
```

### Real-Time Fraud Detection

Score transactions in real-time with streaming anomaly detection:

```rust
use rustkernel::ml::streaming::StreamingIsolationForest;

let detector = StreamingIsolationForest::new(StreamingConfig {
    num_trees: 100,
    sample_size: 256,
    window_size: 10000,
});

// Process incoming transactions
for transaction in transaction_stream {
    let score = detector.score(&transaction.features)?;
    if score > 0.7 {
        block_transaction(&transaction);
    }
}
```

### Process Mining & Digital Twin

Simulate process changes before deployment:

```rust
use rustkernel::procint::simulation::{DigitalTwin, SimulationConfig};

let twin = DigitalTwin::new();

// Simulate adding 2 more resources to bottleneck activity
let what_if = twin.simulate(&process_model, &SimulationConfig {
    num_simulations: 1000,
    resource_overrides: vec![("Review", 5)], // 3 → 5 reviewers
    ..Default::default()
})?;

println!("Projected improvement: {:.1}% faster",
    (1.0 - what_if.avg_completion_time / baseline.avg_completion_time) * 100.0);
```

### Graph Neural Networks for Entity Resolution

Link prediction and entity matching with GNN inference:

```rust
use rustkernel::graph::gnn::{GNNInference, GNNConfig};

let gnn = GNNInference::new();

let embeddings = gnn.infer(&entity_graph, &node_features, &GNNConfig {
    hidden_dim: 64,
    num_layers: 2,
    aggregation: AggregationType::Mean,
})?;

// Find similar entities via embedding similarity
let matches = find_similar_embeddings(&embeddings, threshold: 0.9);
```

---

## Recent Additions

The latest release introduces innovative kernel categories:

| Category | Kernels | Description |
|----------|---------|-------------|
| **Graph Neural Networks** | GNNInference, GraphAttention | Message-passing GNN and multi-head attention for node classification |
| **NLP/Embeddings** | EmbeddingGeneration, SemanticSimilarity | TF-IDF embeddings and document similarity |
| **Federated Learning** | SecureAggregation | Privacy-preserving model aggregation with differential privacy |
| **Healthcare Analytics** | DrugInteractionPrediction, ClinicalPathwayConformance | Clinical decision support kernels |
| **Process Simulation** | DigitalTwin | Monte Carlo process simulation and what-if analysis |
| **Streaming ML** | StreamingIsolationForest, AdaptiveThreshold | Online anomaly detection with concept drift handling |
| **Explainability** | SHAPValues, FeatureImportance | Model interpretability for regulatory compliance |

---

## Enterprise Features (0.2.0)

RustKernels 0.2.0 introduces production-ready enterprise capabilities:

### Security
```rust
use rustkernel_core::security::{SecurityContext, Role, KernelPermission};

let ctx = SecurityContext::new(user_id, tenant_id)
    .with_roles(vec![Role::KernelExecutor])
    .with_permissions(vec![KernelPermission::Execute]);

// Execute with security context
kernel.execute_with_context(&ctx, input).await?;
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

// Execute with circuit breaker protection
cb.call(|| kernel.execute(input)).await?;
```

### Production Configuration
```rust
use rustkernel_core::config::ProductionConfig;

// Load from environment
let config = ProductionConfig::from_env()?;

// Or use presets
let config = ProductionConfig::production();

// Validate before use
config.validate()?;
```

### Service Deployment
```rust
use rustkernel_ecosystem::axum::{KernelRouter, RouterConfig};

let router = KernelRouter::new(registry, RouterConfig::default());
let app = router.into_router();

// Endpoints: /kernels, /execute, /health, /metrics
axum::serve(listener, app).await?;
```

---

## Installation

Add RustKernels to your `Cargo.toml`:

```toml
[dependencies]
rustkernels = "0.2.0"
```

### Feature Flags

Control which domains are compiled to optimize binary size:

```toml
# Default features (graph, ml, compliance, temporal, risk)
rustkernels = "0.2.0"

# Selective domain inclusion
rustkernels = { version = "0.2.0", features = ["graph", "compliance", "procint"] }

# All domains
rustkernels = { version = "0.2.0", features = ["full"] }

# Enterprise ecosystem (REST/gRPC service)
rustkernel-ecosystem = { version = "0.2.0", features = ["axum", "grpc"] }
```

---

## Quick Start

```rust
use rustkernel::prelude::*;
use rustkernel::graph::centrality::PageRank;

#[tokio::main]
async fn main() -> Result<()> {
    // Create kernel instance
    let kernel = PageRank::new();

    // Access metadata
    let metadata = kernel.metadata();
    println!("Kernel: {} ({})", metadata.id, metadata.domain);

    // Build input
    let input = PageRankInput {
        num_nodes: 1000,
        edges: load_edges()?,
        damping_factor: 0.85,
        max_iterations: 100,
        tolerance: 1e-6,
    };

    // Execute
    let result = kernel.execute(input).await?;

    println!("Converged in {} iterations", result.iterations);
    println!("Top node: {} (score: {:.4})",
        result.top_node(),
        result.scores[result.top_node()]);

    Ok(())
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         rustkernels                              │
│                    (facade crate, re-exports)                    │
├─────────────────────────────────────────────────────────────────┤
│  rustkernel-core (0.2.0)     │  rustkernel-ecosystem (0.2.0)    │
│  ├── traits (Gpu/Batch/Ring) │  ├── axum (REST API)             │
│  ├── security (auth, RBAC)   │  ├── tower (middleware)          │
│  ├── observability (metrics) │  ├── grpc (Tonic server)         │
│  ├── resilience (circuit)    │  └── actix (actors)              │
│  ├── runtime (lifecycle)     ├──────────────────────────────────┤
│  ├── memory (pooling)        │  rustkernel-derive               │
│  ├── config (production)     │  - #[gpu_kernel] macro           │
│  └── k2k (coordination)      │  - #[derive(RingMessage)]        │
├─────────────────────────────────────────────────────────────────┤
│                    Domain Crates (14)                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │  graph  │ │   ml    │ │complianc│ │temporal │ │  risk   │   │
│  │  (28)   │ │  (17)   │ │  (11)   │ │   (7)   │ │   (5)   │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │procint  │ │behavior │ │treasury │ │clearing │ │accounting│   │
│  │   (7)   │ │   (6)   │ │   (5)   │ │   (5)   │ │   (9)   │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐               │
│  │ banking │ │orderbook│ │payments │ │  audit  │               │
│  │   (1)   │ │   (1)   │ │   (2)   │ │   (2)   │               │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘               │
├─────────────────────────────────────────────────────────────────┤
│                    RustCompute / RingKernel 0.3.1                │
│                    (GPU execution framework)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Rust | 1.85+ | Edition 2024 features required |
| RustCompute | Latest | RingKernel framework (path dependency) |
| CUDA Toolkit | 12.0+ | Optional; falls back to CPU if unavailable |

---

## Building and Testing

```bash
# Build entire workspace
cargo build --workspace

# Run all tests
cargo test --workspace

# Test specific domain
cargo test --package rustkernel-graph
cargo test --package rustkernel-ml

# Run benchmarks
cargo bench --package rustkernel

# Generate documentation
cargo doc --workspace --no-deps --open

# Lint
cargo clippy --all-targets --all-features -- -D warnings
```

---

## Documentation

- **[Online Documentation](https://mivertowski.github.io/RustKernels/)** - Comprehensive guides and API reference
- **[Kernel Catalogue](https://mivertowski.github.io/RustKernels/domains/)** - Complete listing of all 106 kernels
- **[Architecture Guide](https://mivertowski.github.io/RustKernels/architecture/overview.html)** - System design and patterns

---

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](docs/src/appendix/contributing.md) for guidelines.

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

**Author**: Michael Ivertowski
**Version**: 0.2.0
**Kernels**: 106 across 14 domains
**Crates**: 19 (including rustkernel-ecosystem)
