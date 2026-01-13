# RustKernels

**High-performance GPU kernel library for financial services, compliance, and enterprise analytics.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

---

## Overview

RustKernels provides **106 GPU-accelerated algorithms** across **14 domain-specific crates**, purpose-built for financial institutions, compliance teams, and enterprise analytics platforms. The library delivers consistent, production-grade implementations of algorithms previously developed in C# against Orleans grains, now ported to Rust with the RingKernel framework for maximum performance.

This is not a general-purpose compute library. RustKernels exists to provide a unified Rust interface to specialized algorithms for graph analytics, machine learning, AML/compliance, risk calculations, process mining, and financial operations.

## Key Capabilities

| Capability | Description |
|------------|-------------|
| **14 Domain Categories** | Graph analytics, ML/AI, compliance, risk, treasury, process intelligence, healthcare, and more |
| **106 Production Kernels** | Comprehensive coverage from PageRank to drug interaction prediction |
| **Dual Execution Modes** | Batch (CPU-orchestrated) and Ring (GPU-persistent actors) |
| **Sub-millisecond Latency** | Ring mode delivers 100-500ns message handling |
| **K2K Coordination** | Built-in patterns for cross-kernel workflows |
| **Enterprise Licensing** | Domain-based feature gating for commercial deployment |

## Domain Coverage

| Domain | Crate | Kernels | Highlights |
|--------|-------|---------|------------|
| **Graph Analytics** | `rustkernel-graph` | 28 | PageRank, community detection, GNN inference, graph attention networks |
| **Statistical ML** | `rustkernel-ml` | 17 | Clustering, anomaly detection, NLP embeddings, federated learning, healthcare analytics |
| **Compliance** | `rustkernel-compliance` | 11 | AML pattern detection, KYC scoring, sanctions screening, entity resolution |
| **Temporal Analysis** | `rustkernel-temporal` | 7 | ARIMA forecasting, seasonal decomposition, change point detection |
| **Risk Analytics** | `rustkernel-risk` | 5 | Monte Carlo VaR, credit scoring, stress testing, real-time correlation |
| **Process Intelligence** | `rustkernel-procint` | 7 | DFG construction, conformance checking, digital twin simulation |
| **Behavioral Analytics** | `rustkernel-behavioral` | 6 | Profiling, forensic queries, causal graph analysis |
| **Banking** | `rustkernel-banking` | 1 | Fraud pattern matching |
| **Order Matching** | `rustkernel-orderbook` | 1 | High-frequency order book engine |
| **Clearing** | `rustkernel-clearing` | 5 | Netting calculation, DVP matching, settlement execution |
| **Treasury** | `rustkernel-treasury` | 5 | Liquidity optimization, FX hedging, cash flow forecasting |
| **Accounting** | `rustkernel-accounting` | 9 | Network generation, GL reconciliation, GAAP violation detection |
| **Payments** | `rustkernel-payments` | 2 | Payment processing, flow analysis |
| **Audit** | `rustkernel-audit` | 2 | Feature extraction, hypergraph construction |

## Execution Modes

Kernels operate in one of two modes, selected based on latency requirements and workload characteristics:

### Batch Mode
- **Latency**: 10-50μs launch overhead
- **State**: CPU memory, transferred to GPU per execution
- **Use Case**: Heavy periodic computation, large batch processing
- **Example**: End-of-day risk aggregation, batch AML screening

### Ring Mode
- **Latency**: 100-500ns per message
- **State**: Persistent in GPU memory
- **Use Case**: High-frequency streaming, real-time analytics
- **Example**: Order book matching, real-time fraud scoring

Most kernels support both modes. The `KernelMetadata` struct indicates supported modes for each kernel.

## Recent Additions

The latest release includes several innovative kernel categories:

- **Graph Neural Networks**: `GNNInference` and `GraphAttention` for node classification and link prediction using message passing and multi-head attention
- **NLP/LLM Integration**: `EmbeddingGeneration` and `SemanticSimilarity` for text processing, document similarity, and semantic search
- **Federated Learning**: `SecureAggregation` with differential privacy for privacy-preserving distributed model training
- **Healthcare Analytics**: `DrugInteractionPrediction` and `ClinicalPathwayConformance` for clinical decision support
- **Process Simulation**: `DigitalTwin` for Monte Carlo process simulation and what-if analysis
- **Streaming ML**: `StreamingIsolationForest` and `AdaptiveThreshold` for online anomaly detection
- **Explainability**: `SHAPValues` and `FeatureImportance` for model interpretability

## Installation

Add RustKernels to your `Cargo.toml`:

```toml
[dependencies]
rustkernel = "0.1.0"
```

### Feature Flags

Control which domains are compiled to optimize binary size:

```toml
# Default features (graph, ml, compliance, temporal, risk)
rustkernel = "0.1.0"

# Selective domain inclusion
rustkernel = { version = "0.1.0", features = ["graph", "risk", "procint"] }

# All domains
rustkernel = { version = "0.1.0", features = ["full"] }
```

## Quick Start

```rust
use rustkernel::prelude::*;
use rustkernel::graph::centrality::PageRank;

// Create kernel instance
let kernel = PageRank::new();

// Access metadata
let metadata = kernel.metadata();
println!("Kernel: {}", metadata.id);
println!("Domain: {:?}", metadata.domain);

// Execute in batch mode
let input = PageRankInput {
    num_nodes: 1000,
    edges: edges,
    damping_factor: 0.85,
    max_iterations: 100,
    tolerance: 1e-6,
};

let result = kernel.execute(input).await?;
println!("Converged in {} iterations", result.iterations);
```

## Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Rust | 1.85+ | Edition 2024 features required |
| RustCompute | Latest | RingKernel framework (expected at `../../RustCompute/RustCompute/`) |
| CUDA Toolkit | 12.0+ | Optional; falls back to CPU execution if unavailable |

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

## Project Structure

```
crates/
├── rustkernel/              # Facade crate, re-exports all domains
├── rustkernel-core/         # Core traits, registry, licensing, K2K coordination
├── rustkernel-derive/       # Procedural macros (#[gpu_kernel], #[derive(KernelMessage)])
├── rustkernel-cli/          # Command-line interface for kernel management
└── rustkernel-{domain}/     # 14 domain-specific implementation crates
    ├── src/
    │   ├── lib.rs           # Module exports and registration
    │   ├── messages.rs      # Batch kernel I/O types
    │   ├── ring_messages.rs # Ring message types
    │   ├── types.rs         # Common domain types
    │   └── {feature}.rs     # Kernel implementations
    └── Cargo.toml
```

## Documentation

- **[Online Documentation](https://mivertowski.github.io/RustKernels/)** - Comprehensive guides and API reference
- **[Kernel Catalogue](https://mivertowski.github.io/RustKernels/domains/)** - Complete listing of all 106 kernels
- **[Architecture Guide](https://mivertowski.github.io/RustKernels/architecture/overview.html)** - System design and patterns

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](docs/src/appendix/contributing.md) for guidelines.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

**Author**: Michael Ivertowski
**Version**: 0.1.0
