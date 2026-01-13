# RustKernels

**GPU-accelerated kernel library for financial services and analytics**

---

## Overview

RustKernels provides **106 GPU-accelerated algorithms** across **14 domain-specific crates**, designed for financial services, compliance, and enterprise analytics. Ported from the DotCompute C# implementation to Rust, using the RingKernel framework.

<div class="warning">

This is a specialized compute library for financial and enterprise workloads, not a general-purpose GPU compute framework.

</div>

## Key Features

| Feature | Description |
|---------|-------------|
| **14 Domain Categories** | Graph analytics, ML, compliance, risk, treasury, and more |
| **106 Kernels** | Comprehensive coverage of financial algorithms |
| **Dual Execution Modes** | Batch (CPU-orchestrated) and Ring (GPU-persistent) |
| **Enterprise Ready** | Apache-2.0 license, domain-based feature gating |
| **K2K Messaging** | Cross-kernel coordination patterns |
| **Fixed-Point Arithmetic** | Exact financial calculations |

## Execution Modes

Kernels operate in one of two modes:

| Mode | Latency | Overhead | State Location | Best For |
|------|---------|----------|----------------|----------|
| **Batch** | 10-50μs | Higher | CPU memory | Heavy periodic computation |
| **Ring** | 100-500ns | Minimal | GPU memory | High-frequency streaming |

Most kernels support both modes. Choose based on your latency requirements.

## Domains at a Glance

| Domain | Crate | Kernels | Description |
|--------|-------|---------|-------------|
| Graph Analytics | `rustkernel-graph` | 28 | PageRank, community detection, GNN inference, graph attention |
| Statistical ML | `rustkernel-ml` | 17 | Clustering, NLP embeddings, federated learning, healthcare analytics |
| Compliance | `rustkernel-compliance` | 11 | AML patterns, KYC, sanctions screening |
| Temporal Analysis | `rustkernel-temporal` | 7 | Forecasting, anomaly detection, decomposition |
| Risk Analytics | `rustkernel-risk` | 5 | Credit scoring, VaR, stress testing, correlation |
| Banking | `rustkernel-banking` | 1 | Fraud pattern matching |
| Behavioral Analytics | `rustkernel-behavioral` | 6 | Profiling, forensics, event correlation |
| Order Matching | `rustkernel-orderbook` | 1 | Order book matching engine |
| Process Intelligence | `rustkernel-procint` | 7 | DFG, conformance, digital twin simulation |
| Clearing | `rustkernel-clearing` | 5 | Netting, settlement, DVP matching |
| Treasury | `rustkernel-treasury` | 5 | Cash flow, FX hedging, liquidity |
| Accounting | `rustkernel-accounting` | 9 | Network generation, reconciliation |
| Payments | `rustkernel-payments` | 2 | Payment processing, flow analysis |
| Audit | `rustkernel-audit` | 2 | Feature extraction, hypergraph construction |

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernel = "0.1.0"
```

Basic usage:

```rust
use rustkernel::prelude::*;
use rustkernel::graph::centrality::PageRank;

// Create a kernel instance
let kernel = PageRank::new();

// Access kernel metadata
let metadata = kernel.metadata();
println!("Kernel: {}", metadata.id);
println!("Domain: {:?}", metadata.domain);

// Execute (batch mode)
let result = kernel.execute(input).await?;
```

### Feature Flags

Control which domains are compiled:

```toml
# Only what you need
rustkernel = { version = "0.1.0", features = ["graph", "risk"] }

# Everything
rustkernel = { version = "0.1.0", features = ["full"] }
```

Default features: `graph`, `ml`, `compliance`, `temporal`, `risk`.

## Requirements

- **Rust 1.85** or later
- **RustCompute** (RingKernel framework)
- **CUDA toolkit** (optional, falls back to CPU execution)

## Project Structure

```
crates/
├── rustkernel/           # Facade crate, re-exports all domains
├── rustkernel-core/      # Core traits, registry, licensing
├── rustkernel-derive/    # Procedural macros
├── rustkernel-cli/       # Command-line interface
└── rustkernel-{domain}/  # 14 domain-specific crates
```

## Building

```bash
# Build entire workspace
cargo build --workspace

# Run all tests
cargo test --workspace

# Test single domain
cargo test --package rustkernel-graph

# Generate API documentation
cargo doc --workspace --no-deps --open
```

## License

Licensed under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). See LICENSE for details.

---

<div style="text-align: center; margin-top: 2rem;">

[Getting Started](getting-started/installation.md) | [Kernel Catalogue](domains/README.md) | [Articles](articles/README.md)

</div>
