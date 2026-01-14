# rustkernels

[![Crates.io](https://img.shields.io/crates/v/rustkernels.svg)](https://crates.io/crates/rustkernels)
[![Documentation](https://docs.rs/rustkernels/badge.svg)](https://docs.rs/rustkernels)
[![License](https://img.shields.io/crates/l/rustkernels.svg)](LICENSE)

GPU-accelerated kernel library for financial services, analytics, and compliance workloads.

RustKernels is a Rust port of the DotCompute GPU kernel library, leveraging the RustCompute (RingKernel) framework for GPU-native persistent actors.

## Features

- **14 Domain Categories**: Graph analytics, ML, compliance, risk, temporal analysis, and more
- **106+ Kernels**: Comprehensive coverage of financial and analytical algorithms
- **Dual Execution Modes**:
  - **Batch**: CPU-orchestrated, 10-50us overhead, for periodic heavy computation
  - **Ring**: GPU-persistent actor, 100-500ns latency, for high-frequency operations
- **Enterprise Licensing**: Domain-based licensing and feature gating
- **Multi-Backend**: CUDA, WebGPU, and CPU backends via RustCompute

## Domain Crates

| Domain | Crate | Kernels |
|--------|-------|---------|
| Graph Analytics | `rustkernel-graph` | 28 |
| Statistical ML | `rustkernel-ml` | 17 |
| Compliance | `rustkernel-compliance` | 11 |
| Temporal Analysis | `rustkernel-temporal` | 7 |
| Risk Analytics | `rustkernel-risk` | 5 |
| Process Intelligence | `rustkernel-procint` | 7 |
| Behavioral Analytics | `rustkernel-behavioral` | 6 |
| Banking | `rustkernel-banking` | 1 |
| Order Matching | `rustkernel-orderbook` | 1 |
| Clearing | `rustkernel-clearing` | 5 |
| Treasury | `rustkernel-treasury` | 5 |
| Accounting | `rustkernel-accounting` | 9 |
| Payments | `rustkernel-payments` | 2 |
| Audit | `rustkernel-audit` | 2 |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernels = "0.1.1"
```

## Quick Start

```rust
use rustkernels::prelude::*;

// Create a kernel registry
let registry = KernelRegistry::new();

// Register all kernels
rustkernels::register_all(&registry)?;

// Use a specific kernel
let pagerank = PageRank::new();
pagerank.initialize(graph, 0.85);
let score = pagerank.query_score(node_id);
```

## License

Apache-2.0
