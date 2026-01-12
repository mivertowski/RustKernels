# RustKernels

GPU kernel library for financial services and analytics. Ported from the DotCompute C# implementation to Rust, using the RingKernel framework.

**Version**: 0.1.0
**Author**: Michael Ivertowski
**License**: Apache-2.0

## What This Is

A collection of GPU-accelerated algorithms organized into 14 domain-specific crates. The kernels cover graph analytics, machine learning, compliance/AML, risk calculations, and various financial operations.

This is not a general-purpose compute library. It exists to provide a consistent Rust interface to algorithms that were previously implemented in C# against Orleans grains.

## Execution Modes

Kernels operate in one of two modes:

- **Batch**: CPU-orchestrated execution. Higher launch overhead (10-50μs), but simpler to reason about. State lives in CPU memory.
- **Ring**: GPU-persistent actors via RingKernel. Lower latency (100-500ns per message), but requires understanding the actor model. State remains on GPU.

Most kernels support both modes. Choose based on your latency requirements.

## Domains

| Domain | Crate | Kernels |
|--------|-------|---------|
| Graph Analytics | `rustkernel-graph` | 15 |
| Statistical ML | `rustkernel-ml` | 6 |
| Compliance | `rustkernel-compliance` | 9 |
| Temporal Analysis | `rustkernel-temporal` | 7 |
| Risk Analytics | `rustkernel-risk` | 4 |
| Banking | `rustkernel-banking` | 1 |
| Behavioral Analytics | `rustkernel-behavioral` | 6 |
| Order Matching | `rustkernel-orderbook` | 1 |
| Process Intelligence | `rustkernel-procint` | 4 |
| Clearing | `rustkernel-clearing` | 5 |
| Treasury | `rustkernel-treasury` | 5 |
| Accounting | `rustkernel-accounting` | 5 |
| Payments | `rustkernel-payments` | 2 |
| Audit | `rustkernel-audit` | 2 |

## Usage

```toml
[dependencies]
rustkernel = "0.1.0"
```

```rust
use rustkernel::prelude::*;
use rustkernel::graph::centrality::PageRank;

let kernel = PageRank::new();
let metadata = kernel.metadata();
```

Feature flags control which domains are compiled:

```toml
# Only what you need
rustkernel = { version = "0.1.0", features = ["graph", "risk"] }

# Everything
rustkernel = { version = "0.1.0", features = ["full"] }
```

Default features include: `graph`, `ml`, `compliance`, `temporal`, `risk`.

## Requirements

- Rust 1.85 or later
- RustCompute (RingKernel) — expected at `../../RustCompute/RustCompute/` relative to this workspace
- CUDA toolkit if you want actual GPU execution; otherwise falls back to CPU

## Building and Testing

```bash
cargo build --workspace
cargo test --workspace
cargo test --package rustkernel-graph  # single domain
```

## Project Structure

```
crates/
├── rustkernel/           # Facade, re-exports domains
├── rustkernel-core/      # Traits, registry, licensing
├── rustkernel-derive/    # Proc macros
├── rustkernel-cli/       # Command-line tool
└── rustkernel-{domain}/  # 14 domain crates
```

## Status

The port is functionally complete. All 72 kernels have been implemented with both BatchKernel and RingKernelHandler traits. K2K (kernel-to-kernel) messaging is in place for cross-kernel coordination patterns.

Test coverage exists for all domains. Some edge cases in the behavioral analytics causal graph module remain flaky.

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE) for details.
