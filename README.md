# RustKernels

GPU-accelerated kernel library for financial services, analytics, and compliance workloads.

RustKernels is a Rust port of the DotCompute GPU kernel library, leveraging the RustCompute (RingKernel) framework for GPU-native persistent actors.

## Features

- **14 Domain Categories**: Graph analytics, ML, compliance, risk, temporal analysis, and more
- **72+ Kernels**: Comprehensive coverage of financial and analytical algorithms
- **Dual Execution Modes**:
  - **Batch**: CPU-orchestrated, 10-50us overhead, for periodic heavy computation
  - **Ring**: GPU-persistent actor, 100-500ns latency, for high-frequency operations
- **Enterprise Licensing**: Domain-based licensing and feature gating
- **Multi-Backend**: CUDA, WebGPU, and CPU backends via RustCompute

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernel = "0.1"
```

Basic usage:

```rust
use rustkernel::prelude::*;
use rustkernel::graph::centrality::PageRank;

fn main() {
    // Create a kernel
    let kernel = PageRank::new();

    // Check metadata
    let metadata = kernel.metadata();
    println!("Kernel: {} ({:?})", metadata.id, metadata.domain);
}
```

## Domains

### Priority 1 (High Value)

| Domain | Feature Flag | Kernels | Description |
|--------|-------------|---------|-------------|
| Graph Analytics | `graph` | 15 | Centrality, community detection, motifs, similarity |
| Statistical ML | `ml` | 6 | Clustering, anomaly detection, regression |
| Compliance | `compliance` | 9 | AML, KYC, sanctions screening |
| Temporal Analysis | `temporal` | 7 | Forecasting, change detection, decomposition |
| Risk Analytics | `risk` | 4 | Credit risk, VaR, portfolio risk |

### Priority 2 (Medium)

| Domain | Feature Flag | Kernels | Description |
|--------|-------------|---------|-------------|
| Banking | `banking` | 1 | Fraud detection |
| Behavioral Analytics | `behavioral` | 6 | Profiling, forensics |
| Order Matching | `orderbook` | 1 | HFT order book |
| Process Intelligence | `procint` | 4 | Process mining |
| Clearing | `clearing` | 5 | Settlement, netting |

### Priority 3 (Lower)

| Domain | Feature Flag | Kernels | Description |
|--------|-------------|---------|-------------|
| Treasury Management | `treasury` | 5 | Cash flow, hedging |
| Accounting | `accounting` | 5 | Chart of accounts, reconciliation |
| Payment Processing | `payments` | 2 | Transaction execution |
| Financial Audit | `audit` | 2 | Feature extraction |

## Feature Flags

Enable domains via Cargo features:

```toml
# Default: P1 domains
rustkernel = "0.1"

# Specific domains
rustkernel = { version = "0.1", features = ["graph", "ml", "risk"] }

# All domains
rustkernel = { version = "0.1", features = ["full"] }
```

Available features:
- `default`: graph, ml, compliance, temporal, risk
- `full`: All 14 domains
- Individual: `graph`, `ml`, `compliance`, `temporal`, `risk`, `banking`, `behavioral`, `orderbook`, `procint`, `clearing`, `treasury`, `accounting`, `payments`, `audit`

## Examples

Run the examples:

```bash
# PageRank centrality
cargo run --example graph_pagerank --features graph

# K-Means clustering
cargo run --example ml_kmeans --features ml

# AML pattern detection
cargo run --example compliance_aml --features compliance

# Monte Carlo VaR
cargo run --example risk_var --features risk

# Order matching engine
cargo run --example orderbook_matching --features orderbook
```

## CLI Tool

Install the CLI:

```bash
cargo install --path crates/rustkernel-cli
```

Commands:

```bash
# List all kernels
rustkernel list

# List kernels by domain
rustkernel list --domain graph

# Show kernel info
rustkernel info graph/pagerank

# Show domains
rustkernel domains

# Check system compatibility
rustkernel check --all-backends
```

## Licensing

RustKernels supports tiered licensing:

| Tier | Domains | GPU-Native | Max Kernels |
|------|---------|------------|-------------|
| Development | All | Yes | Unlimited |
| Community | Core, Graph, ML | No | 5 |
| Professional | Configurable | No | 50 |
| Enterprise | All | Yes | Unlimited |

```rust
use rustkernel::core::license::{License, StandardLicenseValidator, LicenseValidator};

// Development (all features, local use)
let dev_license = rustkernel::core::license::dev_license();

// Community (limited features)
let community = License::community("User");
let validator = StandardLicenseValidator::new(community);

// Enterprise (all features)
let enterprise = License::enterprise("Corp", None);
```

## Architecture

```
RustKernels/
├── crates/
│   ├── rustkernel/              # Facade crate
│   ├── rustkernel-core/         # Core traits and registry
│   ├── rustkernel-derive/       # Proc macros
│   ├── rustkernel-graph/        # Graph Analytics
│   ├── rustkernel-ml/           # Statistical ML
│   ├── rustkernel-compliance/   # Compliance
│   ├── rustkernel-temporal/     # Temporal Analysis
│   ├── rustkernel-risk/         # Risk Analytics
│   ├── rustkernel-banking/      # Banking
│   ├── rustkernel-behavioral/   # Behavioral Analytics
│   ├── rustkernel-orderbook/    # Order Matching
│   ├── rustkernel-procint/      # Process Intelligence
│   ├── rustkernel-clearing/     # Clearing
│   ├── rustkernel-treasury/     # Treasury Management
│   ├── rustkernel-accounting/   # Accounting
│   ├── rustkernel-payments/     # Payment Processing
│   ├── rustkernel-audit/        # Financial Audit
│   └── rustkernel-cli/          # CLI tool
└── examples/                    # Usage examples
```

## Benchmarks

Run benchmarks:

```bash
# All benchmarks
cargo bench --package rustkernel

# Specific domain
cargo bench --package rustkernel-graph
```

## Testing

```bash
# All tests
cargo test --workspace

# With all features
cargo test --workspace --all-features

# Specific domain
cargo test --package rustkernel-graph
```

## Requirements

- Rust 1.85+
- RustCompute/RingKernel 0.1+
- Optional: CUDA toolkit (for GPU acceleration)

## License

MIT OR Apache-2.0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `cargo test --workspace`
5. Submit a pull request
