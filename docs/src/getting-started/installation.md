# Installation

This guide covers installing RustKernels and its dependencies.

## Prerequisites

### Rust Toolchain

RustKernels requires **Rust 1.85** or later:

```bash
# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Update to latest stable
rustup update stable

# Verify version
rustc --version  # Should be 1.85.0 or higher
```

### RingKernel Framework

RustKernels depends on [RingKernel 0.4.2](https://crates.io/crates/ringkernel-core) for GPU execution. RingKernel is published on crates.io and is resolved automatically by Cargo — no manual installation is required.

### CUDA Toolkit (Optional)

For GPU acceleration, install the CUDA toolkit:

- **Linux**: Install via your package manager or from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
- **Windows**: Download the installer from NVIDIA
- **macOS**: Not supported for CUDA (CPU fallback only)

```bash
# Verify CUDA installation
nvcc --version
nvidia-smi
```

If CUDA is not available, RustKernels falls back to CPU execution automatically.

## Adding RustKernels to Your Project

### Basic Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernels = "0.4.0"
```

This includes the default feature set: `graph`, `ml`, `compliance`, `temporal`, `risk`.

### Selective Installation

Include only the domains you need to reduce compile time and binary size:

```toml
[dependencies]
rustkernels = { version = "0.4.0", default-features = false, features = ["graph", "accounting"] }
```

### Full Installation

Include all 14 domains:

```toml
[dependencies]
rustkernels = { version = "0.4.0", features = ["full"] }
```

### Service Deployment

For deploying kernels as REST or gRPC services:

```toml
[dependencies]
rustkernel-ecosystem = { version = "0.4.0", features = ["axum", "grpc"] }
```

## Available Features

| Feature | Domain | Description |
|---------|--------|-------------|
| `graph` | Graph Analytics | Centrality, community detection, GNN inference |
| `ml` | Statistical ML | Clustering, anomaly detection, NLP embeddings |
| `compliance` | Compliance | AML, KYC, sanctions screening |
| `temporal` | Temporal Analysis | Forecasting, anomaly detection, decomposition |
| `risk` | Risk Analytics | Credit scoring, VaR, stress testing |
| `banking` | Banking | Fraud pattern detection |
| `behavioral` | Behavioral | Profiling, forensics |
| `orderbook` | Order Matching | Order book engine |
| `procint` | Process Intelligence | DFG, conformance checking, digital twin |
| `clearing` | Clearing | Netting, settlement |
| `treasury` | Treasury | Cash flow, FX hedging |
| `accounting` | Accounting | Network generation, reconciliation |
| `payments` | Payments | Payment processing |
| `audit` | Audit | Feature extraction |
| `full` | All | Enables all domains |

## Building from Source

Clone and build the entire workspace:

```bash
# Clone the repository
git clone https://github.com/mivertowski/RustKernels.git
cd RustKernels

# Build all crates
cargo build --workspace

# Build in release mode
cargo build --workspace --release

# Run all tests (895 tests)
cargo test --workspace

# Lint
cargo clippy --all-targets --all-features -- -D warnings
```

## Verifying Installation

Create a simple test file:

```rust
// src/main.rs
use rustkernels::prelude::*;

fn main() {
    println!("RustKernels v0.4.0 installed successfully!");
    println!("RingKernel 0.4.2 — GPU-native persistent actor runtime");
}
```

Run with:

```bash
cargo run
```

## Troubleshooting

### CUDA Not Detected

If GPU execution is not working:

1. Verify CUDA installation with `nvcc --version`
2. Check GPU availability with `nvidia-smi`
3. Ensure CUDA libraries are in your PATH
4. RustKernels falls back to CPU automatically if CUDA is not available

### Compilation Errors

For Rust version issues:

```bash
# Ensure you are on the correct toolchain
rustup override set stable
rustup update
```

### Dependency Resolution

RingKernel 0.4.2 is resolved from crates.io. If you encounter resolution issues:

```bash
# Update the Cargo registry index
cargo update

# Clear the build cache if needed
cargo clean && cargo build --workspace
```

## Next Steps

- [Quick Start](quick-start.md) — Run your first kernel
- [Execution Modes](../architecture/execution-modes.md) — Understand Batch vs Ring modes
- [Kernel Catalogue](../domains/README.md) — Browse available kernels
