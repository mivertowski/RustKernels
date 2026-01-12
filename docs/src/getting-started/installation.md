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

### RustCompute Framework

RustKernels depends on the RustCompute (RingKernel) framework for GPU execution:

```bash
# Clone RustCompute alongside RustKernels
cd /path/to/your/projects
git clone https://github.com/mivertowski/RustCompute.git

# Directory structure should be:
# projects/
# ├── RustCompute/
# │   └── RustCompute/
# └── RustKernels/
#     └── RustKernels/
```

### CUDA Toolkit (Optional)

For GPU acceleration, install the CUDA toolkit:

- **Linux**: Install via your package manager or from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
- **Windows**: Download installer from NVIDIA
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
rustkernel = "0.1.0"
```

This includes the default feature set: `graph`, `ml`, `compliance`, `temporal`, `risk`.

### Selective Installation

Only include the domains you need to reduce compile time and binary size:

```toml
[dependencies]
rustkernel = { version = "0.1.0", default-features = false, features = ["graph", "accounting"] }
```

### Full Installation

Include all 14 domains:

```toml
[dependencies]
rustkernel = { version = "0.1.0", features = ["full"] }
```

## Available Features

| Feature | Domain | Description |
|---------|--------|-------------|
| `graph` | Graph Analytics | Centrality, community detection, similarity |
| `ml` | Statistical ML | Clustering, anomaly detection, regression |
| `compliance` | Compliance | AML, KYC, sanctions screening |
| `temporal` | Temporal Analysis | Forecasting, anomaly detection |
| `risk` | Risk Analytics | Credit scoring, VaR, stress testing |
| `banking` | Banking | Fraud pattern detection |
| `behavioral` | Behavioral | Profiling, forensics |
| `orderbook` | Order Matching | Order book engine |
| `procint` | Process Intelligence | DFG, conformance checking |
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

# Run tests
cargo test --workspace
```

## Verifying Installation

Create a simple test file:

```rust
// src/main.rs
use rustkernel::prelude::*;

fn main() {
    println!("RustKernels installed successfully!");

    // List available domains
    let domains = [
        "Graph Analytics",
        "Statistical ML",
        "Compliance",
        "Temporal Analysis",
        "Risk Analytics",
    ];

    for domain in domains {
        println!("  - {}", domain);
    }
}
```

Run with:

```bash
cargo run
```

## Troubleshooting

### RustCompute Not Found

If you see path errors related to RustCompute:

1. Ensure RustCompute is cloned at the expected location
2. Check that the directory structure matches what's expected in `Cargo.toml`
3. Verify the RustCompute workspace builds independently

### CUDA Not Detected

If GPU execution isn't working:

1. Verify CUDA installation with `nvcc --version`
2. Check GPU availability with `nvidia-smi`
3. Ensure CUDA libraries are in your PATH
4. RustKernels will fall back to CPU if CUDA isn't available

### Compilation Errors

For Rust version issues:

```bash
# Ensure you're on the correct toolchain
rustup override set stable
rustup update
```

## Next Steps

- [Quick Start](quick-start.md) - Run your first kernel
- [Execution Modes](../architecture/execution-modes.md) - Understand Batch vs Ring modes
- [Kernel Catalogue](../domains/README.md) - Browse available kernels
