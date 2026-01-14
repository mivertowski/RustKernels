# rustkernel-core

[![Crates.io](https://img.shields.io/crates/v/rustkernel-core.svg)](https://crates.io/crates/rustkernel-core)
[![Documentation](https://docs.rs/rustkernel-core/badge.svg)](https://docs.rs/rustkernel-core)
[![License](https://img.shields.io/crates/l/rustkernel-core.svg)](LICENSE)

Core abstractions, traits, and registry for the RustKernels GPU kernel library.

## Features

- **Domain and Kernel Types**: Type-safe domain categorization and kernel metadata
- **Kernel Traits**: `BatchKernel` and `RingKernelHandler` trait definitions
- **Kernel Registry**: Auto-discovery and registration of kernels
- **Licensing System**: Enterprise licensing and feature gating
- **K2K Coordination**: Kernel-to-kernel messaging patterns (iterative, scatter-gather, fan-out, pipeline)
- **Actix Integration**: GPU-backed actor support

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernel-core = "0.1.0"
```

## Usage

```rust
use rustkernel_core::{
    domain::Domain,
    kernel::KernelMetadata,
    traits::{GpuKernel, BatchKernel},
};

// Define a kernel with metadata
let metadata = KernelMetadata::batch("my-domain/my-kernel", Domain::GraphAnalytics)
    .with_description("My custom kernel")
    .with_throughput(10_000)
    .with_latency_us(100.0);
```

## License

Apache-2.0
