# rustkernel-derive

[![Crates.io](https://img.shields.io/crates/v/rustkernel-derive.svg)](https://crates.io/crates/rustkernel-derive)
[![Documentation](https://docs.rs/rustkernel-derive/badge.svg)](https://docs.rs/rustkernel-derive)
[![License](https://img.shields.io/crates/l/rustkernel-derive.svg)](LICENSE)

Procedural macros for the RustKernels GPU kernel library.

## Features

- **`#[gpu_kernel]`**: Define a GPU kernel with metadata
- **`#[derive(KernelMessage)]`**: Derive serialization for kernel messages

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernel-derive = "0.1.0"
```

## Usage

```rust
use rustkernel_derive::gpu_kernel;

#[gpu_kernel(
    id = "graph/pagerank",
    mode = "ring",
    domain = "GraphAnalytics",
    throughput = 100_000,
    latency_us = 1.0
)]
pub struct PageRank {
    // kernel fields
}
```

## License

Apache-2.0
