# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RustKernels is a GPU-accelerated kernel library for financial services, analytics, and compliance workloads. It's a Rust port of the DotCompute GPU kernel library, built on the RustCompute (RingKernel) framework.

**Key dependency**: RustCompute is located at `../../RustCompute/RustCompute/` (relative path from workspace root).

## Build Commands

```bash
# Build entire workspace
cargo build --workspace

# Build specific domain crate
cargo build --package rustkernel-graph

# Check with all features
cargo check --all-features

# Format code
cargo fmt --all

# Lint
cargo clippy --all-targets --all-features -- -D warnings
```

## Test Commands

```bash
# Run all tests
cargo test --workspace

# Run tests for specific domain
cargo test --package rustkernel-graph
cargo test --package rustkernel-ml
cargo test --package rustkernel-compliance
cargo test --package rustkernel-risk

# Run single test
cargo test --package rustkernel-graph test_pagerank_metadata

# Run tests with all features
cargo test --workspace --all-features

# Run benchmarks
cargo bench --package rustkernel
```

## Architecture

### Workspace Structure

18 crates organized by concern:

- **`rustkernel`** - Facade crate, re-exports all domains
- **`rustkernel-core`** - Core traits, registry, licensing, K2K coordination
- **`rustkernel-derive`** - Proc macros (`#[gpu_kernel]`, `#[derive(KernelMessage)]`)
- **`rustkernel-cli`** - CLI tool for kernel management
- **14 domain crates** - One per business domain (graph, ml, compliance, etc.)

### Kernel Execution Modes

Two execution modes with different latency/overhead tradeoffs:

1. **Batch Kernels** (`BatchKernel<I, O>` trait)
   - CPU-orchestrated, 10-50μs launch overhead
   - State in CPU memory, launched on-demand
   - Use for heavy periodic computation

2. **Ring Kernels** (`RingKernelHandler<M, R>` trait)
   - GPU-persistent actors, 100-500ns message latency
   - State permanently in GPU memory
   - Use for high-frequency operations

### Core Traits (`rustkernel-core/src/traits.rs`)

```rust
// All kernels implement GpuKernel for metadata
trait GpuKernel: Send + Sync + Debug {
    fn metadata(&self) -> &KernelMetadata;
    fn validate(&self) -> Result<()>;
}

// Batch execution
trait BatchKernel<I, O>: GpuKernel {
    async fn execute(&self, input: I) -> Result<O>;
}

// Ring (persistent actor) execution
trait RingKernelHandler<M, R>: GpuKernel
where M: RingMessage, R: RingMessage {
    async fn handle(&self, ctx: &mut RingContext, msg: M) -> Result<R>;
}

// Multi-pass algorithms (PageRank, K-Means)
trait IterativeKernel<S, I, O>: GpuKernel {
    fn initial_state(&self, input: &I) -> S;
    async fn iterate(&self, state: &mut S, input: &I) -> Result<IterationResult<O>>;
    fn converged(&self, state: &S, threshold: f64) -> bool;
}
```

### K2K (Kernel-to-Kernel) Messaging

Cross-kernel coordination in `rustkernel-core/src/k2k.rs`:

- `IterativeState` - Track convergence across iterations
- `ScatterGatherState` - Parallel worker patterns
- `FanOutTracker` - Broadcast patterns
- `PipelineTracker` - Multi-stage processing

### Ring Message Type IDs

Each domain has a reserved range for Ring message type IDs:
- Graph: 200-299
- Compliance: 300-399
- Temporal: 400-499
- Risk: 600-699
- ML: 700-799

### Domain Crate Structure

Each domain crate follows this pattern:
```
rustkernel-{domain}/
├── src/
│   ├── lib.rs           # Module exports, register_all()
│   ├── messages.rs      # Batch kernel input/output types
│   ├── ring_messages.rs # Ring message types with #[derive(RingMessage)]
│   ├── types.rs         # Common domain types
│   └── {feature}.rs     # Kernel implementations
```

### Ring Message Definition Pattern

```rust
use ringkernel_derive::RingMessage;
use rkyv::{Archive, Serialize, Deserialize};

#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 200)]  // Unique within domain range
pub struct MyRequest {
    #[message(id)]
    pub id: MessageId,
    pub data: u64,
}
```

**Important**: `MessageId` is a tuple struct. Use `MessageId(value)` not `MessageId::new()`.

### Fixed-Point Arithmetic

Ring messages use fixed-point for GPU-compatible numerics:
```rust
// 8 decimal places
fn to_fixed_point(value: f64) -> i64 { (value * 100_000_000.0) as i64 }
fn from_fixed_point(fp: i64) -> f64 { fp as f64 / 100_000_000.0 }
```

## Adding a New Kernel

1. Define kernel struct implementing `GpuKernel`
2. Implement `BatchKernel<I, O>` or `RingKernelHandler<M, R>`
3. Add input/output types to `messages.rs`
4. For Ring kernels, add messages to `ring_messages.rs` with unique type IDs
5. Add tests

## Licensing System

Enterprise licensing in `rustkernel-core/src/license.rs`:
- `DevelopmentLicense` - All features enabled (default for local dev)
- Domain-based validation via `LicenseValidator` trait
- Feature gating at kernel registration and activation time
