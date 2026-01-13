# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RustKernels is a GPU-accelerated kernel library for financial services, analytics, and compliance workloads. It's a Rust port of the DotCompute GPU kernel library, built on the RustCompute (RingKernel) framework.

**Current State**: 106 kernels across 14 domain crates, fully implemented with both Batch and Ring execution modes.

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
cargo test --package rustkernel-procint

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
- **14 domain crates** - One per business domain

### Domain Crates and Kernel Counts

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
5. Register in `lib.rs` `register_all()` function
6. Update test count in `lib.rs` test
7. Add comprehensive tests

### Example: Adding a Batch Kernel

```rust
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

#[derive(Debug)]
pub struct MyNewKernel {
    metadata: KernelMetadata,
}

impl MyNewKernel {
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("domain/my-kernel", Domain::MyDomain)
                .with_description("Description of what this kernel does")
                .with_throughput(10_000)
                .with_latency_us(100.0),
        }
    }

    pub fn compute(input: &MyInput) -> MyOutput {
        // Implementation
    }
}

impl GpuKernel for MyNewKernel {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }

    fn validate(&self) -> rustkernel_core::error::Result<()> {
        Ok(())
    }
}
```

## Licensing System

Enterprise licensing in `rustkernel-core/src/license.rs`:
- `DevelopmentLicense` - All features enabled (default for local dev)
- Domain-based validation via `LicenseValidator` trait
- Feature gating at kernel registration and activation time

## Recent Kernel Additions

The following kernel categories were recently added:

### Graph (rustkernel-graph)
- `GNNInference` - Message passing neural network inference
- `GraphAttention` - Graph Attention Network with multi-head attention

### ML (rustkernel-ml)
- `EmbeddingGeneration` - Hash-based text embeddings for NLP
- `SemanticSimilarity` - Multi-metric semantic similarity search
- `SecureAggregation` - Federated learning with differential privacy
- `DrugInteractionPrediction` - Multi-drug interaction prediction
- `ClinicalPathwayConformance` - Treatment guideline checking
- `StreamingIsolationForest` - Online anomaly detection
- `AdaptiveThreshold` - Self-adjusting thresholds with drift detection
- `SHAPValues` - Kernel SHAP for feature explanations
- `FeatureImportance` - Permutation-based feature importance

### Process Intelligence (rustkernel-procint)
- `DigitalTwin` - Monte Carlo process simulation
- `NextActivityPrediction` - Markov/N-gram next activity prediction
- `EventLogImputation` - Event log quality detection and repair

## Documentation

- **Docs Site**: `docs/` directory contains mdBook documentation
- **Build Docs**: `cd docs && mdbook build`
- **Serve Locally**: `cd docs && mdbook serve`
- **GitHub Pages**: Deployed automatically via GitHub Actions
