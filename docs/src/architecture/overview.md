# Architecture Overview

RustKernels is designed as a modular, high-performance GPU kernel library for financial and enterprise workloads. This document explains the system architecture and key design decisions.

## System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        rustkernel (facade)                       │
│                    Re-exports all domain crates                  │
└─────────────────────────────────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ rustkernel-core │   │rustkernel-derive│   │  rustkernel-cli │
│                 │   │                 │   │                 │
│ - Traits        │   │ - #[gpu_kernel] │   │ - CLI tool      │
│ - Registry      │   │ - #[derive(...)]│   │ - Management    │
│ - K2K messaging │   │                 │   │                 │
│ - Licensing     │   │                 │   │                 │
└─────────────────┘   └─────────────────┘   └─────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     14 Domain Crates                             │
│                                                                  │
│  graph  │  ml  │ compliance │ temporal │ risk │ banking │ ...   │
│                                                                  │
│  Each domain implements domain-specific kernels using the core   │
│  traits and infrastructure                                       │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RustCompute (RingKernel)                      │
│                    GPU execution framework                       │
└─────────────────────────────────────────────────────────────────┘
```

## Workspace Structure

The workspace contains 18 crates organized by concern:

### Infrastructure Crates

| Crate | Purpose |
|-------|---------|
| `rustkernel` | Facade crate, re-exports all domains |
| `rustkernel-core` | Core traits, registry, licensing, K2K coordination |
| `rustkernel-derive` | Procedural macros for kernel definition |
| `rustkernel-cli` | Command-line interface for kernel management |

### Domain Crates

14 domain-specific crates, each containing kernels for a particular business area:

```
crates/
├── rustkernel-graph/        # Graph analytics (21 kernels)
├── rustkernel-ml/           # Statistical ML (8 kernels)
├── rustkernel-compliance/   # AML/KYC (9 kernels)
├── rustkernel-temporal/     # Time series (7 kernels)
├── rustkernel-risk/         # Risk analytics (4 kernels)
├── rustkernel-banking/      # Banking (1 kernel)
├── rustkernel-behavioral/   # Behavioral (6 kernels)
├── rustkernel-orderbook/    # Order matching (1 kernel)
├── rustkernel-procint/      # Process intelligence (4 kernels)
├── rustkernel-clearing/     # Clearing/settlement (5 kernels)
├── rustkernel-treasury/     # Treasury (5 kernels)
├── rustkernel-accounting/   # Accounting (7 kernels)
├── rustkernel-payments/     # Payments (2 kernels)
└── rustkernel-audit/        # Audit (2 kernels)
```

## Core Traits

All kernels are built on a set of core traits defined in `rustkernel-core`:

### GpuKernel

The base trait for all kernels:

```rust
pub trait GpuKernel: Send + Sync + Debug {
    /// Returns kernel metadata (ID, domain, mode, etc.)
    fn metadata(&self) -> &KernelMetadata;

    /// Validates kernel configuration
    fn validate(&self) -> Result<()>;
}
```

### BatchKernel

For CPU-orchestrated batch execution:

```rust
pub trait BatchKernel<I, O>: GpuKernel {
    /// Execute the kernel with given input
    async fn execute(&self, input: I) -> Result<O>;
}
```

### RingKernelHandler

For GPU-persistent actor execution:

```rust
pub trait RingKernelHandler<M, R>: GpuKernel
where
    M: RingMessage,
    R: RingMessage,
{
    /// Handle a message and produce a response
    async fn handle(&self, ctx: &mut RingContext, msg: M) -> Result<R>;
}
```

### IterativeKernel

For multi-pass algorithms (PageRank, K-Means, etc.):

```rust
pub trait IterativeKernel<S, I, O>: GpuKernel {
    /// Create initial state from input
    fn initial_state(&self, input: &I) -> S;

    /// Perform one iteration
    async fn iterate(&self, state: &mut S, input: &I) -> Result<IterationResult<O>>;

    /// Check if algorithm has converged
    fn converged(&self, state: &S, threshold: f64) -> bool;
}
```

## Kernel Metadata

Every kernel has associated metadata:

```rust
pub struct KernelMetadata {
    /// Unique identifier (e.g., "graph/pagerank")
    pub id: String,

    /// Execution mode
    pub mode: KernelMode,

    /// Business domain
    pub domain: Domain,

    /// Human-readable description
    pub description: String,

    /// Expected throughput (ops/sec)
    pub expected_throughput: u64,

    /// Target latency in microseconds
    pub target_latency_us: f64,

    /// Whether GPU-native execution is required
    pub requires_gpu_native: bool,

    /// Kernel version
    pub version: u32,
}
```

## K2K (Kernel-to-Kernel) Messaging

RustKernels supports cross-kernel coordination through K2K messaging patterns:

### Available Patterns

| Pattern | Use Case |
|---------|----------|
| `IterativeState` | Track convergence across iterations |
| `ScatterGatherState` | Parallel worker coordination |
| `FanOutTracker` | Broadcast patterns |
| `PipelineTracker` | Multi-stage processing |

### Example: Iterative Coordination

```rust
use rustkernel_core::k2k::IterativeState;

let mut state = IterativeState::new(max_iterations);

while !state.converged() {
    // Execute iteration across kernels
    let results = execute_iteration(&mut state).await?;

    // Update convergence tracking
    state.update(results.delta);
}
```

## Domain Crate Structure

Each domain crate follows a consistent structure:

```
rustkernel-{domain}/
├── Cargo.toml
└── src/
    ├── lib.rs           # Module exports, register_all()
    ├── messages.rs      # Batch kernel input/output types
    ├── ring_messages.rs # Ring message types
    ├── types.rs         # Common domain types
    └── {feature}.rs     # Kernel implementations
```

### Example: Graph Analytics Crate

```
rustkernel-graph/
└── src/
    ├── lib.rs
    ├── messages.rs
    ├── ring_messages.rs
    ├── types.rs
    ├── centrality.rs    # PageRank, Betweenness, etc.
    ├── community.rs     # Louvain, Label Propagation
    ├── similarity.rs    # Jaccard, Cosine, Adamic-Adar
    ├── metrics.rs       # Density, Clustering Coefficient
    └── motif.rs         # Triangle counting, k-cliques
```

## Ring Message Type IDs

Each domain has a reserved range for Ring message type IDs to avoid collisions:

| Domain | Range |
|--------|-------|
| Graph | 200-299 |
| Compliance | 300-399 |
| Temporal | 400-499 |
| Risk | 600-699 |
| ML | 700-799 |

## Licensing System

RustKernels includes an enterprise licensing system:

- **DevelopmentLicense**: All features enabled (default for local development)
- **ProductionLicense**: Domain-based feature gating
- Validation occurs at kernel registration and activation time

```rust
use rustkernel_core::license::{LicenseValidator, DevelopmentLicense};

let validator = DevelopmentLicense::new();
assert!(validator.is_domain_licensed(Domain::GraphAnalytics));
```

## Fixed-Point Arithmetic

For GPU-compatible and exact financial calculations, Ring messages use fixed-point arithmetic:

```rust
// 18 decimal places (accounting kernels)
const SCALE: i128 = 1_000_000_000_000_000_000;

pub struct FixedPoint128 {
    pub value: i128,
}

impl FixedPoint128 {
    pub fn from_f64(v: f64) -> Self {
        Self { value: (v * SCALE as f64) as i128 }
    }

    pub fn to_f64(&self) -> f64 {
        self.value as f64 / SCALE as f64
    }
}
```

## Next Steps

- [Execution Modes](execution-modes.md) - Deep dive into Batch vs Ring
- [Kernel Catalogue](../domains/README.md) - Browse available kernels
- [Quick Start](../getting-started/quick-start.md) - Run your first kernel
