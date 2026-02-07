# Architecture Overview

RustKernels is a modular, high-performance GPU kernel library for financial and enterprise workloads. This document describes the system architecture and key design decisions.

## System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                       rustkernels (facade)                       │
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
│ - Enterprise    │   │                 │   │                 │
│   modules       │   │                 │   │                 │
└─────────────────┘   └─────────────────┘   └─────────────────┘
          │
          ├──────────────────────────────────────┐
          │                                      │
          ▼                                      ▼
┌───────────────────────────────────┐   ┌─────────────────┐
│          14 Domain Crates         │   │   rustkernel-   │
│                                   │   │   ecosystem     │
│  graph │ ml │ compliance │ risk  │   │                 │
│  temporal │ banking │ procint    │   │ - Axum REST     │
│  behavioral │ orderbook │ ...   │   │ - Tower         │
│                                   │   │ - Tonic gRPC    │
│  Each implements domain-specific  │   │ - Actix actors  │
│  kernels using core traits        │   │                 │
└───────────────────────────────────┘   └─────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RingKernel 0.4.2 (crates.io)                  │
│          GPU-native persistent actor runtime framework           │
└─────────────────────────────────────────────────────────────────┘
```

## Workspace Structure

The workspace contains **19 crates** organized by concern:

### Infrastructure Crates

| Crate | Purpose |
|-------|---------|
| `rustkernels` | Facade crate — re-exports all domains |
| `rustkernel-core` | Core traits, registry, licensing, K2K coordination, enterprise modules |
| `rustkernel-derive` | Procedural macros for kernel definition |
| `rustkernel-ecosystem` | Service integrations (Axum, Tower, Tonic, Actix) |
| `rustkernel-cli` | Command-line interface for kernel management |

### Domain Crates

14 domain-specific crates, each containing kernels for a particular business area:

```
crates/
├── rustkernel-graph/        # Graph analytics (28 kernels)
├── rustkernel-ml/           # Statistical ML (17 kernels)
├── rustkernel-compliance/   # AML/KYC (11 kernels)
├── rustkernel-temporal/     # Time series (7 kernels)
├── rustkernel-risk/         # Risk analytics (5 kernels)
├── rustkernel-banking/      # Banking (1 kernel)
├── rustkernel-behavioral/   # Behavioral (6 kernels)
├── rustkernel-orderbook/    # Order matching (1 kernel)
├── rustkernel-procint/      # Process intelligence (7 kernels)
├── rustkernel-clearing/     # Clearing/settlement (5 kernels)
├── rustkernel-treasury/     # Treasury (5 kernels)
├── rustkernel-accounting/   # Accounting (9 kernels)
├── rustkernel-payments/     # Payments (2 kernels)
└── rustkernel-audit/        # Audit (2 kernels)
```

## Core Traits

All kernels are built on a set of core traits defined in `rustkernel-core`:

### GpuKernel

The base trait for all kernels:

```rust
pub trait GpuKernel: Send + Sync + Debug {
    /// Returns kernel metadata (ID, domain, mode, performance targets)
    fn metadata(&self) -> &KernelMetadata;

    /// Validates kernel configuration
    fn validate(&self) -> Result<()>;

    /// Health check (enterprise)
    fn health_check(&self) -> HealthStatus { HealthStatus::Healthy }

    /// Graceful shutdown
    async fn shutdown(&self) -> Result<()> { Ok(()) }

    /// Hot-reload configuration
    fn refresh_config(&mut self, config: &KernelConfig) -> Result<()> { Ok(()) }
}
```

### BatchKernel

For CPU-orchestrated batch execution:

```rust
pub trait BatchKernel<I, O>: GpuKernel {
    /// Execute the kernel with typed input
    async fn execute(&self, input: I) -> Result<O>;

    /// Execute with auth, tenant, and tracing context
    async fn execute_with_context(&self, ctx: &ExecutionContext, input: I) -> Result<O>;

    /// Validate input before execution
    fn validate_input(&self, input: &I) -> Result<()> { Ok(()) }
}
```

### BatchKernelDyn and TypeErasedBatchKernel

For type-erased execution via REST/gRPC:

```rust
/// Dynamic dispatch trait — JSON bytes in, JSON bytes out
pub trait BatchKernelDyn: GpuKernel {
    async fn execute_dyn(&self, input: &[u8]) -> Result<Vec<u8>>;
}

/// Bridges typed BatchKernel<I,O> to BatchKernelDyn via JSON serialization
pub struct TypeErasedBatchKernel<K, I, O> { /* ... */ }
```

Kernels registered via `register_batch_typed()` are automatically wrapped in `TypeErasedBatchKernel`, enabling execution through the ecosystem service layer without compile-time knowledge of input and output types.

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

    /// Handle with security context
    async fn handle_secure(&self, ctx: &mut SecureRingContext, msg: M) -> Result<R>;
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

    /// Check convergence
    fn converged(&self, state: &S, threshold: f64) -> bool;
}
```

### Additional Traits

| Trait | Purpose |
|-------|---------|
| `CheckpointableKernel` | Save/restore kernel state for recovery |
| `DegradableKernel` | Graceful degradation under pressure |

## Kernel Registration

The `KernelRegistry` provides three registration methods:

| Method | Use Case |
|--------|----------|
| `register_batch_typed(factory)` | Kernels with `BatchKernel<I, O>` — full execution support via REST/gRPC |
| `register_batch_metadata_from(factory)` | Batch kernels with `GpuKernel` only — metadata and discovery |
| `register_ring_metadata_from(factory)` | Ring kernels — metadata only (require Ring runtime for execution) |

Example:

```rust
pub fn register_all(registry: &KernelRegistry) -> Result<()> {
    // Full execution support — callable via REST/gRPC
    registry.register_batch_typed(BetweennessCentrality::new)?;

    // Metadata-only — discoverable but not directly executable via REST
    registry.register_batch_metadata_from(GraphDensity::new)?;

    // Ring kernel — requires RingKernel runtime
    registry.register_ring_metadata_from(PageRankRing::new)?;

    Ok(())
}
```

## Kernel Metadata

Every kernel carries associated metadata:

```rust
pub struct KernelMetadata {
    pub id: String,                  // e.g., "graph/pagerank"
    pub mode: KernelMode,           // Batch or Ring
    pub domain: Domain,             // Business domain
    pub description: String,        // Human-readable description
    pub expected_throughput: u64,    // Operations per second
    pub target_latency_us: f64,     // Target latency in microseconds
    pub requires_gpu_native: bool,  // GPU-only or CPU fallback available
    pub version: u32,               // Kernel implementation version
}
```

## K2K (Kernel-to-Kernel) Messaging

Cross-kernel coordination patterns for complex multi-stage computations:

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
    let results = execute_iteration(&mut state).await?;
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
    ├── ring_messages.rs # Ring message types with #[derive(RingMessage)]
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
    ├── centrality.rs    # PageRank, Betweenness, Closeness, etc.
    ├── community.rs     # Louvain, Label Propagation
    ├── similarity.rs    # Jaccard, Cosine, Adamic-Adar
    ├── metrics.rs       # Density, Clustering Coefficient
    ├── motif.rs         # Triangle counting, k-cliques
    ├── topology.rs      # Connected components, cycles, paths
    └── gnn.rs           # GNN inference, graph attention
```

## Ring Message Type IDs

Each domain has a reserved range for Ring message type IDs, aligned with `ringkernel_core::domain::Domain` base offsets (0.4.2):

| Domain | Range | RingKernel Domain |
|--------|-------|-------------------|
| Graph Analytics | 100–199 | `GraphAnalytics` |
| Statistical ML | 200–299 | `StatisticalML` |
| Compliance | 300–399 | `Compliance` |
| Risk Analytics | 400–499 | `RiskManagement` |
| Temporal Analysis | 500–599 | `TimeSeries` |
| Order Matching | 600–699 | `OrderMatching` |
| Clearing | 700–799 | `Clearing` |

## RingKernel 0.4.2 Integration

RustKernels 0.4.0 deeply integrates with RingKernel 0.4.2:

### Domain Conversion

Bidirectional conversion between RustKernels and RingKernel domain types:

```rust
use rustkernel_core::domain::Domain;

let domain = Domain::TemporalAnalysis;
let ring_domain = domain.to_ring_domain();  // → ringkernel_core::domain::Domain::TimeSeries
let back = Domain::from_ring_domain(ring_domain);  // → Domain::TemporalAnalysis
```

### Re-exports from RingKernel

| Type | Description |
|------|-------------|
| `ControlBlock` | GPU control block for persistent kernel state |
| `Backend` | Runtime backend selection (CUDA, CPU, WebGPU) |
| `KernelStatus` | Detailed kernel status information |
| `RuntimeMetrics` | Runtime performance metrics |
| `K2KConfig` | Kernel-to-kernel messaging configuration |
| `Priority` | Message priority levels |

### Submodule Re-exports

| Module | Description |
|--------|-------------|
| `rustkernel_core::checkpoint` | Kernel checkpointing and recovery |
| `rustkernel_core::dispatcher` | Message dispatching |
| `rustkernel_core::health` | Health checking (circuit breaker, degradation) |
| `rustkernel_core::pubsub` | Pub/sub messaging patterns |

## Licensing System

Enterprise licensing in `rustkernel-core/src/license.rs`:

- **DevelopmentLicense**: All features enabled (default for local development)
- **ProductionLicense**: Domain-based feature gating
- Validation occurs at kernel registration and activation time

```rust
use rustkernel_core::license::{LicenseValidator, DevelopmentLicense};

let validator = DevelopmentLicense::new();
assert!(validator.is_domain_licensed(Domain::GraphAnalytics));
```

## Fixed-Point Arithmetic

For GPU-compatible exact financial calculations, Ring messages use fixed-point arithmetic:

```rust
// 8 decimal places (standard kernels)
fn to_fixed_point(value: f64) -> i64 { (value * 100_000_000.0) as i64 }
fn from_fixed_point(fp: i64) -> f64 { fp as f64 / 100_000_000.0 }

// 18 decimal places (accounting kernels)
const SCALE: i128 = 1_000_000_000_000_000_000;

pub struct FixedPoint128 {
    pub value: i128,
}
```

## Next Steps

- [Execution Modes](execution-modes.md) — Deep dive into Batch vs Ring
- [Kernel Catalogue](../domains/README.md) — Browse available kernels
- [Quick Start](../getting-started/quick-start.md) — Run your first kernel
