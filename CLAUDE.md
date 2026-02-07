# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RustKernels is a GPU-accelerated kernel library for financial services, analytics, and compliance workloads. It's a Rust port of the DotCompute GPU kernel library, built on the RustCompute (RingKernel) framework.

**Current State**: 106 kernels across 14 domain crates, fully implemented with both Batch and Ring execution modes.

**Version**: 0.4.0 - Deep integration with RingKernel 0.4.2. Enterprise-ready with security, observability, resilience, and service APIs.

**Key dependency**: RingKernel 0.4.2 (crates.io) - GPU-native persistent actor runtime with enterprise features.

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

19 crates organized by concern:

- **`rustkernel`** - Facade crate, re-exports all domains
- **`rustkernel-core`** - Core traits, registry, licensing, K2K coordination, enterprise modules
- **`rustkernel-derive`** - Proc macros (`#[gpu_kernel]`, `#[derive(KernelMessage)]`)
- **`rustkernel-ecosystem`** - REST/gRPC service integrations (Axum, Tower, Tonic, Actix)
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
    fn health_check(&self) -> HealthStatus { HealthStatus::Healthy }
    async fn shutdown(&self) -> Result<()> { Ok(()) }
    fn refresh_config(&mut self, config: &KernelConfig) -> Result<()> { Ok(()) }
}

// Batch execution
trait BatchKernel<I, O>: GpuKernel {
    async fn execute(&self, input: I) -> Result<O>;
    // With auth/tenant/tracing context
    async fn execute_with_context(&self, ctx: &ExecutionContext, input: I) -> Result<O>;
}

// Ring (persistent actor) execution
trait RingKernelHandler<M, R>: GpuKernel
where M: RingMessage, R: RingMessage {
    async fn handle(&self, ctx: &mut RingContext, msg: M) -> Result<R>;
    // With security context
    async fn handle_secure(&self, ctx: &mut SecureRingContext, msg: M) -> Result<R>;
}

// Multi-pass algorithms (PageRank, K-Means)
trait IterativeKernel<S, I, O>: GpuKernel {
    fn initial_state(&self, input: &I) -> S;
    async fn iterate(&self, state: &mut S, input: &I) -> Result<IterationResult<O>>;
    fn converged(&self, state: &S, threshold: f64) -> bool;
}

// Checkpointable for recovery
trait CheckpointableKernel: GpuKernel {
    type Checkpoint: Serialize + DeserializeOwned;
    async fn checkpoint(&self) -> Result<Self::Checkpoint>;
    async fn restore(&mut self, checkpoint: Self::Checkpoint) -> Result<()>;
}

// Graceful degradation
trait DegradableKernel: GpuKernel {
    fn degrade(&mut self, level: DegradationLevel);
    fn current_degradation(&self) -> DegradationLevel;
}
```

### K2K (Kernel-to-Kernel) Messaging

Cross-kernel coordination in `rustkernel-core/src/k2k.rs`:

- `IterativeState` - Track convergence across iterations
- `ScatterGatherState` - Parallel worker patterns
- `FanOutTracker` - Broadcast patterns
- `PipelineTracker` - Multi-stage processing

### Ring Message Type IDs

Each domain has a reserved range for Ring message type IDs, aligned with
`ringkernel_core::domain::Domain` base offsets (0.4.2):

- Graph (GraphAnalytics): 100-199
- ML (StatisticalML): 200-299
- Compliance: 300-399
- Risk (RiskManagement): 400-499
- OrderMatching: 500-599
- Temporal (TimeSeries): 1100-1199

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
#[message(type_id = 100)]  // Unique within domain range (GraphAnalytics: 100-199)
pub struct MyRequest {
    #[message(id)]
    pub id: MessageId,
    pub data: u64,
}
```

**Important**: `MessageId` supports both `MessageId(value)` and `MessageId::new(value)` (0.4.2+). For auto-generated IDs, use `MessageId::generate()`.

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

## RingKernel 0.4.2 Integration

RustKernels 0.4.0 deeply integrates with RingKernel 0.4.2:

### Domain Conversion

Bidirectional conversion between `rustkernel_core::domain::Domain` and `ringkernel_core::domain::Domain`:

```rust
use rustkernel_core::domain::Domain;

let domain = Domain::TemporalAnalysis;
let ring_domain = domain.to_ring_domain(); // → ringkernel_core::domain::Domain::TimeSeries
let back = Domain::from_ring_domain(ring_domain); // → Domain::TemporalAnalysis

// Naming differences:
// TemporalAnalysis ↔ TimeSeries
// RiskAnalytics ↔ RiskManagement
// Core ↔ General
```

### Direct RingKernel Access

For advanced usage, the full ringkernel-core 0.4.2 API is available:

```rust
use rustkernel_core::ring; // Full ringkernel_core re-export

// New 0.4.2 types in prelude
use rustkernel_core::prelude::{Backend, KernelStatus, RuntimeMetrics, ControlBlock, K2KConfig, Priority};

// Enterprise re-exports from ringkernel-core in each module:
use rustkernel_core::security::ring_security;
use rustkernel_core::observability::ring_observability;
use rustkernel_core::resilience::ring_health;
use rustkernel_core::memory::ring_memory;
```

### New Re-exports

Top-level re-exports from ringkernel-core 0.4.2:

- `ControlBlock` - GPU control block for persistent kernel state
- `Backend` - Runtime backend selection (CUDA, CPU, WebGPU)
- `KernelStatus` - Detailed kernel status information
- `RuntimeMetrics` - Runtime performance metrics
- `K2KConfig` - Kernel-to-kernel messaging configuration
- `DeliveryStatus` - K2K message delivery tracking
- `Priority` - Message priority levels

Submodule re-exports:
- `rustkernel_core::checkpoint` - Kernel checkpointing
- `rustkernel_core::dispatcher` - Message dispatching
- `rustkernel_core::health` - Health checking (circuit breaker, degradation)
- `rustkernel_core::pubsub` - Pub/sub messaging patterns

## Enterprise Modules

### Security (`rustkernel-core/src/security/`)

- **Authentication**: JWT and API key validation via `AuthConfig`
- **RBAC**: Role-based access control with `KernelPermission` (Execute, Configure, Monitor, Admin)
- **Multi-tenancy**: Tenant isolation with `TenantId` and resource quotas
- **Secrets**: `SecretStore` abstraction for credential management

```rust
use rustkernel_core::security::{SecurityContext, AuthConfig, Role};

let ctx = SecurityContext::new(user_id, tenant_id)
    .with_roles(vec![Role::KernelExecutor])
    .with_permissions(vec![KernelPermission::Execute]);
```

### Observability (`rustkernel-core/src/observability/`)

- **Metrics**: Prometheus-compatible metrics via `KernelMetrics`
- **Tracing**: Distributed tracing with OTLP export via `KernelTracing`
- **Logging**: Structured logging with kernel context
- **Alerting**: SLO-based alerts with `AlertRule`

```rust
use rustkernel_core::observability::{ObservabilityConfig, MetricsConfig};

let config = ObservabilityConfig::production()
    .with_metrics(MetricsConfig::default())
    .with_tracing_enabled(true);
```

### Resilience (`rustkernel-core/src/resilience/`)

- **Circuit Breaker**: Failure isolation with `CircuitBreaker`
- **Retry**: Exponential backoff with jitter via `RetryConfig`
- **Timeouts**: Deadline propagation with `DeadlineContext`
- **Health Checks**: Liveness/readiness probes via `HealthProbe`

```rust
use rustkernel_core::resilience::{CircuitBreaker, CircuitBreakerConfig};

let cb = CircuitBreaker::new(CircuitBreakerConfig {
    failure_threshold: 5,
    success_threshold: 2,
    timeout: Duration::from_secs(30),
    ..Default::default()
});
```

### Runtime (`rustkernel-core/src/runtime/`)

- **Lifecycle**: State machine with `LifecycleState` (Starting, Running, Draining, Stopped)
- **Configuration**: `RuntimeConfig` with presets (development, production, high-performance)
- **Graceful Shutdown**: Drain period and connection tracking

```rust
use rustkernel_core::runtime::{RuntimeBuilder, RuntimePreset};

let runtime = RuntimeBuilder::new()
    .preset(RuntimePreset::Production)
    .with_graceful_shutdown(Duration::from_secs(30))
    .build()?;
```

### Memory (`rustkernel-core/src/memory/`)

- **Pooling**: Size-stratified memory pools via `KernelMemoryManager`
- **Pressure Handling**: Configurable thresholds with `PressureLevel`
- **Reductions**: Multi-phase GPU reductions via `InterPhaseReduction`
- **Analytics Contexts**: Workload-specific buffers with `AnalyticsContextManager`

```rust
use rustkernel_core::memory::{MemoryConfig, KernelMemoryManager};

let config = MemoryConfig::high_performance();
let manager = KernelMemoryManager::new(config);
```

### Production Config (`rustkernel-core/src/config/`)

Unified configuration for production deployments:

```rust
use rustkernel_core::config::{ProductionConfig, ProductionConfigBuilder};

// From environment
let config = ProductionConfig::from_env()?;

// From file
let config = ProductionConfig::from_file("config/production.toml")?;

// Using builder
let config = ProductionConfigBuilder::production()
    .service_name("my-service")
    .environment("staging")
    .build()?;
```

### Ecosystem (`rustkernel-ecosystem/`)

Service integrations for deploying RustKernels as a standalone service:

- **Axum**: REST API with `KernelRouter` (endpoints: `/kernels`, `/execute`, `/health`, `/metrics`)
- **Tower**: Middleware (`TimeoutLayer`, `RateLimiterLayer`, `KernelService`)
- **Tonic**: gRPC server via `KernelGrpcServer`
- **Actix**: Actor integration via `KernelActor`

```rust
use rustkernel_ecosystem::axum::{KernelRouter, RouterConfig};

let router = KernelRouter::new(registry, RouterConfig::default());
let app = router.into_router();
// Serve with axum::Server
```

## Documentation

- **Docs Site**: `docs/` directory contains mdBook documentation
- **Build Docs**: `cd docs && mdbook build`
- **Serve Locally**: `cd docs && mdbook serve`
- **GitHub Pages**: Deployed automatically via GitHub Actions
