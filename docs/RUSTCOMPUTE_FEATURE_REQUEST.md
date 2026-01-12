# RustCompute Feature Request: RustKernels Integration

**From**: RustKernels Team
**To**: RustCompute Team
**Date**: 2026-01-12
**Priority**: High

## Executive Summary

RustKernels is porting 173 GPU kernels from DotCompute to Rust using the RustCompute framework. We've completed BatchKernel implementations across all domains and are now implementing Ring-mode transformations for Tier 1 critical-path kernels. We need several enhancements to ringkernel-core to support this.

---

## FR-1: KernelMessage ↔ RingMessage Bridge (HIGH PRIORITY - BLOCKER)

### Current State
- `rustkernel-derive` provides `#[derive(KernelMessage)]` which adds `message_type_id()` method
- `ringkernel-derive` provides `#[derive(RingMessage)]` which implements full RingMessage trait with rkyv serialization
- Both macros use `#[message(...)]` attribute but with incompatible parameters

### Exact Incompatibility (Discovered)
When attempting to derive both on the same struct:

```rust
// This FAILS to compile
#[derive(KernelMessage, RingMessage)]
#[message(type_id = 1000, domain = "OrderMatching")]  // KernelMessage wants domain
pub struct SubmitOrderInput {
    #[message(id)]           // RingMessage expects this
    pub message_id: MessageId,
    #[message(correlation)]  // RingMessage expects this
    pub correlation_id: CorrelationId,
    pub order: Order,
}
```

Errors encountered:
1. `ringkernel-derive::RingMessage` doesn't recognize `domain` attribute
2. `MessageId` and `CorrelationId` don't implement serde `Serialize`/`Deserialize`
3. Field-level attributes (`#[message(id)]`) conflict between the two macros

### Problem
Ring kernels need messages that implement both:
1. `KernelMessage` - for domain metadata (type_id, domain)
2. `RingMessage` - for ring buffer serialization (message_id, correlation_id, priority, serialize/deserialize)

### Requested Solution
Option A (Preferred): Update `ringkernel-derive::RingMessage` to:
1. Accept optional `domain` attribute (ignored but not errored)
2. Add serde derives to `MessageId` and `CorrelationId` types

Option B: Create a unified `#[derive(DomainRingMessage)]` macro in rustkernel-derive that:
1. Uses separate attribute namespaces: `#[kernel(...)]` for domain info, `#[ring(...)]` for ring buffer
2. Generates both trait implementations

Option C: Define a wrapper type pattern:
```rust
// Domain message (serde-compatible)
#[derive(KernelMessage)]
#[message(type_id = 1000, domain = "OrderMatching")]
pub struct SubmitOrderInput { pub order: Order }

// Ring wrapper (rkyv-compatible)
#[derive(RingMessage)]
#[message(type_id = 1000)]
pub struct SubmitOrderRingMsg {
    #[message(id)]
    pub id: MessageId,
    pub inner: SubmitOrderInput,  // Requires rkyv derives
}
```

### Impact
- **BLOCKING**: 48 Ring kernel messages cannot be implemented
- Blocks Phase 6.3 Ring transformations (5 Tier 1 kernels)
- Blocks Phase 6.4-6.5 (10 additional kernels)

---

## FR-2: RingContext Domain Extensions (MEDIUM PRIORITY)

### Current State
`RingContext` provides basic Ring kernel infrastructure but lacks domain-specific helpers.

### Requested Additions

```rust
impl RingContext {
    // 1. Domain metadata access
    pub fn domain(&self) -> Option<&Domain>;

    // 2. Kernel metrics collection
    pub fn record_latency(&mut self, operation: &str, latency_us: u64);
    pub fn record_throughput(&mut self, operation: &str, count: u64);

    // 3. Alert/event emission (for compliance kernels)
    pub fn emit_alert(&mut self, alert: impl Into<Alert>);

    // 4. State snapshot capability
    pub fn snapshot_state<S: GpuState>(&self) -> Result<S>;
}
```

### Use Cases
- **Compliance kernels**: Need to emit AML alerts during `handle()` calls
- **Risk kernels**: Need to track VaR calculation latencies
- **All Ring kernels**: Need domain-aware metrics for monitoring

---

## FR-3: K2K Message Type Registry (MEDIUM PRIORITY)

### Current State
K2K messaging works but type registration is manual.

### Requested Enhancement
Auto-register message types for K2K routing:

```rust
// Declarative registration via attribute
#[derive(RingMessage)]
#[ring_message(type_id = 1000, k2k_routable = true)]
pub struct SubmitOrderInput { ... }

// Runtime discovery
let registry = K2KTypeRegistry::discover();
assert!(registry.is_routable(1000));
```

### Impact
- Enables auto-routing in K2K broker
- Required for Phase 6.4 K2K enhancements

---

## FR-4: ControlBlock State Helpers (LOW PRIORITY)

### Current State
ControlBlock is 128-byte aligned for GPU cache efficiency, but manual serialization is required.

### Requested Enhancement
Helper traits for common state patterns:

```rust
// Auto-derive for simple state
#[derive(ControlBlockState)]
#[repr(C, align(128))]
pub struct OrderBookState {
    pub best_bid: Price,
    pub best_ask: Price,
    pub order_count: u32,
    pub last_trade_time: u64,
}

// Generated impl
impl GpuState for OrderBookState {
    fn to_control_block(&self) -> ControlBlock { ... }
    fn from_control_block(block: &ControlBlock) -> Self { ... }
}
```

---

## FR-5: Variance Reduction Primitives for Monte Carlo (LOW PRIORITY)

### Current State
No built-in variance reduction for Monte Carlo simulations.

### Requested Addition
Add to `ringkernel-core` or new `ringkernel-monte-carlo` crate:

```rust
pub mod variance_reduction {
    /// Antithetic variates: generate pairs of negatively correlated samples
    pub fn antithetic_variates<T, F>(n: usize, sampler: F) -> Vec<T>
    where
        F: Fn(f64) -> T;

    /// Control variates: reduce variance using known expected value
    pub fn control_variates<T>(
        samples: &[T],
        control: &[T],
        expected_control: f64,
    ) -> Vec<T>;

    /// Importance sampling with weight function
    pub fn importance_sampling<T, F, W>(
        n: usize,
        sampler: F,
        weight: W,
    ) -> Vec<(T, f64)>;
}
```

### Impact
- Required for MonteCarloVaR kernel (Risk domain)
- Can significantly improve convergence rate (10-100x fewer samples)

---

## FR-6: Graph Algorithm GPU Primitives (LOW PRIORITY)

### Current State
No built-in graph algorithm primitives.

### Requested Addition
New `ringkernel-graph-primitives` crate or module:

```rust
pub mod graph_primitives {
    /// Parallel BFS from multiple sources
    pub async fn bfs_parallel(
        adjacency: &CsrMatrix,
        sources: &[NodeId],
    ) -> Vec<Distance>;

    /// GPU-accelerated strongly connected components
    pub async fn scc_tarjan(adjacency: &CsrMatrix) -> Vec<ComponentId>;

    /// Parallel union-find for DBSCAN/community detection
    pub async fn union_find_parallel(edges: &[(NodeId, NodeId)]) -> Vec<ComponentId>;

    /// Sparse matrix-vector multiplication (CSR format)
    pub async fn spmv(matrix: &CsrMatrix, vector: &[f64]) -> Vec<f64>;
}
```

### Impact
- Required for Graph Analytics kernels (15 kernels)
- PageRank, community detection, centrality measures all need these

---

## Timeline Alignment

| RustKernels Phase | RustCompute Feature | Blocking? | Status |
|-------------------|---------------------|-----------|--------|
| Phase 6.3 (Tier 1 Ring) | FR-1 KernelMessage bridge | **YES** | BLOCKED |
| Phase 6.4 (Tier 2 K2K) | FR-1, FR-3 K2K Registry | **YES** | Waiting |
| Phase 6.5 (Tier 3 Batch+K2K) | FR-1, FR-2, FR-4 | **YES** | Waiting |
| Ongoing | FR-5, FR-6 | No | Can proceed |

**Critical Path**: FR-1 is blocking ALL Ring kernel implementations (63 kernels total).

### Current Progress (as of 2026-01-12)
- ✅ Phase 6.1: BatchKernel implementations complete (72 kernels)
- ✅ Phase 6.2: KernelMessage derives complete (72 kernels)
- 🚫 Phase 6.3: BLOCKED - waiting for FR-1
- ⏳ Phase 6.4-6.6: Waiting on Phase 6.3

---

## Contact

For questions or clarification:
- Repository: `/home/michael/DEV/Repos/RustKernels/RustKernels`
- Reference Plan: `docs/IMPLEMENTATION_PLAN.md`

Thank you for your support!
