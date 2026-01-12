# Execution Modes

RustKernels supports two execution modes with different performance characteristics. Understanding these modes is essential for choosing the right approach for your workload.

## Overview

| Aspect | Batch Mode | Ring Mode |
|--------|------------|-----------|
| **Latency** | 10-50μs | 100-500ns |
| **Launch Overhead** | Higher | Minimal |
| **State Location** | CPU memory | GPU memory |
| **Programming Model** | Request/response | Actor messages |
| **Best For** | Heavy periodic computation | High-frequency streaming |

## Batch Mode

Batch mode provides CPU-orchestrated kernel execution. The kernel is launched on-demand, executes on the GPU, and returns results to the CPU.

### Characteristics

- **Launch overhead**: 10-50μs per invocation
- **State management**: State lives in CPU memory between calls
- **Execution model**: Synchronous request/response
- **Data transfer**: Input copied to GPU, output copied back

### When to Use Batch Mode

- Heavy computational tasks (matrix operations, large graph processing)
- Periodic batch jobs (nightly risk calculations, weekly reports)
- Tasks where launch overhead is negligible compared to computation time
- When you need to process a complete dataset at once

### Implementation

Batch kernels implement the `BatchKernel` trait:

```rust
use rustkernel_core::traits::{GpuKernel, BatchKernel};
use rustkernel_core::kernel::KernelMetadata;

pub struct MyBatchKernel {
    metadata: KernelMetadata,
}

impl GpuKernel for MyBatchKernel {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

impl BatchKernel<MyInput, MyOutput> for MyBatchKernel {
    async fn execute(&self, input: MyInput) -> Result<MyOutput> {
        // GPU computation here
        Ok(output)
    }
}
```

### Usage Example

```rust
use rustkernel::graph::centrality::PageRank;

let kernel = PageRank::new();

// Prepare large graph input
let input = PageRankInput {
    num_nodes: 1_000_000,
    edges: load_edges_from_file()?,
    damping_factor: 0.85,
    max_iterations: 100,
    tolerance: 1e-6,
};

// Execute - may take seconds for large graphs
let result = kernel.execute(input).await?;

println!("Top node: {} with score {:.4}",
    result.top_node(),
    result.scores[result.top_node()]
);
```

## Ring Mode

Ring mode provides GPU-persistent actors. The kernel maintains state on the GPU and processes messages with minimal latency.

### Characteristics

- **Message latency**: 100-500ns per message
- **State persistence**: State remains on GPU between messages
- **Execution model**: Asynchronous actor messages
- **Data transfer**: Only message payloads transferred

### When to Use Ring Mode

- High-frequency operations (order matching, real-time scoring)
- Streaming workloads (continuous data feeds)
- When sub-millisecond latency is critical
- Incremental updates to persistent state

### Implementation

Ring kernels implement the `RingKernelHandler` trait:

```rust
use rustkernel_core::traits::{GpuKernel, RingKernelHandler};
use rustkernel_core::ring::{RingContext, RingMessage};

pub struct MyRingKernel {
    metadata: KernelMetadata,
}

impl GpuKernel for MyRingKernel {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

impl RingKernelHandler<MyRequest, MyResponse> for MyRingKernel {
    async fn handle(
        &self,
        ctx: &mut RingContext,
        msg: MyRequest,
    ) -> Result<MyResponse> {
        // Process message, update GPU state
        Ok(response)
    }
}
```

### Ring Message Definition

Ring messages use fixed-point arithmetic and rkyv serialization:

```rust
use ringkernel_derive::RingMessage;
use rkyv::{Archive, Serialize, Deserialize};

#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 200)]  // Unique within domain range
pub struct ScoreQueryRequest {
    #[message(id)]
    pub id: MessageId,
    pub node_id: u32,
}

#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 201)]
pub struct ScoreQueryResponse {
    #[message(id)]
    pub id: MessageId,
    pub score_fp: i64,  // Fixed-point score
}
```

### Usage Example

```rust
use rustkernel::graph::centrality::PageRankRing;

// Create ring kernel (initializes GPU state)
let ring = PageRankRing::new();

// Stream of edge updates
for edge in incoming_edges {
    // Low-latency edge addition
    ring.add_edge(edge.from, edge.to).await?;
}

// Query current scores (sub-millisecond)
let scores = ring.query_scores().await?;

// Trigger re-computation if needed
ring.recalculate().await?;
```

## Choosing Between Modes

### Decision Matrix

| Scenario | Recommended Mode | Reason |
|----------|------------------|--------|
| Nightly risk report | Batch | Large computation, latency not critical |
| Real-time fraud scoring | Ring | Sub-ms latency required |
| Graph analysis on static data | Batch | One-time computation |
| Order book matching | Ring | Continuous high-frequency updates |
| ML model inference (bulk) | Batch | Process entire batch at once |
| ML model inference (streaming) | Ring | Incremental predictions |

### Hybrid Approach

Many applications combine both modes:

```rust
// Batch: Initial heavy computation
let graph_kernel = GraphBuilder::new();
let initial_graph = graph_kernel.build(edges).await?;

// Ring: Real-time updates
let ring_kernel = GraphRing::from(initial_graph);

loop {
    // Process streaming updates with low latency
    let update = receive_update().await;
    ring_kernel.handle(update).await?;

    // Periodically sync state back for batch analysis
    if should_sync() {
        let snapshot = ring_kernel.snapshot().await?;
        batch_analysis(snapshot).await?;
    }
}
```

## Performance Considerations

### Batch Mode Optimization

1. **Batch your inputs**: Process multiple items in one call
2. **Minimize data transfer**: Only send required fields
3. **Use async**: Don't block on kernel completion

```rust
// Good: Process many items at once
let results = kernel.execute_batch(items).await?;

// Avoid: Processing one at a time
for item in items {
    let result = kernel.execute(item).await?;  // High overhead
}
```

### Ring Mode Optimization

1. **Keep messages small**: Only transfer deltas
2. **Batch when possible**: Group related messages
3. **Use K2K for coordination**: Avoid CPU round-trips

```rust
// Good: Small incremental update
ring.update_score(node_id, delta).await?;

// Avoid: Transferring full state
ring.set_all_scores(full_score_vector).await?;  // Large transfer
```

## Iterative Kernels

Some algorithms naturally span multiple iterations (PageRank, K-Means). These implement `IterativeKernel`:

```rust
pub trait IterativeKernel<S, I, O>: GpuKernel {
    fn initial_state(&self, input: &I) -> S;
    async fn iterate(&self, state: &mut S, input: &I) -> Result<IterationResult<O>>;
    fn converged(&self, state: &S, threshold: f64) -> bool;
}
```

Usage:

```rust
let kernel = PageRank::new();

// Create initial state
let mut state = kernel.initial_state(&input);

// Iterate until convergence
while !kernel.converged(&state, tolerance) {
    let result = kernel.iterate(&mut state, &input).await?;
    println!("Iteration {}: delta={:.6}", result.iteration, result.delta);
}
```

## Next Steps

- [Architecture Overview](overview.md) - System design and components
- [Kernel Catalogue](../domains/README.md) - Available kernels by domain
- [K2K Messaging](overview.md#k2k-kernel-to-kernel-messaging) - Cross-kernel coordination
