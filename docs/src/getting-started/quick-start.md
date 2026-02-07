# Quick Start

Get up and running with RustKernels in minutes.

## Your First Kernel

Run a betweenness centrality calculation on a simple graph.

### Step 1: Create a New Project

```bash
cargo new my-analytics
cd my-analytics
```

### Step 2: Add Dependencies

Edit `Cargo.toml`:

```toml
[package]
name = "my-analytics"
version = "0.1.0"
edition = "2024"

[dependencies]
rustkernels = { version = "0.4.0", features = ["graph"] }
tokio = { version = "1", features = ["full"] }
```

### Step 3: Write Your Code

Edit `src/main.rs`:

```rust
use rustkernels::prelude::*;
use rustkernels::graph::centrality::{BetweennessCentrality, BetweennessCentralityInput};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create the kernel
    let kernel = BetweennessCentrality::new();

    // Print kernel metadata
    let metadata = kernel.metadata();
    println!("Kernel: {}", metadata.id);
    println!("Domain: {:?}", metadata.domain);
    println!("Mode:   {:?}", metadata.mode);

    // Prepare input: a simple 4-node graph
    let input = BetweennessCentralityInput {
        num_nodes: 4,
        edges: vec![(0, 1), (1, 2), (2, 3), (0, 3)],
        normalized: true,
    };

    // Execute the kernel
    let result = kernel.execute(input).await?;

    // Print results
    println!("\nBetweenness Centrality Scores:");
    for (node, score) in result.scores.iter().enumerate() {
        println!("  Node {}: {:.4}", node, score);
    }

    Ok(())
}
```

### Step 4: Run

```bash
cargo run
```

## Using the Registry

For production deployments, use the `KernelRegistry` to manage kernels centrally:

```rust
use rustkernels::prelude::*;
use rustkernel_core::registry::KernelRegistry;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create the registry and register all domains
    let registry = Arc::new(KernelRegistry::new());
    rustkernels::register_all(&registry)?;

    // Execute via type-erased interface (same path REST/gRPC uses)
    let input_json = serde_json::to_vec(&serde_json::json!({
        "num_nodes": 4,
        "edges": [[0, 1], [1, 2], [2, 3], [0, 3]],
        "normalized": true
    }))?;

    let output_json = registry.execute_batch(
        "graph/betweenness-centrality",
        &input_json,
    ).await?;

    let result: serde_json::Value = serde_json::from_slice(&output_json)?;
    println!("Result: {}", serde_json::to_string_pretty(&result)?);

    Ok(())
}
```

## Deploying as a REST Service

Expose kernels via HTTP using the Axum integration:

```rust
use rustkernel_ecosystem::axum::{KernelRouter, RouterConfig};
use rustkernel_core::registry::KernelRegistry;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let registry = Arc::new(KernelRegistry::new());
    rustkernels::register_all(&registry).unwrap();

    let router = KernelRouter::new(registry, RouterConfig::default());
    let app = router.into_router();

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("Listening on http://0.0.0.0:8080");
    axum::serve(listener, app).await.unwrap();
}
```

Then call it:

```bash
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{
    "kernel_id": "graph/betweenness-centrality",
    "input": {
      "num_nodes": 4,
      "edges": [[0, 1], [1, 2], [2, 3], [0, 3]],
      "normalized": true
    }
  }'
```

## Kernel Configuration

Most kernels accept configuration through their input types:

```rust
use rustkernels::ml::clustering::{KMeans, KMeansInput};

let kernel = KMeans::new();
let input = KMeansInput {
    data: vec![/* data points */],
    k: 5,
    max_iterations: 300,
    tolerance: 1e-4,
};

let result = kernel.execute(input).await?;
```

## Batch vs Ring Mode

### Batch Mode (Default)

CPU-orchestrated execution — best for periodic computations:

```rust
// Batch kernels implement BatchKernel<I, O>
let kernel = BetweennessCentrality::new();
let result = kernel.execute(input).await?;
```

### Ring Mode

GPU-persistent actors for streaming workloads. Ring kernels require the RingKernel runtime:

```rust
// Ring kernels implement RingKernelHandler<M, R>
// They maintain persistent state in GPU memory and communicate
// via lock-free ring buffers with sub-microsecond latency.
// See architecture/execution-modes.md for setup details.
```

See [Execution Modes](../architecture/execution-modes.md) for a detailed comparison.

## Error Handling

RustKernels uses standard Rust error handling:

```rust
use rustkernel_core::error::KernelError;

match kernel.execute(input).await {
    Ok(result) => println!("Success: {:?}", result),
    Err(KernelError::ValidationError(msg)) => {
        eprintln!("Invalid input: {}", msg);
    }
    Err(KernelError::Timeout(duration)) => {
        eprintln!("Timed out after {:?}", duration);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Next Steps

- [Architecture Overview](../architecture/overview.md) — Understand the system design
- [Kernel Catalogue](../domains/README.md) — Explore all 106 kernels across 14 domains
- [Service Deployment](../enterprise/ecosystem.md) — Deploy as REST/gRPC services
- [Accounting Network Generation](../articles/accounting-network-generation.md) — Deep-dive article
