# Quick Start

Get up and running with RustKernels in 5 minutes.

## Your First Kernel

Let's run a PageRank calculation on a simple graph.

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
rustkernel = { version = "0.1.0", features = ["graph"] }
tokio = { version = "1.0", features = ["full"] }
```

### Step 3: Write Your Code

Edit `src/main.rs`:

```rust
use rustkernel::prelude::*;
use rustkernel::graph::centrality::{PageRank, PageRankInput};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a PageRank kernel
    let kernel = PageRank::new();

    // Print kernel metadata
    let metadata = kernel.metadata();
    println!("Kernel: {}", metadata.id);
    println!("Domain: {:?}", metadata.domain);
    println!("Mode: {:?}", metadata.mode);

    // Prepare input: a simple 4-node graph
    // Node 0 -> Node 1, Node 2
    // Node 1 -> Node 2
    // Node 2 -> Node 0, Node 3
    // Node 3 -> Node 0
    let input = PageRankInput {
        num_nodes: 4,
        edges: vec![
            (0, 1), (0, 2),
            (1, 2),
            (2, 0), (2, 3),
            (3, 0),
        ],
        damping_factor: 0.85,
        max_iterations: 100,
        tolerance: 1e-6,
    };

    // Execute the kernel
    let result = kernel.execute(input).await?;

    // Print results
    println!("\nPageRank Scores:");
    for (node, score) in result.scores.iter().enumerate() {
        println!("  Node {}: {:.4}", node, score);
    }
    println!("\nConverged in {} iterations", result.iterations);

    Ok(())
}
```

### Step 4: Run

```bash
cargo run
```

Expected output:

```
Kernel: graph/pagerank
Domain: GraphAnalytics
Mode: Batch

PageRank Scores:
  Node 0: 0.3682
  Node 1: 0.1418
  Node 2: 0.2879
  Node 3: 0.2021

Converged in 23 iterations
```

## Using Multiple Kernels

Combine kernels from different domains:

```rust
use rustkernel::prelude::*;
use rustkernel::graph::centrality::PageRank;
use rustkernel::graph::community::LouvainCommunity;
use rustkernel::graph::metrics::GraphDensity;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Analyze the same graph with multiple kernels
    let edges = vec![
        (0, 1), (0, 2), (1, 2),
        (2, 3), (3, 4), (4, 2),
    ];

    // Centrality analysis
    let pagerank = PageRank::new();
    let pr_result = pagerank.execute(PageRankInput {
        num_nodes: 5,
        edges: edges.clone(),
        damping_factor: 0.85,
        max_iterations: 100,
        tolerance: 1e-6,
    }).await?;

    // Community detection
    let louvain = LouvainCommunity::new();
    let community_result = louvain.execute(LouvainInput {
        num_nodes: 5,
        edges: edges.clone(),
        resolution: 1.0,
    }).await?;

    // Graph metrics
    let density = GraphDensity::new();
    let density_result = density.execute(DensityInput {
        num_nodes: 5,
        num_edges: edges.len(),
    }).await?;

    println!("Analysis complete:");
    println!("  Communities found: {}", community_result.num_communities);
    println!("  Graph density: {:.4}", density_result.density);
    println!("  Most central node: {}", pr_result.top_node());

    Ok(())
}
```

## Kernel Configuration

Most kernels accept configuration options:

```rust
use rustkernel::ml::clustering::{KMeans, KMeansConfig};

let config = KMeansConfig {
    num_clusters: 5,
    max_iterations: 300,
    tolerance: 1e-4,
    initialization: KMeansInit::KMeansPlusPlus,
    ..Default::default()
};

let kernel = KMeans::with_config(config);
```

## Batch vs Ring Mode

### Batch Mode (Default)

CPU-orchestrated execution. Best for periodic computations:

```rust
// Batch kernels implement BatchKernel trait
let kernel = PageRank::new();
let result = kernel.execute(input).await?;
```

### Ring Mode

GPU-persistent actors for streaming workloads:

```rust
// Ring kernels implement RingKernelHandler trait
use rustkernel::graph::centrality::PageRankRing;

// Ring kernels maintain persistent GPU state
let ring = PageRankRing::new();

// Send streaming updates
ring.add_edge(0, 1).await?;
ring.add_edge(1, 2).await?;

// Query current state
let scores = ring.query_scores().await?;
```

See [Execution Modes](../architecture/execution-modes.md) for detailed comparison.

## Error Handling

RustKernels uses standard Rust error handling:

```rust
use rustkernel::prelude::*;
use rustkernel::error::KernelError;

match kernel.execute(input).await {
    Ok(result) => println!("Success: {:?}", result),
    Err(KernelError::InvalidInput(msg)) => {
        eprintln!("Invalid input: {}", msg);
    }
    Err(KernelError::ExecutionFailed(msg)) => {
        eprintln!("Execution failed: {}", msg);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Next Steps

- [Architecture Overview](../architecture/overview.md) - Understand the system design
- [Kernel Catalogue](../domains/README.md) - Explore all 82 kernels
- [Accounting Network Generation](../articles/accounting-network-generation.md) - Deep-dive article
