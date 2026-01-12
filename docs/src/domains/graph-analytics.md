# Graph Analytics

**Crate**: `rustkernel-graph`
**Kernels**: 21
**Feature**: `graph` (included in default features)

Graph analytics kernels for network analysis, social network analysis, and knowledge graph operations.

## Kernel Overview

### Centrality Measures (6)

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| PageRank | `graph/pagerank` | Batch, Ring | Power iteration with teleportation |
| DegreeCentrality | `graph/degree-centrality` | Batch, Ring | In/out/total degree counting |
| BetweennessCentrality | `graph/betweenness-centrality` | Batch | Brandes algorithm |
| ClosenessCentrality | `graph/closeness-centrality` | Batch, Ring | BFS-based distance calculation |
| EigenvectorCentrality | `graph/eigenvector-centrality` | Batch, Ring | Power iteration method |
| KatzCentrality | `graph/katz-centrality` | Batch, Ring | Attenuated path counting |

### Community Detection (3)

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| ModularityScore | `graph/modularity-score` | Batch | Community quality metric |
| LouvainCommunity | `graph/louvain-community` | Batch, Ring | Modularity optimization |
| LabelPropagation | `graph/label-propagation` | Batch, Ring | Fast community detection |

### Similarity Measures (4)

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| JaccardSimilarity | `graph/jaccard-similarity` | Batch, Ring | Neighbor set overlap |
| CosineSimilarity | `graph/cosine-similarity` | Batch, Ring | Vector-based similarity |
| AdamicAdarIndex | `graph/adamic-adar-index` | Batch | Weighted common neighbors |
| CommonNeighbors | `graph/common-neighbors` | Batch, Ring | Shared neighbor counting |

### Graph Metrics (5)

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| GraphDensity | `graph/graph-density` | Batch, Ring | Edge density calculation |
| AveragePathLength | `graph/average-path-length` | Batch | BFS-based distance sampling |
| ClusteringCoefficient | `graph/clustering-coefficient` | Batch, Ring | Local/global clustering |
| ConnectedComponents | `graph/connected-components` | Batch, Ring | Union-Find algorithm |
| FullGraphMetrics | `graph/full-graph-metrics` | Batch | Combined metrics computation |

### Motif Detection (3)

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| TriangleCounting | `graph/triangle-counting` | Batch, Ring | Triangle enumeration |
| MotifDetection | `graph/motif-detection` | Batch | Subgraph pattern matching |
| KCliqueDetection | `graph/k-clique-detection` | Batch | Complete subgraph finding |

---

## Kernel Details

### PageRank

The PageRank algorithm computes the importance of nodes based on link structure.

**ID**: `graph/pagerank`
**Modes**: Batch, Ring
**Throughput**: ~100,000 nodes/sec
**Latency**: 50μs (Batch), 500ns (Ring)

#### Input

```rust
pub struct PageRankInput {
    /// Number of nodes in the graph
    pub num_nodes: u32,
    /// Edges as (from, to) pairs
    pub edges: Vec<(u32, u32)>,
    /// Damping factor (typically 0.85)
    pub damping_factor: f64,
    /// Maximum iterations
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
}
```

#### Output

```rust
pub struct PageRankOutput {
    /// PageRank scores indexed by node ID
    pub scores: Vec<f64>,
    /// Number of iterations performed
    pub iterations: u32,
    /// Final delta (convergence measure)
    pub delta: f64,
}
```

#### Example

```rust
use rustkernel::graph::centrality::{PageRank, PageRankInput};

let kernel = PageRank::new();

let input = PageRankInput {
    num_nodes: 4,
    edges: vec![(0, 1), (0, 2), (1, 2), (2, 0), (2, 3), (3, 0)],
    damping_factor: 0.85,
    max_iterations: 100,
    tolerance: 1e-6,
};

let result = kernel.execute(input).await?;
println!("Top node: {}", result.top_node());
```

---

### LouvainCommunity

Detects communities using the Louvain method for modularity optimization.

**ID**: `graph/louvain-community`
**Modes**: Batch, Ring
**Throughput**: ~50,000 nodes/sec

#### Input

```rust
pub struct LouvainInput {
    pub num_nodes: u32,
    pub edges: Vec<(u32, u32)>,
    /// Resolution parameter (1.0 = standard modularity)
    pub resolution: f64,
}
```

#### Output

```rust
pub struct LouvainOutput {
    /// Community assignment per node
    pub communities: Vec<u32>,
    /// Number of communities found
    pub num_communities: u32,
    /// Final modularity score
    pub modularity: f64,
}
```

---

### TriangleCounting

Counts triangles in the graph, useful for clustering coefficient and network density analysis.

**ID**: `graph/triangle-counting`
**Modes**: Batch, Ring

#### Example

```rust
use rustkernel::graph::motif::{TriangleCounting, TriangleInput};

let kernel = TriangleCounting::new();
let result = kernel.execute(TriangleInput {
    num_nodes: 100,
    edges: edges,
}).await?;

println!("Triangles: {}", result.triangle_count);
println!("Clustering coefficient: {:.4}", result.global_clustering);
```

---

## Ring Mode Usage

For high-frequency graph updates, use Ring mode:

```rust
use rustkernel::graph::centrality::PageRankRing;

let ring = PageRankRing::new();

// Add edges with low latency
ring.add_edge(0, 1).await?;
ring.add_edge(1, 2).await?;

// Query current scores
let score = ring.query_score(0).await?;

// Trigger recalculation
ring.recalculate().await?;
```

## Performance Tips

1. **Use CSR format**: For large static graphs, convert to CSR before processing
2. **Batch edge updates**: When using Ring mode, batch multiple edges when possible
3. **Choose appropriate algorithms**: BetweennessCentrality is O(V*E), consider sampling for large graphs
4. **Leverage GPU**: Ensure CUDA is available for maximum throughput
