# Graph Analytics

**Crate**: `rustkernel-graph`
**Kernels**: 28
**Feature**: `graph` (included in default features)

Graph analytics kernels for network analysis, social network analysis, knowledge graph operations, and AML/fraud detection.

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

### Similarity Measures (5)

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| JaccardSimilarity | `graph/jaccard-similarity` | Batch, Ring | Neighbor set overlap |
| CosineSimilarity | `graph/cosine-similarity` | Batch, Ring | Vector-based similarity |
| AdamicAdarIndex | `graph/adamic-adar-index` | Batch | Weighted common neighbors |
| CommonNeighbors | `graph/common-neighbors` | Batch, Ring | Shared neighbor counting |
| ValueSimilarity | `graph/value-similarity` | Batch | Distribution comparison (JSD/Wasserstein) |

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

### Topology Analysis (2)

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| DegreeRatio | `graph/degree-ratio` | Ring | In/out ratio for source/sink classification |
| StarTopologyScore | `graph/star-topology` | Batch | Hub-and-spoke detection (smurfing) |

### Cycle Detection (1)

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| ShortCycleParticipation | `graph/cycle-participation` | Batch | 2-4 hop cycle detection (AML) |

### Path Analysis (1)

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| ShortestPath | `graph/shortest-path` | Batch | BFS/Delta-Stepping SSSP/APSP |

### Graph Neural Networks (2)

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| GNNInference | `graph/gnn-inference` | Batch | Message-passing neural network inference |
| GraphAttention | `graph/graph-attention` | Batch | Multi-head graph attention networks |

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

### ShortCycleParticipation

Detects participation in short cycles (2-4 hops) which are key indicators for AML.

**ID**: `graph/cycle-participation`
**Modes**: Batch
**Throughput**: ~25,000 nodes/sec

Short cycles are critical AML indicators:
- **2-cycles (reciprocal)**: Immediate return transactions
- **3-cycles (triangles)**: Layering patterns - HIGH AML risk
- **4-cycles (squares)**: Organized laundering - CRITICAL AML risk

#### Example

```rust
use rustkernel::graph::cycles::{ShortCycleParticipation, CycleRiskLevel};

let kernel = ShortCycleParticipation::new();
let results = kernel.compute_all(&graph);

// Find high-risk nodes
for result in &results {
    if matches!(result.risk_level, CycleRiskLevel::High | CycleRiskLevel::Critical) {
        println!("HIGH RISK: Node {} participates in {} 4-cycles",
                 result.node_index, result.cycle_count_4hop);
    }
}

// Count triangles in the graph
let triangles = ShortCycleParticipation::count_triangles(&graph);
```

---

### DegreeRatio

Calculates in-degree/out-degree ratio for node classification.

**ID**: `graph/degree-ratio`
**Modes**: Ring
**Latency**: ~300ns per query

Classifies nodes as:
- **Source**: Mostly outgoing edges (payment originators)
- **Sink**: Mostly incoming edges (collection accounts)
- **Balanced**: Equal in/out (intermediary accounts)

#### Example

```rust
use rustkernel::graph::topology::{DegreeRatio, NodeClassification};

let results = DegreeRatio::compute_batch(&graph);
let roles = DegreeRatio::classify_nodes(&graph);

println!("Sources: {:?}", roles.sources);
println!("Sinks: {:?}", roles.sinks);
```

---

### StarTopologyScore

Detects hub-and-spoke patterns for smurfing and money mule detection.

**ID**: `graph/star-topology`
**Modes**: Batch
**Throughput**: ~20,000 nodes/sec

Star types:
- **In-Star**: Collection pattern (many payers to one receiver)
- **Out-Star**: Distribution pattern (smurfing indicator)
- **Mixed**: Money mule hub

#### Example

```rust
use rustkernel::graph::topology::{StarTopologyScore, StarType};

let kernel = StarTopologyScore::with_min_degree(10);
let hubs = kernel.top_k_hubs(&graph, 10);

// Find potential smurfing accounts (out-stars)
let out_stars = kernel.find_out_stars(&graph, 0.8);
for hub in out_stars {
    println!("POTENTIAL SMURFING: Node {} with score {:.2}",
             hub.node_index, hub.star_score);
}
```

---

### ShortestPath

Computes shortest paths using BFS or Delta-Stepping algorithm.

**ID**: `graph/shortest-path`
**Modes**: Batch
**Throughput**: ~50,000 nodes/sec

Supports:
- Single-source shortest path (SSSP)
- All-pairs shortest path (APSP)
- K-shortest paths (Yen's algorithm)

#### Example

```rust
use rustkernel::graph::paths::ShortestPath;

// Single-source shortest path
let sssp = ShortestPath::compute_sssp_bfs(&graph, source);
println!("Distance to target: {}", sssp[target].distance);

// Reconstruct path
if let Some(path) = ShortestPath::compute_path(&graph, source, target) {
    println!("Path: {:?}", path.node_path);
}

// Graph diameter
let diameter = ShortestPath::compute_diameter(&graph);
```

---

### ValueSimilarity

Compares node value distributions using statistical distance metrics.

**ID**: `graph/value-similarity`
**Modes**: Batch
**Throughput**: ~25,000 pairs/sec

Metrics:
- **Jensen-Shannon Divergence (JSD)**: Symmetric KL divergence
- **Wasserstein Distance**: Earth Mover's Distance

#### Example

```rust
use rustkernel::graph::similarity::{ValueSimilarity, ValueDistribution};

// Create distributions from transaction amounts
let dist = ValueDistribution::from_values(&node_amounts, 50);

// Find similar nodes using JSD
let pairs = ValueSimilarity::compute_all_pairs_jsd(&dist, 0.9, 100);
for pair in &pairs {
    println!("Similar: {} and {} (similarity: {:.3})",
             pair.node_a, pair.node_b, pair.similarity);
}
```

---

### GNNInference

Graph Neural Network inference using message passing.

**ID**: `graph/gnn-inference`
**Modes**: Batch
**Throughput**: ~10,000 nodes/sec

Supports configurable aggregation functions (mean, sum, max) and multiple message passing iterations.

#### Example

```rust
use rustkernel::graph::gnn::{GNNInference, GNNConfig, AggregationType};

let kernel = GNNInference::new();

// Configure the GNN
let config = GNNConfig {
    hidden_dim: 64,
    output_dim: 32,
    num_layers: 2,
    aggregation: AggregationType::Mean,
    activation: ActivationType::ReLU,
};

// Run inference
let node_embeddings = kernel.infer(&graph, &node_features, &config)?;
println!("Node 0 embedding: {:?}", node_embeddings[0]);
```

---

### GraphAttention

Multi-head graph attention network for node classification and link prediction.

**ID**: `graph/graph-attention`
**Modes**: Batch
**Throughput**: ~8,000 nodes/sec

Uses self-attention to learn importance weights for neighboring nodes.

#### Example

```rust
use rustkernel::graph::gnn::{GraphAttention, AttentionConfig};

let kernel = GraphAttention::new();

let config = AttentionConfig {
    num_heads: 4,
    hidden_dim: 64,
    output_dim: 32,
    dropout: 0.1,
    concat_heads: true,
};

let output = kernel.forward(&graph, &node_features, &config)?;
println!("Attention weights: {:?}", output.attention_weights);
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
