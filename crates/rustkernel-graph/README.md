# rustkernel-graph

[![Crates.io](https://img.shields.io/crates/v/rustkernel-graph.svg)](https://crates.io/crates/rustkernel-graph)
[![Documentation](https://docs.rs/rustkernel-graph/badge.svg)](https://docs.rs/rustkernel-graph)
[![License](https://img.shields.io/crates/l/rustkernel-graph.svg)](LICENSE)

GPU-accelerated graph analytics kernels for centrality measures, community detection, motif analysis, and similarity metrics.

## Kernels (28)

### Centrality (6 kernels)
- **DegreeCentrality** - O(1) degree queries
- **BetweennessCentrality** - Brandes algorithm
- **ClosenessCentrality** - BFS-based closeness
- **EigenvectorCentrality** - Power iteration
- **PageRank** - Power iteration with teleport
- **KatzCentrality** - Attenuated paths

### Community Detection (5 kernels)
- **LouvainCommunity** - Modularity optimization
- **LabelPropagation** - Semi-synchronous propagation
- **ConnectedComponents** - Union-find based
- **CoreDecomposition** - K-core decomposition
- **SpectralClustering** - Laplacian-based clustering

### Motif Analysis (2 kernels)
- **TriangleCounting** - Exact triangle enumeration
- **MotifCensus** - 3/4-node motif counting

### Similarity (4 kernels)
- **JaccardSimilarity** - Set-based similarity
- **CosineSimilarity** - Vector-based similarity
- **SimRank** - Recursive structural similarity
- **RoleSimilarity** - Role equivalence

### Paths & Cycles (4 kernels)
- **ShortestPath** - BFS/Delta-stepping SSSP
- **AllPairsShortestPath** - Floyd-Warshall APSP
- **CycleDetection** - Tarjan's algorithm
- **TopologicalSort** - Kahn's algorithm

### GNN (2 kernels)
- **GNNInference** - Message passing neural network
- **GraphAttention** - Multi-head graph attention

### Topology (5 kernels)
- **GraphDiameter** - BFS-based diameter
- **GraphDensity** - Edge density metrics
- **ClusteringCoefficient** - Local/global clustering
- **Eccentricity** - Node eccentricity
- **GraphStatistics** - Comprehensive graph stats

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernel-graph = "0.1.0"
```

## Usage

```rust
use rustkernel_graph::prelude::*;

// Create a PageRank kernel
let pagerank = PageRank::new();

// Initialize with a graph
pagerank.initialize(graph, 0.85);

// Query scores
let score = pagerank.query_score(node_id);
```

## License

Apache-2.0
