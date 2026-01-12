//! Common graph types and data structures.

use serde::{Deserialize, Serialize};

/// Graph representation in Compressed Sparse Row (CSR) format.
///
/// Efficient for GPU processing with coalesced memory access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsrGraph {
    /// Number of nodes.
    pub num_nodes: usize,
    /// Number of edges.
    pub num_edges: usize,
    /// Row offsets (length: num_nodes + 1).
    pub row_offsets: Vec<u64>,
    /// Column indices (length: num_edges).
    pub col_indices: Vec<u64>,
    /// Edge weights (optional, length: num_edges).
    pub weights: Option<Vec<f32>>,
}

impl CsrGraph {
    /// Create an empty graph.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            num_nodes: 0,
            num_edges: 0,
            row_offsets: vec![0],
            col_indices: Vec::new(),
            weights: None,
        }
    }

    /// Create a graph from edge list.
    #[must_use]
    pub fn from_edges(num_nodes: usize, edges: &[(u64, u64)]) -> Self {
        let mut row_counts = vec![0u64; num_nodes];
        for (src, _) in edges {
            row_counts[*src as usize] += 1;
        }

        let mut row_offsets = vec![0u64; num_nodes + 1];
        for i in 0..num_nodes {
            row_offsets[i + 1] = row_offsets[i] + row_counts[i];
        }

        let mut col_indices = vec![0u64; edges.len()];
        let mut current_pos = row_offsets.clone();

        for (src, dst) in edges {
            let pos = current_pos[*src as usize] as usize;
            col_indices[pos] = *dst;
            current_pos[*src as usize] += 1;
        }

        Self {
            num_nodes,
            num_edges: edges.len(),
            row_offsets,
            col_indices,
            weights: None,
        }
    }

    /// Get the out-degree of a node.
    #[must_use]
    pub fn out_degree(&self, node: u64) -> u64 {
        let n = node as usize;
        if n >= self.num_nodes {
            return 0;
        }
        self.row_offsets[n + 1] - self.row_offsets[n]
    }

    /// Get the neighbors of a node.
    #[must_use]
    pub fn neighbors(&self, node: u64) -> &[u64] {
        let n = node as usize;
        if n >= self.num_nodes {
            return &[];
        }
        let start = self.row_offsets[n] as usize;
        let end = self.row_offsets[n + 1] as usize;
        &self.col_indices[start..end]
    }

    /// Check if the graph has weights.
    #[must_use]
    pub fn is_weighted(&self) -> bool {
        self.weights.is_some()
    }

    /// Calculate graph density.
    #[must_use]
    pub fn density(&self) -> f64 {
        if self.num_nodes <= 1 {
            return 0.0;
        }
        let max_edges = self.num_nodes * (self.num_nodes - 1);
        self.num_edges as f64 / max_edges as f64
    }
}

/// Node with centrality score.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NodeScore {
    /// Node ID.
    pub node_id: u64,
    /// Centrality score.
    pub score: f64,
}

/// Centrality result for a graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityResult {
    /// Scores per node.
    pub scores: Vec<NodeScore>,
    /// Number of iterations (for iterative algorithms).
    pub iterations: Option<u32>,
    /// Whether the algorithm converged.
    pub converged: bool,
}

impl CentralityResult {
    /// Get the top-k nodes by score.
    #[must_use]
    pub fn top_k(&self, k: usize) -> Vec<NodeScore> {
        let mut sorted = self.scores.clone();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(k);
        sorted
    }
}

/// Community detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityResult {
    /// Community assignment per node.
    pub assignments: Vec<u64>,
    /// Number of communities found.
    pub num_communities: usize,
    /// Modularity score.
    pub modularity: f64,
}

/// Similarity result between two nodes or sets.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SimilarityScore {
    /// First node/set ID.
    pub id_a: u64,
    /// Second node/set ID.
    pub id_b: u64,
    /// Similarity score (0.0 to 1.0).
    pub similarity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_from_edges() {
        let edges = vec![(0, 1), (0, 2), (1, 2), (2, 0)];
        let graph = CsrGraph::from_edges(3, &edges);

        assert_eq!(graph.num_nodes, 3);
        assert_eq!(graph.num_edges, 4);
        assert_eq!(graph.out_degree(0), 2);
        assert_eq!(graph.out_degree(1), 1);
        assert_eq!(graph.out_degree(2), 1);
    }

    #[test]
    fn test_graph_density() {
        let edges = vec![(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)];
        let graph = CsrGraph::from_edges(3, &edges);

        // Complete graph of 3 nodes has 6 edges, density = 1.0
        assert!((graph.density() - 1.0).abs() < 0.001);
    }
}
