//! Graph motif detection kernels.
//!
//! This module provides algorithms for detecting graph motifs:
//! - Triangle counting (local and global)
//! - Motif detection (k-node subgraph census)

use crate::types::CsrGraph;
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashSet;

// ============================================================================
// Triangle Counting Kernel
// ============================================================================

/// Result of triangle counting.
#[derive(Debug, Clone)]
pub struct TriangleCountResult {
    /// Total number of triangles in the graph.
    pub total_triangles: u64,
    /// Number of triangles per node (node participates in).
    pub per_node_triangles: Vec<u64>,
    /// Clustering coefficient per node.
    pub clustering_coefficients: Vec<f64>,
    /// Global clustering coefficient.
    pub global_clustering_coefficient: f64,
}

/// Triangle counting kernel.
///
/// Counts triangles using the node-iterator algorithm.
/// Each triangle is counted once.
#[derive(Debug, Clone)]
pub struct TriangleCounting {
    metadata: KernelMetadata,
}

impl TriangleCounting {
    /// Create a new triangle counting kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/triangle-counting", Domain::GraphAnalytics)
                .with_description("Triangle counting (node-iterator algorithm)")
                .with_throughput(50_000)
                .with_latency_us(20.0),
        }
    }

    /// Count all triangles in the graph.
    ///
    /// Uses the node-iterator algorithm with degree ordering
    /// to ensure each triangle is counted exactly once.
    pub fn compute(graph: &CsrGraph) -> TriangleCountResult {
        let n = graph.num_nodes;
        let mut total_triangles = 0u64;
        let mut per_node_triangles = vec![0u64; n];

        // For each node, look at pairs of its neighbors and check if they're connected
        for u in 0..n {
            let neighbors_u: HashSet<u64> = graph.neighbors(u as u64).iter().copied().collect();

            for &v in graph.neighbors(u as u64) {
                let v = v as usize;

                // Only count once: process when u < v
                if u >= v {
                    continue;
                }

                // Check common neighbors (nodes that form triangles with u and v)
                for &w in graph.neighbors(v as u64) {
                    let w_usize = w as usize;

                    // Only count once: ensure u < v < w (by node index)
                    if v >= w_usize {
                        continue;
                    }

                    if neighbors_u.contains(&w) {
                        // Found triangle: u-v-w
                        total_triangles += 1;
                        per_node_triangles[u] += 1;
                        per_node_triangles[v] += 1;
                        per_node_triangles[w_usize] += 1;
                    }
                }
            }
        }

        // Compute clustering coefficients
        let mut clustering_coefficients = vec![0.0f64; n];
        let mut total_possible = 0u64;
        let mut total_actual = 0u64;

        for i in 0..n {
            let degree = graph.out_degree(i as u64);
            if degree >= 2 {
                let possible = degree * (degree - 1) / 2;
                total_possible += possible;
                total_actual += per_node_triangles[i];
                clustering_coefficients[i] = per_node_triangles[i] as f64 / possible as f64;
            }
        }

        let global_clustering_coefficient = if total_possible > 0 {
            total_actual as f64 / total_possible as f64
        } else {
            0.0
        };

        TriangleCountResult {
            total_triangles,
            per_node_triangles,
            clustering_coefficients,
            global_clustering_coefficient,
        }
    }

    /// Count triangles for a specific node.
    pub fn count_node_triangles(graph: &CsrGraph, node: u64) -> u64 {
        let neighbors: HashSet<u64> = graph.neighbors(node).iter().copied().collect();

        let mut count = 0u64;

        for &v in graph.neighbors(node) {
            for &w in graph.neighbors(v) {
                if w != node && neighbors.contains(&w) {
                    count += 1;
                }
            }
        }

        // Each triangle is counted twice (once for each direction)
        count / 2
    }
}

impl Default for TriangleCounting {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for TriangleCounting {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Motif Detection Kernel
// ============================================================================

/// Types of 3-node motifs (triads).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TriadType {
    /// No edges (independent nodes)
    Empty,
    /// One edge (a single connection)
    Edge,
    /// Two edges forming a path (wedge)
    Wedge,
    /// Three edges forming a triangle
    Triangle,
}

/// Result of motif detection.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MotifResult {
    /// Count of each motif type.
    pub motif_counts: std::collections::HashMap<String, u64>,
}

/// Motif detection kernel.
///
/// Counts occurrences of small subgraph patterns.
#[derive(Debug, Clone)]
pub struct MotifDetection {
    metadata: KernelMetadata,
}

impl MotifDetection {
    /// Create a new motif detection kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/motif-detection", Domain::GraphAnalytics)
                .with_description("Motif detection (k-node subgraph census)")
                .with_throughput(10_000)
                .with_latency_us(100.0),
        }
    }

    /// Count 3-node motifs (triads) in the graph.
    pub fn count_triads(graph: &CsrGraph) -> MotifResult {
        let n = graph.num_nodes;
        let mut triangles = 0u64;
        let mut wedges = 0u64;
        let mut edges = 0u64;

        // Count triangles and wedges
        for u in 0..n {
            let neighbors_u: HashSet<u64> = graph.neighbors(u as u64).iter().copied().collect();
            let degree_u = neighbors_u.len();

            // Wedges centered at u: C(degree, 2)
            if degree_u >= 2 {
                let potential_wedges = (degree_u * (degree_u - 1)) / 2;

                // Count how many of these are actually triangles
                let mut triangles_at_u = 0u64;
                for &v in graph.neighbors(u as u64) {
                    for &w in graph.neighbors(v) {
                        if w != u as u64 && neighbors_u.contains(&w) && v < w {
                            triangles_at_u += 1;
                        }
                    }
                }

                wedges += potential_wedges as u64 - triangles_at_u;
                triangles += triangles_at_u;
            }
        }

        // Triangles were counted 3 times (once per vertex)
        triangles /= 3;

        // Edges count
        edges = graph.num_edges as u64 / 2; // Undirected edges

        let mut motif_counts = std::collections::HashMap::new();
        motif_counts.insert("triangles".to_string(), triangles);
        motif_counts.insert("wedges".to_string(), wedges);
        motif_counts.insert("edges".to_string(), edges);

        MotifResult { motif_counts }
    }

    /// Classify a triad (set of 3 nodes) by its structure.
    pub fn classify_triad(graph: &CsrGraph, nodes: [u64; 3]) -> TriadType {
        let [a, b, c] = nodes;

        let neighbors_a: HashSet<u64> = graph.neighbors(a).iter().copied().collect();
        let neighbors_b: HashSet<u64> = graph.neighbors(b).iter().copied().collect();

        let ab = neighbors_a.contains(&b);
        let ac = neighbors_a.contains(&c);
        let bc = neighbors_b.contains(&c);

        let edge_count = ab as u8 + ac as u8 + bc as u8;

        match edge_count {
            0 => TriadType::Empty,
            1 => TriadType::Edge,
            2 => TriadType::Wedge,
            3 => TriadType::Triangle,
            _ => unreachable!(),
        }
    }
}

impl Default for MotifDetection {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for MotifDetection {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// K-Clique Detection Kernel
// ============================================================================

/// K-clique detection kernel.
///
/// Finds all cliques of size k in the graph.
#[derive(Debug, Clone)]
pub struct KCliqueDetection {
    metadata: KernelMetadata,
}

impl KCliqueDetection {
    /// Create a new k-clique detection kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/k-clique", Domain::GraphAnalytics)
                .with_description("K-clique detection")
                .with_throughput(1_000)
                .with_latency_us(1000.0),
        }
    }

    /// Find all cliques of size k.
    ///
    /// Uses Bron-Kerbosch algorithm with pivoting.
    pub fn find_cliques(graph: &CsrGraph, k: usize) -> Vec<Vec<u64>> {
        let n = graph.num_nodes;
        let mut cliques = Vec::new();

        // Build adjacency set for each node
        let adj: Vec<HashSet<u64>> = (0..n)
            .map(|i| graph.neighbors(i as u64).iter().copied().collect())
            .collect();

        // Bron-Kerbosch with size limit
        let mut current_clique = Vec::new();
        let candidates: HashSet<u64> = (0..n as u64).collect();
        let excluded: HashSet<u64> = HashSet::new();

        Self::bron_kerbosch(
            &adj,
            &mut current_clique,
            candidates,
            excluded,
            k,
            &mut cliques,
        );

        cliques
    }

    fn bron_kerbosch(
        adj: &[HashSet<u64>],
        current: &mut Vec<u64>,
        mut candidates: HashSet<u64>,
        mut excluded: HashSet<u64>,
        k: usize,
        cliques: &mut Vec<Vec<u64>>,
    ) {
        // Found a clique of size k
        if current.len() == k {
            cliques.push(current.clone());
            return;
        }

        // Can't reach size k
        if current.len() + candidates.len() < k {
            return;
        }

        // No more candidates
        if candidates.is_empty() {
            return;
        }

        // Choose pivot (node with most connections to candidates)
        let pivot = candidates
            .iter()
            .chain(excluded.iter())
            .max_by_key(|&&v| adj[v as usize].intersection(&candidates).count())
            .copied();

        let pivot_neighbors = pivot.map(|p| adj[p as usize].clone()).unwrap_or_default();

        let to_explore: Vec<u64> = candidates.difference(&pivot_neighbors).copied().collect();

        for v in to_explore {
            current.push(v);

            let new_candidates: HashSet<u64> =
                candidates.intersection(&adj[v as usize]).copied().collect();
            let new_excluded: HashSet<u64> =
                excluded.intersection(&adj[v as usize]).copied().collect();

            Self::bron_kerbosch(adj, current, new_candidates, new_excluded, k, cliques);

            current.pop();
            candidates.remove(&v);
            excluded.insert(v);
        }
    }

    /// Count cliques of size k (more efficient than enumerating all).
    pub fn count_cliques(graph: &CsrGraph, k: usize) -> u64 {
        Self::find_cliques(graph, k).len() as u64
    }
}

impl Default for KCliqueDetection {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for KCliqueDetection {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_triangle_graph() -> CsrGraph {
        // Graph with one triangle: 0-1-2-0
        CsrGraph::from_edges(3, &[(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)])
    }

    fn create_square_graph() -> CsrGraph {
        // Graph with square: 0-1-2-3-0 (no triangles)
        CsrGraph::from_edges(
            4,
            &[
                (0, 1),
                (1, 0),
                (1, 2),
                (2, 1),
                (2, 3),
                (3, 2),
                (3, 0),
                (0, 3),
            ],
        )
    }

    #[test]
    fn test_triangle_counting_metadata() {
        let kernel = TriangleCounting::new();
        assert_eq!(kernel.metadata().id, "graph/triangle-counting");
        assert_eq!(kernel.metadata().domain, Domain::GraphAnalytics);
    }

    #[test]
    fn test_triangle_counting() {
        let graph = create_triangle_graph();
        let result = TriangleCounting::compute(&graph);

        assert_eq!(result.total_triangles, 1, "Expected 1 triangle");

        // Each node participates in 1 triangle
        for &count in &result.per_node_triangles {
            assert_eq!(count, 1);
        }

        // Clustering coefficient should be 1.0 for a complete graph
        assert!((result.global_clustering_coefficient - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_no_triangles() {
        let graph = create_square_graph();
        let result = TriangleCounting::compute(&graph);

        assert_eq!(result.total_triangles, 0, "Expected 0 triangles in square");
        assert!((result.global_clustering_coefficient).abs() < 0.01);
    }

    #[test]
    fn test_triad_classification() {
        let graph = create_triangle_graph();

        let triad_type = MotifDetection::classify_triad(&graph, [0, 1, 2]);
        assert_eq!(triad_type, TriadType::Triangle);
    }

    #[test]
    fn test_motif_detection() {
        let graph = create_triangle_graph();
        let result = MotifDetection::count_triads(&graph);

        assert_eq!(result.motif_counts.get("triangles"), Some(&1));
    }

    #[test]
    fn test_k_clique_triangles() {
        let graph = create_triangle_graph();
        let cliques = KCliqueDetection::find_cliques(&graph, 3);

        // Should find one 3-clique (the triangle)
        assert_eq!(cliques.len(), 1);
    }

    #[test]
    fn test_k_clique_edges() {
        let graph = create_square_graph();
        let cliques = KCliqueDetection::find_cliques(&graph, 2);

        // Should find 4 edges (2-cliques)
        assert_eq!(cliques.len(), 4);
    }
}
