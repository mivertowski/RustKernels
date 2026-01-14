//! Shortest path kernels.
//!
//! This module provides shortest path algorithms:
//! - Single-source shortest path (SSSP) via BFS/Delta-Stepping
//! - All-pairs shortest path (APSP)
//! - K-shortest paths (Yen's algorithm)

use crate::types::CsrGraph;
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

// ============================================================================
// Shortest Path Results
// ============================================================================

/// Result of single-source shortest path calculation.
#[derive(Debug, Clone)]
pub struct ShortestPathResult {
    /// Node index.
    pub node_index: usize,
    /// Shortest distance from source (f64::INFINITY if unreachable).
    pub distance: f64,
    /// Predecessor node index on shortest path (-1 if no path).
    pub predecessor: i64,
    /// Whether node is reachable from source.
    pub is_reachable: bool,
    /// Number of hops (for unweighted graphs).
    pub hop_count: u32,
}

/// A single path result.
#[derive(Debug, Clone)]
pub struct PathResult {
    /// Source node index.
    pub source: usize,
    /// Target node index.
    pub target: usize,
    /// Total path length (sum of edge weights).
    pub path_length: f64,
    /// Number of hops (edges) in path.
    pub hop_count: usize,
    /// Ordered list of node indices along the path.
    pub node_path: Vec<usize>,
}

/// All-pairs shortest path result.
#[derive(Debug, Clone)]
pub struct AllPairsResult {
    /// Number of nodes.
    pub node_count: usize,
    /// Distance matrix in row-major order.
    /// distances[i * node_count + j] = shortest distance from node i to node j.
    pub distances: Vec<f64>,
    /// Predecessor matrix for path reconstruction.
    pub predecessors: Vec<i64>,
}

impl AllPairsResult {
    /// Get distance from source to target.
    pub fn distance(&self, source: usize, target: usize) -> f64 {
        self.distances[source * self.node_count + target]
    }

    /// Reconstruct path from source to target.
    pub fn reconstruct_path(&self, source: usize, target: usize) -> Option<Vec<usize>> {
        if !self.distance(source, target).is_finite() {
            return None;
        }

        let mut path = Vec::new();
        let mut current = target;

        while current != source {
            path.push(current);
            let pred = self.predecessors[source * self.node_count + current];
            if pred < 0 {
                return None;
            }
            current = pred as usize;
        }

        path.push(source);
        path.reverse();
        Some(path)
    }
}

// ============================================================================
// Shortest Path Kernel
// ============================================================================

/// Shortest path kernel using BFS (unweighted) or Delta-Stepping (weighted).
#[derive(Debug, Clone)]
pub struct ShortestPath {
    metadata: KernelMetadata,
}

impl Default for ShortestPath {
    fn default() -> Self {
        Self::new()
    }
}

impl ShortestPath {
    /// Create a new shortest path kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/shortest-path", Domain::GraphAnalytics)
                .with_description("Shortest path via BFS/Delta-Stepping")
                .with_throughput(50_000)
                .with_latency_us(80.0),
        }
    }

    /// Compute single-source shortest paths using BFS (for unweighted graphs).
    ///
    /// # Arguments
    /// * `graph` - Input graph (CSR format)
    /// * `source` - Source node index
    pub fn compute_sssp_bfs(graph: &CsrGraph, source: usize) -> Vec<ShortestPathResult> {
        let n = graph.num_nodes;
        let mut distances = vec![f64::INFINITY; n];
        let mut predecessors = vec![-1i64; n];
        let mut hop_counts = vec![0u32; n];

        distances[source] = 0.0;

        let mut queue = VecDeque::new();
        queue.push_back(source);

        while let Some(v) = queue.pop_front() {
            let current_dist = distances[v];

            for &w in graph.neighbors(v as u64) {
                let w = w as usize;
                if distances[w].is_infinite() {
                    distances[w] = current_dist + 1.0;
                    predecessors[w] = v as i64;
                    hop_counts[w] = hop_counts[v] + 1;
                    queue.push_back(w);
                }
            }
        }

        (0..n)
            .map(|i| ShortestPathResult {
                node_index: i,
                distance: distances[i],
                predecessor: predecessors[i],
                is_reachable: distances[i].is_finite(),
                hop_count: hop_counts[i],
            })
            .collect()
    }

    /// Compute single-source shortest paths using Dijkstra (for weighted graphs).
    ///
    /// # Arguments
    /// * `graph` - Input graph (CSR format)
    /// * `source` - Source node index
    /// * `weights` - Edge weights (parallel to graph edges)
    pub fn compute_sssp_dijkstra(
        graph: &CsrGraph,
        source: usize,
        weights: &[f64],
    ) -> Vec<ShortestPathResult> {
        let n = graph.num_nodes;
        let mut distances = vec![f64::INFINITY; n];
        let mut predecessors = vec![-1i64; n];
        let mut hop_counts = vec![0u32; n];

        distances[source] = 0.0;

        // Priority queue: (negative distance, node) - negated for min-heap behavior
        let mut heap = BinaryHeap::new();
        heap.push(HeapNode {
            dist: 0.0,
            node: source,
        });

        while let Some(HeapNode { dist, node: v }) = heap.pop() {
            if dist > distances[v] {
                continue; // Already processed with shorter distance
            }

            let neighbors = graph.neighbors(v as u64);
            let edge_start = if v == 0 {
                0
            } else {
                graph.row_offsets[v] as usize
            };

            for (i, &w) in neighbors.iter().enumerate() {
                let w = w as usize;
                let weight = weights.get(edge_start + i).copied().unwrap_or(1.0);
                let new_dist = distances[v] + weight;

                if new_dist < distances[w] {
                    distances[w] = new_dist;
                    predecessors[w] = v as i64;
                    hop_counts[w] = hop_counts[v] + 1;
                    heap.push(HeapNode {
                        dist: new_dist,
                        node: w,
                    });
                }
            }
        }

        (0..n)
            .map(|i| ShortestPathResult {
                node_index: i,
                distance: distances[i],
                predecessor: predecessors[i],
                is_reachable: distances[i].is_finite(),
                hop_count: hop_counts[i],
            })
            .collect()
    }

    /// Compute all-pairs shortest paths.
    pub fn compute_apsp(graph: &CsrGraph) -> AllPairsResult {
        let n = graph.num_nodes;
        let mut distances = vec![f64::INFINITY; n * n];
        let mut predecessors = vec![-1i64; n * n];

        // Run SSSP from each node
        for source in 0..n {
            let sssp = Self::compute_sssp_bfs(graph, source);

            for result in sssp {
                let idx = source * n + result.node_index;
                distances[idx] = result.distance;
                predecessors[idx] = result.predecessor;
            }
        }

        AllPairsResult {
            node_count: n,
            distances,
            predecessors,
        }
    }

    /// Reconstruct path from source to target.
    pub fn reconstruct_path(
        sssp: &[ShortestPathResult],
        source: usize,
        target: usize,
    ) -> Option<Vec<usize>> {
        if !sssp[target].is_reachable {
            return None;
        }

        let mut path = Vec::new();
        let mut current = target;

        while current != source {
            path.push(current);
            let pred = sssp[current].predecessor;
            if pred < 0 {
                return None;
            }
            current = pred as usize;
        }

        path.push(source);
        path.reverse();
        Some(path)
    }

    /// Compute shortest path between two nodes.
    pub fn compute_path(graph: &CsrGraph, source: usize, target: usize) -> Option<PathResult> {
        let sssp = Self::compute_sssp_bfs(graph, source);

        if !sssp[target].is_reachable {
            return None;
        }

        let node_path = Self::reconstruct_path(&sssp, source, target)?;

        Some(PathResult {
            source,
            target,
            path_length: sssp[target].distance,
            hop_count: node_path.len() - 1,
            node_path,
        })
    }

    /// Find k shortest paths using Yen's algorithm.
    pub fn compute_k_shortest(
        graph: &CsrGraph,
        source: usize,
        target: usize,
        k: usize,
    ) -> Vec<PathResult> {
        let mut result_paths = Vec::new();

        // First, find the shortest path
        if let Some(first_path) = Self::compute_path(graph, source, target) {
            result_paths.push(first_path);
        } else {
            return result_paths;
        }

        // Candidate paths
        let mut candidates: Vec<PathResult> = Vec::new();

        for _i in 1..k {
            let prev_path = &result_paths[result_paths.len() - 1];

            // For each deviation point on the previous path
            for j in 0..(prev_path.node_path.len() - 1) {
                let spur_node = prev_path.node_path[j];
                let root_path: Vec<usize> = prev_path.node_path[..=j].to_vec();

                // Create modified graph (remove edges used by previous paths at this deviation)
                // For simplicity, we'll use a less efficient but correct approach
                let edges_to_avoid = Self::collect_edges_to_avoid(&result_paths, &root_path);

                // Find path in modified graph
                if let Some(spur_path) =
                    Self::compute_path_avoiding(graph, spur_node, target, &edges_to_avoid)
                {
                    let mut total_path = root_path.clone();
                    total_path.extend(spur_path.node_path.into_iter().skip(1));

                    let path_length = (total_path.len() - 1) as f64;
                    let candidate = PathResult {
                        source,
                        target,
                        path_length,
                        hop_count: total_path.len() - 1,
                        node_path: total_path,
                    };

                    // Add if not already in candidates or results
                    if !Self::path_exists(&candidates, &candidate.node_path)
                        && !Self::path_exists_in_results(&result_paths, &candidate.node_path)
                    {
                        candidates.push(candidate);
                    }
                }
            }

            if candidates.is_empty() {
                break;
            }

            // Sort candidates by path length and take the best one
            candidates.sort_by(|a, b| {
                a.path_length
                    .partial_cmp(&b.path_length)
                    .unwrap_or(Ordering::Equal)
            });

            result_paths.push(candidates.remove(0));
        }

        result_paths
    }

    /// Compute path avoiding certain edges.
    fn compute_path_avoiding(
        graph: &CsrGraph,
        source: usize,
        target: usize,
        avoid_edges: &[(usize, usize)],
    ) -> Option<PathResult> {
        let n = graph.num_nodes;
        let mut distances = vec![f64::INFINITY; n];
        let mut predecessors = vec![-1i64; n];

        distances[source] = 0.0;

        let mut queue = VecDeque::new();
        queue.push_back(source);

        while let Some(v) = queue.pop_front() {
            if v == target {
                break;
            }

            let current_dist = distances[v];

            for &w in graph.neighbors(v as u64) {
                let w = w as usize;

                // Skip avoided edges
                if avoid_edges.contains(&(v, w)) {
                    continue;
                }

                if distances[w].is_infinite() {
                    distances[w] = current_dist + 1.0;
                    predecessors[w] = v as i64;
                    queue.push_back(w);
                }
            }
        }

        if distances[target].is_infinite() {
            return None;
        }

        // Reconstruct path
        let mut path = Vec::new();
        let mut current = target;

        while current != source {
            path.push(current);
            let pred = predecessors[current];
            if pred < 0 {
                return None;
            }
            current = pred as usize;
        }

        path.push(source);
        path.reverse();

        Some(PathResult {
            source,
            target,
            path_length: distances[target],
            hop_count: path.len() - 1,
            node_path: path,
        })
    }

    fn collect_edges_to_avoid(
        result_paths: &[PathResult],
        root_path: &[usize],
    ) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();

        for path in result_paths {
            // Check if this path shares the root
            if path.node_path.len() >= root_path.len()
                && path.node_path[..root_path.len()] == *root_path
            {
                // Add the edge right after root_path
                if path.node_path.len() > root_path.len() {
                    let from = root_path[root_path.len() - 1];
                    let to = path.node_path[root_path.len()];
                    edges.push((from, to));
                }
            }
        }

        edges
    }

    fn path_exists(candidates: &[PathResult], path: &[usize]) -> bool {
        candidates.iter().any(|c| c.node_path == path)
    }

    fn path_exists_in_results(results: &[PathResult], path: &[usize]) -> bool {
        results.iter().any(|r| r.node_path == path)
    }

    /// Compute eccentricity for each node (max distance to any other node).
    pub fn compute_eccentricity(graph: &CsrGraph) -> Vec<f64> {
        let n = graph.num_nodes;
        let mut eccentricities = vec![0.0; n];

        for source in 0..n {
            let sssp = Self::compute_sssp_bfs(graph, source);
            let max_dist = sssp
                .iter()
                .filter(|r| r.is_reachable)
                .map(|r| r.distance)
                .fold(0.0, f64::max);
            eccentricities[source] = max_dist;
        }

        eccentricities
    }

    /// Compute graph diameter (max eccentricity).
    pub fn compute_diameter(graph: &CsrGraph) -> f64 {
        Self::compute_eccentricity(graph)
            .into_iter()
            .fold(0.0, f64::max)
    }

    /// Compute graph radius (min eccentricity).
    pub fn compute_radius(graph: &CsrGraph) -> f64 {
        Self::compute_eccentricity(graph)
            .into_iter()
            .filter(|&e| e > 0.0)
            .fold(f64::INFINITY, f64::min)
    }
}

impl GpuKernel for ShortestPath {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Helper struct for Dijkstra's priority queue.
#[derive(Clone, PartialEq)]
struct HeapNode {
    dist: f64,
    node: usize,
}

impl Eq for HeapNode {}

impl Ord for HeapNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for HeapNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_line_graph() -> CsrGraph {
        // Line: 0 - 1 - 2 - 3
        CsrGraph::from_edges(4, &[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)])
    }

    fn create_complete_graph() -> CsrGraph {
        // Complete graph K4
        CsrGraph::from_edges(
            4,
            &[
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (2, 3),
                (3, 0),
                (3, 1),
                (3, 2),
            ],
        )
    }

    fn create_disconnected_graph() -> CsrGraph {
        // Two disconnected pairs: 0-1 and 2-3
        CsrGraph::from_edges(4, &[(0, 1), (1, 0), (2, 3), (3, 2)])
    }

    #[test]
    fn test_shortest_path_metadata() {
        let kernel = ShortestPath::new();
        assert_eq!(kernel.metadata().id, "graph/shortest-path");
        assert_eq!(kernel.metadata().domain, Domain::GraphAnalytics);
    }

    #[test]
    fn test_sssp_bfs_line() {
        let graph = create_line_graph();
        let sssp = ShortestPath::compute_sssp_bfs(&graph, 0);

        assert_eq!(sssp[0].distance, 0.0);
        assert_eq!(sssp[1].distance, 1.0);
        assert_eq!(sssp[2].distance, 2.0);
        assert_eq!(sssp[3].distance, 3.0);
    }

    #[test]
    fn test_sssp_bfs_complete() {
        let graph = create_complete_graph();
        let sssp = ShortestPath::compute_sssp_bfs(&graph, 0);

        // In complete graph, all nodes are distance 1 from any other
        assert_eq!(sssp[0].distance, 0.0);
        assert_eq!(sssp[1].distance, 1.0);
        assert_eq!(sssp[2].distance, 1.0);
        assert_eq!(sssp[3].distance, 1.0);
    }

    #[test]
    fn test_sssp_disconnected() {
        let graph = create_disconnected_graph();
        let sssp = ShortestPath::compute_sssp_bfs(&graph, 0);

        assert!(sssp[0].is_reachable);
        assert!(sssp[1].is_reachable);
        assert!(!sssp[2].is_reachable);
        assert!(!sssp[3].is_reachable);
    }

    #[test]
    fn test_reconstruct_path() {
        let graph = create_line_graph();
        let sssp = ShortestPath::compute_sssp_bfs(&graph, 0);

        let path = ShortestPath::reconstruct_path(&sssp, 0, 3);
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_compute_path() {
        let graph = create_line_graph();
        let path = ShortestPath::compute_path(&graph, 0, 3);

        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.hop_count, 3);
        assert_eq!(path.node_path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_apsp() {
        let graph = create_line_graph();
        let apsp = ShortestPath::compute_apsp(&graph);

        assert_eq!(apsp.distance(0, 3), 3.0);
        assert_eq!(apsp.distance(1, 2), 1.0);
        assert_eq!(apsp.distance(0, 0), 0.0);
    }

    #[test]
    fn test_diameter() {
        let graph = create_line_graph();
        let diameter = ShortestPath::compute_diameter(&graph);

        assert_eq!(diameter, 3.0);
    }

    #[test]
    fn test_k_shortest() {
        let graph = create_complete_graph();
        let paths = ShortestPath::compute_k_shortest(&graph, 0, 3, 3);

        assert!(paths.len() >= 1);
        assert_eq!(paths[0].hop_count, 1); // Direct edge
    }
}
