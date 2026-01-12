//! Graph-level metrics kernels.
//!
//! This module provides graph-wide metrics:
//! - Graph density
//! - Average path length
//! - Diameter
//! - Average clustering coefficient

use crate::types::CsrGraph;
use rustkernel_core::{
    domain::Domain,
    kernel::KernelMetadata,
    traits::GpuKernel,
};
use std::collections::VecDeque;

/// Result of graph metrics computation.
#[derive(Debug, Clone)]
pub struct GraphMetricsResult {
    /// Graph density: 2E / (V * (V-1))
    pub density: f64,
    /// Average shortest path length.
    pub average_path_length: f64,
    /// Graph diameter (longest shortest path).
    pub diameter: u64,
    /// Average clustering coefficient.
    pub average_clustering_coefficient: f64,
    /// Number of connected components.
    pub num_components: usize,
    /// Size of largest component.
    pub largest_component_size: usize,
    /// Is the graph connected?
    pub is_connected: bool,
}

// ============================================================================
// Graph Density Kernel
// ============================================================================

/// Graph density kernel.
///
/// Computes graph density: 2E / (V * (V-1)) for undirected graphs.
#[derive(Debug, Clone)]
pub struct GraphDensity {
    metadata: KernelMetadata,
}

impl GraphDensity {
    /// Create a new graph density kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/density", Domain::GraphAnalytics)
                .with_description("Graph density calculation")
                .with_throughput(1_000_000)
                .with_latency_us(1.0),
        }
    }

    /// Compute graph density.
    pub fn compute(graph: &CsrGraph) -> f64 {
        graph.density()
    }
}

impl Default for GraphDensity {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for GraphDensity {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Average Path Length Kernel
// ============================================================================

/// Average path length kernel.
///
/// Computes the average shortest path length using BFS from each node.
#[derive(Debug, Clone)]
pub struct AveragePathLength {
    metadata: KernelMetadata,
}

impl AveragePathLength {
    /// Create a new average path length kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/average-path-length", Domain::GraphAnalytics)
                .with_description("Average shortest path length (BFS)")
                .with_throughput(10_000)
                .with_latency_us(100.0),
        }
    }

    /// Compute average path length and diameter.
    ///
    /// Returns (average_path_length, diameter).
    pub fn compute(graph: &CsrGraph) -> (f64, u64) {
        let n = graph.num_nodes;
        if n <= 1 {
            return (0.0, 0);
        }

        let mut total_distance = 0u64;
        let mut num_pairs = 0u64;
        let mut diameter = 0u64;

        for source in 0..n {
            let distances = Self::bfs_distances(graph, source);

            for (target, &dist) in distances.iter().enumerate() {
                if target != source && dist > 0 {
                    total_distance += dist as u64;
                    num_pairs += 1;
                    if dist as u64 > diameter {
                        diameter = dist as u64;
                    }
                }
            }
        }

        let average = if num_pairs > 0 {
            total_distance as f64 / num_pairs as f64
        } else {
            0.0
        };

        (average, diameter)
    }

    /// BFS to compute distances from source to all other nodes.
    fn bfs_distances(graph: &CsrGraph, source: usize) -> Vec<i64> {
        let n = graph.num_nodes;
        let mut distances = vec![-1i64; n];
        distances[source] = 0;

        let mut queue = VecDeque::new();
        queue.push_back(source);

        while let Some(v) = queue.pop_front() {
            for &w in graph.neighbors(v as u64) {
                let w = w as usize;
                if distances[w] < 0 {
                    distances[w] = distances[v] + 1;
                    queue.push_back(w);
                }
            }
        }

        distances
    }
}

impl Default for AveragePathLength {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for AveragePathLength {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Clustering Coefficient Kernel
// ============================================================================

/// Clustering coefficient kernel.
///
/// Computes local and global clustering coefficients.
#[derive(Debug, Clone)]
pub struct ClusteringCoefficient {
    metadata: KernelMetadata,
}

impl ClusteringCoefficient {
    /// Create a new clustering coefficient kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/clustering-coefficient", Domain::GraphAnalytics)
                .with_description("Clustering coefficient (local and global)")
                .with_throughput(50_000)
                .with_latency_us(20.0),
        }
    }

    /// Compute local clustering coefficient for a single node.
    ///
    /// CC(v) = 2 * triangles(v) / (degree(v) * (degree(v) - 1))
    pub fn compute_local(graph: &CsrGraph, node: u64) -> f64 {
        let degree = graph.out_degree(node) as usize;
        if degree < 2 {
            return 0.0;
        }

        let neighbors: std::collections::HashSet<u64> = graph.neighbors(node).iter().copied().collect();

        let mut triangles = 0u64;
        for &v in graph.neighbors(node) {
            for &w in graph.neighbors(v) {
                if w != node && neighbors.contains(&w) {
                    triangles += 1;
                }
            }
        }

        // Each triangle is counted twice
        triangles /= 2;

        let possible = (degree * (degree - 1)) / 2;
        triangles as f64 / possible as f64
    }

    /// Compute average clustering coefficient for the entire graph.
    pub fn compute_average(graph: &CsrGraph) -> f64 {
        let n = graph.num_nodes;
        if n == 0 {
            return 0.0;
        }

        let sum: f64 = (0..n)
            .map(|i| Self::compute_local(graph, i as u64))
            .sum();

        sum / n as f64
    }

    /// Compute global clustering coefficient.
    ///
    /// 3 * triangles / wedges
    pub fn compute_global(graph: &CsrGraph) -> f64 {
        let n = graph.num_nodes;
        let mut triangles = 0u64;
        let mut wedges = 0u64;

        for u in 0..n {
            let neighbors: std::collections::HashSet<u64> = graph.neighbors(u as u64).iter().copied().collect();
            let degree = neighbors.len();

            if degree >= 2 {
                // Count wedges centered at u
                wedges += (degree * (degree - 1)) as u64 / 2;

                // Count triangles
                for &v in graph.neighbors(u as u64) {
                    for &w in graph.neighbors(v) {
                        if w != u as u64 && neighbors.contains(&w) && v < w {
                            triangles += 1;
                        }
                    }
                }
            }
        }

        if wedges == 0 {
            0.0
        } else {
            triangles as f64 / wedges as f64
        }
    }
}

impl Default for ClusteringCoefficient {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for ClusteringCoefficient {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Connected Components Kernel
// ============================================================================

/// Connected components kernel.
///
/// Finds connected components using BFS.
#[derive(Debug, Clone)]
pub struct ConnectedComponents {
    metadata: KernelMetadata,
}

impl ConnectedComponents {
    /// Create a new connected components kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/connected-components", Domain::GraphAnalytics)
                .with_description("Connected components (BFS)")
                .with_throughput(100_000)
                .with_latency_us(10.0),
        }
    }

    /// Find connected components.
    ///
    /// Returns component labels for each node.
    pub fn compute(graph: &CsrGraph) -> Vec<u64> {
        let n = graph.num_nodes;
        let mut labels = vec![u64::MAX; n];
        let mut current_label = 0u64;

        for start in 0..n {
            if labels[start] != u64::MAX {
                continue;
            }

            // BFS from this node
            let mut queue = VecDeque::new();
            queue.push_back(start);
            labels[start] = current_label;

            while let Some(v) = queue.pop_front() {
                for &w in graph.neighbors(v as u64) {
                    let w = w as usize;
                    if labels[w] == u64::MAX {
                        labels[w] = current_label;
                        queue.push_back(w);
                    }
                }
            }

            current_label += 1;
        }

        labels
    }

    /// Count the number of connected components.
    pub fn count_components(graph: &CsrGraph) -> usize {
        let labels = Self::compute(graph);
        labels.iter().copied().max().map_or(0, |m| m as usize + 1)
    }

    /// Get component sizes.
    pub fn component_sizes(graph: &CsrGraph) -> Vec<usize> {
        let labels = Self::compute(graph);
        let num_components = labels.iter().copied().max().map_or(0, |m| m as usize + 1);

        let mut sizes = vec![0usize; num_components];
        for label in labels {
            sizes[label as usize] += 1;
        }

        sizes
    }
}

impl Default for ConnectedComponents {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for ConnectedComponents {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Full Graph Metrics Kernel
// ============================================================================

/// Full graph metrics kernel.
///
/// Computes all graph-level metrics at once.
#[derive(Debug, Clone)]
pub struct FullGraphMetrics {
    metadata: KernelMetadata,
}

impl FullGraphMetrics {
    /// Create a new full graph metrics kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/full-metrics", Domain::GraphAnalytics)
                .with_description("Complete graph metrics")
                .with_throughput(1_000)
                .with_latency_us(1000.0),
        }
    }

    /// Compute all graph metrics.
    pub fn compute(graph: &CsrGraph) -> GraphMetricsResult {
        let density = GraphDensity::compute(graph);
        let (average_path_length, diameter) = AveragePathLength::compute(graph);
        let average_clustering_coefficient = ClusteringCoefficient::compute_average(graph);

        let component_sizes = ConnectedComponents::component_sizes(graph);
        let num_components = component_sizes.len();
        let largest_component_size = component_sizes.iter().copied().max().unwrap_or(0);
        let is_connected = num_components <= 1;

        GraphMetricsResult {
            density,
            average_path_length,
            diameter,
            average_clustering_coefficient,
            num_components,
            largest_component_size,
            is_connected,
        }
    }
}

impl Default for FullGraphMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for FullGraphMetrics {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_complete_graph() -> CsrGraph {
        // Complete graph K4
        CsrGraph::from_edges(4, &[
            (0, 1), (0, 2), (0, 3),
            (1, 0), (1, 2), (1, 3),
            (2, 0), (2, 1), (2, 3),
            (3, 0), (3, 1), (3, 2),
        ])
    }

    fn create_line_graph() -> CsrGraph {
        // Line: 0 - 1 - 2 - 3
        CsrGraph::from_edges(4, &[
            (0, 1), (1, 0),
            (1, 2), (2, 1),
            (2, 3), (3, 2),
        ])
    }

    fn create_disconnected_graph() -> CsrGraph {
        // Two disconnected components: 0-1 and 2-3
        CsrGraph::from_edges(4, &[
            (0, 1), (1, 0),
            (2, 3), (3, 2),
        ])
    }

    #[test]
    fn test_graph_density_complete() {
        let graph = create_complete_graph();
        let density = GraphDensity::compute(&graph);

        // Complete graph should have density 1.0
        assert!((density - 1.0).abs() < 0.01, "Expected 1.0, got {}", density);
    }

    #[test]
    fn test_graph_density_line() {
        let graph = create_line_graph();
        let density = GraphDensity::compute(&graph);

        // Line graph: 3 edges out of 6 possible = 0.5
        assert!((density - 0.5).abs() < 0.01, "Expected 0.5, got {}", density);
    }

    #[test]
    fn test_average_path_length_complete() {
        let graph = create_complete_graph();
        let (avg, diameter) = AveragePathLength::compute(&graph);

        // In complete graph, all paths are length 1
        assert!((avg - 1.0).abs() < 0.01, "Expected 1.0, got {}", avg);
        assert_eq!(diameter, 1);
    }

    #[test]
    fn test_average_path_length_line() {
        let graph = create_line_graph();
        let (avg, diameter) = AveragePathLength::compute(&graph);

        // Diameter of line 0-1-2-3 is 3
        assert_eq!(diameter, 3);

        // Average: (1+2+3+1+1+2+2+1+1+3+2+1) / 12 = 20/12 = 1.67
        assert!(avg > 1.0 && avg < 3.0);
    }

    #[test]
    fn test_clustering_coefficient_complete() {
        let graph = create_complete_graph();
        let cc = ClusteringCoefficient::compute_average(&graph);

        // Complete graph has CC = 1.0
        assert!((cc - 1.0).abs() < 0.01, "Expected 1.0, got {}", cc);
    }

    #[test]
    fn test_clustering_coefficient_line() {
        let graph = create_line_graph();
        let cc = ClusteringCoefficient::compute_average(&graph);

        // Line graph has no triangles, CC = 0
        assert!((cc).abs() < 0.01, "Expected 0.0, got {}", cc);
    }

    #[test]
    fn test_connected_components() {
        let graph = create_disconnected_graph();
        let num = ConnectedComponents::count_components(&graph);

        assert_eq!(num, 2);
    }

    #[test]
    fn test_connected_graph() {
        let graph = create_line_graph();
        let num = ConnectedComponents::count_components(&graph);

        assert_eq!(num, 1);
    }

    #[test]
    fn test_full_metrics() {
        let graph = create_complete_graph();
        let metrics = FullGraphMetrics::compute(&graph);

        assert!((metrics.density - 1.0).abs() < 0.01);
        assert!(metrics.is_connected);
        assert_eq!(metrics.num_components, 1);
    }
}
