//! Graph topology analysis kernels.
//!
//! This module provides topology analysis:
//! - Degree ratio (in/out degree analysis)
//! - Star topology score (hub-and-spoke detection)

use crate::types::CsrGraph;
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashSet;

// ============================================================================
// Degree Ratio Kernel
// ============================================================================

/// Degree ratio result for a node.
#[derive(Debug, Clone)]
pub struct DegreeRatioResult {
    /// Node index.
    pub node_index: usize,
    /// In-degree (incoming edges).
    pub in_degree: u32,
    /// Out-degree (outgoing edges).
    pub out_degree: u32,
    /// In/out ratio (infinity if out-degree=0).
    pub ratio: f64,
    /// Degree variance among neighbors.
    pub variance: f64,
    /// Classification based on ratio.
    pub classification: NodeClassification,
}

/// Node classification based on degree ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeClassification {
    /// Source node (ratio ≈ 0, mostly outgoing).
    Source,
    /// Balanced node (ratio ≈ 1, equal in/out).
    Balanced,
    /// Sink node (ratio > threshold, mostly incoming).
    Sink,
    /// Isolated node (no edges).
    Isolated,
}

/// Degree ratio kernel.
///
/// Calculates in-degree/out-degree ratio for source/sink classification.
/// Critical for cash flow analysis and account role identification.
#[derive(Debug, Clone)]
pub struct DegreeRatio {
    metadata: KernelMetadata,
}

impl Default for DegreeRatio {
    fn default() -> Self {
        Self::new()
    }
}

impl DegreeRatio {
    /// Create a new degree ratio kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("graph/degree-ratio", Domain::GraphAnalytics)
                .with_description("In/out degree ratio for node classification")
                .with_throughput(1_000_000)
                .with_latency_us(0.3),
        }
    }

    /// Compute degree ratio for a single node.
    ///
    /// # Arguments
    /// * `graph` - Input graph (CSR format)
    /// * `in_degrees` - Precomputed in-degrees (or None to compute on-the-fly)
    /// * `node` - Node index
    pub fn compute_single(
        graph: &CsrGraph,
        in_degrees: Option<&[u32]>,
        node: usize,
    ) -> DegreeRatioResult {
        let out_degree = graph.out_degree(node as u64) as u32;

        let in_degree = if let Some(in_deg) = in_degrees {
            in_deg.get(node).copied().unwrap_or(0)
        } else {
            // Compute in-degree by scanning all edges
            Self::compute_in_degree(graph, node)
        };

        let (ratio, classification) = if out_degree == 0 && in_degree == 0 {
            (0.0, NodeClassification::Isolated)
        } else if out_degree == 0 {
            (f64::INFINITY, NodeClassification::Sink)
        } else {
            let r = in_degree as f64 / out_degree as f64;
            let class = if r < 0.2 {
                NodeClassification::Source
            } else if r > 5.0 {
                NodeClassification::Sink
            } else {
                NodeClassification::Balanced
            };
            (r, class)
        };

        // Compute degree variance among neighbors
        let variance = Self::compute_neighbor_degree_variance(graph, node);

        DegreeRatioResult {
            node_index: node,
            in_degree,
            out_degree,
            ratio,
            variance,
            classification,
        }
    }

    /// Compute in-degree for a node by scanning all edges.
    fn compute_in_degree(graph: &CsrGraph, target: usize) -> u32 {
        let mut in_deg = 0u32;
        for source in 0..graph.num_nodes {
            for &neighbor in graph.neighbors(source as u64) {
                if neighbor as usize == target {
                    in_deg += 1;
                }
            }
        }
        in_deg
    }

    /// Compute degree variance among neighbors.
    fn compute_neighbor_degree_variance(graph: &CsrGraph, node: usize) -> f64 {
        let neighbors = graph.neighbors(node as u64);
        if neighbors.is_empty() {
            return 0.0;
        }

        let degrees: Vec<f64> = neighbors
            .iter()
            .map(|&n| graph.out_degree(n) as f64)
            .collect();

        let mean = degrees.iter().sum::<f64>() / degrees.len() as f64;
        let variance = degrees.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / degrees.len() as f64;

        variance
    }

    /// Compute degree ratios for all nodes.
    pub fn compute_batch(graph: &CsrGraph) -> Vec<DegreeRatioResult> {
        // Precompute in-degrees for efficiency
        let in_degrees = Self::compute_all_in_degrees(graph);

        (0..graph.num_nodes)
            .map(|node| Self::compute_single(graph, Some(&in_degrees), node))
            .collect()
    }

    /// Compute in-degrees for all nodes.
    pub fn compute_all_in_degrees(graph: &CsrGraph) -> Vec<u32> {
        let mut in_degrees = vec![0u32; graph.num_nodes];

        for source in 0..graph.num_nodes {
            for &target in graph.neighbors(source as u64) {
                in_degrees[target as usize] += 1;
            }
        }

        in_degrees
    }

    /// Classify nodes by their role (source, balanced, sink).
    pub fn classify_nodes(graph: &CsrGraph) -> NodeRoleDistribution {
        let results = Self::compute_batch(graph);

        let mut sources = Vec::new();
        let mut balanced = Vec::new();
        let mut sinks = Vec::new();
        let mut isolated = Vec::new();

        for result in results {
            match result.classification {
                NodeClassification::Source => sources.push(result.node_index),
                NodeClassification::Balanced => balanced.push(result.node_index),
                NodeClassification::Sink => sinks.push(result.node_index),
                NodeClassification::Isolated => isolated.push(result.node_index),
            }
        }

        NodeRoleDistribution {
            sources,
            balanced,
            sinks,
            isolated,
        }
    }
}

impl GpuKernel for DegreeRatio {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Distribution of node roles in the graph.
#[derive(Debug, Clone)]
pub struct NodeRoleDistribution {
    /// Source nodes (mostly outgoing edges).
    pub sources: Vec<usize>,
    /// Balanced nodes (equal in/out).
    pub balanced: Vec<usize>,
    /// Sink nodes (mostly incoming edges).
    pub sinks: Vec<usize>,
    /// Isolated nodes (no edges).
    pub isolated: Vec<usize>,
}

// ============================================================================
// Star Topology Score Kernel
// ============================================================================

/// Star type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StarType {
    /// Not a star (low score or low degree).
    None,
    /// In-Star: Most edges point TO this node (collection pattern).
    InStar,
    /// Out-Star: Most edges point FROM this node (distribution pattern).
    OutStar,
    /// Mixed-Star: Both incoming and outgoing edges (money mule hub).
    Mixed,
}

/// Star topology result for a node.
#[derive(Debug, Clone)]
pub struct StarTopologyResult {
    /// Node index.
    pub node_index: usize,
    /// Star score [0,1] - 1.0 = perfect star, 0.0 = not a star.
    pub star_score: f64,
    /// Type of star topology.
    pub star_type: StarType,
    /// Number of spokes (neighbors).
    pub spoke_count: usize,
    /// Number of inter-spoke edges (ideally 0 for perfect star).
    pub inter_spoke_edges: usize,
}

/// Star topology score kernel.
///
/// Detects and scores star/hub-and-spoke topology patterns.
/// Critical for AML (smurfing detection) and fraud (money mule identification).
#[derive(Debug, Clone)]
pub struct StarTopologyScore {
    metadata: KernelMetadata,
    /// Minimum degree to be considered a potential hub.
    pub min_degree: usize,
}

impl Default for StarTopologyScore {
    fn default() -> Self {
        Self::new()
    }
}

impl StarTopologyScore {
    /// Create a new star topology score kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/star-topology", Domain::GraphAnalytics)
                .with_description("Star/hub-and-spoke topology score")
                .with_throughput(20_000)
                .with_latency_us(150.0),
            min_degree: 3,
        }
    }

    /// Create with custom minimum degree threshold.
    #[must_use]
    pub fn with_min_degree(min_degree: usize) -> Self {
        Self {
            min_degree,
            ..Self::new()
        }
    }

    /// Compute star topology score for a single node.
    pub fn compute_single(
        graph: &CsrGraph,
        in_degrees: Option<&[u32]>,
        node: usize,
        min_degree: usize,
    ) -> StarTopologyResult {
        let neighbors = graph.neighbors(node as u64);
        let spoke_count = neighbors.len();

        // Check minimum degree
        if spoke_count < min_degree {
            return StarTopologyResult {
                node_index: node,
                star_score: 0.0,
                star_type: StarType::None,
                spoke_count,
                inter_spoke_edges: 0,
            };
        }

        // Create spoke set for O(1) lookup
        let spoke_set: HashSet<u64> = neighbors.iter().copied().collect();

        // Count inter-spoke edges
        let mut inter_spoke_edges = 0usize;
        for &spoke in neighbors {
            for &neighbor_of_spoke in graph.neighbors(spoke) {
                if neighbor_of_spoke != node as u64 && spoke_set.contains(&neighbor_of_spoke) {
                    inter_spoke_edges += 1;
                }
            }
        }
        // Each edge counted twice
        inter_spoke_edges /= 2;

        // Maximum possible inter-spoke edges (complete graph among spokes)
        let max_inter_spoke = (spoke_count * (spoke_count - 1)) / 2;

        // Star score: 1.0 = perfect star (no inter-spoke edges), 0.0 = complete graph
        let star_score = if max_inter_spoke == 0 {
            0.0
        } else {
            1.0 - (inter_spoke_edges as f64 / max_inter_spoke as f64)
        };

        // Classify star type based on edge directionality
        let out_degree = graph.out_degree(node as u64) as usize;
        let in_degree = if let Some(in_deg) = in_degrees {
            in_deg.get(node).copied().unwrap_or(0) as usize
        } else {
            DegreeRatio::compute_in_degree(graph, node) as usize
        };

        let total_edges = in_degree + out_degree;
        let star_type = if total_edges == 0 {
            StarType::None
        } else if in_degree as f64 > 0.9 * total_edges as f64 {
            StarType::InStar
        } else if out_degree as f64 > 0.9 * total_edges as f64 {
            StarType::OutStar
        } else {
            StarType::Mixed
        };

        StarTopologyResult {
            node_index: node,
            star_score,
            star_type,
            spoke_count,
            inter_spoke_edges,
        }
    }

    /// Compute star topology scores for all nodes.
    pub fn compute_batch(&self, graph: &CsrGraph) -> Vec<StarTopologyResult> {
        let in_degrees = DegreeRatio::compute_all_in_degrees(graph);

        (0..graph.num_nodes)
            .map(|node| Self::compute_single(graph, Some(&in_degrees), node, self.min_degree))
            .collect()
    }

    /// Find top-K star hubs.
    pub fn top_k_hubs(&self, graph: &CsrGraph, k: usize) -> Vec<StarTopologyResult> {
        let mut results = self.compute_batch(graph);

        // Sort by star score descending
        results.sort_by(|a, b| {
            b.star_score
                .partial_cmp(&a.star_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.into_iter().take(k).collect()
    }

    /// Find in-star hubs (collection accounts).
    pub fn find_in_stars(&self, graph: &CsrGraph, min_score: f64) -> Vec<StarTopologyResult> {
        self.compute_batch(graph)
            .into_iter()
            .filter(|r| r.star_type == StarType::InStar && r.star_score >= min_score)
            .collect()
    }

    /// Find out-star hubs (distribution accounts - potential smurfing).
    pub fn find_out_stars(&self, graph: &CsrGraph, min_score: f64) -> Vec<StarTopologyResult> {
        self.compute_batch(graph)
            .into_iter()
            .filter(|r| r.star_type == StarType::OutStar && r.star_score >= min_score)
            .collect()
    }
}

impl GpuKernel for StarTopologyScore {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_star_graph() -> CsrGraph {
        // Star graph: node 0 is hub, connected to 1,2,3,4
        // 0 -> 1, 0 -> 2, 0 -> 3, 0 -> 4 (out-star)
        CsrGraph::from_edges(
            5,
            &[(0, 1), (0, 2), (0, 3), (0, 4)],
        )
    }

    fn create_bidirectional_star() -> CsrGraph {
        // Bidirectional star: node 0 is hub, connected to 1,2,3,4 in both directions
        CsrGraph::from_edges(
            5,
            &[
                (0, 1), (0, 2), (0, 3), (0, 4),
                (1, 0), (2, 0), (3, 0), (4, 0),
            ],
        )
    }

    fn create_partial_star() -> CsrGraph {
        // Partial star with some inter-spoke edges
        CsrGraph::from_edges(
            5,
            &[
                (0, 1), (0, 2), (0, 3), (0, 4), // Hub edges
                (1, 2), (2, 1), // Inter-spoke edge (bidirectional)
            ],
        )
    }

    #[test]
    fn test_degree_ratio_metadata() {
        let kernel = DegreeRatio::new();
        assert_eq!(kernel.metadata().id, "graph/degree-ratio");
        assert_eq!(kernel.metadata().domain, Domain::GraphAnalytics);
    }

    #[test]
    fn test_degree_ratio_out_star() {
        let graph = create_star_graph();
        let result = DegreeRatio::compute_single(&graph, None, 0);

        assert_eq!(result.out_degree, 4);
        assert_eq!(result.in_degree, 0);
        assert_eq!(result.ratio, 0.0);
        assert_eq!(result.classification, NodeClassification::Source);
    }

    #[test]
    fn test_degree_ratio_bidirectional() {
        let graph = create_bidirectional_star();
        let result = DegreeRatio::compute_single(&graph, None, 0);

        assert_eq!(result.out_degree, 4);
        assert_eq!(result.in_degree, 4);
        assert!((result.ratio - 1.0).abs() < 0.01);
        assert_eq!(result.classification, NodeClassification::Balanced);
    }

    #[test]
    fn test_star_topology_perfect_star() {
        let graph = create_star_graph();
        let result = StarTopologyScore::compute_single(&graph, None, 0, 3);

        assert_eq!(result.spoke_count, 4);
        assert_eq!(result.inter_spoke_edges, 0);
        assert!((result.star_score - 1.0).abs() < 0.01);
        assert_eq!(result.star_type, StarType::OutStar);
    }

    #[test]
    fn test_star_topology_bidirectional() {
        let graph = create_bidirectional_star();
        let result = StarTopologyScore::compute_single(&graph, None, 0, 3);

        // Bidirectional star - should be Mixed type
        assert_eq!(result.spoke_count, 4);
        assert_eq!(result.inter_spoke_edges, 0);
        assert!((result.star_score - 1.0).abs() < 0.01, "Star score should be 1.0 for perfect star");
        assert_eq!(result.star_type, StarType::Mixed, "Bidirectional star should be Mixed type");
    }

    #[test]
    fn test_star_topology_partial() {
        let graph = create_partial_star();
        let result = StarTopologyScore::compute_single(&graph, None, 0, 3);

        assert_eq!(result.spoke_count, 4);
        // One bidirectional inter-spoke edge between 1 and 2
        assert!(result.inter_spoke_edges >= 1, "Should detect inter-spoke edge");
        // 6 possible inter-spoke edges, 1 exists -> score = 1 - 1/6 ≈ 0.83
        assert!(result.star_score > 0.5 && result.star_score < 1.0,
                "Star score {} should be between 0.5 and 1.0", result.star_score);
    }

    #[test]
    fn test_star_topology_top_k() {
        let graph = create_star_graph();
        let kernel = StarTopologyScore::with_min_degree(2);
        let top = kernel.top_k_hubs(&graph, 2);

        assert!(!top.is_empty());
        assert_eq!(top[0].node_index, 0); // Hub should be first
    }

    #[test]
    fn test_node_classification() {
        let graph = create_star_graph();
        let roles = DegreeRatio::classify_nodes(&graph);

        assert!(roles.sources.contains(&0)); // Hub is source
        assert!(roles.sinks.len() + roles.isolated.len() >= 1); // Spoke nodes
    }
}
