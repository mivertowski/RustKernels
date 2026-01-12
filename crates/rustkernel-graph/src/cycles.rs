//! Short cycle detection kernels.
//!
//! This module provides cycle participation analysis:
//! - 2-cycle (reciprocal edge) detection
//! - 3-cycle (triangle) detection - KEY AML INDICATOR
//! - 4-cycle (square) detection - CRITICAL AML INDICATOR

use crate::types::CsrGraph;
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashSet;

// ============================================================================
// Cycle Participation Result
// ============================================================================

/// Cycle participation result for a node.
#[derive(Debug, Clone)]
pub struct CycleParticipationResult {
    /// Node index.
    pub node_index: usize,
    /// Number of 2-cycles (reciprocal edges) this node participates in.
    pub cycle_count_2hop: u32,
    /// Number of 3-cycles (triangles) this node participates in - KEY AML INDICATOR.
    pub cycle_count_3hop: u32,
    /// Number of 4-cycles (squares) this node participates in - CRITICAL AML INDICATOR.
    pub cycle_count_4hop: u32,
    /// Total weight (sum of edge weights) in all cycles.
    pub total_cycle_weight: f64,
    /// Cycle ratio: fraction of edges that participate in cycles [0,1].
    pub cycle_ratio: f64,
    /// Risk level based on cycle participation.
    pub risk_level: CycleRiskLevel,
}

/// Risk level based on cycle participation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleRiskLevel {
    /// Baseline - zero or minimal 2-cycles.
    Baseline,
    /// Low risk - 1-2 triangles (could be legitimate).
    Low,
    /// Medium risk - 3+ triangles (investigate for layering).
    Medium,
    /// High risk - ANY 4-cycles (structured laundering pattern).
    High,
    /// Critical risk - Multiple 4-cycles (professional laundering ring).
    Critical,
}

// ============================================================================
// Short Cycle Participation Kernel
// ============================================================================

/// Short cycle participation kernel.
///
/// Detects participation in 2-4 hop cycles. Critical for AML:
/// - Triangles (3-cycles) indicate layering patterns
/// - Squares (4-cycles) indicate organized money laundering
#[derive(Debug, Clone)]
pub struct ShortCycleParticipation {
    metadata: KernelMetadata,
}

impl Default for ShortCycleParticipation {
    fn default() -> Self {
        Self::new()
    }
}

impl ShortCycleParticipation {
    /// Create a new short cycle participation kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/cycle-participation", Domain::GraphAnalytics)
                .with_description("Short cycle (2-4 hop) participation detection")
                .with_throughput(25_000)
                .with_latency_us(200.0),
        }
    }

    /// Detect 2-cycles (reciprocal edges) for all nodes.
    pub fn detect_2_cycles(graph: &CsrGraph) -> Vec<u32> {
        let n = graph.num_nodes;
        let mut cycle_counts = vec![0u32; n];

        // For each edge, check if reciprocal exists
        for u in 0..n {
            let neighbors_u: HashSet<u64> = graph.neighbors(u as u64).iter().copied().collect();

            for &v in graph.neighbors(u as u64) {
                // Check if edge v -> u exists (reciprocal)
                if graph.neighbors(v).contains(&(u as u64)) {
                    cycle_counts[u] += 1;
                }
            }
        }

        // Each reciprocal edge is counted once per endpoint
        // (already normalized since we check u->v and v->u separately)

        cycle_counts
    }

    /// Detect 3-cycles (triangles) for all nodes.
    ///
    /// This is the KEY AML INDICATOR for layering detection.
    pub fn detect_3_cycles(graph: &CsrGraph) -> Vec<u32> {
        let n = graph.num_nodes;
        let mut cycle_counts = vec![0u32; n];

        for u in 0..n {
            let neighbors_u: HashSet<u64> = graph.neighbors(u as u64).iter().copied().collect();

            for &v in graph.neighbors(u as u64) {
                if v as usize <= u {
                    continue; // Avoid counting twice
                }

                // Find common neighbors (triangles)
                for &w in graph.neighbors(v) {
                    if w as usize <= v as usize {
                        continue;
                    }

                    if neighbors_u.contains(&w) {
                        // Found triangle u-v-w
                        cycle_counts[u] += 1;
                        cycle_counts[v as usize] += 1;
                        cycle_counts[w as usize] += 1;
                    }
                }
            }
        }

        cycle_counts
    }

    /// Detect 4-cycles (squares) for all nodes.
    ///
    /// This is the CRITICAL AML INDICATOR for organized laundering.
    pub fn detect_4_cycles(graph: &CsrGraph) -> Vec<u32> {
        let n = graph.num_nodes;
        let mut cycle_counts = vec![0u32; n];

        // For efficiency, limit to smaller graphs or use sampling
        if n > 1000 {
            // Use sampling for large graphs
            return Self::detect_4_cycles_sampled(graph, 0.1);
        }

        for u in 0..n {
            let neighbors_u: HashSet<u64> = graph.neighbors(u as u64).iter().copied().collect();

            // Enumerate 2-paths starting from u: u -> v -> w
            for &v in graph.neighbors(u as u64) {
                for &w in graph.neighbors(v) {
                    if w as usize == u {
                        continue; // Skip immediate back-edge
                    }

                    // Look for x such that w -> x -> u forms a 4-cycle
                    for &x in graph.neighbors(w) {
                        if x as usize != u && x != v && neighbors_u.contains(&x) {
                            // Found 4-cycle: u -> v -> w -> x -> u
                            cycle_counts[u] += 1;
                        }
                    }
                }
            }
        }

        // Each 4-cycle is counted 4 times (once per node), normalize
        for count in &mut cycle_counts {
            *count /= 4;
        }

        cycle_counts
    }

    /// Detect 4-cycles using sampling for large graphs.
    fn detect_4_cycles_sampled(graph: &CsrGraph, sample_rate: f64) -> Vec<u32> {
        let n = graph.num_nodes;
        let mut cycle_counts = vec![0u32; n];

        let sample_count = (n as f64 * sample_rate).max(1.0) as usize;
        let step = (n / sample_count).max(1);

        for u in (0..n).step_by(step) {
            let neighbors_u: HashSet<u64> = graph.neighbors(u as u64).iter().copied().collect();

            for &v in graph.neighbors(u as u64) {
                for &w in graph.neighbors(v) {
                    if w as usize == u {
                        continue;
                    }

                    for &x in graph.neighbors(w) {
                        if x as usize != u && x != v && neighbors_u.contains(&x) {
                            cycle_counts[u] += 1;
                        }
                    }
                }
            }
        }

        // Scale up for sampling
        let scale = 1.0 / sample_rate;
        for count in &mut cycle_counts {
            *count = (*count as f64 * scale) as u32 / 4;
        }

        cycle_counts
    }

    /// Compute full cycle participation for all nodes.
    pub fn compute_all(graph: &CsrGraph) -> Vec<CycleParticipationResult> {
        let cycles_2 = Self::detect_2_cycles(graph);
        let cycles_3 = Self::detect_3_cycles(graph);
        let cycles_4 = Self::detect_4_cycles(graph);

        let n = graph.num_nodes;

        (0..n)
            .map(|i| {
                let c2 = cycles_2[i];
                let c3 = cycles_3[i];
                let c4 = cycles_4[i];

                let degree = graph.out_degree(i as u64);
                let cycle_ratio = if degree > 0 {
                    (c2 + c3 + c4) as f64 / degree as f64
                } else {
                    0.0
                };

                let risk_level = Self::calculate_risk_level(c2, c3, c4);

                CycleParticipationResult {
                    node_index: i,
                    cycle_count_2hop: c2,
                    cycle_count_3hop: c3,
                    cycle_count_4hop: c4,
                    total_cycle_weight: 0.0, // Would need weighted graph
                    cycle_ratio,
                    risk_level,
                }
            })
            .collect()
    }

    /// Calculate risk level based on cycle participation.
    fn calculate_risk_level(cycles_2: u32, cycles_3: u32, cycles_4: u32) -> CycleRiskLevel {
        if cycles_4 > 3 {
            CycleRiskLevel::Critical
        } else if cycles_4 > 0 {
            CycleRiskLevel::High
        } else if cycles_3 >= 3 {
            CycleRiskLevel::Medium
        } else if cycles_3 >= 1 || cycles_2 >= 3 {
            CycleRiskLevel::Low
        } else {
            CycleRiskLevel::Baseline
        }
    }

    /// Find high-risk nodes based on cycle participation.
    pub fn find_high_risk_nodes(graph: &CsrGraph) -> Vec<CycleParticipationResult> {
        Self::compute_all(graph)
            .into_iter()
            .filter(|r| matches!(r.risk_level, CycleRiskLevel::High | CycleRiskLevel::Critical))
            .collect()
    }

    /// Find nodes with any 4-cycle participation.
    pub fn find_4_cycle_nodes(graph: &CsrGraph) -> Vec<CycleParticipationResult> {
        Self::compute_all(graph)
            .into_iter()
            .filter(|r| r.cycle_count_4hop > 0)
            .collect()
    }

    /// Count total triangles in the graph.
    pub fn count_triangles(graph: &CsrGraph) -> u64 {
        let cycles_3 = Self::detect_3_cycles(graph);
        // Each triangle is counted 3 times (once per vertex)
        cycles_3.iter().map(|&c| c as u64).sum::<u64>() / 3
    }
}

impl GpuKernel for ShortCycleParticipation {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_triangle_graph() -> CsrGraph {
        // Undirected triangle: 0 - 1 - 2 - 0 (bidirectional edges)
        CsrGraph::from_edges(
            3,
            &[(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)],
        )
    }

    fn create_square_graph() -> CsrGraph {
        // Square: 0 -> 1 -> 2 -> 3 -> 0
        CsrGraph::from_edges(
            4,
            &[(0, 1), (1, 2), (2, 3), (3, 0)],
        )
    }

    fn create_reciprocal_graph() -> CsrGraph {
        // Bidirectional edges: 0 <-> 1, 1 <-> 2
        CsrGraph::from_edges(
            3,
            &[(0, 1), (1, 0), (1, 2), (2, 1)],
        )
    }

    fn create_complex_graph() -> CsrGraph {
        // Graph with two triangles (undirected)
        CsrGraph::from_edges(
            5,
            &[
                // Triangle 0-1-2
                (0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0),
                // Triangle 2-3-4
                (2, 3), (3, 2), (3, 4), (4, 3), (2, 4), (4, 2),
            ],
        )
    }

    #[test]
    fn test_cycle_participation_metadata() {
        let kernel = ShortCycleParticipation::new();
        assert_eq!(kernel.metadata().id, "graph/cycle-participation");
        assert_eq!(kernel.metadata().domain, Domain::GraphAnalytics);
    }

    #[test]
    fn test_detect_2_cycles() {
        let graph = create_reciprocal_graph();
        let counts = ShortCycleParticipation::detect_2_cycles(&graph);

        // Node 1 has reciprocal edges with both 0 and 2
        assert!(counts[1] >= 2);
    }

    #[test]
    fn test_detect_3_cycles() {
        let graph = create_triangle_graph();
        let counts = ShortCycleParticipation::detect_3_cycles(&graph);

        // All three nodes participate in one triangle
        assert!(counts[0] >= 1, "Node 0 should participate in triangle: got {}", counts[0]);
        assert!(counts[1] >= 1, "Node 1 should participate in triangle: got {}", counts[1]);
        assert!(counts[2] >= 1, "Node 2 should participate in triangle: got {}", counts[2]);
    }

    #[test]
    fn test_count_triangles() {
        let graph = create_triangle_graph();
        let count = ShortCycleParticipation::count_triangles(&graph);

        assert_eq!(count, 1, "Should find 1 triangle in undirected triangle graph");
    }

    #[test]
    fn test_complex_graph_triangles() {
        let graph = create_complex_graph();
        let count = ShortCycleParticipation::count_triangles(&graph);

        assert_eq!(count, 2, "Should find 2 triangles"); // Two triangles
    }

    #[test]
    fn test_risk_level_baseline() {
        let level = ShortCycleParticipation::calculate_risk_level(0, 0, 0);
        assert_eq!(level, CycleRiskLevel::Baseline);
    }

    #[test]
    fn test_risk_level_low() {
        let level = ShortCycleParticipation::calculate_risk_level(0, 1, 0);
        assert_eq!(level, CycleRiskLevel::Low);
    }

    #[test]
    fn test_risk_level_medium() {
        let level = ShortCycleParticipation::calculate_risk_level(0, 3, 0);
        assert_eq!(level, CycleRiskLevel::Medium);
    }

    #[test]
    fn test_risk_level_high() {
        let level = ShortCycleParticipation::calculate_risk_level(0, 0, 1);
        assert_eq!(level, CycleRiskLevel::High);
    }

    #[test]
    fn test_risk_level_critical() {
        let level = ShortCycleParticipation::calculate_risk_level(0, 0, 5);
        assert_eq!(level, CycleRiskLevel::Critical);
    }

    #[test]
    fn test_find_high_risk_nodes() {
        let graph = create_triangle_graph();
        let high_risk = ShortCycleParticipation::find_high_risk_nodes(&graph);

        // No 4-cycles, so no high-risk nodes
        assert!(high_risk.is_empty());
    }

    #[test]
    fn test_compute_all() {
        let graph = create_triangle_graph();
        let results = ShortCycleParticipation::compute_all(&graph);

        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.cycle_count_3hop >= 1, "Each node should have at least 1 triangle participation");
            assert_eq!(result.cycle_count_4hop, 0);
        }
    }
}
