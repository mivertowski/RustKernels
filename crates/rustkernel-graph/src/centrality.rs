//! Centrality measure kernels.
//!
//! This module provides GPU-accelerated centrality algorithms:
//! - Degree centrality
//! - Betweenness centrality (Brandes algorithm)
//! - Closeness centrality (BFS-based)
//! - Eigenvector centrality (power iteration)
//! - PageRank (power iteration with teleport)
//! - Katz centrality (attenuated paths)

use crate::types::{CentralityResult, CsrGraph, NodeScore};
use rustkernel_core::{
    domain::Domain,
    error::Result,
    kernel::{KernelMetadata, KernelMode},
    traits::GpuKernel,
};
use serde::{Deserialize, Serialize};

// ============================================================================
// PageRank Kernel
// ============================================================================

/// PageRank operation type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PageRankOp {
    /// Query the current PageRank score for a node.
    Query,
    /// Perform one iteration of PageRank.
    Iterate,
    /// Reset all scores.
    Reset,
    /// Initialize with a graph.
    Initialize,
}

/// PageRank request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRankRequest {
    /// Node ID to query (for Query operation).
    pub node_id: Option<u64>,
    /// Operation type.
    pub operation: PageRankOp,
    /// Graph data (for Initialize operation).
    pub graph: Option<CsrGraph>,
    /// Damping factor (default: 0.85).
    pub damping: Option<f32>,
}

/// PageRank response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRankResponse {
    /// Score for the queried node.
    pub score: Option<f64>,
    /// Whether the algorithm has converged.
    pub converged: bool,
    /// Current iteration count.
    pub iteration: u32,
    /// Full result (for Query after convergence).
    pub result: Option<CentralityResult>,
}

/// PageRank kernel state.
#[derive(Debug, Clone, Default)]
pub struct PageRankState {
    /// Current scores.
    pub scores: Vec<f64>,
    /// Previous scores (for convergence check).
    pub prev_scores: Vec<f64>,
    /// Graph in CSR format.
    pub graph: Option<CsrGraph>,
    /// Damping factor.
    pub damping: f32,
    /// Current iteration.
    pub iteration: u32,
    /// Whether converged.
    pub converged: bool,
}

/// PageRank centrality kernel.
///
/// Calculates PageRank centrality using power iteration with teleportation.
/// This is a Ring kernel for low-latency queries after graph is loaded.
#[derive(Debug, Clone)]
pub struct PageRank {
    metadata: KernelMetadata,
}

impl PageRank {
    /// Create a new PageRank kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("graph/pagerank", Domain::GraphAnalytics)
                .with_description("PageRank centrality via power iteration")
                .with_throughput(100_000)
                .with_latency_us(1.0)
                .with_gpu_native(true),
        }
    }

    /// Perform one iteration of PageRank on the given state.
    pub fn iterate_step(state: &mut PageRankState) -> f64 {
        let Some(ref graph) = state.graph else {
            return 0.0;
        };

        let n = graph.num_nodes;
        if n == 0 {
            return 0.0;
        }

        let d = state.damping as f64;
        let teleport = (1.0 - d) / n as f64;

        // Swap buffers
        std::mem::swap(&mut state.scores, &mut state.prev_scores);

        // Calculate new scores
        let mut max_diff = 0.0f64;

        for i in 0..n {
            let mut rank_sum = 0.0f64;

            // Sum contributions from incoming edges
            for &neighbor in graph.neighbors(i as u64) {
                let out_degree = graph.out_degree(neighbor) as f64;
                if out_degree > 0.0 {
                    rank_sum += state.prev_scores[neighbor as usize] / out_degree;
                }
            }

            let new_score = teleport + d * rank_sum;
            state.scores[i] = new_score;

            let diff = (new_score - state.prev_scores[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        state.iteration += 1;
        max_diff
    }

    /// Initialize state for a graph.
    pub fn initialize_state(graph: CsrGraph, damping: f32) -> PageRankState {
        let n = graph.num_nodes;
        PageRankState {
            scores: vec![1.0 / n as f64; n],
            prev_scores: vec![0.0; n],
            graph: Some(graph),
            damping,
            iteration: 0,
            converged: false,
        }
    }

    /// Run PageRank to convergence.
    pub fn run_to_convergence(
        graph: CsrGraph,
        damping: f32,
        max_iterations: u32,
        threshold: f64,
    ) -> Result<CentralityResult> {
        let mut state = Self::initialize_state(graph, damping);

        for _ in 0..max_iterations {
            let diff = Self::iterate_step(&mut state);
            if diff < threshold {
                state.converged = true;
                break;
            }
        }

        Ok(CentralityResult {
            scores: state
                .scores
                .iter()
                .enumerate()
                .map(|(i, &score)| NodeScore {
                    node_id: i as u64,
                    score,
                })
                .collect(),
            iterations: Some(state.iteration),
            converged: state.converged,
        })
    }
}

impl Default for PageRank {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for PageRank {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Degree Centrality Kernel
// ============================================================================

/// Degree centrality kernel.
///
/// Simple O(1) lookup of node degrees after graph is loaded.
#[derive(Debug, Clone)]
pub struct DegreeCentrality {
    metadata: KernelMetadata,
}

impl DegreeCentrality {
    /// Create a new degree centrality kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("graph/degree-centrality", Domain::GraphAnalytics)
                .with_description("Degree centrality (O(1) lookup)")
                .with_throughput(1_000_000)
                .with_latency_us(0.1),
        }
    }
}

impl Default for DegreeCentrality {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for DegreeCentrality {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Betweenness Centrality Kernel
// ============================================================================

/// Betweenness centrality kernel.
///
/// Uses Brandes algorithm for efficient computation.
#[derive(Debug, Clone)]
pub struct BetweennessCentrality {
    metadata: KernelMetadata,
}

impl BetweennessCentrality {
    /// Create a new betweenness centrality kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("graph/betweenness-centrality", Domain::GraphAnalytics)
                .with_description("Betweenness centrality (Brandes algorithm)")
                .with_throughput(10_000)
                .with_latency_us(100.0),
        }
    }
}

impl Default for BetweennessCentrality {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for BetweennessCentrality {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Closeness Centrality Kernel
// ============================================================================

/// Closeness centrality kernel.
///
/// BFS-based closeness centrality calculation.
#[derive(Debug, Clone)]
pub struct ClosenessCentrality {
    metadata: KernelMetadata,
}

impl ClosenessCentrality {
    /// Create a new closeness centrality kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("graph/closeness-centrality", Domain::GraphAnalytics)
                .with_description("Closeness centrality (BFS-based)")
                .with_throughput(10_000)
                .with_latency_us(100.0),
        }
    }
}

impl Default for ClosenessCentrality {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for ClosenessCentrality {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Eigenvector Centrality Kernel
// ============================================================================

/// Eigenvector centrality kernel.
///
/// Power iteration method for eigenvector centrality.
#[derive(Debug, Clone)]
pub struct EigenvectorCentrality {
    metadata: KernelMetadata,
}

impl EigenvectorCentrality {
    /// Create a new eigenvector centrality kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("graph/eigenvector-centrality", Domain::GraphAnalytics)
                .with_description("Eigenvector centrality (power iteration)")
                .with_throughput(50_000)
                .with_latency_us(10.0),
        }
    }
}

impl Default for EigenvectorCentrality {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for EigenvectorCentrality {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Katz Centrality Kernel
// ============================================================================

/// Katz centrality kernel.
///
/// Measures influence through attenuated paths.
#[derive(Debug, Clone)]
pub struct KatzCentrality {
    metadata: KernelMetadata,
}

impl KatzCentrality {
    /// Create a new Katz centrality kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("graph/katz-centrality", Domain::GraphAnalytics)
                .with_description("Katz centrality (attenuated paths)")
                .with_throughput(50_000)
                .with_latency_us(10.0),
        }
    }
}

impl Default for KatzCentrality {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for KatzCentrality {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pagerank_metadata() {
        let kernel = PageRank::new();
        assert_eq!(kernel.metadata().id, "graph/pagerank");
        assert_eq!(kernel.metadata().mode, KernelMode::Ring);
        assert_eq!(kernel.metadata().domain, Domain::GraphAnalytics);
    }

    #[test]
    fn test_pagerank_iteration() {
        let graph = CsrGraph::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]);
        let mut state = PageRank::initialize_state(graph, 0.85);

        let diff = PageRank::iterate_step(&mut state);
        assert!(diff >= 0.0);
        assert_eq!(state.iteration, 1);
    }

    #[test]
    fn test_pagerank_convergence() {
        let graph = CsrGraph::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]);
        let result = PageRank::run_to_convergence(graph, 0.85, 100, 1e-6).unwrap();

        assert!(result.converged);
        assert_eq!(result.scores.len(), 4);
    }
}
