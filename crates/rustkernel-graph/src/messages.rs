//! Ring message types for Graph Analytics kernels.
//!
//! This module defines request/response message types for GPU-native
//! persistent actor communication.

use crate::motif::MotifResult;
use crate::types::{CentralityResult, CommunityResult, CsrGraph, SimilarityResult};
use rustkernel_core::messages::CorrelationId;
use serde::{Deserialize, Serialize};

// ============================================================================
// PageRank Messages
// ============================================================================

/// PageRank operation type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PageRankOp {
    /// Query the current PageRank score for a node.
    Query { node_id: u64 },
    /// Perform one iteration of PageRank.
    Iterate,
    /// Initialize with a new graph.
    Initialize,
    /// Reset all scores to initial values.
    Reset,
    /// Run until convergence with threshold.
    ConvergeUntil { threshold: f64, max_iterations: u32 },
}

/// PageRank request message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRankRequest {
    /// Correlation ID for request-response pairing.
    pub correlation_id: CorrelationId,
    /// The operation to perform.
    pub operation: PageRankOp,
    /// Graph data (for Initialize operation).
    pub graph: Option<CsrGraph>,
    /// Damping factor (default: 0.85).
    pub damping: Option<f32>,
}

impl PageRankRequest {
    /// Create a query request for a specific node.
    pub fn query(node_id: u64) -> Self {
        Self {
            correlation_id: CorrelationId::new(),
            operation: PageRankOp::Query { node_id },
            graph: None,
            damping: None,
        }
    }

    /// Create an iterate request.
    pub fn iterate() -> Self {
        Self {
            correlation_id: CorrelationId::new(),
            operation: PageRankOp::Iterate,
            graph: None,
            damping: None,
        }
    }

    /// Create an initialize request with graph data.
    pub fn initialize(graph: CsrGraph, damping: f32) -> Self {
        Self {
            correlation_id: CorrelationId::new(),
            operation: PageRankOp::Initialize,
            graph: Some(graph),
            damping: Some(damping),
        }
    }

    /// Create a converge request.
    pub fn converge(threshold: f64, max_iterations: u32) -> Self {
        Self {
            correlation_id: CorrelationId::new(),
            operation: PageRankOp::ConvergeUntil {
                threshold,
                max_iterations,
            },
            graph: None,
            damping: None,
        }
    }
}

/// PageRank response message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRankResponse {
    /// Correlation ID matching the request.
    pub correlation_id: CorrelationId,
    /// Score for the queried node (Query operation).
    pub score: Option<f64>,
    /// Whether the algorithm has converged.
    pub converged: bool,
    /// Current iteration count.
    pub iteration: u32,
    /// Full centrality result (for converged operations).
    pub result: Option<CentralityResult>,
    /// Error message if operation failed.
    pub error: Option<String>,
}

impl PageRankResponse {
    /// Create a successful query response.
    pub fn score(correlation_id: CorrelationId, score: f64, iteration: u32) -> Self {
        Self {
            correlation_id,
            score: Some(score),
            converged: false,
            iteration,
            result: None,
            error: None,
        }
    }

    /// Create a convergence response with full result.
    pub fn converged(
        correlation_id: CorrelationId,
        result: CentralityResult,
        iteration: u32,
    ) -> Self {
        Self {
            correlation_id,
            score: None,
            converged: true,
            iteration,
            result: Some(result),
            error: None,
        }
    }

    /// Create an error response.
    pub fn error(correlation_id: CorrelationId, error: impl Into<String>) -> Self {
        Self {
            correlation_id,
            score: None,
            converged: false,
            iteration: 0,
            result: None,
            error: Some(error.into()),
        }
    }
}

// ============================================================================
// Centrality Batch Input/Output Types
// ============================================================================

/// Input for batch centrality computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityInput {
    /// The graph to analyze.
    pub graph: CsrGraph,
    /// Whether to normalize the results.
    pub normalize: bool,
    /// Maximum iterations (for iterative algorithms).
    pub max_iterations: Option<u32>,
    /// Convergence tolerance (for iterative algorithms).
    pub tolerance: Option<f64>,
    /// Algorithm-specific parameters.
    pub params: CentralityParams,
}

/// Algorithm-specific parameters for centrality computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CentralityParams {
    /// PageRank parameters.
    PageRank { damping: f32 },
    /// Degree centrality (no extra params).
    Degree,
    /// Betweenness centrality (no extra params).
    Betweenness,
    /// Closeness centrality parameters.
    Closeness { harmonic: bool },
    /// Eigenvector centrality (no extra params).
    Eigenvector,
    /// Katz centrality parameters.
    Katz { alpha: f64, beta: f64 },
}

impl Default for CentralityInput {
    fn default() -> Self {
        Self {
            graph: CsrGraph::empty(),
            normalize: true,
            max_iterations: Some(100),
            tolerance: Some(1e-6),
            params: CentralityParams::Degree,
        }
    }
}

/// Output from batch centrality computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityOutput {
    /// The centrality result.
    pub result: CentralityResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Community Detection Messages
// ============================================================================

/// Community detection algorithm.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CommunityAlgorithm {
    /// Louvain algorithm (modularity optimization).
    Louvain,
    /// Modularity score calculation.
    ModularityScore,
}

/// Input for community detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityInput {
    /// The graph to analyze.
    pub graph: CsrGraph,
    /// Algorithm to use.
    pub algorithm: CommunityAlgorithm,
    /// Resolution parameter for Louvain (default: 1.0).
    pub resolution: f64,
}

/// Output from community detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityOutput {
    /// The community detection result.
    pub result: CommunityResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Similarity Messages
// ============================================================================

/// Similarity metric type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SimilarityMetric {
    /// Jaccard similarity coefficient.
    Jaccard,
    /// Cosine similarity.
    Cosine,
    /// Adamic-Adar index.
    AdamicAdar,
}

/// Input for similarity computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityInput {
    /// The graph to analyze.
    pub graph: CsrGraph,
    /// Similarity metric to use.
    pub metric: SimilarityMetric,
    /// Node pairs to compute similarity for (if None, compute all pairs).
    pub node_pairs: Option<Vec<(u64, u64)>>,
}

/// Output from similarity computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityOutput {
    /// The similarity result.
    pub result: SimilarityResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Motif Detection Messages
// ============================================================================

/// Input for motif/triangle detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifInput {
    /// The graph to analyze.
    pub graph: CsrGraph,
    /// Motif size (3 for triangles).
    pub motif_size: usize,
    /// Whether to enumerate all instances.
    pub enumerate: bool,
}

/// Output from motif detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifOutput {
    /// The motif detection result.
    pub result: MotifResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pagerank_request_query() {
        let req = PageRankRequest::query(42);
        assert!(matches!(req.operation, PageRankOp::Query { node_id: 42 }));
    }

    #[test]
    fn test_pagerank_request_converge() {
        let req = PageRankRequest::converge(1e-6, 100);
        assert!(matches!(
            req.operation,
            PageRankOp::ConvergeUntil {
                threshold: _,
                max_iterations: 100
            }
        ));
    }

    #[test]
    fn test_pagerank_response_score() {
        let cid = CorrelationId::new();
        let resp = PageRankResponse::score(cid, 0.5, 10);
        assert_eq!(resp.score, Some(0.5));
        assert!(!resp.converged);
    }
}
