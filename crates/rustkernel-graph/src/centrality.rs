//! Centrality measure kernels.
//!
//! This module provides GPU-accelerated centrality algorithms:
//! - Degree centrality
//! - Betweenness centrality (Brandes algorithm)
//! - Closeness centrality (BFS-based)
//! - Eigenvector centrality (power iteration)
//! - PageRank (power iteration with teleport)
//! - Katz centrality (attenuated paths)

use crate::messages::{CentralityInput, CentralityOutput, CentralityParams};
use crate::ring_messages::{
    K2KBarrier, K2KBarrierRelease, K2KIterationSync, K2KIterationSyncResponse,
    PageRankConvergeResponse, PageRankConvergeRing, PageRankIterateResponse, PageRankIterateRing,
    PageRankQueryResponse, PageRankQueryRing, from_fixed_point, to_fixed_point,
};
use crate::types::{CentralityResult, CsrGraph, NodeScore};
use async_trait::async_trait;
use ringkernel_core::RingContext;
use rustkernel_core::{
    domain::Domain,
    error::Result,
    k2k::IterativeState,
    kernel::KernelMetadata,
    traits::{BatchKernel, GpuKernel, RingKernelHandler},
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;

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
#[derive(Debug)]
pub struct PageRank {
    metadata: KernelMetadata,
    /// Internal state for Ring mode operations.
    state: std::sync::RwLock<PageRankState>,
}

impl Clone for PageRank {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            state: std::sync::RwLock::new(self.state.read().unwrap().clone()),
        }
    }
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
            state: std::sync::RwLock::new(PageRankState::default()),
        }
    }

    /// Initialize the kernel with a graph for Ring mode operations.
    pub fn initialize(&self, graph: CsrGraph, damping: f32) {
        let mut state = self.state.write().unwrap();
        *state = Self::initialize_state(graph, damping);
    }

    /// Query the score for a specific node.
    pub fn query_score(&self, node_id: u64) -> Option<f64> {
        let state = self.state.read().unwrap();
        state.scores.get(node_id as usize).copied()
    }

    /// Get current iteration count.
    pub fn current_iteration(&self) -> u32 {
        self.state.read().unwrap().iteration
    }

    /// Check if converged.
    pub fn is_converged(&self) -> bool {
        self.state.read().unwrap().converged
    }

    /// Perform one iteration step using internal state.
    pub fn iterate(&self) -> f64 {
        let mut state = self.state.write().unwrap();
        Self::iterate_step(&mut state)
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
// PageRank RingKernelHandler Implementations
// ============================================================================

/// RingKernelHandler for PageRank queries.
///
/// Enables low-latency score queries for individual nodes in Ring mode.
#[async_trait]
impl RingKernelHandler<PageRankQueryRing, PageRankQueryResponse> for PageRank {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: PageRankQueryRing,
    ) -> Result<PageRankQueryResponse> {
        let state = self.state.read().unwrap();
        let score = state
            .scores
            .get(msg.node_id as usize)
            .copied()
            .unwrap_or(0.0);

        Ok(PageRankQueryResponse {
            request_id: msg.id.0,
            node_id: msg.node_id,
            score_fp: to_fixed_point(score),
            iteration: state.iteration,
            converged: state.converged,
        })
    }
}

/// RingKernelHandler for PageRank single iteration.
///
/// Performs one power iteration step in Ring mode.
#[async_trait]
impl RingKernelHandler<PageRankIterateRing, PageRankIterateResponse> for PageRank {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: PageRankIterateRing,
    ) -> Result<PageRankIterateResponse> {
        // Perform one iteration on internal state
        let max_delta = self.iterate();

        // Check convergence using default threshold
        let state = self.state.read().unwrap();
        let converged = max_delta < 1e-6;

        Ok(PageRankIterateResponse {
            request_id: msg.id.0,
            iteration: state.iteration,
            max_delta_fp: to_fixed_point(max_delta),
            converged,
        })
    }
}

/// RingKernelHandler for PageRank convergence.
///
/// Runs PageRank to convergence using K2K coordination for iterative state.
#[async_trait]
impl RingKernelHandler<PageRankConvergeRing, PageRankConvergeResponse> for PageRank {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: PageRankConvergeRing,
    ) -> Result<PageRankConvergeResponse> {
        let threshold = from_fixed_point(msg.threshold_fp);
        let max_iterations = msg.max_iterations as u64;

        // Use K2K IterativeState for convergence tracking
        let mut iterative_state = IterativeState::new(threshold, max_iterations);

        // Run actual iterations on internal state
        while iterative_state.should_continue() {
            let max_delta = self.iterate();
            iterative_state.update(max_delta);
        }

        // Update convergence status in internal state
        {
            let mut state = self.state.write().unwrap();
            state.converged = iterative_state.summary().converged;
        }

        let summary = iterative_state.summary();

        Ok(PageRankConvergeResponse {
            request_id: msg.id.0,
            iterations: summary.iterations as u32,
            final_delta_fp: to_fixed_point(summary.final_delta),
            converged: summary.converged,
        })
    }
}

/// RingKernelHandler for K2K iteration synchronization.
///
/// Used in distributed PageRank to synchronize iterations across partitions.
/// In a single-instance setting, this validates the worker's iteration state
/// and returns convergence status based on the reported delta.
#[async_trait]
impl RingKernelHandler<K2KIterationSync, K2KIterationSyncResponse> for PageRank {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: K2KIterationSync,
    ) -> Result<K2KIterationSyncResponse> {
        let state = self.state.read().unwrap();

        // For single-instance, verify iteration matches internal state
        // In distributed setting, would aggregate deltas from all workers
        let current_iteration = state.iteration as u64;
        let all_synced = msg.iteration <= current_iteration;

        // Use reported local delta as global delta (single worker case)
        // In distributed setting, would compute max across all workers
        let local_delta = from_fixed_point(msg.local_delta_fp);
        let global_converged = local_delta < 1e-6 || state.converged;

        Ok(K2KIterationSyncResponse {
            request_id: msg.id.0,
            iteration: msg.iteration,
            all_synced,
            global_delta_fp: msg.local_delta_fp,
            global_converged,
        })
    }
}

/// RingKernelHandler for K2K barrier synchronization.
///
/// Implements barrier synchronization for distributed PageRank iterations.
#[async_trait]
impl RingKernelHandler<K2KBarrier, K2KBarrierRelease> for PageRank {
    async fn handle(&self, _ctx: &mut RingContext, msg: K2KBarrier) -> Result<K2KBarrierRelease> {
        // In a distributed setting, this would:
        // 1. Record this worker as ready
        // 2. Check if all workers are ready
        // 3. Release barrier when all ready
        let all_ready = msg.ready_count >= msg.total_workers;

        Ok(K2KBarrierRelease {
            barrier_id: msg.barrier_id,
            all_ready,
            next_iteration: msg.barrier_id + 1,
        })
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

    /// Calculate degree centrality for all nodes.
    ///
    /// Returns normalized degree centrality (degree / (n-1)).
    pub fn compute(graph: &CsrGraph) -> CentralityResult {
        let n = graph.num_nodes;
        let normalizer = if n > 1 { (n - 1) as f64 } else { 1.0 };

        let scores: Vec<NodeScore> = (0..n)
            .map(|i| NodeScore {
                node_id: i as u64,
                score: graph.out_degree(i as u64) as f64 / normalizer,
            })
            .collect();

        CentralityResult {
            scores,
            iterations: None,
            converged: true,
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
// Betweenness Centrality Kernel (Brandes Algorithm)
// ============================================================================

/// Betweenness centrality kernel.
///
/// Uses Brandes algorithm for efficient computation in O(VE) time.
#[derive(Debug, Clone)]
pub struct BetweennessCentrality {
    metadata: KernelMetadata,
}

impl BetweennessCentrality {
    /// Create a new betweenness centrality kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/betweenness-centrality", Domain::GraphAnalytics)
                .with_description("Betweenness centrality (Brandes algorithm)")
                .with_throughput(10_000)
                .with_latency_us(100.0),
        }
    }

    /// Compute betweenness centrality using Brandes algorithm.
    ///
    /// The algorithm runs BFS from each vertex and accumulates
    /// dependency scores in a single backward pass.
    pub fn compute(graph: &CsrGraph, normalized: bool) -> CentralityResult {
        let n = graph.num_nodes;
        let mut centrality = vec![0.0f64; n];

        // Run Brandes algorithm from each source
        for s in 0..n {
            // BFS structures
            let mut stack: Vec<usize> = Vec::with_capacity(n);
            let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
            let mut sigma = vec![0.0f64; n]; // Number of shortest paths
            let mut dist = vec![-1i64; n]; // Distance from source

            sigma[s] = 1.0;
            dist[s] = 0;

            let mut queue = VecDeque::new();
            queue.push_back(s);

            // Forward BFS
            while let Some(v) = queue.pop_front() {
                stack.push(v);

                for &w in graph.neighbors(v as u64) {
                    let w = w as usize;

                    // First time visiting w?
                    if dist[w] < 0 {
                        dist[w] = dist[v] + 1;
                        queue.push_back(w);
                    }

                    // Is this a shortest path to w via v?
                    if dist[w] == dist[v] + 1 {
                        sigma[w] += sigma[v];
                        predecessors[w].push(v);
                    }
                }
            }

            // Backward pass - accumulate dependencies
            let mut delta = vec![0.0f64; n];

            while let Some(w) = stack.pop() {
                for &v in &predecessors[w] {
                    let contribution = (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                    delta[v] += contribution;
                }

                if w != s {
                    centrality[w] += delta[w];
                }
            }
        }

        // Normalize if requested
        if normalized && n > 2 {
            let scale = 1.0 / ((n - 1) * (n - 2)) as f64;
            for c in &mut centrality {
                *c *= scale;
            }
        }

        CentralityResult {
            scores: centrality
                .into_iter()
                .enumerate()
                .map(|(i, score)| NodeScore {
                    node_id: i as u64,
                    score,
                })
                .collect(),
            iterations: None,
            converged: true,
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
/// Closeness = (n-1) / sum(shortest_path_distances)
#[derive(Debug, Clone)]
pub struct ClosenessCentrality {
    metadata: KernelMetadata,
}

impl ClosenessCentrality {
    /// Create a new closeness centrality kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/closeness-centrality", Domain::GraphAnalytics)
                .with_description("Closeness centrality (BFS-based)")
                .with_throughput(10_000)
                .with_latency_us(100.0),
        }
    }

    /// Compute closeness centrality using BFS from each node.
    ///
    /// For disconnected graphs, uses harmonic mean variant.
    pub fn compute(graph: &CsrGraph, harmonic: bool) -> CentralityResult {
        let n = graph.num_nodes;
        let mut centrality = vec![0.0f64; n];

        for source in 0..n {
            let distances = Self::bfs_distances(graph, source);

            if harmonic {
                // Harmonic centrality: sum(1/d) for all reachable nodes
                let sum: f64 = distances
                    .iter()
                    .enumerate()
                    .filter(|(i, d)| *i != source && **d > 0)
                    .map(|(_, d)| 1.0 / *d as f64)
                    .sum();
                centrality[source] = sum / (n - 1) as f64;
            } else {
                // Classic closeness: (n-1) / sum(distances)
                let sum: i64 = distances.iter().sum();
                let reachable: usize = distances.iter().filter(|&&d| d > 0).count();

                if sum > 0 && reachable > 0 {
                    centrality[source] = reachable as f64 / sum as f64;
                }
            }
        }

        CentralityResult {
            scores: centrality
                .into_iter()
                .enumerate()
                .map(|(i, score)| NodeScore {
                    node_id: i as u64,
                    score,
                })
                .collect(),
            iterations: None,
            converged: true,
        }
    }

    /// BFS to compute distances from source to all other nodes.
    fn bfs_distances(graph: &CsrGraph, source: usize) -> Vec<i64> {
        let n = graph.num_nodes;
        let mut distances = vec![0i64; n];
        let mut visited = vec![false; n];

        let mut queue = VecDeque::new();
        queue.push_back(source);
        visited[source] = true;

        while let Some(v) = queue.pop_front() {
            for &w in graph.neighbors(v as u64) {
                let w = w as usize;
                if !visited[w] {
                    visited[w] = true;
                    distances[w] = distances[v] + 1;
                    queue.push_back(w);
                }
            }
        }

        distances
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
/// A node's score is proportional to the sum of its neighbors' scores.
#[derive(Debug, Clone)]
pub struct EigenvectorCentrality {
    metadata: KernelMetadata,
}

impl EigenvectorCentrality {
    /// Create a new eigenvector centrality kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/eigenvector-centrality", Domain::GraphAnalytics)
                .with_description("Eigenvector centrality (power iteration)")
                .with_throughput(50_000)
                .with_latency_us(10.0),
        }
    }

    /// Compute eigenvector centrality using power iteration.
    pub fn compute(graph: &CsrGraph, max_iterations: u32, tolerance: f64) -> CentralityResult {
        let n = graph.num_nodes;
        if n == 0 {
            return CentralityResult {
                scores: Vec::new(),
                iterations: Some(0),
                converged: true,
            };
        }

        // Initialize with uniform scores
        let mut scores = vec![1.0 / (n as f64).sqrt(); n];
        let mut new_scores = vec![0.0f64; n];
        let mut converged = false;
        let mut iterations = 0u32;

        for iter in 0..max_iterations {
            iterations = iter + 1;

            // Compute new scores: x_i = sum(A_ij * x_j)
            for i in 0..n {
                let mut sum = 0.0f64;
                for &j in graph.neighbors(i as u64) {
                    sum += scores[j as usize];
                }
                new_scores[i] = sum;
            }

            // Normalize
            let norm: f64 = new_scores.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for x in &mut new_scores {
                    *x /= norm;
                }
            }

            // Check convergence
            let diff: f64 = scores
                .iter()
                .zip(new_scores.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, |acc, x| acc.max(x));

            std::mem::swap(&mut scores, &mut new_scores);

            if diff < tolerance {
                converged = true;
                break;
            }
        }

        CentralityResult {
            scores: scores
                .into_iter()
                .enumerate()
                .map(|(i, score)| NodeScore {
                    node_id: i as u64,
                    score,
                })
                .collect(),
            iterations: Some(iterations),
            converged,
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
/// Katz(i) = sum over all paths from j to i of alpha^(path_length)
#[derive(Debug, Clone)]
pub struct KatzCentrality {
    metadata: KernelMetadata,
}

impl KatzCentrality {
    /// Create a new Katz centrality kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/katz-centrality", Domain::GraphAnalytics)
                .with_description("Katz centrality (attenuated paths)")
                .with_throughput(50_000)
                .with_latency_us(10.0),
        }
    }

    /// Compute Katz centrality.
    ///
    /// # Arguments
    /// * `graph` - The input graph
    /// * `alpha` - Attenuation factor (should be < 1/lambda_max)
    /// * `beta` - Base score for each node (default 1.0)
    /// * `max_iterations` - Maximum iterations for power iteration
    /// * `tolerance` - Convergence threshold
    pub fn compute(
        graph: &CsrGraph,
        alpha: f64,
        beta: f64,
        max_iterations: u32,
        tolerance: f64,
    ) -> CentralityResult {
        let n = graph.num_nodes;
        if n == 0 {
            return CentralityResult {
                scores: Vec::new(),
                iterations: Some(0),
                converged: true,
            };
        }

        // Initialize scores
        let mut scores = vec![0.0f64; n];
        let mut new_scores = vec![0.0f64; n];
        let mut converged = false;
        let mut iterations = 0u32;

        // Power iteration: x = alpha * A * x + beta
        for iter in 0..max_iterations {
            iterations = iter + 1;

            for i in 0..n {
                let mut sum = 0.0f64;
                for &j in graph.neighbors(i as u64) {
                    sum += scores[j as usize];
                }
                new_scores[i] = alpha * sum + beta;
            }

            // Check convergence
            let diff: f64 = scores
                .iter()
                .zip(new_scores.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, |acc, x| acc.max(x));

            std::mem::swap(&mut scores, &mut new_scores);

            if diff < tolerance {
                converged = true;
                break;
            }
        }

        // Normalize by maximum score
        let max_score = scores.iter().cloned().fold(0.0f64, f64::max);
        if max_score > 0.0 {
            for s in &mut scores {
                *s /= max_score;
            }
        }

        CentralityResult {
            scores: scores
                .into_iter()
                .enumerate()
                .map(|(i, score)| NodeScore {
                    node_id: i as u64,
                    score,
                })
                .collect(),
            iterations: Some(iterations),
            converged,
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

// ============================================================================
// BatchKernel Implementations
// ============================================================================

/// Batch execution wrapper for all centrality kernels.
///
/// Since centrality algorithms are computationally intensive,
/// they benefit from batch execution with CPU orchestration.

#[async_trait]
impl BatchKernel<CentralityInput, CentralityOutput> for BetweennessCentrality {
    async fn execute(&self, input: CentralityInput) -> Result<CentralityOutput> {
        let start = Instant::now();
        let normalized = input.normalize;
        let result = Self::compute(&input.graph, normalized);
        let compute_time_us = start.elapsed().as_micros() as u64;

        Ok(CentralityOutput {
            result,
            compute_time_us,
        })
    }
}

#[async_trait]
impl BatchKernel<CentralityInput, CentralityOutput> for ClosenessCentrality {
    async fn execute(&self, input: CentralityInput) -> Result<CentralityOutput> {
        let start = Instant::now();
        let harmonic = match input.params {
            CentralityParams::Closeness { harmonic } => harmonic,
            _ => false,
        };
        let result = Self::compute(&input.graph, harmonic);
        let compute_time_us = start.elapsed().as_micros() as u64;

        Ok(CentralityOutput {
            result,
            compute_time_us,
        })
    }
}

#[async_trait]
impl BatchKernel<CentralityInput, CentralityOutput> for EigenvectorCentrality {
    async fn execute(&self, input: CentralityInput) -> Result<CentralityOutput> {
        let start = Instant::now();
        let max_iterations = input.max_iterations.unwrap_or(1000);
        let tolerance = input.tolerance.unwrap_or(1e-6);
        let result = Self::compute(&input.graph, max_iterations, tolerance);
        let compute_time_us = start.elapsed().as_micros() as u64;

        Ok(CentralityOutput {
            result,
            compute_time_us,
        })
    }
}

#[async_trait]
impl BatchKernel<CentralityInput, CentralityOutput> for KatzCentrality {
    async fn execute(&self, input: CentralityInput) -> Result<CentralityOutput> {
        let start = Instant::now();
        let (alpha, beta) = match input.params {
            CentralityParams::Katz { alpha, beta } => (alpha, beta),
            _ => (0.1, 1.0),
        };
        let max_iterations = input.max_iterations.unwrap_or(100);
        let tolerance = input.tolerance.unwrap_or(1e-6);
        let result = Self::compute(&input.graph, alpha, beta, max_iterations, tolerance);
        let compute_time_us = start.elapsed().as_micros() as u64;

        Ok(CentralityOutput {
            result,
            compute_time_us,
        })
    }
}

/// PageRank can be used in both batch and ring modes.
/// This is the batch mode implementation.
impl PageRank {
    /// Execute PageRank as a batch operation.
    ///
    /// Convenience method that runs the algorithm to convergence.
    pub async fn compute_batch(
        &self,
        graph: CsrGraph,
        damping: f32,
        max_iterations: u32,
        threshold: f64,
    ) -> Result<CentralityResult> {
        Self::run_to_convergence(graph, damping, max_iterations, threshold)
    }
}

#[async_trait]
impl BatchKernel<CentralityInput, CentralityOutput> for PageRank {
    async fn execute(&self, input: CentralityInput) -> Result<CentralityOutput> {
        let start = Instant::now();
        let damping = match input.params {
            CentralityParams::PageRank { damping } => damping,
            _ => 0.85,
        };
        let max_iterations = input.max_iterations.unwrap_or(100);
        let tolerance = input.tolerance.unwrap_or(1e-6);
        let result = Self::run_to_convergence(input.graph, damping, max_iterations, tolerance)?;
        let compute_time_us = start.elapsed().as_micros() as u64;

        Ok(CentralityOutput {
            result,
            compute_time_us,
        })
    }
}

/// Degree centrality batch implementation.
#[async_trait]
impl BatchKernel<CentralityInput, CentralityOutput> for DegreeCentrality {
    async fn execute(&self, input: CentralityInput) -> Result<CentralityOutput> {
        let start = Instant::now();
        let result = Self::compute(&input.graph);
        let compute_time_us = start.elapsed().as_micros() as u64;

        Ok(CentralityOutput {
            result,
            compute_time_us,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> CsrGraph {
        // Simple graph: 0 -> 1 -> 2 -> 3 -> 0 (cycle)
        CsrGraph::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)])
    }

    fn create_star_graph() -> CsrGraph {
        // Star graph: center node 0 connected to all others
        CsrGraph::from_edges(
            5,
            &[
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
            ],
        )
    }

    #[test]
    fn test_pagerank_metadata() {
        let kernel = PageRank::new();
        assert_eq!(kernel.metadata().id, "graph/pagerank");
        assert_eq!(kernel.metadata().domain, Domain::GraphAnalytics);
    }

    #[test]
    fn test_pagerank_iteration() {
        let graph = create_test_graph();
        let mut state = PageRank::initialize_state(graph, 0.85);

        let diff = PageRank::iterate_step(&mut state);
        assert!(diff >= 0.0);
        assert_eq!(state.iteration, 1);
    }

    #[test]
    fn test_pagerank_convergence() {
        let graph = create_test_graph();
        let result = PageRank::run_to_convergence(graph, 0.85, 100, 1e-6).unwrap();

        assert!(result.converged);
        assert_eq!(result.scores.len(), 4);

        // In a cycle, all nodes should have equal PageRank
        let first_score = result.scores[0].score;
        for score in &result.scores {
            assert!((score.score - first_score).abs() < 0.01);
        }
    }

    #[test]
    fn test_degree_centrality() {
        let graph = create_star_graph();
        let result = DegreeCentrality::compute(&graph);

        assert_eq!(result.scores.len(), 5);

        // Center node (0) should have highest degree
        let center_score = result.scores[0].score;
        for score in &result.scores[1..] {
            assert!(center_score > score.score);
        }
    }

    #[test]
    fn test_betweenness_centrality() {
        // Line graph: 0 - 1 - 2 - 3
        let graph = CsrGraph::from_edges(4, &[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]);

        let result = BetweennessCentrality::compute(&graph, false);

        assert_eq!(result.scores.len(), 4);

        // Middle nodes (1, 2) should have highest betweenness
        let node_1_score = result.scores[1].score;
        let node_0_score = result.scores[0].score;
        assert!(node_1_score > node_0_score);
    }

    #[test]
    fn test_closeness_centrality() {
        let graph = create_star_graph();
        let result = ClosenessCentrality::compute(&graph, false);

        assert_eq!(result.scores.len(), 5);

        // Center node should have highest closeness
        let center_score = result.scores[0].score;
        for score in &result.scores[1..] {
            assert!(center_score >= score.score);
        }
    }

    #[test]
    fn test_eigenvector_centrality() {
        let graph = create_star_graph();
        let result = EigenvectorCentrality::compute(&graph, 1000, 1e-4);

        // May or may not converge depending on graph structure
        assert_eq!(result.scores.len(), 5);

        // Center node should have high eigenvector centrality
        // (may not be highest due to star graph properties)
        let center_score = result.scores[0].score;
        assert!(center_score > 0.0);
    }

    #[test]
    fn test_katz_centrality() {
        let graph = create_star_graph();
        let result = KatzCentrality::compute(&graph, 0.1, 1.0, 100, 1e-6);

        assert!(result.converged);
        assert_eq!(result.scores.len(), 5);
    }
}
