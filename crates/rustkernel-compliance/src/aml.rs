//! Anti-Money Laundering (AML) kernels.
//!
//! This module provides AML detection algorithms:
//! - Circular flow detection (SCC-based)
//! - Reciprocity analysis
//! - Rapid movement (velocity) analysis
//! - Multi-pattern AML detection

use crate::messages::{
    AMLPatternInput, AMLPatternOutput, CircularFlowInput, CircularFlowOutput, RapidMovementInput,
    RapidMovementOutput, ReciprocityFlowInput, ReciprocityFlowOutput,
};
use crate::ring_messages::{
    AddGraphEdgeResponse, AddGraphEdgeRing, MatchPatternResponse, MatchPatternRing,
    QueryCircularRatioResponse, QueryCircularRatioRing,
};
use crate::types::{
    AMLPattern, AMLPatternResult, CircularFlowResult, PatternDetail, RapidMovementResult,
    ReciprocityResult, TimeWindow, Transaction,
};
use async_trait::async_trait;
use ringkernel_core::RingContext;
use rustkernel_core::error::Result;
use rustkernel_core::traits::{BatchKernel, RingKernelHandler};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

// ============================================================================
// Circular Flow Ratio Kernel
// ============================================================================

/// Per-entity circular flow state.
#[derive(Debug, Clone, Default)]
pub struct EntityCircularState {
    /// Total outgoing volume.
    pub outgoing_volume: f64,
    /// Total incoming volume.
    pub incoming_volume: f64,
    /// Volume in circular flows.
    pub circular_volume: f64,
    /// Number of outgoing edges.
    pub out_degree: u32,
    /// Number of incoming edges.
    pub in_degree: u32,
    /// Whether entity is in an SCC.
    pub in_scc: bool,
}

/// Circular flow state for Ring mode operations.
#[derive(Debug, Clone, Default)]
pub struct CircularFlowState {
    /// Transaction graph: source -> [(dest, amount)]
    pub graph: HashMap<u64, Vec<(u64, f64)>>,
    /// Per-entity state.
    pub entities: HashMap<u64, EntityCircularState>,
    /// Cached SCCs.
    pub sccs: Vec<Vec<u64>>,
    /// Total transaction volume.
    pub total_volume: f64,
    /// Total circular volume.
    pub circular_volume: f64,
    /// Whether SCCs need recalculation.
    pub sccs_stale: bool,
}

/// Circular flow detection kernel.
///
/// Detects circular transactions using Strongly Connected Components (SCC).
/// High circular flow ratio indicates potential money laundering.
#[derive(Debug)]
pub struct CircularFlowRatio {
    metadata: KernelMetadata,
    /// Internal state for Ring mode operations.
    state: std::sync::RwLock<CircularFlowState>,
}

impl Clone for CircularFlowRatio {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            state: std::sync::RwLock::new(self.state.read().unwrap().clone()),
        }
    }
}

impl Default for CircularFlowRatio {
    fn default() -> Self {
        Self::new()
    }
}

impl CircularFlowRatio {
    /// Create a new circular flow detection kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("compliance/circular-flow", Domain::Compliance)
                .with_description("Circular flow detection via SCC")
                .with_throughput(50_000)
                .with_latency_us(100.0),
            state: std::sync::RwLock::new(CircularFlowState::default()),
        }
    }

    /// Add an edge to the transaction graph.
    /// Returns (cycle_detected, cycle_size, source_ratio).
    pub fn add_edge(&self, source_id: u64, dest_id: u64, amount: f64) -> (bool, u32, f64) {
        let mut state = self.state.write().unwrap();

        // Add edge to graph
        state
            .graph
            .entry(source_id)
            .or_default()
            .push((dest_id, amount));
        state.total_volume += amount;
        state.sccs_stale = true;

        // Update entity states
        let source_state = state.entities.entry(source_id).or_default();
        source_state.outgoing_volume += amount;
        source_state.out_degree += 1;

        let dest_state = state.entities.entry(dest_id).or_default();
        dest_state.incoming_volume += amount;
        dest_state.in_degree += 1;

        // Quick cycle check using DFS from dest back to source
        let cycle_detected = Self::has_path(&state.graph, dest_id, source_id);

        let cycle_size = if cycle_detected {
            // Estimate cycle size (full SCC would be more accurate)
            Self::estimate_cycle_size(&state.graph, source_id, dest_id)
        } else {
            0
        };

        let source_ratio = {
            let s = state.entities.get(&source_id).unwrap();
            if s.outgoing_volume > 0.0 {
                s.circular_volume / s.outgoing_volume
            } else {
                0.0
            }
        };

        (cycle_detected, cycle_size, source_ratio)
    }

    /// Query circular flow ratio for an entity.
    pub fn query_entity(&self, entity_id: u64) -> (f64, u32, u64) {
        let state = self.state.read().unwrap();

        if let Some(entity_state) = state.entities.get(&entity_id) {
            let ratio = if entity_state.outgoing_volume > 0.0 {
                entity_state.circular_volume / entity_state.outgoing_volume
            } else {
                0.0
            };
            let scc_count = state
                .sccs
                .iter()
                .filter(|scc| scc.contains(&entity_id))
                .count() as u32;
            let cycle_volume = (entity_state.circular_volume * 100_000_000.0) as u64;
            (ratio, scc_count, cycle_volume)
        } else {
            (0.0, 0, 0)
        }
    }

    /// Check if there's a path from src to dst in the graph.
    fn has_path(graph: &HashMap<u64, Vec<(u64, f64)>>, src: u64, dst: u64) -> bool {
        let mut visited = HashSet::new();
        let mut stack = vec![src];

        while let Some(node) = stack.pop() {
            if node == dst {
                return true;
            }
            if visited.insert(node) {
                if let Some(neighbors) = graph.get(&node) {
                    for &(neighbor, _) in neighbors {
                        if !visited.contains(&neighbor) {
                            stack.push(neighbor);
                        }
                    }
                }
            }
        }
        false
    }

    /// Estimate cycle size using BFS.
    fn estimate_cycle_size(graph: &HashMap<u64, Vec<(u64, f64)>>, start: u64, end: u64) -> u32 {
        let mut visited = HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((end, 1u32));

        while let Some((node, depth)) = queue.pop_front() {
            if node == start {
                return depth;
            }
            if visited.insert(node) {
                if let Some(neighbors) = graph.get(&node) {
                    for &(neighbor, _) in neighbors {
                        if !visited.contains(&neighbor) {
                            queue.push_back((neighbor, depth + 1));
                        }
                    }
                }
            }
        }
        0
    }

    /// Detect circular flows in transactions.
    ///
    /// # Arguments
    /// * `transactions` - List of transactions to analyze
    /// * `min_amount` - Minimum total amount for a cycle to be flagged
    pub fn compute(transactions: &[Transaction], min_amount: f64) -> CircularFlowResult {
        if transactions.is_empty() {
            return CircularFlowResult {
                circular_ratio: 0.0,
                sccs: Vec::new(),
                circular_amount: 0.0,
                total_amount: 0.0,
            };
        }

        // Build transaction graph
        let mut graph: HashMap<u64, Vec<(u64, f64)>> = HashMap::new();
        let mut total_amount = 0.0;

        for tx in transactions {
            graph
                .entry(tx.source_id)
                .or_default()
                .push((tx.dest_id, tx.amount));
            total_amount += tx.amount;
        }

        // Find SCCs using Tarjan's algorithm
        let sccs = Self::tarjan_scc(&graph);

        // Calculate circular amount (amount in non-trivial SCCs)
        let mut circular_amount = 0.0;
        let mut significant_sccs = Vec::new();

        for scc in &sccs {
            if scc.len() > 1 {
                // Calculate amount flowing within this SCC
                let scc_set: HashSet<u64> = scc.iter().copied().collect();
                let mut scc_amount = 0.0;

                for tx in transactions {
                    if scc_set.contains(&tx.source_id) && scc_set.contains(&tx.dest_id) {
                        scc_amount += tx.amount;
                    }
                }

                if scc_amount >= min_amount {
                    circular_amount += scc_amount;
                    significant_sccs.push(scc.clone());
                }
            }
        }

        let circular_ratio = if total_amount > 0.0 {
            circular_amount / total_amount
        } else {
            0.0
        };

        CircularFlowResult {
            circular_ratio,
            sccs: significant_sccs,
            circular_amount,
            total_amount,
        }
    }

    /// Tarjan's SCC algorithm.
    fn tarjan_scc(graph: &HashMap<u64, Vec<(u64, f64)>>) -> Vec<Vec<u64>> {
        let mut index_counter = 0u64;
        let mut stack = Vec::new();
        let mut on_stack: HashSet<u64> = HashSet::new();
        let mut indices: HashMap<u64, u64> = HashMap::new();
        let mut lowlinks: HashMap<u64, u64> = HashMap::new();
        let mut sccs = Vec::new();

        // Get all nodes
        let mut nodes: HashSet<u64> = graph.keys().copied().collect();
        for edges in graph.values() {
            for (dest, _) in edges {
                nodes.insert(*dest);
            }
        }

        fn strongconnect(
            v: u64,
            graph: &HashMap<u64, Vec<(u64, f64)>>,
            index_counter: &mut u64,
            stack: &mut Vec<u64>,
            on_stack: &mut HashSet<u64>,
            indices: &mut HashMap<u64, u64>,
            lowlinks: &mut HashMap<u64, u64>,
            sccs: &mut Vec<Vec<u64>>,
        ) {
            indices.insert(v, *index_counter);
            lowlinks.insert(v, *index_counter);
            *index_counter += 1;
            stack.push(v);
            on_stack.insert(v);

            if let Some(neighbors) = graph.get(&v) {
                for (w, _) in neighbors {
                    if !indices.contains_key(w) {
                        strongconnect(
                            *w,
                            graph,
                            index_counter,
                            stack,
                            on_stack,
                            indices,
                            lowlinks,
                            sccs,
                        );
                        let lowlink_w = lowlinks[w];
                        if let Some(ll) = lowlinks.get_mut(&v) {
                            *ll = (*ll).min(lowlink_w);
                        }
                    } else if on_stack.contains(w) {
                        let index_w = indices[w];
                        if let Some(ll) = lowlinks.get_mut(&v) {
                            *ll = (*ll).min(index_w);
                        }
                    }
                }
            }

            if lowlinks[&v] == indices[&v] {
                let mut scc = Vec::new();
                loop {
                    let w = stack.pop().unwrap();
                    on_stack.remove(&w);
                    scc.push(w);
                    if w == v {
                        break;
                    }
                }
                sccs.push(scc);
            }
        }

        for node in nodes {
            if !indices.contains_key(&node) {
                strongconnect(
                    node,
                    graph,
                    &mut index_counter,
                    &mut stack,
                    &mut on_stack,
                    &mut indices,
                    &mut lowlinks,
                    &mut sccs,
                );
            }
        }

        sccs
    }
}

impl GpuKernel for CircularFlowRatio {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[async_trait]
impl BatchKernel<CircularFlowInput, CircularFlowOutput> for CircularFlowRatio {
    async fn execute(&self, input: CircularFlowInput) -> Result<CircularFlowOutput> {
        let start = Instant::now();
        let result = Self::compute(&input.transactions, input.min_amount);
        Ok(CircularFlowOutput {
            result,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

// ----------------------------------------------------------------------------
// Ring Kernel Handler Implementation for CircularFlowRatio
// ----------------------------------------------------------------------------

/// Ring handler for adding edges to the transaction graph.
#[async_trait]
impl RingKernelHandler<AddGraphEdgeRing, AddGraphEdgeResponse> for CircularFlowRatio {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: AddGraphEdgeRing,
    ) -> Result<AddGraphEdgeResponse> {
        // Add edge to internal graph state
        let amount = msg.amount as f64 / 100_000_000.0;
        let (cycle_detected, cycle_size, source_ratio) =
            self.add_edge(msg.source_id, msg.dest_id, amount);

        Ok(AddGraphEdgeResponse {
            correlation_id: msg.correlation_id,
            cycle_detected,
            cycle_size,
            source_ratio: source_ratio as f32,
        })
    }
}

/// Ring handler for querying circular flow ratio.
#[async_trait]
impl RingKernelHandler<QueryCircularRatioRing, QueryCircularRatioResponse> for CircularFlowRatio {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: QueryCircularRatioRing,
    ) -> Result<QueryCircularRatioResponse> {
        // Query internal state for this entity
        let (ratio, scc_count, cycle_volume) = self.query_entity(msg.entity_id);

        Ok(QueryCircularRatioResponse {
            correlation_id: msg.correlation_id,
            entity_id: msg.entity_id,
            ratio: ratio as f32,
            scc_count,
            cycle_volume: cycle_volume as i64,
        })
    }
}

// ============================================================================
// Reciprocity Flow Ratio Kernel
// ============================================================================

/// Reciprocity flow detection kernel.
///
/// Detects mutual/reciprocal transactions between entities.
/// High reciprocity can indicate layering or round-tripping.
#[derive(Debug, Clone)]
pub struct ReciprocityFlowRatio {
    metadata: KernelMetadata,
}

impl Default for ReciprocityFlowRatio {
    fn default() -> Self {
        Self::new()
    }
}

impl ReciprocityFlowRatio {
    /// Create a new reciprocity detection kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("compliance/reciprocity-flow", Domain::Compliance)
                .with_description("Reciprocal transaction detection")
                .with_throughput(100_000)
                .with_latency_us(50.0),
        }
    }

    /// Detect reciprocal transactions.
    ///
    /// # Arguments
    /// * `transactions` - List of transactions
    /// * `window` - Time window for reciprocity (None = all time)
    /// * `min_amount` - Minimum amount to consider
    pub fn compute(
        transactions: &[Transaction],
        window: Option<TimeWindow>,
        min_amount: f64,
    ) -> ReciprocityResult {
        if transactions.is_empty() {
            return ReciprocityResult {
                reciprocity_ratio: 0.0,
                reciprocal_pairs: Vec::new(),
                reciprocal_amount: 0.0,
            };
        }

        // Filter by time window if specified
        let txs: Vec<&Transaction> = transactions
            .iter()
            .filter(|tx| window.map(|w| w.contains(tx.timestamp)).unwrap_or(true))
            .collect();

        // Build directed edge map: (src, dst) -> total amount
        let mut edge_amounts: HashMap<(u64, u64), f64> = HashMap::new();

        for tx in &txs {
            *edge_amounts.entry((tx.source_id, tx.dest_id)).or_default() += tx.amount;
        }

        // Find reciprocal pairs
        let mut reciprocal_pairs = Vec::new();
        let mut reciprocal_amount = 0.0;
        let mut processed: HashSet<(u64, u64)> = HashSet::new();

        for (&(src, dst), &amount) in &edge_amounts {
            if processed.contains(&(src, dst)) || processed.contains(&(dst, src)) {
                continue;
            }

            if let Some(&reverse_amount) = edge_amounts.get(&(dst, src)) {
                let min_reciprocal = amount.min(reverse_amount);
                if min_reciprocal >= min_amount {
                    reciprocal_pairs.push((src, dst));
                    reciprocal_amount += min_reciprocal * 2.0;
                    processed.insert((src, dst));
                    processed.insert((dst, src));
                }
            }
        }

        let total_amount: f64 = txs.iter().map(|tx| tx.amount).sum();
        let reciprocity_ratio = if total_amount > 0.0 {
            reciprocal_amount / total_amount
        } else {
            0.0
        };

        ReciprocityResult {
            reciprocity_ratio,
            reciprocal_pairs,
            reciprocal_amount,
        }
    }
}

impl GpuKernel for ReciprocityFlowRatio {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[async_trait]
impl BatchKernel<ReciprocityFlowInput, ReciprocityFlowOutput> for ReciprocityFlowRatio {
    async fn execute(&self, input: ReciprocityFlowInput) -> Result<ReciprocityFlowOutput> {
        let start = Instant::now();
        let result = Self::compute(&input.transactions, input.window, input.min_amount);
        Ok(ReciprocityFlowOutput {
            result,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

// ============================================================================
// Rapid Movement Kernel
// ============================================================================

/// Rapid movement (velocity) analysis kernel.
///
/// Detects accounts with unusually high transaction velocity,
/// which can indicate structuring or rapid movement schemes.
#[derive(Debug, Clone)]
pub struct RapidMovement {
    metadata: KernelMetadata,
}

impl Default for RapidMovement {
    fn default() -> Self {
        Self::new()
    }
}

impl RapidMovement {
    /// Create a new rapid movement kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("compliance/rapid-movement", Domain::Compliance)
                .with_description("Velocity analysis for rapid movement detection")
                .with_throughput(200_000)
                .with_latency_us(20.0),
        }
    }

    /// Detect rapid movement patterns.
    ///
    /// # Arguments
    /// * `transactions` - List of transactions
    /// * `window_hours` - Time window in hours for velocity calculation
    /// * `velocity_threshold` - Transactions per hour threshold
    /// * `amount_threshold` - Minimum amount threshold
    pub fn compute(
        transactions: &[Transaction],
        window_hours: f64,
        velocity_threshold: f64,
        amount_threshold: f64,
    ) -> RapidMovementResult {
        if transactions.is_empty() || window_hours <= 0.0 {
            return RapidMovementResult {
                flagged_entities: Vec::new(),
                velocity_metrics: Vec::new(),
                rapid_amount: 0.0,
            };
        }

        let window_seconds = (window_hours * 3600.0) as u64;

        // Group transactions by entity (both as source and dest)
        let mut entity_txs: HashMap<u64, Vec<&Transaction>> = HashMap::new();

        for tx in transactions {
            entity_txs.entry(tx.source_id).or_default().push(tx);
            entity_txs.entry(tx.dest_id).or_default().push(tx);
        }

        let mut flagged_entities = Vec::new();
        let mut velocity_metrics = Vec::new();
        let mut rapid_amount = 0.0;

        for (entity_id, txs) in entity_txs {
            if txs.is_empty() {
                continue;
            }

            // Sort by timestamp
            let mut sorted_txs: Vec<_> = txs.into_iter().collect();
            sorted_txs.sort_by_key(|tx| tx.timestamp);

            // Calculate velocity using sliding window
            let mut max_velocity = 0.0f64;
            let mut max_window_amount = 0.0f64;

            for (i, tx) in sorted_txs.iter().enumerate() {
                let window_start = tx.timestamp;
                let window_end = window_start + window_seconds;

                let window_txs: Vec<_> = sorted_txs[i..]
                    .iter()
                    .take_while(|t| t.timestamp < window_end)
                    .collect();

                let count = window_txs.len();
                let velocity = count as f64 / window_hours;
                let window_amount: f64 = window_txs.iter().map(|t| t.amount).sum();

                if velocity > max_velocity {
                    max_velocity = velocity;
                    max_window_amount = window_amount;
                }
            }

            velocity_metrics.push((entity_id, max_velocity));

            if max_velocity >= velocity_threshold && max_window_amount >= amount_threshold {
                flagged_entities.push(entity_id);
                rapid_amount += max_window_amount;
            }
        }

        // Sort by velocity descending
        velocity_metrics.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        RapidMovementResult {
            flagged_entities,
            velocity_metrics,
            rapid_amount,
        }
    }
}

impl GpuKernel for RapidMovement {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[async_trait]
impl BatchKernel<RapidMovementInput, RapidMovementOutput> for RapidMovement {
    async fn execute(&self, input: RapidMovementInput) -> Result<RapidMovementOutput> {
        let start = Instant::now();
        let result = Self::compute(
            &input.transactions,
            input.window_hours,
            input.velocity_threshold,
            input.amount_threshold,
        );
        Ok(RapidMovementOutput {
            result,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

// ============================================================================
// AML Pattern Detection Kernel
// ============================================================================

/// Multi-pattern AML detection kernel.
///
/// Detects various AML patterns including structuring, layering,
/// funnel accounts, and fan-out patterns.
#[derive(Debug, Clone)]
pub struct AMLPatternDetection {
    metadata: KernelMetadata,
}

impl Default for AMLPatternDetection {
    fn default() -> Self {
        Self::new()
    }
}

impl AMLPatternDetection {
    /// Create a new AML pattern detection kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("compliance/aml-patterns", Domain::Compliance)
                .with_description("Multi-pattern AML detection")
                .with_throughput(30_000)
                .with_latency_us(200.0),
        }
    }

    /// Detect AML patterns in transactions.
    ///
    /// # Arguments
    /// * `transactions` - List of transactions
    /// * `structuring_threshold` - Amount threshold for structuring (e.g., 10000)
    /// * `structuring_window_hours` - Time window for structuring detection
    pub fn compute(
        transactions: &[Transaction],
        structuring_threshold: f64,
        structuring_window_hours: f64,
    ) -> AMLPatternResult {
        if transactions.is_empty() {
            return AMLPatternResult {
                patterns: Vec::new(),
                risk_score: 0.0,
                pattern_details: Vec::new(),
            };
        }

        let mut patterns = Vec::new();
        let mut pattern_details = Vec::new();
        let mut total_risk_score = 0.0;

        // Detect structuring
        let structuring = Self::detect_structuring(
            transactions,
            structuring_threshold,
            structuring_window_hours,
        );
        if !structuring.is_empty() {
            for detail in structuring {
                total_risk_score += detail.confidence * 25.0;
                patterns.push((AMLPattern::Structuring, detail.entities.clone()));
                pattern_details.push(detail);
            }
        }

        // Detect funnel accounts
        let funnels = Self::detect_funnel_accounts(transactions);
        if !funnels.is_empty() {
            for detail in funnels {
                total_risk_score += detail.confidence * 20.0;
                patterns.push((AMLPattern::FunnelAccount, detail.entities.clone()));
                pattern_details.push(detail);
            }
        }

        // Detect fan-out patterns
        let fanouts = Self::detect_fan_out(transactions);
        if !fanouts.is_empty() {
            for detail in fanouts {
                total_risk_score += detail.confidence * 15.0;
                patterns.push((AMLPattern::FanOut, detail.entities.clone()));
                pattern_details.push(detail);
            }
        }

        let risk_score = total_risk_score.min(100.0);

        AMLPatternResult {
            patterns,
            risk_score,
            pattern_details,
        }
    }

    /// Detect structuring (smurfing) patterns.
    fn detect_structuring(
        transactions: &[Transaction],
        threshold: f64,
        window_hours: f64,
    ) -> Vec<PatternDetail> {
        let window_seconds = (window_hours * 3600.0) as u64;
        let mut results = Vec::new();

        // Group by source entity
        let mut by_source: HashMap<u64, Vec<&Transaction>> = HashMap::new();
        for tx in transactions {
            by_source.entry(tx.source_id).or_default().push(tx);
        }

        for (source_id, txs) in by_source {
            if txs.len() < 3 {
                continue;
            }

            let mut sorted_txs: Vec<_> = txs.into_iter().collect();
            sorted_txs.sort_by_key(|tx| tx.timestamp);

            // Look for multiple transactions just under threshold
            for (i, tx) in sorted_txs.iter().enumerate() {
                let window_end = tx.timestamp + window_seconds;

                let window_txs: Vec<_> = sorted_txs[i..]
                    .iter()
                    .take_while(|t| t.timestamp < window_end)
                    .filter(|t| t.amount < threshold && t.amount > threshold * 0.7)
                    .collect();

                if window_txs.len() >= 3 {
                    let total_amount: f64 = window_txs.iter().map(|t| t.amount).sum();
                    if total_amount > threshold {
                        let confidence = (window_txs.len() as f64 / 10.0).min(1.0);
                        let dests: HashSet<u64> = window_txs.iter().map(|t| t.dest_id).collect();
                        let mut entities = vec![source_id];
                        entities.extend(dests);

                        results.push(PatternDetail {
                            pattern: AMLPattern::Structuring,
                            entities,
                            amount: total_amount,
                            confidence,
                            time_span: TimeWindow::new(
                                tx.timestamp,
                                window_txs
                                    .last()
                                    .map(|t| t.timestamp)
                                    .unwrap_or(tx.timestamp),
                            ),
                        });
                        break; // One pattern per source
                    }
                }
            }
        }

        results
    }

    /// Detect funnel accounts (many sources to one destination).
    fn detect_funnel_accounts(transactions: &[Transaction]) -> Vec<PatternDetail> {
        let mut results = Vec::new();

        // Count incoming transactions per destination
        let mut incoming: HashMap<u64, Vec<&Transaction>> = HashMap::new();
        for tx in transactions {
            incoming.entry(tx.dest_id).or_default().push(tx);
        }

        for (dest_id, txs) in incoming {
            let unique_sources: HashSet<u64> = txs.iter().map(|tx| tx.source_id).collect();

            // Flag if many sources funnel to one destination
            if unique_sources.len() >= 5 {
                let total_amount: f64 = txs.iter().map(|t| t.amount).sum();
                let confidence = (unique_sources.len() as f64 / 20.0).min(1.0);

                let mut entities = vec![dest_id];
                entities.extend(unique_sources.iter().take(10));

                let timestamps: Vec<u64> = txs.iter().map(|t| t.timestamp).collect();

                results.push(PatternDetail {
                    pattern: AMLPattern::FunnelAccount,
                    entities,
                    amount: total_amount,
                    confidence,
                    time_span: TimeWindow::new(
                        *timestamps.iter().min().unwrap_or(&0),
                        *timestamps.iter().max().unwrap_or(&0),
                    ),
                });
            }
        }

        results
    }

    /// Detect fan-out patterns (one source to many destinations).
    fn detect_fan_out(transactions: &[Transaction]) -> Vec<PatternDetail> {
        let mut results = Vec::new();

        // Count outgoing transactions per source
        let mut outgoing: HashMap<u64, Vec<&Transaction>> = HashMap::new();
        for tx in transactions {
            outgoing.entry(tx.source_id).or_default().push(tx);
        }

        for (source_id, txs) in outgoing {
            let unique_dests: HashSet<u64> = txs.iter().map(|tx| tx.dest_id).collect();

            // Flag if one source fans out to many destinations
            if unique_dests.len() >= 5 {
                let total_amount: f64 = txs.iter().map(|t| t.amount).sum();
                let confidence = (unique_dests.len() as f64 / 20.0).min(1.0);

                let mut entities = vec![source_id];
                entities.extend(unique_dests.iter().take(10));

                let timestamps: Vec<u64> = txs.iter().map(|t| t.timestamp).collect();

                results.push(PatternDetail {
                    pattern: AMLPattern::FanOut,
                    entities,
                    amount: total_amount,
                    confidence,
                    time_span: TimeWindow::new(
                        *timestamps.iter().min().unwrap_or(&0),
                        *timestamps.iter().max().unwrap_or(&0),
                    ),
                });
            }
        }

        results
    }
}

impl GpuKernel for AMLPatternDetection {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[async_trait]
impl BatchKernel<AMLPatternInput, AMLPatternOutput> for AMLPatternDetection {
    async fn execute(&self, input: AMLPatternInput) -> Result<AMLPatternOutput> {
        let start = Instant::now();
        let result = Self::compute(
            &input.transactions,
            input.structuring_threshold,
            input.structuring_window_hours,
        );
        Ok(AMLPatternOutput {
            result,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

// ----------------------------------------------------------------------------
// Ring Kernel Handler Implementation for AMLPatternDetection
// ----------------------------------------------------------------------------

/// Ring handler for pattern matching on streaming transactions.
#[async_trait]
impl RingKernelHandler<MatchPatternRing, MatchPatternResponse> for AMLPatternDetection {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: MatchPatternRing,
    ) -> Result<MatchPatternResponse> {
        // Convert Ring message to domain Transaction
        let transaction = Transaction {
            id: msg.tx_id,
            source_id: msg.source_id,
            dest_id: msg.dest_id,
            amount: msg.amount as f64 / 100_000_000.0,
            timestamp: msg.timestamp,
            currency: "USD".to_string(),
            tx_type: match msg.tx_type {
                0 => "wire",
                1 => "ach",
                2 => "check",
                _ => "other",
            }
            .to_string(),
        };

        // Default thresholds for streaming analysis
        let structuring_threshold = 10_000.0;
        let structuring_window_hours = 24.0;

        let result = Self::compute(
            &[transaction],
            structuring_threshold,
            structuring_window_hours,
        );

        // Build patterns_matched bitmask
        let mut patterns_matched = 0u64;
        for (pattern, _) in &result.patterns {
            let bit = match pattern {
                AMLPattern::Structuring => 0,
                AMLPattern::Layering => 1,
                AMLPattern::RapidMovement => 2,
                AMLPattern::RoundTripping => 3,
                AMLPattern::FunnelAccount => 4,
                AMLPattern::FanOut => 5,
            };
            patterns_matched |= 1u64 << bit;
        }

        let max_score = result
            .pattern_details
            .iter()
            .map(|d| d.confidence)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0) as f32;

        Ok(MatchPatternResponse {
            correlation_id: msg.correlation_id,
            tx_id: msg.tx_id,
            patterns_matched,
            max_score,
            match_count: result.patterns.len() as u32,
        })
    }
}

// ============================================================================
// Flow Reversal Pattern Kernel
// ============================================================================

/// Flow reversal pattern detection kernel.
///
/// Detects transactions that are reversed (A->B followed by B->A).
/// Critical for detecting wash trading, round-tripping, and layering.
#[derive(Debug, Clone)]
pub struct FlowReversalPattern {
    metadata: KernelMetadata,
}

impl Default for FlowReversalPattern {
    fn default() -> Self {
        Self::new()
    }
}

impl FlowReversalPattern {
    /// Create a new flow reversal pattern kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("compliance/flow-reversal", Domain::Compliance)
                .with_description("Transaction reversal pattern detection")
                .with_throughput(80_000)
                .with_latency_us(50.0),
        }
    }

    /// Detect flow reversal patterns.
    ///
    /// # Arguments
    /// * `transactions` - List of transactions to analyze
    /// * `config` - Reversal detection configuration
    pub fn compute(
        transactions: &[Transaction],
        config: &FlowReversalConfig,
    ) -> crate::types::FlowReversalResult {
        use crate::types::{FlowReversalPair, FlowReversalResult, ReversalRiskLevel};

        if transactions.is_empty() {
            return FlowReversalResult {
                reversals: Vec::new(),
                reversal_volume: 0.0,
                reversal_ratio: 0.0,
                repeat_offenders: Vec::new(),
                risk_score: 0.0,
            };
        }

        // Sort transactions by timestamp
        let mut sorted_txs: Vec<_> = transactions.iter().collect();
        sorted_txs.sort_by_key(|tx| tx.timestamp);

        // Build edge map for quick lookup: (src, dst) -> Vec<(tx_id, amount, timestamp)>
        let mut forward_edges: HashMap<(u64, u64), Vec<(u64, f64, u64)>> = HashMap::new();
        for tx in &sorted_txs {
            forward_edges
                .entry((tx.source_id, tx.dest_id))
                .or_default()
                .push((tx.id, tx.amount, tx.timestamp));
        }

        let mut reversals = Vec::new();
        let mut processed_pairs: HashSet<(u64, u64)> = HashSet::new();
        let mut entity_reversal_count: HashMap<u64, u32> = HashMap::new();

        // For each transaction A->B, look for B->A within the time window
        for tx in &sorted_txs {
            let reverse_key = (tx.dest_id, tx.source_id);

            if let Some(reverse_txs) = forward_edges.get(&reverse_key) {
                for &(rev_id, rev_amount, rev_timestamp) in reverse_txs {
                    // Skip if already processed this pair
                    if processed_pairs.contains(&(tx.id, rev_id))
                        || processed_pairs.contains(&(rev_id, tx.id))
                    {
                        continue;
                    }

                    // Check time window (reversal must come after original)
                    if rev_timestamp <= tx.timestamp {
                        continue;
                    }

                    let time_delta = rev_timestamp - tx.timestamp;
                    if time_delta > config.max_window_seconds {
                        continue;
                    }

                    // Check amount tolerance
                    let amount_ratio = if tx.amount > 0.0 {
                        (rev_amount / tx.amount).min(tx.amount / rev_amount)
                    } else {
                        0.0
                    };

                    if amount_ratio < config.min_amount_match_ratio {
                        continue;
                    }

                    // Calculate risk level
                    let risk_level = Self::calculate_risk_level(time_delta, amount_ratio, config);

                    reversals.push(FlowReversalPair {
                        original_tx_id: tx.id,
                        reversal_tx_id: rev_id,
                        entity_a: tx.source_id,
                        entity_b: tx.dest_id,
                        original_amount: tx.amount,
                        reversal_amount: rev_amount,
                        time_delta,
                        amount_match_ratio: amount_ratio,
                        risk_level,
                    });

                    processed_pairs.insert((tx.id, rev_id));

                    // Track repeat offenders
                    *entity_reversal_count.entry(tx.source_id).or_insert(0) += 1;
                    *entity_reversal_count.entry(tx.dest_id).or_insert(0) += 1;
                }
            }
        }

        // Calculate metrics
        let reversal_volume: f64 = reversals
            .iter()
            .map(|r| r.original_amount + r.reversal_amount)
            .sum();
        let total_volume: f64 = transactions.iter().map(|tx| tx.amount).sum();
        let reversal_ratio = if total_volume > 0.0 {
            reversal_volume / total_volume
        } else {
            0.0
        };

        // Find repeat offenders (entities with 2+ reversals)
        let mut repeat_offenders: Vec<_> = entity_reversal_count
            .into_iter()
            .filter(|(_, count)| *count >= 2)
            .collect();
        repeat_offenders.sort_by(|a, b| b.1.cmp(&a.1));

        // Calculate overall risk score
        let high_risk_count = reversals
            .iter()
            .filter(|r| {
                matches!(
                    r.risk_level,
                    ReversalRiskLevel::High | ReversalRiskLevel::Critical
                )
            })
            .count();

        let risk_score = ((reversals.len() as f64 * 5.0)
            + (high_risk_count as f64 * 15.0)
            + (repeat_offenders.len() as f64 * 10.0)
            + (reversal_ratio * 50.0))
            .min(100.0);

        FlowReversalResult {
            reversals,
            reversal_volume,
            reversal_ratio,
            repeat_offenders,
            risk_score,
        }
    }

    /// Calculate risk level for a reversal.
    fn calculate_risk_level(
        time_delta: u64,
        amount_ratio: f64,
        config: &FlowReversalConfig,
    ) -> crate::types::ReversalRiskLevel {
        use crate::types::ReversalRiskLevel;

        let is_exact_match = amount_ratio >= 0.99;
        let is_fast = time_delta < config.suspicious_window_seconds;
        let is_very_fast = time_delta < config.critical_window_seconds;

        match (is_very_fast, is_fast, is_exact_match) {
            (true, _, true) => ReversalRiskLevel::Critical,
            (true, _, _) => ReversalRiskLevel::High,
            (_, true, true) => ReversalRiskLevel::High,
            (_, true, _) => ReversalRiskLevel::Suspicious,
            (_, _, true) => ReversalRiskLevel::Suspicious,
            _ => ReversalRiskLevel::Normal,
        }
    }
}

impl GpuKernel for FlowReversalPattern {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Configuration for flow reversal detection.
#[derive(Debug, Clone)]
pub struct FlowReversalConfig {
    /// Maximum time window to consider reversals (seconds).
    pub max_window_seconds: u64,
    /// Time threshold for suspicious reversals (seconds).
    pub suspicious_window_seconds: u64,
    /// Time threshold for critical reversals (seconds).
    pub critical_window_seconds: u64,
    /// Minimum amount match ratio (0-1) to consider a reversal.
    pub min_amount_match_ratio: f64,
}

impl Default for FlowReversalConfig {
    fn default() -> Self {
        Self {
            max_window_seconds: 86400, // 24 hours
            suspicious_window_seconds: 3600, // 1 hour
            critical_window_seconds: 300, // 5 minutes
            min_amount_match_ratio: 0.9, // 90% match
        }
    }
}

// ============================================================================
// Flow Split Ratio Kernel
// ============================================================================

/// Flow split ratio detection kernel.
///
/// Detects structuring patterns where transactions are split to avoid
/// reporting thresholds (e.g., BSA $10,000 threshold).
#[derive(Debug, Clone)]
pub struct FlowSplitRatio {
    metadata: KernelMetadata,
}

impl Default for FlowSplitRatio {
    fn default() -> Self {
        Self::new()
    }
}

impl FlowSplitRatio {
    /// Create a new flow split ratio kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("compliance/flow-split", Domain::Compliance)
                .with_description("Transaction splitting and structuring detection")
                .with_throughput(60_000)
                .with_latency_us(80.0),
        }
    }

    /// Detect flow split patterns.
    ///
    /// # Arguments
    /// * `transactions` - List of transactions to analyze
    /// * `config` - Split detection configuration
    pub fn compute(
        transactions: &[Transaction],
        config: &FlowSplitConfig,
    ) -> crate::types::FlowSplitResult {
        use crate::types::{FlowSplitPattern, FlowSplitResult, SplitRiskLevel};

        if transactions.is_empty() {
            return FlowSplitResult {
                splits: Vec::new(),
                structuring_entities: Vec::new(),
                split_volume: 0.0,
                split_ratio: 0.0,
                risk_score: 0.0,
            };
        }

        // Group transactions by source entity
        let mut by_source: HashMap<u64, Vec<&Transaction>> = HashMap::new();
        for tx in transactions {
            by_source.entry(tx.source_id).or_default().push(tx);
        }

        let mut splits = Vec::new();
        let mut structuring_entities: HashSet<u64> = HashSet::new();

        for (source_id, source_txs) in by_source {
            if source_txs.len() < config.min_split_count {
                continue;
            }

            // Sort by timestamp
            let mut sorted_txs: Vec<_> = source_txs.into_iter().collect();
            sorted_txs.sort_by_key(|tx| tx.timestamp);

            // Sliding window to find split patterns
            for start_idx in 0..sorted_txs.len() {
                let start_tx = sorted_txs[start_idx];
                let window_end_time = start_tx.timestamp + config.window_seconds;

                // Collect transactions in window
                let window_txs: Vec<_> = sorted_txs[start_idx..]
                    .iter()
                    .take_while(|tx| tx.timestamp <= window_end_time)
                    .copied()
                    .collect();

                if window_txs.len() < config.min_split_count {
                    continue;
                }

                // Check if transactions are below threshold but sum near/above it
                let under_threshold: Vec<_> = window_txs
                    .iter()
                    .filter(|tx| tx.amount < config.reporting_threshold)
                    .copied()
                    .collect();

                if under_threshold.len() < config.min_split_count {
                    continue;
                }

                let total_amount: f64 = under_threshold.iter().map(|tx| tx.amount).sum();

                // Check if total is near or above reporting threshold
                if total_amount < config.reporting_threshold * 0.8 {
                    continue;
                }

                // Check for structuring indicators
                let amounts: Vec<f64> = under_threshold.iter().map(|tx| tx.amount).collect();
                let avg_amount = total_amount / amounts.len() as f64;
                let is_uniform = amounts
                    .iter()
                    .all(|a| (*a - avg_amount).abs() / avg_amount < 0.3);
                let near_threshold = amounts
                    .iter()
                    .filter(|a| **a > config.reporting_threshold * 0.7)
                    .count();

                // Calculate risk level
                let risk_level = Self::calculate_risk_level(
                    total_amount,
                    &amounts,
                    is_uniform,
                    near_threshold,
                    config,
                );

                if matches!(risk_level, SplitRiskLevel::Normal) {
                    continue;
                }

                let time_span = under_threshold
                    .last()
                    .map(|tx| tx.timestamp)
                    .unwrap_or(start_tx.timestamp)
                    .saturating_sub(start_tx.timestamp);

                let dest_entities: Vec<u64> = under_threshold
                    .iter()
                    .map(|tx| tx.dest_id)
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect();

                splits.push(FlowSplitPattern {
                    source_entity: source_id,
                    dest_entities,
                    transaction_ids: under_threshold.iter().map(|tx| tx.id).collect(),
                    amounts: amounts.clone(),
                    total_amount,
                    time_span,
                    estimated_threshold: config.reporting_threshold,
                    risk_level,
                });

                if matches!(
                    risk_level,
                    SplitRiskLevel::High | SplitRiskLevel::Critical
                ) {
                    structuring_entities.insert(source_id);
                }
            }
        }

        // Deduplicate overlapping patterns (keep highest risk)
        let splits = Self::deduplicate_splits(splits);

        // Calculate metrics
        let split_volume: f64 = splits.iter().map(|s| s.total_amount).sum();
        let total_volume: f64 = transactions.iter().map(|tx| tx.amount).sum();
        let split_ratio = if total_volume > 0.0 {
            split_volume / total_volume
        } else {
            0.0
        };

        // Calculate risk score
        let critical_count = splits
            .iter()
            .filter(|s| matches!(s.risk_level, SplitRiskLevel::Critical))
            .count();
        let high_count = splits
            .iter()
            .filter(|s| matches!(s.risk_level, SplitRiskLevel::High))
            .count();

        let risk_score = ((splits.len() as f64 * 5.0)
            + (high_count as f64 * 15.0)
            + (critical_count as f64 * 25.0)
            + (structuring_entities.len() as f64 * 10.0))
            .min(100.0);

        FlowSplitResult {
            splits,
            structuring_entities: structuring_entities.into_iter().collect(),
            split_volume,
            split_ratio,
            risk_score,
        }
    }

    /// Calculate risk level for a split pattern.
    fn calculate_risk_level(
        total_amount: f64,
        amounts: &[f64],
        is_uniform: bool,
        near_threshold_count: usize,
        config: &FlowSplitConfig,
    ) -> crate::types::SplitRiskLevel {
        use crate::types::SplitRiskLevel;

        let threshold = config.reporting_threshold;
        let just_under = total_amount >= threshold * 0.95 && total_amount <= threshold * 1.05;
        let multiple_near = near_threshold_count >= 3;

        match (just_under, multiple_near, is_uniform) {
            (true, true, true) => SplitRiskLevel::Critical,
            (true, true, _) => SplitRiskLevel::Critical,
            (true, _, true) => SplitRiskLevel::High,
            (true, _, _) => SplitRiskLevel::High,
            (_, true, _) => SplitRiskLevel::High,
            _ if amounts.len() >= 5 && total_amount > threshold => SplitRiskLevel::Elevated,
            _ => SplitRiskLevel::Normal,
        }
    }

    /// Deduplicate overlapping split patterns.
    fn deduplicate_splits(
        mut splits: Vec<crate::types::FlowSplitPattern>,
    ) -> Vec<crate::types::FlowSplitPattern> {
        // Sort by risk level (highest first), then by total amount
        splits.sort_by(|a, b| {
            let risk_ord = Self::risk_level_ord(&b.risk_level).cmp(&Self::risk_level_ord(&a.risk_level));
            if risk_ord == std::cmp::Ordering::Equal {
                b.total_amount
                    .partial_cmp(&a.total_amount)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                risk_ord
            }
        });

        let mut kept: Vec<crate::types::FlowSplitPattern> = Vec::new();
        let mut used_tx_ids: HashSet<u64> = HashSet::new();

        for split in splits {
            // Check overlap with already kept patterns
            let overlap = split
                .transaction_ids
                .iter()
                .filter(|id| used_tx_ids.contains(id))
                .count();

            // Keep if less than 50% overlap
            if overlap < split.transaction_ids.len() / 2 {
                for id in &split.transaction_ids {
                    used_tx_ids.insert(*id);
                }
                kept.push(split);
            }
        }

        kept
    }

    /// Convert risk level to ordinal for sorting.
    fn risk_level_ord(level: &crate::types::SplitRiskLevel) -> u8 {
        use crate::types::SplitRiskLevel;
        match level {
            SplitRiskLevel::Normal => 0,
            SplitRiskLevel::Elevated => 1,
            SplitRiskLevel::High => 2,
            SplitRiskLevel::Critical => 3,
        }
    }
}

impl GpuKernel for FlowSplitRatio {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Configuration for flow split detection.
#[derive(Debug, Clone)]
pub struct FlowSplitConfig {
    /// Reporting threshold to detect structuring around (e.g., $10,000).
    pub reporting_threshold: f64,
    /// Time window to look for split transactions (seconds).
    pub window_seconds: u64,
    /// Minimum number of transactions to constitute a split.
    pub min_split_count: usize,
}

impl Default for FlowSplitConfig {
    fn default() -> Self {
        Self {
            reporting_threshold: 10_000.0, // BSA threshold
            window_seconds: 86400,         // 24 hours
            min_split_count: 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_circular_transactions() -> Vec<Transaction> {
        // A -> B -> C -> A circular flow
        vec![
            Transaction {
                id: 1,
                source_id: 1,
                dest_id: 2,
                amount: 1000.0,
                timestamp: 100,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
            Transaction {
                id: 2,
                source_id: 2,
                dest_id: 3,
                amount: 950.0,
                timestamp: 200,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
            Transaction {
                id: 3,
                source_id: 3,
                dest_id: 1,
                amount: 900.0,
                timestamp: 300,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
        ]
    }

    fn create_reciprocal_transactions() -> Vec<Transaction> {
        vec![
            Transaction {
                id: 1,
                source_id: 1,
                dest_id: 2,
                amount: 5000.0,
                timestamp: 100,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
            Transaction {
                id: 2,
                source_id: 2,
                dest_id: 1,
                amount: 4800.0,
                timestamp: 200,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
        ]
    }

    #[test]
    fn test_circular_flow_metadata() {
        let kernel = CircularFlowRatio::new();
        assert_eq!(kernel.metadata().id, "compliance/circular-flow");
        assert_eq!(kernel.metadata().domain, Domain::Compliance);
    }

    #[test]
    fn test_circular_flow_detection() {
        let txs = create_circular_transactions();
        let result = CircularFlowRatio::compute(&txs, 100.0);

        assert!(result.circular_ratio > 0.0);
        assert!(!result.sccs.is_empty());
        assert!(result.circular_amount > 0.0);
    }

    #[test]
    fn test_reciprocity_detection() {
        let txs = create_reciprocal_transactions();
        let result = ReciprocityFlowRatio::compute(&txs, None, 100.0);

        assert!(result.reciprocity_ratio > 0.0);
        assert!(!result.reciprocal_pairs.is_empty());
    }

    #[test]
    fn test_rapid_movement_detection() {
        // Many transactions in short time
        let txs: Vec<Transaction> = (0..20)
            .map(|i| Transaction {
                id: i as u64,
                source_id: 1,
                dest_id: 2,
                amount: 500.0,
                timestamp: 100 + i as u64 * 60, // One minute apart
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            })
            .collect();

        let result = RapidMovement::compute(&txs, 1.0, 10.0, 1000.0);

        assert!(!result.flagged_entities.is_empty());
        assert!(result.rapid_amount > 0.0);
    }

    #[test]
    fn test_aml_pattern_funnel() {
        // Many sources to one destination
        let txs: Vec<Transaction> = (0..10)
            .map(|i| Transaction {
                id: i as u64,
                source_id: i as u64 + 100, // Different sources
                dest_id: 1,                // Same destination
                amount: 500.0,
                timestamp: 100 + i as u64,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            })
            .collect();

        let result = AMLPatternDetection::compute(&txs, 10000.0, 24.0);

        let funnel_patterns: Vec<_> = result
            .patterns
            .iter()
            .filter(|(p, _)| *p == AMLPattern::FunnelAccount)
            .collect();

        assert!(!funnel_patterns.is_empty());
    }

    #[test]
    fn test_empty_transactions() {
        let empty: Vec<Transaction> = vec![];

        let circ = CircularFlowRatio::compute(&empty, 100.0);
        assert!(circ.sccs.is_empty());

        let recip = ReciprocityFlowRatio::compute(&empty, None, 100.0);
        assert!(recip.reciprocal_pairs.is_empty());

        let rapid = RapidMovement::compute(&empty, 1.0, 10.0, 1000.0);
        assert!(rapid.flagged_entities.is_empty());

        let aml = AMLPatternDetection::compute(&empty, 10000.0, 24.0);
        assert!(aml.patterns.is_empty());

        let reversal = FlowReversalPattern::compute(&empty, &FlowReversalConfig::default());
        assert!(reversal.reversals.is_empty());

        let split = FlowSplitRatio::compute(&empty, &FlowSplitConfig::default());
        assert!(split.splits.is_empty());
    }

    // ========================================================================
    // Flow Reversal Pattern Tests
    // ========================================================================

    fn create_reversal_transactions() -> Vec<Transaction> {
        vec![
            // A -> B: 5000 at t=100
            Transaction {
                id: 1,
                source_id: 1,
                dest_id: 2,
                amount: 5000.0,
                timestamp: 100,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
            // B -> A: 5000 at t=200 (reversal with exact match)
            Transaction {
                id: 2,
                source_id: 2,
                dest_id: 1,
                amount: 5000.0,
                timestamp: 200,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
            // C -> D: 3000 at t=300
            Transaction {
                id: 3,
                source_id: 3,
                dest_id: 4,
                amount: 3000.0,
                timestamp: 300,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
            // D -> C: 2800 at t=400 (reversal with ~93% match)
            Transaction {
                id: 4,
                source_id: 4,
                dest_id: 3,
                amount: 2800.0,
                timestamp: 400,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
        ]
    }

    #[test]
    fn test_flow_reversal_metadata() {
        let kernel = FlowReversalPattern::new();
        assert_eq!(kernel.metadata().id, "compliance/flow-reversal");
        assert_eq!(kernel.metadata().domain, Domain::Compliance);
    }

    #[test]
    fn test_flow_reversal_detection() {
        let txs = create_reversal_transactions();
        let config = FlowReversalConfig::default();
        let result = FlowReversalPattern::compute(&txs, &config);

        assert_eq!(result.reversals.len(), 2);
        assert!(result.reversal_volume > 0.0);
        assert!(result.reversal_ratio > 0.0);
    }

    #[test]
    fn test_flow_reversal_exact_match() {
        let txs = vec![
            Transaction {
                id: 1,
                source_id: 1,
                dest_id: 2,
                amount: 1000.0,
                timestamp: 100,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
            Transaction {
                id: 2,
                source_id: 2,
                dest_id: 1,
                amount: 1000.0,
                timestamp: 150, // Very fast reversal
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
        ];

        let config = FlowReversalConfig {
            critical_window_seconds: 100,
            suspicious_window_seconds: 600,
            ..Default::default()
        };
        let result = FlowReversalPattern::compute(&txs, &config);

        assert_eq!(result.reversals.len(), 1);
        assert_eq!(result.reversals[0].amount_match_ratio, 1.0);
        // Should be Critical due to very fast + exact match
        assert!(matches!(
            result.reversals[0].risk_level,
            crate::types::ReversalRiskLevel::Critical
        ));
    }

    #[test]
    fn test_flow_reversal_outside_window() {
        let txs = vec![
            Transaction {
                id: 1,
                source_id: 1,
                dest_id: 2,
                amount: 1000.0,
                timestamp: 100,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
            Transaction {
                id: 2,
                source_id: 2,
                dest_id: 1,
                amount: 1000.0,
                timestamp: 100000, // Way outside window
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
        ];

        let config = FlowReversalConfig {
            max_window_seconds: 3600, // 1 hour window
            ..Default::default()
        };
        let result = FlowReversalPattern::compute(&txs, &config);

        assert!(result.reversals.is_empty());
    }

    // ========================================================================
    // Flow Split Ratio Tests
    // ========================================================================

    fn create_structuring_transactions() -> Vec<Transaction> {
        // Classic structuring: 5 transactions of $2500 each = $12,500 total
        // All just under the $10,000 threshold indicator
        (0..5)
            .map(|i| Transaction {
                id: i as u64,
                source_id: 1,
                dest_id: (i + 10) as u64,
                amount: 2500.0,
                timestamp: 1000 + i as u64 * 100,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            })
            .collect()
    }

    #[test]
    fn test_flow_split_metadata() {
        let kernel = FlowSplitRatio::new();
        assert_eq!(kernel.metadata().id, "compliance/flow-split");
        assert_eq!(kernel.metadata().domain, Domain::Compliance);
    }

    #[test]
    fn test_flow_split_detection() {
        let txs = create_structuring_transactions();
        let config = FlowSplitConfig::default();
        let result = FlowSplitRatio::compute(&txs, &config);

        // Should detect structuring pattern
        assert!(!result.splits.is_empty());
        assert!(result.split_volume > 0.0);
    }

    #[test]
    fn test_flow_split_near_threshold() {
        // 4 transactions of $2450 each = $9800, just under $10k
        let txs: Vec<Transaction> = (0..4)
            .map(|i| Transaction {
                id: i as u64,
                source_id: 1,
                dest_id: (i + 10) as u64,
                amount: 2450.0,
                timestamp: 1000 + i as u64 * 100,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            })
            .collect();

        let config = FlowSplitConfig {
            reporting_threshold: 10_000.0,
            min_split_count: 3,
            ..Default::default()
        };
        let result = FlowSplitRatio::compute(&txs, &config);

        // Total is $9800, should detect as structuring
        assert!(!result.splits.is_empty());

        // Should flag the source entity
        if !result.structuring_entities.is_empty() {
            assert!(result.structuring_entities.contains(&1));
        }
    }

    #[test]
    fn test_flow_split_below_threshold() {
        // Only 2 small transactions - should NOT be flagged
        let txs: Vec<Transaction> = (0..2)
            .map(|i| Transaction {
                id: i as u64,
                source_id: 1,
                dest_id: (i + 10) as u64,
                amount: 500.0, // Small amounts
                timestamp: 1000 + i as u64 * 100,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            })
            .collect();

        let config = FlowSplitConfig::default();
        let result = FlowSplitRatio::compute(&txs, &config);

        // Should not detect any structuring
        assert!(result.splits.is_empty());
    }

    #[test]
    fn test_flow_split_risk_levels() {
        // Create transactions targeting exactly $10k (critical structuring)
        let txs: Vec<Transaction> = (0..4)
            .map(|i| Transaction {
                id: i as u64,
                source_id: 1,
                dest_id: (i + 10) as u64,
                amount: 2500.0, // 4 x 2500 = 10000 exactly
                timestamp: 1000 + i as u64 * 100,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            })
            .collect();

        let config = FlowSplitConfig {
            reporting_threshold: 10_000.0,
            min_split_count: 3,
            ..Default::default()
        };
        let result = FlowSplitRatio::compute(&txs, &config);

        // Should detect high/critical risk
        if !result.splits.is_empty() {
            let has_high_risk = result.splits.iter().any(|s| {
                matches!(
                    s.risk_level,
                    crate::types::SplitRiskLevel::High | crate::types::SplitRiskLevel::Critical
                )
            });
            assert!(has_high_risk);
        }
    }
}
