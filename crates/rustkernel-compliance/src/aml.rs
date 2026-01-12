//! Anti-Money Laundering (AML) kernels.
//!
//! This module provides AML detection algorithms:
//! - Circular flow detection (SCC-based)
//! - Reciprocity analysis
//! - Rapid movement (velocity) analysis
//! - Multi-pattern AML detection

use crate::types::{
    AMLPattern, AMLPatternResult, CircularFlowResult, PatternDetail, RapidMovementResult,
    ReciprocityResult, TimeWindow, Transaction,
};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Circular Flow Ratio Kernel
// ============================================================================

/// Circular flow detection kernel.
///
/// Detects circular transactions using Strongly Connected Components (SCC).
/// High circular flow ratio indicates potential money laundering.
#[derive(Debug, Clone)]
pub struct CircularFlowRatio {
    metadata: KernelMetadata,
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
        }
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
            .filter(|tx| {
                window
                    .map(|w| w.contains(tx.timestamp))
                    .unwrap_or(true)
            })
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
                                window_txs.last().map(|t| t.timestamp).unwrap_or(tx.timestamp),
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
    }
}
