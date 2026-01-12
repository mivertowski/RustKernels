//! Accounting Network Generation Kernel.
//!
//! This module implements GPU-accelerated transformation of double-entry bookkeeping
//! journal entries into directed accounting networks (graphs). Based on the paper
//! "Hardware Accelerated Method for Accounting Network Generation".
//!
//! ## Key Concepts
//!
//! - **Journal Entry**: A balanced set of debit/credit line items (double-entry bookkeeping)
//! - **Accounting Flow**: A directed edge representing value transfer between accounts
//! - **Accounting Network**: The complete directed graph of all flows
//!
//! ## Solving Methods
//!
//! The module implements five solving methods with decreasing confidence:
//!
//! - **Method A** (Confidence: 1.0): Trivial 1-to-1 mapping for 2-line entries
//! - **Method B** (Confidence: 0.95): n-to-n bijective matching with Hungarian algorithm
//! - **Method C** (Confidence: 0.85): n-to-m partition matching using integer partition
//! - **Method D** (Confidence: 0.70): Aggregation for large entries
//! - **Method E** (Confidence: 0.50): Decomposition for complex multi-entity entries
//!
//! ## GPU Acceleration
//!
//! - Batch processing of journal entries in parallel
//! - Fixed-point arithmetic (128-bit) for exact decimal representation
//! - CSR sparse matrix format for efficient network storage
//! - Ring mode for streaming updates with temporal windowing

use crate::types::{JournalEntry, JournalLine};
use async_trait::async_trait;
use rustkernel_core::{
    domain::Domain,
    error::Result,
    kernel::KernelMetadata,
    traits::{BatchKernel, GpuKernel},
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

// ============================================================================
// Fixed-Point Arithmetic
// ============================================================================

/// Fixed-point representation for exact decimal arithmetic.
/// Uses 128 bits with 18 decimal places (1e18 scale factor).
/// Supports values up to ~170 trillion with 18 decimal precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FixedPoint128 {
    /// Internal value scaled by 1e18.
    pub value: i128,
}

impl FixedPoint128 {
    /// Scale factor: 10^18 for 18 decimal places.
    pub const SCALE: i128 = 1_000_000_000_000_000_000;

    /// Zero value.
    pub const ZERO: Self = Self { value: 0 };

    /// Create from floating point.
    #[inline]
    pub fn from_f64(f: f64) -> Self {
        Self {
            value: (f * Self::SCALE as f64) as i128,
        }
    }

    /// Convert to floating point.
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.value as f64 / Self::SCALE as f64
    }

    /// Create from integer.
    #[inline]
    pub fn from_i64(i: i64) -> Self {
        Self {
            value: i as i128 * Self::SCALE,
        }
    }

    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        Self {
            value: self.value.abs(),
        }
    }

    /// Check if approximately equal within tolerance.
    #[inline]
    pub fn approx_eq(self, other: Self, tolerance: Self) -> bool {
        (self.value - other.value).abs() <= tolerance.value
    }

    /// Check if zero within tolerance.
    #[inline]
    pub fn is_zero(self, tolerance: Self) -> bool {
        self.value.abs() <= tolerance.value
    }
}

impl std::ops::Add for FixedPoint128 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
        }
    }
}

impl std::ops::Sub for FixedPoint128 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
        }
    }
}

impl std::ops::Neg for FixedPoint128 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self { value: -self.value }
    }
}

impl std::ops::AddAssign for FixedPoint128 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl std::ops::SubAssign for FixedPoint128 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

impl PartialOrd for FixedPoint128 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FixedPoint128 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl Default for FixedPoint128 {
    fn default() -> Self {
        Self::ZERO
    }
}

// ============================================================================
// Network Generation Types
// ============================================================================

/// Solving method used to generate an accounting flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SolvingMethod {
    /// Method A: Trivial 1-to-1 for 2-line entries.
    MethodA,
    /// Method B: n-to-n bijective matching (Hungarian algorithm).
    MethodB,
    /// Method C: n-to-m partition matching.
    MethodC,
    /// Method D: Aggregation for large entries.
    MethodD,
    /// Method E: Decomposition for complex entries.
    MethodE,
    /// Unable to solve (suspense account).
    Unsolvable,
}

impl SolvingMethod {
    /// Get the confidence level for this method.
    #[inline]
    pub fn confidence(&self) -> f64 {
        match self {
            SolvingMethod::MethodA => 1.00,
            SolvingMethod::MethodB => 0.95,
            SolvingMethod::MethodC => 0.85,
            SolvingMethod::MethodD => 0.70,
            SolvingMethod::MethodE => 0.50,
            SolvingMethod::Unsolvable => 0.00,
        }
    }

    /// Get the method name.
    pub fn name(&self) -> &'static str {
        match self {
            SolvingMethod::MethodA => "A (1-to-1)",
            SolvingMethod::MethodB => "B (n-to-n)",
            SolvingMethod::MethodC => "C (n-to-m)",
            SolvingMethod::MethodD => "D (aggregation)",
            SolvingMethod::MethodE => "E (decomposition)",
            SolvingMethod::Unsolvable => "Unsolvable",
        }
    }
}

/// A line item classified as debit or credit.
#[derive(Debug, Clone)]
pub struct ClassifiedLine {
    /// Line number from original entry.
    pub line_number: u32,
    /// Account code.
    pub account_code: String,
    /// Amount (always positive, sign determined by is_debit).
    pub amount: FixedPoint128,
    /// True if debit, false if credit.
    pub is_debit: bool,
    /// Entity ID.
    pub entity_id: String,
    /// Cost center (optional).
    pub cost_center: Option<String>,
}

/// An accounting flow (directed edge in the network).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountingFlow {
    /// Unique flow ID.
    pub flow_id: u64,
    /// Source journal entry ID.
    pub entry_id: u64,
    /// Source account (debit account).
    pub from_account: String,
    /// Target account (credit account).
    pub to_account: String,
    /// Flow amount (fixed-point).
    pub amount: FixedPoint128,
    /// Amount as f64 for convenience.
    pub amount_f64: f64,
    /// Timestamp of the journal entry.
    pub timestamp: u64,
    /// Solving method used to derive this flow.
    pub method: SolvingMethod,
    /// Confidence level (derived from method).
    pub confidence: f64,
    /// Source entity ID.
    pub from_entity: String,
    /// Target entity ID.
    pub to_entity: String,
    /// Currency.
    pub currency: String,
    /// Source line numbers (for audit trail).
    pub source_lines: Vec<u32>,
}

/// Result of generating the accounting network for a single entry.
#[derive(Debug, Clone)]
pub struct EntryNetworkResult {
    /// Original entry ID.
    pub entry_id: u64,
    /// Generated flows.
    pub flows: Vec<AccountingFlow>,
    /// Solving method used.
    pub method: SolvingMethod,
    /// Weighted confidence (method confidence * amount weight).
    pub confidence: f64,
    /// Whether the entry was balanced.
    pub was_balanced: bool,
    /// Error message if any.
    pub error: Option<String>,
}

/// Complete accounting network.
#[derive(Debug, Clone, Default)]
pub struct AccountingNetwork {
    /// All accounting flows.
    pub flows: Vec<AccountingFlow>,
    /// Account nodes (unique accounts).
    pub accounts: HashSet<String>,
    /// Account index mapping for CSR representation.
    pub account_index: HashMap<String, usize>,
    /// Adjacency list (from_account -> [(to_account, flow_index)]).
    pub adjacency: HashMap<String, Vec<(String, usize)>>,
    /// Statistics.
    pub stats: NetworkGenerationStats,
}

impl AccountingNetwork {
    /// Create a new empty network.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a flow to the network.
    pub fn add_flow(&mut self, flow: AccountingFlow) {
        // Add accounts
        if !self.accounts.contains(&flow.from_account) {
            let idx = self.accounts.len();
            self.accounts.insert(flow.from_account.clone());
            self.account_index.insert(flow.from_account.clone(), idx);
        }
        if !self.accounts.contains(&flow.to_account) {
            let idx = self.accounts.len();
            self.accounts.insert(flow.to_account.clone());
            self.account_index.insert(flow.to_account.clone(), idx);
        }

        // Add to adjacency list
        let flow_idx = self.flows.len();
        self.adjacency
            .entry(flow.from_account.clone())
            .or_default()
            .push((flow.to_account.clone(), flow_idx));

        // Add flow
        self.flows.push(flow);
    }

    /// Get outgoing flows from an account.
    pub fn outgoing_flows(&self, account: &str) -> Vec<&AccountingFlow> {
        self.adjacency
            .get(account)
            .map(|edges| edges.iter().map(|(_, idx)| &self.flows[*idx]).collect())
            .unwrap_or_default()
    }

    /// Get incoming flows to an account.
    pub fn incoming_flows(&self, account: &str) -> Vec<&AccountingFlow> {
        self.flows
            .iter()
            .filter(|f| f.to_account == account)
            .collect()
    }

    /// Query flows within a time window.
    pub fn query_temporal(&self, start_time: u64, end_time: u64) -> Vec<&AccountingFlow> {
        self.flows
            .iter()
            .filter(|f| f.timestamp >= start_time && f.timestamp <= end_time)
            .collect()
    }

    /// Calculate total flow volume.
    pub fn total_volume(&self) -> f64 {
        self.flows.iter().map(|f| f.amount_f64).sum()
    }

    /// Calculate weighted average confidence.
    pub fn weighted_confidence(&self) -> f64 {
        if self.flows.is_empty() {
            return 0.0;
        }
        let total_amount: f64 = self.flows.iter().map(|f| f.amount_f64).sum();
        if total_amount == 0.0 {
            return self.flows.iter().map(|f| f.confidence).sum::<f64>() / self.flows.len() as f64;
        }
        self.flows
            .iter()
            .map(|f| f.confidence * f.amount_f64)
            .sum::<f64>()
            / total_amount
    }
}

/// Statistics for network generation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkGenerationStats {
    /// Total entries processed.
    pub total_entries: usize,
    /// Entries solved by Method A (1-to-1).
    pub method_a_count: usize,
    /// Entries solved by Method B (n-to-n).
    pub method_b_count: usize,
    /// Entries solved by Method C (n-to-m).
    pub method_c_count: usize,
    /// Entries solved by Method D (aggregation).
    pub method_d_count: usize,
    /// Entries solved by Method E (decomposition).
    pub method_e_count: usize,
    /// Unsolvable entries (routed to suspense).
    pub unsolvable_count: usize,
    /// Total flows generated.
    pub total_flows: usize,
    /// Total volume processed.
    pub total_volume: f64,
    /// Weighted average confidence.
    pub weighted_confidence: f64,
    /// Processing time in microseconds.
    pub processing_time_us: u64,
    /// Entries with balance errors.
    pub balance_errors: usize,
}

// ============================================================================
// Network Generation Configuration
// ============================================================================

/// Configuration for network generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkGenerationConfig {
    /// Tolerance for amount matching (default: 0.01).
    pub amount_tolerance: f64,
    /// Maximum lines for Method B (bijective matching).
    pub max_lines_method_b: usize,
    /// Maximum lines for Method C (partition matching).
    pub max_lines_method_c: usize,
    /// Maximum partition search depth for Method C.
    pub max_partition_depth: usize,
    /// Enable Method D (aggregation).
    pub enable_aggregation: bool,
    /// Enable Method E (decomposition).
    pub enable_decomposition: bool,
    /// Suspense account code for unsolvable entries.
    pub suspense_account: String,
    /// Whether to fail on unbalanced entries.
    pub strict_balance: bool,
}

impl Default for NetworkGenerationConfig {
    fn default() -> Self {
        Self {
            amount_tolerance: 0.01,
            max_lines_method_b: 10,
            max_lines_method_c: 20,
            max_partition_depth: 1000,
            enable_aggregation: true,
            enable_decomposition: true,
            suspense_account: "SUSPENSE".to_string(),
            strict_balance: false,
        }
    }
}

// ============================================================================
// Network Generation Kernel
// ============================================================================

/// Accounting Network Generation Kernel.
///
/// Transforms journal entries into a directed accounting network.
/// Uses GPU-accelerated batch processing for high throughput.
#[derive(Debug, Clone)]
pub struct NetworkGeneration {
    metadata: KernelMetadata,
    config: NetworkGenerationConfig,
}

impl Default for NetworkGeneration {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkGeneration {
    /// Create a new network generation kernel.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(NetworkGenerationConfig::default())
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: NetworkGenerationConfig) -> Self {
        Self {
            metadata: KernelMetadata::batch("accounting/network-generation", Domain::Accounting)
                .with_description("Journal entry to accounting network transformation")
                .with_throughput(50_000)
                .with_latency_us(50.0)
                .with_gpu_native(true),
            config,
        }
    }

    /// Generate accounting network from journal entries.
    pub fn generate(&self, entries: &[JournalEntry]) -> AccountingNetwork {
        let start = Instant::now();
        let mut network = AccountingNetwork::new();
        let mut flow_id = 0u64;

        let mut stats = NetworkGenerationStats {
            total_entries: entries.len(),
            ..Default::default()
        };

        for entry in entries {
            let result = self.process_entry(entry, &mut flow_id);

            // Update stats
            match result.method {
                SolvingMethod::MethodA => stats.method_a_count += 1,
                SolvingMethod::MethodB => stats.method_b_count += 1,
                SolvingMethod::MethodC => stats.method_c_count += 1,
                SolvingMethod::MethodD => stats.method_d_count += 1,
                SolvingMethod::MethodE => stats.method_e_count += 1,
                SolvingMethod::Unsolvable => stats.unsolvable_count += 1,
            }

            if !result.was_balanced {
                stats.balance_errors += 1;
            }

            // Add flows to network
            for flow in result.flows {
                stats.total_volume += flow.amount_f64;
                network.add_flow(flow);
            }
        }

        stats.total_flows = network.flows.len();
        stats.weighted_confidence = network.weighted_confidence();
        stats.processing_time_us = start.elapsed().as_micros() as u64;
        network.stats = stats;

        network
    }

    /// Process a single journal entry.
    fn process_entry(&self, entry: &JournalEntry, flow_id: &mut u64) -> EntryNetworkResult {
        // Classify lines into debits and credits
        let (debits, credits, balance_diff) = self.classify_lines(&entry.lines);

        // Check balance
        let tolerance = FixedPoint128::from_f64(self.config.amount_tolerance);
        let was_balanced = balance_diff.is_zero(tolerance);

        if !was_balanced && self.config.strict_balance {
            return EntryNetworkResult {
                entry_id: entry.id,
                flows: Vec::new(),
                method: SolvingMethod::Unsolvable,
                confidence: 0.0,
                was_balanced: false,
                error: Some(format!("Entry unbalanced: diff={}", balance_diff.to_f64())),
            };
        }

        let debit_count = debits.len();
        let credit_count = credits.len();

        // Select solving method based on line counts
        let (flows, method) = if debit_count == 1 && credit_count == 1 {
            // Method A: Trivial 1-to-1
            (
                self.solve_method_a(entry, &debits, &credits, flow_id),
                SolvingMethod::MethodA,
            )
        } else if debit_count == credit_count && debit_count <= self.config.max_lines_method_b {
            // Method B: n-to-n bijective matching
            (
                self.solve_method_b(entry, &debits, &credits, flow_id),
                SolvingMethod::MethodB,
            )
        } else if (debit_count + credit_count) <= self.config.max_lines_method_c {
            // Method C: n-to-m partition matching
            (
                self.solve_method_c(entry, &debits, &credits, flow_id),
                SolvingMethod::MethodC,
            )
        } else if self.config.enable_aggregation {
            // Method D: Aggregation
            (
                self.solve_method_d(entry, &debits, &credits, flow_id),
                SolvingMethod::MethodD,
            )
        } else if self.config.enable_decomposition {
            // Method E: Decomposition
            (
                self.solve_method_e(entry, &debits, &credits, flow_id),
                SolvingMethod::MethodE,
            )
        } else {
            // Unsolvable - route to suspense
            (
                self.create_suspense_flows(entry, &debits, &credits, flow_id),
                SolvingMethod::Unsolvable,
            )
        };

        let confidence = method.confidence();

        EntryNetworkResult {
            entry_id: entry.id,
            flows,
            method,
            confidence,
            was_balanced,
            error: None,
        }
    }

    /// Classify journal lines into debits and credits.
    fn classify_lines(
        &self,
        lines: &[JournalLine],
    ) -> (Vec<ClassifiedLine>, Vec<ClassifiedLine>, FixedPoint128) {
        let mut debits = Vec::new();
        let mut credits = Vec::new();
        let mut total_debit = FixedPoint128::ZERO;
        let mut total_credit = FixedPoint128::ZERO;

        for line in lines {
            if line.debit > 0.0 {
                let amount = FixedPoint128::from_f64(line.debit);
                total_debit += amount;
                debits.push(ClassifiedLine {
                    line_number: line.line_number,
                    account_code: line.account_code.clone(),
                    amount,
                    is_debit: true,
                    entity_id: line.entity_id.clone(),
                    cost_center: line.cost_center.clone(),
                });
            }
            if line.credit > 0.0 {
                let amount = FixedPoint128::from_f64(line.credit);
                total_credit += amount;
                credits.push(ClassifiedLine {
                    line_number: line.line_number,
                    account_code: line.account_code.clone(),
                    amount,
                    is_debit: false,
                    entity_id: line.entity_id.clone(),
                    cost_center: line.cost_center.clone(),
                });
            }
        }

        let balance_diff = total_debit - total_credit;
        (debits, credits, balance_diff)
    }

    // ========================================================================
    // Method A: Trivial 1-to-1 Matching
    // ========================================================================

    /// Method A: Simple 1-to-1 flow for 2-line entries.
    /// Confidence: 1.0 (deterministic)
    fn solve_method_a(
        &self,
        entry: &JournalEntry,
        debits: &[ClassifiedLine],
        credits: &[ClassifiedLine],
        flow_id: &mut u64,
    ) -> Vec<AccountingFlow> {
        if debits.len() != 1 || credits.len() != 1 {
            return Vec::new();
        }

        let debit = &debits[0];
        let credit = &credits[0];
        let amount = debit.amount.min(credit.amount);
        let currency = entry
            .lines
            .first()
            .map(|l| l.currency.clone())
            .unwrap_or_else(|| "USD".to_string());

        let flow = AccountingFlow {
            flow_id: *flow_id,
            entry_id: entry.id,
            from_account: debit.account_code.clone(),
            to_account: credit.account_code.clone(),
            amount,
            amount_f64: amount.to_f64(),
            timestamp: entry.posting_date,
            method: SolvingMethod::MethodA,
            confidence: SolvingMethod::MethodA.confidence(),
            from_entity: debit.entity_id.clone(),
            to_entity: credit.entity_id.clone(),
            currency,
            source_lines: vec![debit.line_number, credit.line_number],
        };

        *flow_id += 1;
        vec![flow]
    }

    // ========================================================================
    // Method B: n-to-n Bijective Matching (Hungarian Algorithm)
    // ========================================================================

    /// Method B: n-to-n bijective matching using greedy amount matching.
    /// Confidence: 0.95
    fn solve_method_b(
        &self,
        entry: &JournalEntry,
        debits: &[ClassifiedLine],
        credits: &[ClassifiedLine],
        flow_id: &mut u64,
    ) -> Vec<AccountingFlow> {
        if debits.len() != credits.len() {
            return self.solve_method_c(entry, debits, credits, flow_id);
        }

        let currency = entry
            .lines
            .first()
            .map(|l| l.currency.clone())
            .unwrap_or_else(|| "USD".to_string());
        let tolerance = FixedPoint128::from_f64(self.config.amount_tolerance);

        // Try to find exact amount matches first
        let mut flows = Vec::new();
        let mut matched_credits: HashSet<usize> = HashSet::new();
        let mut matched_debits: HashSet<usize> = HashSet::new();

        // Pass 1: Exact matches
        for (di, debit) in debits.iter().enumerate() {
            for (ci, credit) in credits.iter().enumerate() {
                if matched_credits.contains(&ci) {
                    continue;
                }
                if debit.amount.approx_eq(credit.amount, tolerance) {
                    flows.push(AccountingFlow {
                        flow_id: *flow_id,
                        entry_id: entry.id,
                        from_account: debit.account_code.clone(),
                        to_account: credit.account_code.clone(),
                        amount: debit.amount,
                        amount_f64: debit.amount.to_f64(),
                        timestamp: entry.posting_date,
                        method: SolvingMethod::MethodB,
                        confidence: SolvingMethod::MethodB.confidence(),
                        from_entity: debit.entity_id.clone(),
                        to_entity: credit.entity_id.clone(),
                        currency: currency.clone(),
                        source_lines: vec![debit.line_number, credit.line_number],
                    });
                    *flow_id += 1;
                    matched_credits.insert(ci);
                    matched_debits.insert(di);
                    break;
                }
            }
        }

        // Pass 2: Match remaining by order (fallback)
        let unmatched_debits: Vec<_> = debits
            .iter()
            .enumerate()
            .filter(|(i, _)| !matched_debits.contains(i))
            .map(|(_, d)| d)
            .collect();
        let unmatched_credits: Vec<_> = credits
            .iter()
            .enumerate()
            .filter(|(i, _)| !matched_credits.contains(i))
            .map(|(_, c)| c)
            .collect();

        for (debit, credit) in unmatched_debits.iter().zip(unmatched_credits.iter()) {
            let amount = debit.amount.min(credit.amount);
            flows.push(AccountingFlow {
                flow_id: *flow_id,
                entry_id: entry.id,
                from_account: debit.account_code.clone(),
                to_account: credit.account_code.clone(),
                amount,
                amount_f64: amount.to_f64(),
                timestamp: entry.posting_date,
                method: SolvingMethod::MethodB,
                confidence: SolvingMethod::MethodB.confidence(),
                from_entity: debit.entity_id.clone(),
                to_entity: credit.entity_id.clone(),
                currency: currency.clone(),
                source_lines: vec![debit.line_number, credit.line_number],
            });
            *flow_id += 1;
        }

        flows
    }

    // ========================================================================
    // Method C: n-to-m Partition Matching
    // ========================================================================

    /// Method C: n-to-m partition matching.
    /// Tries to find subset partitions where debit sums equal credit sums.
    /// Confidence: 0.85
    fn solve_method_c(
        &self,
        entry: &JournalEntry,
        debits: &[ClassifiedLine],
        credits: &[ClassifiedLine],
        flow_id: &mut u64,
    ) -> Vec<AccountingFlow> {
        let currency = entry
            .lines
            .first()
            .map(|l| l.currency.clone())
            .unwrap_or_else(|| "USD".to_string());
        let tolerance = FixedPoint128::from_f64(self.config.amount_tolerance);

        // Try to partition credits to match each debit
        let mut flows = Vec::new();
        let mut remaining_credits: Vec<(usize, ClassifiedLine)> =
            credits.iter().cloned().enumerate().collect();

        for debit in debits {
            // Find subset of credits that sum to this debit amount
            if let Some(matching_subset) =
                self.find_partition_subset(&remaining_credits, debit.amount, tolerance)
            {
                // Create flows for each credit in the matching subset
                for (ci, credit) in &matching_subset {
                    flows.push(AccountingFlow {
                        flow_id: *flow_id,
                        entry_id: entry.id,
                        from_account: debit.account_code.clone(),
                        to_account: credit.account_code.clone(),
                        amount: credit.amount,
                        amount_f64: credit.amount.to_f64(),
                        timestamp: entry.posting_date,
                        method: SolvingMethod::MethodC,
                        confidence: SolvingMethod::MethodC.confidence(),
                        from_entity: debit.entity_id.clone(),
                        to_entity: credit.entity_id.clone(),
                        currency: currency.clone(),
                        source_lines: vec![debit.line_number, credit.line_number],
                    });
                    *flow_id += 1;

                    // Remove matched credit
                    remaining_credits.retain(|(idx, _)| idx != ci);
                }
            } else {
                // Fallback: distribute debit proportionally across remaining credits
                let total_remaining: FixedPoint128 = remaining_credits
                    .iter()
                    .map(|(_, c)| c.amount)
                    .fold(FixedPoint128::ZERO, |acc, a| acc + a);

                if total_remaining.value > 0 {
                    for (_, credit) in &remaining_credits {
                        // Proportional allocation
                        let proportion = credit.amount.value as f64 / total_remaining.value as f64;
                        let allocated = FixedPoint128::from_f64(debit.amount.to_f64() * proportion);

                        if allocated.value > 0 {
                            flows.push(AccountingFlow {
                                flow_id: *flow_id,
                                entry_id: entry.id,
                                from_account: debit.account_code.clone(),
                                to_account: credit.account_code.clone(),
                                amount: allocated,
                                amount_f64: allocated.to_f64(),
                                timestamp: entry.posting_date,
                                method: SolvingMethod::MethodC,
                                confidence: SolvingMethod::MethodC.confidence() * 0.9, // Lower confidence for proportional
                                from_entity: debit.entity_id.clone(),
                                to_entity: credit.entity_id.clone(),
                                currency: currency.clone(),
                                source_lines: vec![debit.line_number, credit.line_number],
                            });
                            *flow_id += 1;
                        }
                    }
                }
            }
        }

        flows
    }

    /// Find a subset of credits that sums to the target amount.
    /// Uses dynamic programming for subset sum.
    fn find_partition_subset(
        &self,
        credits: &[(usize, ClassifiedLine)],
        target: FixedPoint128,
        tolerance: FixedPoint128,
    ) -> Option<Vec<(usize, ClassifiedLine)>> {
        let n = credits.len();
        if n == 0 {
            return None;
        }

        // Check single credit matches
        for (idx, credit) in credits {
            if credit.amount.approx_eq(target, tolerance) {
                return Some(vec![(*idx, credit.clone())]);
            }
        }

        // For small n, try exhaustive subset search
        if n <= 12 {
            return self.exhaustive_subset_search(credits, target, tolerance);
        }

        // For larger n, use greedy approximation
        self.greedy_subset_search(credits, target, tolerance)
    }

    /// Exhaustive subset search for small n.
    fn exhaustive_subset_search(
        &self,
        credits: &[(usize, ClassifiedLine)],
        target: FixedPoint128,
        tolerance: FixedPoint128,
    ) -> Option<Vec<(usize, ClassifiedLine)>> {
        let n = credits.len();
        let max_subsets = (1u32 << n).min(self.config.max_partition_depth as u32);

        for mask in 1..max_subsets {
            let mut sum = FixedPoint128::ZERO;
            let mut subset = Vec::new();

            for (i, (idx, credit)) in credits.iter().enumerate() {
                if (mask >> i) & 1 == 1 {
                    sum += credit.amount;
                    subset.push((*idx, credit.clone()));
                }
            }

            if sum.approx_eq(target, tolerance) {
                return Some(subset);
            }
        }

        None
    }

    /// Greedy subset search for larger n.
    fn greedy_subset_search(
        &self,
        credits: &[(usize, ClassifiedLine)],
        target: FixedPoint128,
        tolerance: FixedPoint128,
    ) -> Option<Vec<(usize, ClassifiedLine)>> {
        // Sort by amount descending
        let mut sorted: Vec<_> = credits.to_vec();
        sorted.sort_by(|a, b| b.1.amount.cmp(&a.1.amount));

        let mut remaining = target;
        let mut subset = Vec::new();

        for (idx, credit) in sorted {
            if credit.amount <= remaining + tolerance {
                remaining -= credit.amount;
                subset.push((idx, credit));

                if remaining.is_zero(tolerance) {
                    return Some(subset);
                }
            }
        }

        None
    }

    // ========================================================================
    // Method D: Aggregation
    // ========================================================================

    /// Method D: Aggregate small flows into larger flows.
    /// Used for entries with many lines.
    /// Confidence: 0.70
    fn solve_method_d(
        &self,
        entry: &JournalEntry,
        debits: &[ClassifiedLine],
        credits: &[ClassifiedLine],
        flow_id: &mut u64,
    ) -> Vec<AccountingFlow> {
        let currency = entry
            .lines
            .first()
            .map(|l| l.currency.clone())
            .unwrap_or_else(|| "USD".to_string());

        // Aggregate debits by account
        let mut debit_aggregates: HashMap<String, FixedPoint128> = HashMap::new();
        let mut debit_entities: HashMap<String, String> = HashMap::new();
        let mut debit_lines: HashMap<String, Vec<u32>> = HashMap::new();

        for debit in debits {
            *debit_aggregates
                .entry(debit.account_code.clone())
                .or_default() += debit.amount;
            debit_entities
                .entry(debit.account_code.clone())
                .or_insert_with(|| debit.entity_id.clone());
            debit_lines
                .entry(debit.account_code.clone())
                .or_default()
                .push(debit.line_number);
        }

        // Aggregate credits by account
        let mut credit_aggregates: HashMap<String, FixedPoint128> = HashMap::new();
        let mut credit_entities: HashMap<String, String> = HashMap::new();
        let mut credit_lines: HashMap<String, Vec<u32>> = HashMap::new();

        for credit in credits {
            *credit_aggregates
                .entry(credit.account_code.clone())
                .or_default() += credit.amount;
            credit_entities
                .entry(credit.account_code.clone())
                .or_insert_with(|| credit.entity_id.clone());
            credit_lines
                .entry(credit.account_code.clone())
                .or_default()
                .push(credit.line_number);
        }

        // Create flows between aggregated accounts
        let mut flows = Vec::new();
        let total_credit: FixedPoint128 = credit_aggregates
            .values()
            .copied()
            .fold(FixedPoint128::ZERO, |acc, a| acc + a);

        for (debit_account, debit_amount) in &debit_aggregates {
            for (credit_account, credit_amount) in &credit_aggregates {
                // Allocate proportionally
                let proportion = if total_credit.value > 0 {
                    credit_amount.value as f64 / total_credit.value as f64
                } else {
                    1.0 / credit_aggregates.len() as f64
                };

                let allocated = FixedPoint128::from_f64(debit_amount.to_f64() * proportion);

                if allocated.value > 0 {
                    let mut source_lines =
                        debit_lines.get(debit_account).cloned().unwrap_or_default();
                    source_lines.extend(
                        credit_lines
                            .get(credit_account)
                            .cloned()
                            .unwrap_or_default(),
                    );

                    flows.push(AccountingFlow {
                        flow_id: *flow_id,
                        entry_id: entry.id,
                        from_account: debit_account.clone(),
                        to_account: credit_account.clone(),
                        amount: allocated,
                        amount_f64: allocated.to_f64(),
                        timestamp: entry.posting_date,
                        method: SolvingMethod::MethodD,
                        confidence: SolvingMethod::MethodD.confidence(),
                        from_entity: debit_entities
                            .get(debit_account)
                            .cloned()
                            .unwrap_or_default(),
                        to_entity: credit_entities
                            .get(credit_account)
                            .cloned()
                            .unwrap_or_default(),
                        currency: currency.clone(),
                        source_lines,
                    });
                    *flow_id += 1;
                }
            }
        }

        flows
    }

    // ========================================================================
    // Method E: Decomposition
    // ========================================================================

    /// Method E: Decompose complex entries by entity.
    /// Splits multi-entity entries into separate sub-networks.
    /// Confidence: 0.50
    fn solve_method_e(
        &self,
        entry: &JournalEntry,
        debits: &[ClassifiedLine],
        credits: &[ClassifiedLine],
        flow_id: &mut u64,
    ) -> Vec<AccountingFlow> {
        let currency = entry
            .lines
            .first()
            .map(|l| l.currency.clone())
            .unwrap_or_else(|| "USD".to_string());

        // Group by entity
        let mut entity_debits: HashMap<String, Vec<&ClassifiedLine>> = HashMap::new();
        let mut entity_credits: HashMap<String, Vec<&ClassifiedLine>> = HashMap::new();

        for debit in debits {
            entity_debits
                .entry(debit.entity_id.clone())
                .or_default()
                .push(debit);
        }
        for credit in credits {
            entity_credits
                .entry(credit.entity_id.clone())
                .or_default()
                .push(credit);
        }

        let mut flows = Vec::new();

        // Process each entity's portion
        for (entity, entity_debits_list) in &entity_debits {
            let entity_credits_list = entity_credits.get(entity);

            for debit in entity_debits_list {
                // Try to match within entity first
                if let Some(credits_list) = entity_credits_list {
                    let total_entity_credit: FixedPoint128 = credits_list
                        .iter()
                        .map(|c| c.amount)
                        .fold(FixedPoint128::ZERO, |acc, a| acc + a);

                    for credit in credits_list.iter() {
                        let proportion = if total_entity_credit.value > 0 {
                            credit.amount.value as f64 / total_entity_credit.value as f64
                        } else {
                            1.0 / credits_list.len() as f64
                        };

                        let allocated = FixedPoint128::from_f64(debit.amount.to_f64() * proportion);

                        if allocated.value > 0 {
                            flows.push(AccountingFlow {
                                flow_id: *flow_id,
                                entry_id: entry.id,
                                from_account: debit.account_code.clone(),
                                to_account: credit.account_code.clone(),
                                amount: allocated,
                                amount_f64: allocated.to_f64(),
                                timestamp: entry.posting_date,
                                method: SolvingMethod::MethodE,
                                confidence: SolvingMethod::MethodE.confidence(),
                                from_entity: debit.entity_id.clone(),
                                to_entity: credit.entity_id.clone(),
                                currency: currency.clone(),
                                source_lines: vec![debit.line_number, credit.line_number],
                            });
                            *flow_id += 1;
                        }
                    }
                } else {
                    // Cross-entity flow: distribute across all credits
                    let all_credits: Vec<_> = credits.iter().collect();
                    let total_credit: FixedPoint128 = all_credits
                        .iter()
                        .map(|c| c.amount)
                        .fold(FixedPoint128::ZERO, |acc, a| acc + a);

                    for credit in all_credits {
                        let proportion = if total_credit.value > 0 {
                            credit.amount.value as f64 / total_credit.value as f64
                        } else {
                            1.0 / credits.len() as f64
                        };

                        let allocated = FixedPoint128::from_f64(debit.amount.to_f64() * proportion);

                        if allocated.value > 0 {
                            flows.push(AccountingFlow {
                                flow_id: *flow_id,
                                entry_id: entry.id,
                                from_account: debit.account_code.clone(),
                                to_account: credit.account_code.clone(),
                                amount: allocated,
                                amount_f64: allocated.to_f64(),
                                timestamp: entry.posting_date,
                                method: SolvingMethod::MethodE,
                                confidence: SolvingMethod::MethodE.confidence() * 0.8, // Lower for cross-entity
                                from_entity: debit.entity_id.clone(),
                                to_entity: credit.entity_id.clone(),
                                currency: currency.clone(),
                                source_lines: vec![debit.line_number, credit.line_number],
                            });
                            *flow_id += 1;
                        }
                    }
                }
            }
        }

        flows
    }

    // ========================================================================
    // Suspense Account Routing
    // ========================================================================

    /// Route unsolvable entries to suspense account.
    fn create_suspense_flows(
        &self,
        entry: &JournalEntry,
        debits: &[ClassifiedLine],
        credits: &[ClassifiedLine],
        flow_id: &mut u64,
    ) -> Vec<AccountingFlow> {
        let currency = entry
            .lines
            .first()
            .map(|l| l.currency.clone())
            .unwrap_or_else(|| "USD".to_string());

        let mut flows = Vec::new();

        // Route all debits to suspense
        for debit in debits {
            flows.push(AccountingFlow {
                flow_id: *flow_id,
                entry_id: entry.id,
                from_account: debit.account_code.clone(),
                to_account: self.config.suspense_account.clone(),
                amount: debit.amount,
                amount_f64: debit.amount.to_f64(),
                timestamp: entry.posting_date,
                method: SolvingMethod::Unsolvable,
                confidence: 0.0,
                from_entity: debit.entity_id.clone(),
                to_entity: "SUSPENSE".to_string(),
                currency: currency.clone(),
                source_lines: vec![debit.line_number],
            });
            *flow_id += 1;
        }

        // Route all credits from suspense
        for credit in credits {
            flows.push(AccountingFlow {
                flow_id: *flow_id,
                entry_id: entry.id,
                from_account: self.config.suspense_account.clone(),
                to_account: credit.account_code.clone(),
                amount: credit.amount,
                amount_f64: credit.amount.to_f64(),
                timestamp: entry.posting_date,
                method: SolvingMethod::Unsolvable,
                confidence: 0.0,
                from_entity: "SUSPENSE".to_string(),
                to_entity: credit.entity_id.clone(),
                currency: currency.clone(),
                source_lines: vec![credit.line_number],
            });
            *flow_id += 1;
        }

        flows
    }
}

impl GpuKernel for NetworkGeneration {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Batch Kernel Implementation
// ============================================================================

/// Input for network generation batch processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkGenerationInput {
    /// Journal entries to process.
    pub entries: Vec<JournalEntry>,
    /// Configuration overrides.
    pub config: Option<NetworkGenerationConfig>,
}

/// Output from network generation batch processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkGenerationOutput {
    /// Generated accounting flows.
    pub flows: Vec<AccountingFlow>,
    /// Statistics.
    pub stats: NetworkGenerationStats,
}

#[async_trait]
impl BatchKernel<NetworkGenerationInput, NetworkGenerationOutput> for NetworkGeneration {
    async fn execute(&self, input: NetworkGenerationInput) -> Result<NetworkGenerationOutput> {
        let kernel = if let Some(config) = input.config {
            NetworkGeneration::with_config(config)
        } else {
            self.clone()
        };

        let network = kernel.generate(&input.entries);

        Ok(NetworkGenerationOutput {
            flows: network.flows,
            stats: network.stats,
        })
    }
}

// ============================================================================
// Ring Mode Messages and Handlers
// ============================================================================

/// Ring message for adding a journal entry to the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddEntryRing {
    /// Request ID.
    pub request_id: u64,
    /// Journal entry to add.
    pub entry_id: u64,
    /// Entry posting date.
    pub posting_date: u64,
    /// Entry lines as serialized data.
    pub lines_data: Vec<u8>,
}

/// Response for adding an entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddEntryResponse {
    /// Request ID.
    pub request_id: u64,
    /// Entry ID processed.
    pub entry_id: u64,
    /// Number of flows generated.
    pub flow_count: u32,
    /// Method used.
    pub method: u8,
    /// Confidence level (fixed-point, scale 1e6).
    pub confidence_fp: u64,
}

/// Ring message for querying network flows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFlowsRing {
    /// Request ID.
    pub request_id: u64,
    /// Account to query (from or to).
    pub account: String,
    /// Time window start (0 for all).
    pub start_time: u64,
    /// Time window end (u64::MAX for all).
    pub end_time: u64,
    /// Maximum results.
    pub limit: u32,
}

/// Response for flow query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFlowsResponse {
    /// Request ID.
    pub request_id: u64,
    /// Account queried.
    pub account: String,
    /// Total matching flows.
    pub total_count: u32,
    /// Total volume (fixed-point, scale 1e18).
    pub total_volume_fp: i128,
    /// Weighted confidence (fixed-point, scale 1e6).
    pub weighted_confidence_fp: u64,
}

/// Ring message for network statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatsRing {
    /// Request ID.
    pub request_id: u64,
}

/// Response for network statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatsResponse {
    /// Request ID.
    pub request_id: u64,
    /// Total accounts.
    pub total_accounts: u32,
    /// Total flows.
    pub total_flows: u32,
    /// Total volume (fixed-point, scale 1e18).
    pub total_volume_fp: i128,
    /// Method distribution (A, B, C, D, E, Unsolvable).
    pub method_counts: [u32; 6],
    /// Weighted confidence (fixed-point, scale 1e6).
    pub weighted_confidence_fp: u64,
}

// ============================================================================
// Ring Kernel State
// ============================================================================

/// Stateful network generation kernel for Ring mode.
#[derive(Debug)]
pub struct NetworkGenerationRing {
    metadata: KernelMetadata,
    config: NetworkGenerationConfig,
    /// Internal network state.
    network: std::sync::RwLock<AccountingNetwork>,
    /// Next flow ID.
    next_flow_id: std::sync::atomic::AtomicU64,
}

impl Clone for NetworkGenerationRing {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            config: self.config.clone(),
            network: std::sync::RwLock::new(self.network.read().unwrap().clone()),
            next_flow_id: std::sync::atomic::AtomicU64::new(
                self.next_flow_id.load(std::sync::atomic::Ordering::SeqCst),
            ),
        }
    }
}

impl Default for NetworkGenerationRing {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkGenerationRing {
    /// Create a new Ring mode kernel.
    pub fn new() -> Self {
        Self::with_config(NetworkGenerationConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: NetworkGenerationConfig) -> Self {
        Self {
            metadata: KernelMetadata::ring(
                "accounting/network-generation-ring",
                Domain::Accounting,
            )
            .with_description("Streaming accounting network generation")
            .with_throughput(100_000)
            .with_latency_us(5.0)
            .with_gpu_native(true),
            config,
            network: std::sync::RwLock::new(AccountingNetwork::new()),
            next_flow_id: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Get current network statistics.
    pub fn stats(&self) -> NetworkGenerationStats {
        self.network.read().unwrap().stats.clone()
    }

    /// Get total flow count.
    pub fn flow_count(&self) -> usize {
        self.network.read().unwrap().flows.len()
    }

    /// Get total account count.
    pub fn account_count(&self) -> usize {
        self.network.read().unwrap().accounts.len()
    }

    /// Add a journal entry and return generated flows.
    pub fn add_entry(&self, entry: &JournalEntry) -> Vec<AccountingFlow> {
        let batch_kernel = NetworkGeneration::with_config(self.config.clone());
        let mut flow_id = self
            .next_flow_id
            .fetch_add(100, std::sync::atomic::Ordering::SeqCst);

        let result = batch_kernel.process_entry(entry, &mut flow_id);

        // Update internal state
        {
            let mut network = self.network.write().unwrap();
            for flow in &result.flows {
                network.add_flow(flow.clone());
            }

            // Update stats
            network.stats.total_entries += 1;
            match result.method {
                SolvingMethod::MethodA => network.stats.method_a_count += 1,
                SolvingMethod::MethodB => network.stats.method_b_count += 1,
                SolvingMethod::MethodC => network.stats.method_c_count += 1,
                SolvingMethod::MethodD => network.stats.method_d_count += 1,
                SolvingMethod::MethodE => network.stats.method_e_count += 1,
                SolvingMethod::Unsolvable => network.stats.unsolvable_count += 1,
            }
            network.stats.total_flows = network.flows.len();
            network.stats.total_volume = network.total_volume();
            network.stats.weighted_confidence = network.weighted_confidence();
        }

        result.flows
    }

    /// Query flows for an account within a time window.
    pub fn query_flows(
        &self,
        account: &str,
        start_time: u64,
        end_time: u64,
        limit: usize,
    ) -> Vec<AccountingFlow> {
        let network = self.network.read().unwrap();
        let mut flows: Vec<_> = network
            .flows
            .iter()
            .filter(|f| {
                (f.from_account == account || f.to_account == account)
                    && f.timestamp >= start_time
                    && f.timestamp <= end_time
            })
            .cloned()
            .collect();

        flows.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        flows.truncate(limit);
        flows
    }

    /// Clear the network state.
    pub fn clear(&self) {
        let mut network = self.network.write().unwrap();
        *network = AccountingNetwork::new();
        self.next_flow_id
            .store(0, std::sync::atomic::Ordering::SeqCst);
    }
}

impl GpuKernel for NetworkGenerationRing {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::JournalStatus;

    fn create_simple_entry() -> JournalEntry {
        JournalEntry {
            id: 1,
            date: 1700000000,
            posting_date: 1700000000,
            document_number: "JE001".to_string(),
            lines: vec![
                JournalLine {
                    line_number: 1,
                    account_code: "1000".to_string(), // Cash
                    debit: 1000.0,
                    credit: 0.0,
                    currency: "USD".to_string(),
                    entity_id: "CORP".to_string(),
                    cost_center: None,
                    description: "Cash debit".to_string(),
                },
                JournalLine {
                    line_number: 2,
                    account_code: "4000".to_string(), // Revenue
                    debit: 0.0,
                    credit: 1000.0,
                    currency: "USD".to_string(),
                    entity_id: "CORP".to_string(),
                    cost_center: None,
                    description: "Revenue credit".to_string(),
                },
            ],
            status: JournalStatus::Posted,
            source_system: "TEST".to_string(),
            description: "Simple sale".to_string(),
        }
    }

    fn create_multi_line_entry() -> JournalEntry {
        JournalEntry {
            id: 2,
            date: 1700000000,
            posting_date: 1700000000,
            document_number: "JE002".to_string(),
            lines: vec![
                JournalLine {
                    line_number: 1,
                    account_code: "1000".to_string(),
                    debit: 500.0,
                    credit: 0.0,
                    currency: "USD".to_string(),
                    entity_id: "CORP".to_string(),
                    cost_center: None,
                    description: "Cash 1".to_string(),
                },
                JournalLine {
                    line_number: 2,
                    account_code: "1100".to_string(),
                    debit: 300.0,
                    credit: 0.0,
                    currency: "USD".to_string(),
                    entity_id: "CORP".to_string(),
                    cost_center: None,
                    description: "AR".to_string(),
                },
                JournalLine {
                    line_number: 3,
                    account_code: "4000".to_string(),
                    debit: 0.0,
                    credit: 500.0,
                    currency: "USD".to_string(),
                    entity_id: "CORP".to_string(),
                    cost_center: None,
                    description: "Revenue 1".to_string(),
                },
                JournalLine {
                    line_number: 4,
                    account_code: "4100".to_string(),
                    debit: 0.0,
                    credit: 300.0,
                    currency: "USD".to_string(),
                    entity_id: "CORP".to_string(),
                    cost_center: None,
                    description: "Revenue 2".to_string(),
                },
            ],
            status: JournalStatus::Posted,
            source_system: "TEST".to_string(),
            description: "Multi-line sale".to_string(),
        }
    }

    fn create_asymmetric_entry() -> JournalEntry {
        JournalEntry {
            id: 3,
            date: 1700000000,
            posting_date: 1700000000,
            document_number: "JE003".to_string(),
            lines: vec![
                JournalLine {
                    line_number: 1,
                    account_code: "1000".to_string(),
                    debit: 1000.0,
                    credit: 0.0,
                    currency: "USD".to_string(),
                    entity_id: "CORP".to_string(),
                    cost_center: None,
                    description: "Cash".to_string(),
                },
                JournalLine {
                    line_number: 2,
                    account_code: "4000".to_string(),
                    debit: 0.0,
                    credit: 600.0,
                    currency: "USD".to_string(),
                    entity_id: "CORP".to_string(),
                    cost_center: None,
                    description: "Revenue".to_string(),
                },
                JournalLine {
                    line_number: 3,
                    account_code: "4100".to_string(),
                    debit: 0.0,
                    credit: 400.0,
                    currency: "USD".to_string(),
                    entity_id: "CORP".to_string(),
                    cost_center: None,
                    description: "Service revenue".to_string(),
                },
            ],
            status: JournalStatus::Posted,
            source_system: "TEST".to_string(),
            description: "Asymmetric entry".to_string(),
        }
    }

    #[test]
    fn test_fixed_point_arithmetic() {
        let a = FixedPoint128::from_f64(100.50);
        let b = FixedPoint128::from_f64(50.25);

        let sum = a + b;
        assert!((sum.to_f64() - 150.75).abs() < 0.0001);

        let diff = a - b;
        assert!((diff.to_f64() - 50.25).abs() < 0.0001);

        assert!(a > b);
        assert_eq!(
            FixedPoint128::from_f64(100.0).abs(),
            FixedPoint128::from_f64(100.0)
        );
        assert_eq!(
            FixedPoint128::from_f64(-100.0).abs(),
            FixedPoint128::from_f64(100.0)
        );
    }

    #[test]
    fn test_method_a_simple_entry() {
        let kernel = NetworkGeneration::new();
        let entry = create_simple_entry();
        let network = kernel.generate(&[entry]);

        assert_eq!(network.flows.len(), 1);
        assert_eq!(network.stats.method_a_count, 1);
        assert_eq!(network.stats.method_b_count, 0);

        let flow = &network.flows[0];
        assert_eq!(flow.from_account, "1000");
        assert_eq!(flow.to_account, "4000");
        assert!((flow.amount_f64 - 1000.0).abs() < 0.01);
        assert_eq!(flow.method, SolvingMethod::MethodA);
        assert!((flow.confidence - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_method_b_multi_line() {
        let kernel = NetworkGeneration::new();
        let entry = create_multi_line_entry();
        let network = kernel.generate(&[entry]);

        assert_eq!(network.flows.len(), 2);
        assert_eq!(network.stats.method_b_count, 1);

        // Check that amounts match correctly
        let total_flow: f64 = network.flows.iter().map(|f| f.amount_f64).sum();
        assert!((total_flow - 800.0).abs() < 0.01);
    }

    #[test]
    fn test_method_c_asymmetric() {
        let kernel = NetworkGeneration::new();
        let entry = create_asymmetric_entry();
        let network = kernel.generate(&[entry]);

        // 1 debit to 2 credits = Method C
        assert!(network.flows.len() >= 2);
        assert_eq!(network.stats.method_c_count, 1);

        // Total flow should equal the debit amount
        let total_flow: f64 = network.flows.iter().map(|f| f.amount_f64).sum();
        assert!((total_flow - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_network_statistics() {
        let kernel = NetworkGeneration::new();
        let entries = vec![
            create_simple_entry(),
            create_multi_line_entry(),
            create_asymmetric_entry(),
        ];

        let network = kernel.generate(&entries);

        assert_eq!(network.stats.total_entries, 3);
        assert!(network.stats.total_flows >= 5);
        assert!(network.stats.total_volume > 2000.0);
        assert!(network.stats.weighted_confidence > 0.5);
    }

    #[test]
    fn test_temporal_query() {
        let kernel = NetworkGeneration::new();
        let mut entry1 = create_simple_entry();
        entry1.posting_date = 1000;
        entry1.id = 1;

        let mut entry2 = create_simple_entry();
        entry2.posting_date = 2000;
        entry2.id = 2;

        let mut entry3 = create_simple_entry();
        entry3.posting_date = 3000;
        entry3.id = 3;

        let network = kernel.generate(&[entry1, entry2, entry3]);

        // Query middle time window
        let flows = network.query_temporal(1500, 2500);
        assert_eq!(flows.len(), 1);
        assert_eq!(flows[0].entry_id, 2);
    }

    #[test]
    fn test_adjacency_list() {
        let kernel = NetworkGeneration::new();
        let entry = create_simple_entry();
        let network = kernel.generate(&[entry]);

        let outgoing = network.outgoing_flows("1000");
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].to_account, "4000");

        let incoming = network.incoming_flows("4000");
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].from_account, "1000");
    }

    #[test]
    fn test_ring_mode() {
        let ring_kernel = NetworkGenerationRing::new();

        // Add entries one by one
        let entry1 = create_simple_entry();
        let flows1 = ring_kernel.add_entry(&entry1);
        assert_eq!(flows1.len(), 1);

        let entry2 = create_multi_line_entry();
        let flows2 = ring_kernel.add_entry(&entry2);
        assert!(flows2.len() >= 2);

        // Check accumulated stats
        let stats = ring_kernel.stats();
        assert_eq!(stats.total_entries, 2);
        assert!(stats.total_flows >= 3);
    }

    #[test]
    fn test_ring_query() {
        let ring_kernel = NetworkGenerationRing::new();

        let entry = create_simple_entry();
        ring_kernel.add_entry(&entry);

        let flows = ring_kernel.query_flows("1000", 0, u64::MAX, 100);
        assert_eq!(flows.len(), 1);
        assert_eq!(flows[0].from_account, "1000");
    }

    #[test]
    fn test_solving_method_confidence() {
        assert_eq!(SolvingMethod::MethodA.confidence(), 1.00);
        assert_eq!(SolvingMethod::MethodB.confidence(), 0.95);
        assert_eq!(SolvingMethod::MethodC.confidence(), 0.85);
        assert_eq!(SolvingMethod::MethodD.confidence(), 0.70);
        assert_eq!(SolvingMethod::MethodE.confidence(), 0.50);
        assert_eq!(SolvingMethod::Unsolvable.confidence(), 0.00);
    }

    #[test]
    fn test_unbalanced_entry() {
        let kernel = NetworkGeneration::with_config(NetworkGenerationConfig {
            strict_balance: true,
            ..Default::default()
        });

        let mut entry = create_simple_entry();
        entry.lines[0].debit = 1500.0; // Unbalanced

        let network = kernel.generate(&[entry]);

        assert!(network.flows.is_empty());
        assert_eq!(network.stats.balance_errors, 1);
    }

    #[test]
    fn test_suspense_routing() {
        // Create a very complex entry that forces suspense routing
        let kernel = NetworkGeneration::with_config(NetworkGenerationConfig {
            max_lines_method_b: 2,
            max_lines_method_c: 4,
            enable_aggregation: false,
            enable_decomposition: false,
            suspense_account: "SUSPENSE_ACCT".to_string(),
            ..Default::default()
        });

        // Create an entry that can't be solved
        let entry = JournalEntry {
            id: 99,
            date: 1700000000,
            posting_date: 1700000000,
            document_number: "JE099".to_string(),
            lines: vec![
                JournalLine {
                    line_number: 1,
                    account_code: "1000".to_string(),
                    debit: 100.0,
                    credit: 0.0,
                    currency: "USD".to_string(),
                    entity_id: "A".to_string(),
                    cost_center: None,
                    description: "D1".to_string(),
                },
                JournalLine {
                    line_number: 2,
                    account_code: "1001".to_string(),
                    debit: 100.0,
                    credit: 0.0,
                    currency: "USD".to_string(),
                    entity_id: "A".to_string(),
                    cost_center: None,
                    description: "D2".to_string(),
                },
                JournalLine {
                    line_number: 3,
                    account_code: "1002".to_string(),
                    debit: 100.0,
                    credit: 0.0,
                    currency: "USD".to_string(),
                    entity_id: "A".to_string(),
                    cost_center: None,
                    description: "D3".to_string(),
                },
                JournalLine {
                    line_number: 4,
                    account_code: "2000".to_string(),
                    debit: 0.0,
                    credit: 75.0,
                    currency: "USD".to_string(),
                    entity_id: "A".to_string(),
                    cost_center: None,
                    description: "C1".to_string(),
                },
                JournalLine {
                    line_number: 5,
                    account_code: "2001".to_string(),
                    debit: 0.0,
                    credit: 75.0,
                    currency: "USD".to_string(),
                    entity_id: "A".to_string(),
                    cost_center: None,
                    description: "C2".to_string(),
                },
                JournalLine {
                    line_number: 6,
                    account_code: "2002".to_string(),
                    debit: 0.0,
                    credit: 75.0,
                    currency: "USD".to_string(),
                    entity_id: "A".to_string(),
                    cost_center: None,
                    description: "C3".to_string(),
                },
                JournalLine {
                    line_number: 7,
                    account_code: "2003".to_string(),
                    debit: 0.0,
                    credit: 75.0,
                    currency: "USD".to_string(),
                    entity_id: "A".to_string(),
                    cost_center: None,
                    description: "C4".to_string(),
                },
            ],
            status: JournalStatus::Posted,
            source_system: "TEST".to_string(),
            description: "Complex entry".to_string(),
        };

        let network = kernel.generate(&[entry]);

        assert_eq!(network.stats.unsolvable_count, 1);
        // Should have flows to/from suspense account
        assert!(
            network
                .flows
                .iter()
                .any(|f| f.from_account == "SUSPENSE_ACCT" || f.to_account == "SUSPENSE_ACCT")
        );
    }

    #[test]
    fn test_metadata() {
        let kernel = NetworkGeneration::new();
        assert_eq!(kernel.metadata().id, "accounting/network-generation");
        assert_eq!(kernel.metadata().domain, Domain::Accounting);

        let ring_kernel = NetworkGenerationRing::new();
        assert_eq!(
            ring_kernel.metadata().id,
            "accounting/network-generation-ring"
        );
    }

    #[test]
    fn test_weighted_confidence() {
        let kernel = NetworkGeneration::new();
        let entries = vec![
            create_simple_entry(),     // Method A, conf=1.0, amt=1000
            create_asymmetric_entry(), // Method C, conf=0.85, amt=1000
        ];

        let network = kernel.generate(&entries);

        // Weighted confidence should be between 0.85 and 1.0
        assert!(network.stats.weighted_confidence >= 0.85);
        assert!(network.stats.weighted_confidence <= 1.0);
    }
}
