//! Account detection kernels for suspense accounts and GAAP violations.
//!
//! This module provides detection algorithms for:
//! - Suspense account identification using centrality analysis
//! - GAAP violation detection for prohibited transaction patterns

use crate::types::{
    AccountType, GaapViolation, GaapViolationResult, GaapViolationSeverity, GaapViolationType,
    JournalEntry, JournalLine, SuspenseAccountCandidate, SuspenseAccountResult, SuspenseIndicator,
    SuspenseRiskLevel,
};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Suspense Account Detection Kernel
// ============================================================================

/// Suspense account detection kernel.
///
/// Detects accounts that exhibit suspense account characteristics using
/// centrality-based analysis on the account transaction graph.
#[derive(Debug, Clone)]
pub struct SuspenseAccountDetection {
    metadata: KernelMetadata,
}

impl Default for SuspenseAccountDetection {
    fn default() -> Self {
        Self::new()
    }
}

impl SuspenseAccountDetection {
    /// Create a new suspense account detection kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("accounting/suspense-detection", Domain::Accounting)
                .with_description("Centrality-based suspense account detection")
                .with_throughput(20_000)
                .with_latency_us(150.0),
        }
    }

    /// Detect suspense accounts from journal entries.
    pub fn detect(
        entries: &[JournalEntry],
        config: &SuspenseDetectionConfig,
    ) -> SuspenseAccountResult {
        if entries.is_empty() {
            return SuspenseAccountResult {
                candidates: Vec::new(),
                high_risk_accounts: Vec::new(),
                accounts_analyzed: 0,
                risk_score: 0.0,
            };
        }

        // Build account graph from journal entries
        let account_graph = Self::build_account_graph(entries);

        // Calculate metrics for each account
        let mut candidates = Vec::new();
        let mut high_risk_accounts = Vec::new();

        for (account_code, metrics) in &account_graph.account_metrics {
            let indicators = Self::check_indicators(metrics, config);

            if indicators.is_empty() {
                continue;
            }

            let suspense_score = Self::calculate_suspense_score(&indicators, metrics);
            let risk_level = Self::determine_risk_level(suspense_score, &indicators);

            let candidate = SuspenseAccountCandidate {
                account_code: account_code.clone(),
                account_name: metrics.account_name.clone(),
                suspense_score,
                centrality_score: metrics.betweenness_centrality,
                turnover_volume: metrics.total_debit + metrics.total_credit,
                avg_holding_period: metrics.avg_holding_days,
                counterparty_count: metrics.counterparty_count,
                balance_ratio: metrics.balance_ratio,
                risk_level,
                indicators: indicators.clone(),
            };

            if matches!(risk_level, SuspenseRiskLevel::High | SuspenseRiskLevel::Critical) {
                high_risk_accounts.push(account_code.clone());
            }

            candidates.push(candidate);
        }

        // Sort by suspense score descending
        candidates.sort_by(|a, b| {
            b.suspense_score
                .partial_cmp(&a.suspense_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let risk_score = if candidates.is_empty() {
            0.0
        } else {
            (high_risk_accounts.len() as f64 / candidates.len().max(1) as f64 * 50.0
                + candidates
                    .iter()
                    .map(|c| c.suspense_score)
                    .sum::<f64>()
                    / candidates.len().max(1) as f64)
                .min(100.0)
        };

        SuspenseAccountResult {
            candidates,
            high_risk_accounts,
            accounts_analyzed: account_graph.account_metrics.len(),
            risk_score,
        }
    }

    /// Build account graph from journal entries.
    fn build_account_graph(entries: &[JournalEntry]) -> AccountGraph {
        let mut graph = AccountGraph::new();

        for entry in entries {
            // Extract debit and credit accounts
            let debits: Vec<_> = entry
                .lines
                .iter()
                .filter(|l| l.debit > 0.0)
                .collect();
            let credits: Vec<_> = entry
                .lines
                .iter()
                .filter(|l| l.credit > 0.0)
                .collect();

            // Create edges between debit and credit accounts
            for debit_line in &debits {
                for credit_line in &credits {
                    let amount = debit_line.debit.min(credit_line.credit);
                    graph.add_edge(
                        &credit_line.account_code,
                        &debit_line.account_code,
                        amount,
                        entry.posting_date,
                    );
                }
            }

            // Update account metrics
            for line in &entry.lines {
                graph.update_account_metrics(line, entry.posting_date);
            }
        }

        // Calculate centrality
        graph.calculate_betweenness_centrality();

        graph
    }

    /// Check which suspense indicators apply to an account.
    fn check_indicators(
        metrics: &AccountMetrics,
        config: &SuspenseDetectionConfig,
    ) -> Vec<SuspenseIndicator> {
        let mut indicators = Vec::new();

        // Check centrality
        if metrics.betweenness_centrality >= config.centrality_threshold {
            indicators.push(SuspenseIndicator::HighCentrality);
        }

        // Check turnover ratio
        let turnover = metrics.total_debit + metrics.total_credit;
        let avg_balance = (metrics.total_debit - metrics.total_credit).abs() / 2.0;
        if avg_balance > 0.0 && turnover / avg_balance >= config.turnover_ratio_threshold {
            indicators.push(SuspenseIndicator::HighTurnover);
        }

        // Check holding period
        if metrics.avg_holding_days <= config.holding_period_threshold {
            indicators.push(SuspenseIndicator::ShortHoldingPeriod);
        }

        // Check balance ratio (debit/credit balance)
        if metrics.balance_ratio >= config.balance_ratio_threshold {
            indicators.push(SuspenseIndicator::BalancedFlows);
        }

        // Check counterparty count
        if metrics.counterparty_count >= config.counterparty_threshold {
            indicators.push(SuspenseIndicator::ManyCounterparties);
        }

        // Check for zero end balance
        if metrics.end_balance.abs() < config.zero_balance_threshold {
            indicators.push(SuspenseIndicator::ZeroEndBalance);
        }

        // Check naming
        let name_lower = metrics.account_name.to_lowercase();
        if name_lower.contains("suspense")
            || name_lower.contains("clearing")
            || name_lower.contains("holding")
            || name_lower.contains("temporary")
            || name_lower.contains("wash")
        {
            indicators.push(SuspenseIndicator::SuspenseNaming);
        }

        indicators
    }

    /// Calculate suspense score from indicators.
    fn calculate_suspense_score(
        indicators: &[SuspenseIndicator],
        metrics: &AccountMetrics,
    ) -> f64 {
        let mut score = 0.0;

        for indicator in indicators {
            score += match indicator {
                SuspenseIndicator::HighCentrality => 20.0,
                SuspenseIndicator::HighTurnover => 15.0,
                SuspenseIndicator::ShortHoldingPeriod => 15.0,
                SuspenseIndicator::BalancedFlows => 15.0,
                SuspenseIndicator::ManyCounterparties => 10.0,
                SuspenseIndicator::ZeroEndBalance => 15.0,
                SuspenseIndicator::SuspenseNaming => 10.0,
            };
        }

        // Bonus for high centrality combined with other factors
        if indicators.contains(&SuspenseIndicator::HighCentrality) && indicators.len() >= 3 {
            score += metrics.betweenness_centrality * 10.0;
        }

        score.min(100.0)
    }

    /// Determine risk level from score and indicators.
    fn determine_risk_level(score: f64, indicators: &[SuspenseIndicator]) -> SuspenseRiskLevel {
        let has_critical_indicators = indicators.contains(&SuspenseIndicator::HighCentrality)
            && indicators.contains(&SuspenseIndicator::BalancedFlows)
            && indicators.contains(&SuspenseIndicator::ZeroEndBalance);

        if has_critical_indicators || score >= 80.0 {
            SuspenseRiskLevel::Critical
        } else if score >= 60.0 {
            SuspenseRiskLevel::High
        } else if score >= 40.0 {
            SuspenseRiskLevel::Medium
        } else {
            SuspenseRiskLevel::Low
        }
    }
}

impl GpuKernel for SuspenseAccountDetection {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Configuration for suspense account detection.
#[derive(Debug, Clone)]
pub struct SuspenseDetectionConfig {
    /// Minimum betweenness centrality to flag.
    pub centrality_threshold: f64,
    /// Minimum turnover ratio (turnover/balance).
    pub turnover_ratio_threshold: f64,
    /// Maximum average holding period (days).
    pub holding_period_threshold: f64,
    /// Minimum balance ratio to consider balanced (0-1).
    pub balance_ratio_threshold: f64,
    /// Minimum counterparty count to flag.
    pub counterparty_threshold: usize,
    /// Maximum balance to consider "zero".
    pub zero_balance_threshold: f64,
}

impl Default for SuspenseDetectionConfig {
    fn default() -> Self {
        Self {
            centrality_threshold: 0.1,
            turnover_ratio_threshold: 10.0,
            holding_period_threshold: 7.0,
            balance_ratio_threshold: 0.9,
            counterparty_threshold: 5,
            zero_balance_threshold: 100.0,
        }
    }
}

// ============================================================================
// GAAP Violation Detection Kernel
// ============================================================================

/// GAAP violation detection kernel.
///
/// Detects prohibited transaction patterns that violate GAAP principles.
#[derive(Debug, Clone)]
pub struct GaapViolationDetection {
    metadata: KernelMetadata,
}

impl Default for GaapViolationDetection {
    fn default() -> Self {
        Self::new()
    }
}

impl GaapViolationDetection {
    /// Create a new GAAP violation detection kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("accounting/gaap-violation", Domain::Accounting)
                .with_description("GAAP prohibited flow pattern detection")
                .with_throughput(15_000)
                .with_latency_us(200.0),
        }
    }

    /// Detect GAAP violations from journal entries.
    pub fn detect(
        entries: &[JournalEntry],
        account_types: &HashMap<String, AccountType>,
        config: &GaapDetectionConfig,
    ) -> GaapViolationResult {
        if entries.is_empty() {
            return GaapViolationResult {
                violations: Vec::new(),
                entries_analyzed: 0,
                amount_at_risk: 0.0,
                compliance_score: 100.0,
                violation_counts: HashMap::new(),
            };
        }

        let mut violations = Vec::new();
        let mut violation_id = 1;

        // Check each entry for violations
        for entry in entries {
            // Check direct revenue-to-expense transfers
            let rev_exp = Self::check_revenue_expense_transfer(entry, account_types);
            if let Some(mut v) = rev_exp {
                v.id = format!("GAAP{:05}", violation_id);
                violation_id += 1;
                violations.push(v);
            }

            // Check improper asset-to-expense
            let asset_exp = Self::check_improper_asset_expense(entry, account_types);
            if let Some(mut v) = asset_exp {
                v.id = format!("GAAP{:05}", violation_id);
                violation_id += 1;
                violations.push(v);
            }

            // Check suspense account misuse
            let suspense_misuse = Self::check_suspense_misuse(entry, config);
            if let Some(mut v) = suspense_misuse {
                v.id = format!("GAAP{:05}", violation_id);
                violation_id += 1;
                violations.push(v);
            }
        }

        // Check for circular flows across entries
        let circular = Self::check_circular_flows(entries, account_types);
        for mut v in circular {
            v.id = format!("GAAP{:05}", violation_id);
            violation_id += 1;
            violations.push(v);
        }

        // Calculate metrics
        let amount_at_risk: f64 = violations.iter().map(|v| v.amount).sum();
        let entries_analyzed = entries.len();

        // Calculate violation counts by type
        let mut violation_counts: HashMap<String, usize> = HashMap::new();
        for v in &violations {
            let type_name = format!("{:?}", v.violation_type);
            *violation_counts.entry(type_name).or_insert(0) += 1;
        }

        // Calculate compliance score
        let major_violations = violations
            .iter()
            .filter(|v| {
                matches!(
                    v.severity,
                    GaapViolationSeverity::Major | GaapViolationSeverity::Critical
                )
            })
            .count();

        let compliance_score = (100.0
            - (violations.len() as f64 * 2.0)
            - (major_violations as f64 * 10.0))
            .max(0.0);

        GaapViolationResult {
            violations,
            entries_analyzed,
            amount_at_risk,
            compliance_score,
            violation_counts,
        }
    }

    /// Check for direct revenue-to-expense transfers.
    fn check_revenue_expense_transfer(
        entry: &JournalEntry,
        account_types: &HashMap<String, AccountType>,
    ) -> Option<GaapViolation> {
        let debits: Vec<_> = entry.lines.iter().filter(|l| l.debit > 0.0).collect();
        let credits: Vec<_> = entry.lines.iter().filter(|l| l.credit > 0.0).collect();

        for debit in &debits {
            for credit in &credits {
                let debit_type = account_types.get(&debit.account_code);
                let credit_type = account_types.get(&credit.account_code);

                // Revenue credited directly to expense
                if matches!(credit_type, Some(AccountType::Revenue))
                    && matches!(debit_type, Some(AccountType::Expense))
                {
                    return Some(GaapViolation {
                        id: String::new(),
                        violation_type: GaapViolationType::DirectRevenueExpense,
                        accounts: vec![credit.account_code.clone(), debit.account_code.clone()],
                        entry_ids: vec![entry.id],
                        amount: debit.debit.min(credit.credit),
                        description: format!(
                            "Direct transfer from revenue ({}) to expense ({}) without capital account",
                            credit.account_code, debit.account_code
                        ),
                        severity: GaapViolationSeverity::Major,
                        remediation: "Route through retained earnings or appropriate capital account"
                            .to_string(),
                    });
                }
            }
        }

        None
    }

    /// Check for improper asset-to-expense (without depreciation).
    fn check_improper_asset_expense(
        entry: &JournalEntry,
        account_types: &HashMap<String, AccountType>,
    ) -> Option<GaapViolation> {
        let debits: Vec<_> = entry.lines.iter().filter(|l| l.debit > 0.0).collect();
        let credits: Vec<_> = entry.lines.iter().filter(|l| l.credit > 0.0).collect();

        for debit in &debits {
            for credit in &credits {
                let debit_type = account_types.get(&debit.account_code);
                let credit_type = account_types.get(&credit.account_code);

                // Asset credited directly to expense without depreciation
                if matches!(credit_type, Some(AccountType::Asset))
                    && matches!(debit_type, Some(AccountType::Expense))
                {
                    // Check if it's a large amount (potential capitalization issue)
                    let amount = debit.debit.min(credit.credit);
                    if amount > 5000.0 {
                        return Some(GaapViolation {
                            id: String::new(),
                            violation_type: GaapViolationType::ImproperAssetExpense,
                            accounts: vec![credit.account_code.clone(), debit.account_code.clone()],
                            entry_ids: vec![entry.id],
                            amount,
                            description: format!(
                                "Large asset ({}) expensed directly to {} without depreciation",
                                credit.account_code, debit.account_code
                            ),
                            severity: GaapViolationSeverity::Moderate,
                            remediation:
                                "Use depreciation schedule for asset disposal or verify expensing is appropriate"
                                    .to_string(),
                        });
                    }
                }
            }
        }

        None
    }

    /// Check for suspense account misuse.
    fn check_suspense_misuse(entry: &JournalEntry, config: &GaapDetectionConfig) -> Option<GaapViolation> {
        let suspense_keywords = ["suspense", "clearing", "holding", "temporary"];

        for line in &entry.lines {
            let account_lower = line.account_code.to_lowercase();
            let is_suspense = suspense_keywords.iter().any(|kw| account_lower.contains(kw));

            if is_suspense {
                let amount = line.debit.max(line.credit);
                if amount > config.suspense_amount_threshold {
                    return Some(GaapViolation {
                        id: String::new(),
                        violation_type: GaapViolationType::SuspenseAccountMisuse,
                        accounts: vec![line.account_code.clone()],
                        entry_ids: vec![entry.id],
                        amount,
                        description: format!(
                            "Large amount ({:.2}) posted to suspense account {}",
                            amount, line.account_code
                        ),
                        severity: GaapViolationSeverity::Minor,
                        remediation: "Clear suspense account to proper account within reporting period"
                            .to_string(),
                    });
                }
            }
        }

        None
    }

    /// Check for circular flows that may inflate revenue.
    fn check_circular_flows(
        entries: &[JournalEntry],
        account_types: &HashMap<String, AccountType>,
    ) -> Vec<GaapViolation> {
        let mut violations = Vec::new();

        // Build flow graph
        let mut flows: HashMap<(String, String), (f64, Vec<u64>)> = HashMap::new();

        for entry in entries {
            let debits: Vec<_> = entry.lines.iter().filter(|l| l.debit > 0.0).collect();
            let credits: Vec<_> = entry.lines.iter().filter(|l| l.credit > 0.0).collect();

            for debit in &debits {
                for credit in &credits {
                    let key = (credit.account_code.clone(), debit.account_code.clone());
                    let amount = debit.debit.min(credit.credit);

                    let entry_data = flows.entry(key).or_insert((0.0, Vec::new()));
                    entry_data.0 += amount;
                    entry_data.1.push(entry.id);
                }
            }
        }

        // Find circular patterns (A -> B and B -> A)
        let mut checked: HashSet<(String, String)> = HashSet::new();

        for ((from, to), (amount, entry_ids)) in &flows {
            if checked.contains(&(from.clone(), to.clone()))
                || checked.contains(&(to.clone(), from.clone()))
            {
                continue;
            }

            if let Some((reverse_amount, reverse_ids)) = flows.get(&(to.clone(), from.clone())) {
                // Check if one is revenue and this creates inflation
                let from_type = account_types.get(from);
                let to_type = account_types.get(to);

                let involves_revenue = matches!(from_type, Some(AccountType::Revenue))
                    || matches!(to_type, Some(AccountType::Revenue));

                if involves_revenue && *amount > 1000.0 && *reverse_amount > 1000.0 {
                    let min_amount = amount.min(*reverse_amount);
                    let mut all_entries = entry_ids.clone();
                    all_entries.extend(reverse_ids.iter());

                    violations.push(GaapViolation {
                        id: String::new(),
                        violation_type: GaapViolationType::RevenueInflation,
                        accounts: vec![from.clone(), to.clone()],
                        entry_ids: all_entries,
                        amount: min_amount,
                        description: format!(
                            "Circular flow detected between {} and {} involving revenue accounts",
                            from, to
                        ),
                        severity: GaapViolationSeverity::Critical,
                        remediation: "Review entries for potential revenue inflation or wash transactions"
                            .to_string(),
                    });
                }

                checked.insert((from.clone(), to.clone()));
            }
        }

        violations
    }
}

impl GpuKernel for GaapViolationDetection {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Configuration for GAAP violation detection.
#[derive(Debug, Clone)]
pub struct GaapDetectionConfig {
    /// Threshold for suspense account amounts.
    pub suspense_amount_threshold: f64,
    /// Minimum amount for asset-to-expense flag.
    pub asset_expense_threshold: f64,
    /// Minimum circular flow amount.
    pub circular_flow_threshold: f64,
}

impl Default for GaapDetectionConfig {
    fn default() -> Self {
        Self {
            suspense_amount_threshold: 10_000.0,
            asset_expense_threshold: 5_000.0,
            circular_flow_threshold: 1_000.0,
        }
    }
}

// ============================================================================
// Internal Types
// ============================================================================

/// Account graph for suspense detection.
struct AccountGraph {
    /// Edges: (from, to) -> total amount
    edges: HashMap<(String, String), f64>,
    /// Account metrics.
    account_metrics: HashMap<String, AccountMetrics>,
}

impl AccountGraph {
    fn new() -> Self {
        Self {
            edges: HashMap::new(),
            account_metrics: HashMap::new(),
        }
    }

    fn add_edge(&mut self, from: &str, to: &str, amount: f64, _timestamp: u64) {
        *self
            .edges
            .entry((from.to_string(), to.to_string()))
            .or_insert(0.0) += amount;

        // Update counterparty counts
        self.account_metrics
            .entry(from.to_string())
            .or_default()
            .outgoing_counterparties
            .insert(to.to_string());
        self.account_metrics
            .entry(to.to_string())
            .or_default()
            .incoming_counterparties
            .insert(from.to_string());
    }

    fn update_account_metrics(&mut self, line: &JournalLine, timestamp: u64) {
        let metrics = self
            .account_metrics
            .entry(line.account_code.clone())
            .or_default();

        metrics.account_name = line.description.clone();
        metrics.total_debit += line.debit;
        metrics.total_credit += line.credit;
        metrics.transaction_count += 1;
        metrics.last_activity = metrics.last_activity.max(timestamp);
        if metrics.first_activity == 0 {
            metrics.first_activity = timestamp;
        } else {
            metrics.first_activity = metrics.first_activity.min(timestamp);
        }
    }

    fn calculate_betweenness_centrality(&mut self) {
        // Simplified betweenness: ratio of paths going through account
        let total_paths = self.edges.len() as f64;

        if total_paths == 0.0 {
            return;
        }

        for (account_code, metrics) in &mut self.account_metrics {
            // Count paths through this account
            let paths_through: f64 = self
                .edges
                .keys()
                .filter(|(from, to)| from == account_code || to == account_code)
                .count() as f64;

            metrics.betweenness_centrality = paths_through / total_paths;
        }

        // Finalize other metrics
        for metrics in self.account_metrics.values_mut() {
            metrics.counterparty_count = metrics
                .incoming_counterparties
                .len()
                .max(metrics.outgoing_counterparties.len());

            // Calculate balance ratio
            let total = metrics.total_debit + metrics.total_credit;
            if total > 0.0 {
                let diff = (metrics.total_debit - metrics.total_credit).abs();
                metrics.balance_ratio = 1.0 - (diff / total);
            }

            // Calculate average holding period
            if metrics.first_activity > 0 && metrics.last_activity > metrics.first_activity {
                let days =
                    (metrics.last_activity - metrics.first_activity) as f64 / 86400.0;
                metrics.avg_holding_days = days / metrics.transaction_count.max(1) as f64;
            }

            // Calculate end balance
            metrics.end_balance = metrics.total_debit - metrics.total_credit;
        }
    }
}

/// Metrics for an account.
#[derive(Debug, Clone, Default)]
struct AccountMetrics {
    account_name: String,
    total_debit: f64,
    total_credit: f64,
    end_balance: f64,
    transaction_count: usize,
    betweenness_centrality: f64,
    balance_ratio: f64,
    avg_holding_days: f64,
    counterparty_count: usize,
    first_activity: u64,
    last_activity: u64,
    incoming_counterparties: HashSet<String>,
    outgoing_counterparties: HashSet<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::JournalStatus;

    fn create_test_entry(
        id: u64,
        debit_account: &str,
        credit_account: &str,
        amount: f64,
    ) -> JournalEntry {
        JournalEntry {
            id,
            date: 1700000000,
            posting_date: 1700000000,
            document_number: format!("DOC{}", id),
            lines: vec![
                JournalLine {
                    line_number: 1,
                    account_code: debit_account.to_string(),
                    debit: amount,
                    credit: 0.0,
                    currency: "USD".to_string(),
                    entity_id: "CORP".to_string(),
                    cost_center: None,
                    description: debit_account.to_string(),
                },
                JournalLine {
                    line_number: 2,
                    account_code: credit_account.to_string(),
                    debit: 0.0,
                    credit: amount,
                    currency: "USD".to_string(),
                    entity_id: "CORP".to_string(),
                    cost_center: None,
                    description: credit_account.to_string(),
                },
            ],
            status: JournalStatus::Posted,
            source_system: "TEST".to_string(),
            description: "Test entry".to_string(),
        }
    }

    #[test]
    fn test_suspense_detection_metadata() {
        let kernel = SuspenseAccountDetection::new();
        assert_eq!(kernel.metadata().id, "accounting/suspense-detection");
        assert_eq!(kernel.metadata().domain, Domain::Accounting);
    }

    #[test]
    fn test_gaap_violation_metadata() {
        let kernel = GaapViolationDetection::new();
        assert_eq!(kernel.metadata().id, "accounting/gaap-violation");
        assert_eq!(kernel.metadata().domain, Domain::Accounting);
    }

    #[test]
    fn test_suspense_detection_empty() {
        let entries: Vec<JournalEntry> = vec![];
        let config = SuspenseDetectionConfig::default();
        let result = SuspenseAccountDetection::detect(&entries, &config);

        assert!(result.candidates.is_empty());
        assert_eq!(result.accounts_analyzed, 0);
    }

    #[test]
    fn test_suspense_detection_naming() {
        let entries = vec![
            create_test_entry(1, "EXPENSE", "SUSPENSE_CLEARING", 5000.0),
            create_test_entry(2, "CASH", "SUSPENSE_CLEARING", 3000.0),
            create_test_entry(3, "SUSPENSE_CLEARING", "PAYABLES", 4000.0),
            create_test_entry(4, "SUSPENSE_CLEARING", "RECEIVABLES", 4000.0),
        ];

        let config = SuspenseDetectionConfig::default();
        let result = SuspenseAccountDetection::detect(&entries, &config);

        // Should detect suspense_clearing as candidate due to naming
        let suspense_candidate = result
            .candidates
            .iter()
            .find(|c| c.account_code == "SUSPENSE_CLEARING");
        assert!(suspense_candidate.is_some());

        let candidate = suspense_candidate.unwrap();
        assert!(candidate.indicators.contains(&SuspenseIndicator::SuspenseNaming));
    }

    #[test]
    fn test_gaap_violation_empty() {
        let entries: Vec<JournalEntry> = vec![];
        let account_types = HashMap::new();
        let config = GaapDetectionConfig::default();
        let result = GaapViolationDetection::detect(&entries, &account_types, &config);

        assert!(result.violations.is_empty());
        assert_eq!(result.compliance_score, 100.0);
    }

    #[test]
    fn test_gaap_direct_revenue_expense() {
        let entries = vec![create_test_entry(1, "SALARIES_EXPENSE", "SALES_REVENUE", 10000.0)];

        let mut account_types = HashMap::new();
        account_types.insert("SALES_REVENUE".to_string(), AccountType::Revenue);
        account_types.insert("SALARIES_EXPENSE".to_string(), AccountType::Expense);

        let config = GaapDetectionConfig::default();
        let result = GaapViolationDetection::detect(&entries, &account_types, &config);

        assert!(!result.violations.is_empty());

        let rev_exp_violation = result
            .violations
            .iter()
            .find(|v| v.violation_type == GaapViolationType::DirectRevenueExpense);
        assert!(rev_exp_violation.is_some());
    }

    #[test]
    fn test_gaap_suspense_misuse() {
        let entries = vec![create_test_entry(1, "EXPENSE", "SUSPENSE_ACCOUNT", 50000.0)];

        let account_types = HashMap::new();
        let config = GaapDetectionConfig {
            suspense_amount_threshold: 10_000.0,
            ..Default::default()
        };

        let result = GaapViolationDetection::detect(&entries, &account_types, &config);

        let suspense_violation = result
            .violations
            .iter()
            .find(|v| v.violation_type == GaapViolationType::SuspenseAccountMisuse);
        assert!(suspense_violation.is_some());
    }

    #[test]
    fn test_gaap_circular_flow() {
        // A -> B and B -> A with revenue
        let entries = vec![
            create_test_entry(1, "ACCOUNT_B", "SALES_REVENUE", 5000.0),
            create_test_entry(2, "SALES_REVENUE", "ACCOUNT_B", 5000.0),
        ];

        let mut account_types = HashMap::new();
        account_types.insert("SALES_REVENUE".to_string(), AccountType::Revenue);
        account_types.insert("ACCOUNT_B".to_string(), AccountType::Asset);

        let config = GaapDetectionConfig::default();
        let result = GaapViolationDetection::detect(&entries, &account_types, &config);

        let circular_violation = result
            .violations
            .iter()
            .find(|v| v.violation_type == GaapViolationType::RevenueInflation);
        assert!(circular_violation.is_some());
    }

    #[test]
    fn test_gaap_improper_asset_expense() {
        let entries = vec![create_test_entry(1, "OFFICE_EXPENSE", "EQUIPMENT_ASSET", 15000.0)];

        let mut account_types = HashMap::new();
        account_types.insert("EQUIPMENT_ASSET".to_string(), AccountType::Asset);
        account_types.insert("OFFICE_EXPENSE".to_string(), AccountType::Expense);

        let config = GaapDetectionConfig::default();
        let result = GaapViolationDetection::detect(&entries, &account_types, &config);

        let asset_exp_violation = result
            .violations
            .iter()
            .find(|v| v.violation_type == GaapViolationType::ImproperAssetExpense);
        assert!(asset_exp_violation.is_some());
    }

    #[test]
    fn test_compliance_score() {
        // Entry with no violations
        let entries = vec![create_test_entry(1, "CASH", "RECEIVABLES", 1000.0)];

        let account_types = HashMap::new();
        let config = GaapDetectionConfig::default();
        let result = GaapViolationDetection::detect(&entries, &account_types, &config);

        assert!(result.compliance_score >= 90.0);
    }
}
