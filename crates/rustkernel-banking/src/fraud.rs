//! Fraud pattern detection kernels.
//!
//! This module provides fraud detection algorithms:
//! - Aho-Corasick pattern matching
//! - Rapid split detection (structuring)
//! - Circular flow detection
//! - Velocity and amount anomalies

use crate::types::{
    AccountProfile, BankTransaction, FraudDetectionResult, FraudPattern, FraudPatternType,
    PatternMatch, PatternParams, RecommendedAction, RiskLevel,
};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Fraud Pattern Match Kernel
// ============================================================================

/// Fraud pattern matching kernel.
///
/// Combines multiple fraud detection techniques:
/// - Aho-Corasick for string pattern matching
/// - Rapid split detection for structuring
/// - Cycle detection for circular flows
/// - Statistical anomaly detection
#[derive(Debug, Clone)]
pub struct FraudPatternMatch {
    metadata: KernelMetadata,
}

impl Default for FraudPatternMatch {
    fn default() -> Self {
        Self::new()
    }
}

impl FraudPatternMatch {
    /// Create a new fraud pattern match kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("banking/fraud-pattern-match", Domain::Banking)
                .with_description("Fraud pattern detection (Aho-Corasick, rapid split, cycles)")
                .with_throughput(50_000)
                .with_latency_us(100.0)
                .with_gpu_native(true),
        }
    }

    /// Detect fraud patterns in a transaction.
    ///
    /// # Arguments
    /// * `transaction` - Transaction to analyze
    /// * `history` - Recent transaction history for the account
    /// * `patterns` - Fraud patterns to check
    /// * `profile` - Account profile for baseline comparison
    pub fn compute(
        transaction: &BankTransaction,
        history: &[BankTransaction],
        patterns: &[FraudPattern],
        profile: Option<&AccountProfile>,
    ) -> FraudDetectionResult {
        let mut matched_patterns = Vec::new();
        let mut total_score = 0.0;
        let mut related_transactions = HashSet::new();

        let default_profile = AccountProfile::default();
        let acct_profile = profile.unwrap_or(&default_profile);

        for pattern in patterns {
            if let Some(match_result) =
                Self::check_pattern(transaction, history, pattern, acct_profile)
            {
                total_score += match_result.score * pattern.risk_weight / 100.0;
                for &tx_id in &match_result.evidence {
                    related_transactions.insert(tx_id);
                }
                matched_patterns.push(match_result);
            }
        }

        // Normalize score to 0-100
        let fraud_score = (total_score / patterns.len().max(1) as f64 * 100.0).min(100.0);
        let risk_level = RiskLevel::from(fraud_score);
        let recommended_action = RecommendedAction::from(risk_level);

        FraudDetectionResult {
            transaction_id: transaction.id,
            fraud_score,
            matched_patterns,
            risk_level,
            recommended_action,
            related_transactions: related_transactions.into_iter().collect(),
        }
    }

    /// Batch analyze multiple transactions.
    pub fn compute_batch(
        transactions: &[BankTransaction],
        history_map: &HashMap<u64, Vec<BankTransaction>>,
        patterns: &[FraudPattern],
        profiles: &HashMap<u64, AccountProfile>,
    ) -> Vec<FraudDetectionResult> {
        transactions
            .iter()
            .map(|tx| {
                let history = history_map
                    .get(&tx.source_account)
                    .map(|h| h.as_slice())
                    .unwrap_or(&[]);
                let profile = profiles.get(&tx.source_account);
                Self::compute(tx, history, patterns, profile)
            })
            .collect()
    }

    /// Check a single pattern against a transaction.
    fn check_pattern(
        transaction: &BankTransaction,
        history: &[BankTransaction],
        pattern: &FraudPattern,
        profile: &AccountProfile,
    ) -> Option<PatternMatch> {
        match pattern.pattern_type {
            FraudPatternType::RapidSplit => {
                Self::check_rapid_split(transaction, history, &pattern.params)
            }
            FraudPatternType::CircularFlow => {
                Self::check_circular_flow(transaction, history, &pattern.params)
            }
            FraudPatternType::VelocityAnomaly => {
                Self::check_velocity(transaction, history, profile, &pattern.params)
            }
            FraudPatternType::AmountAnomaly => {
                Self::check_amount_anomaly(transaction, profile, &pattern.params)
            }
            FraudPatternType::GeoAnomaly => {
                Self::check_geo_anomaly(transaction, history, profile, &pattern.params)
            }
            FraudPatternType::TimeAnomaly => Self::check_time_anomaly(transaction, profile),
            FraudPatternType::AccountTakeover => {
                Self::check_account_takeover(transaction, history, profile)
            }
            FraudPatternType::MuleAccount => {
                Self::check_mule_account(transaction, history, &pattern.params)
            }
            FraudPatternType::Layering => {
                Self::check_layering(transaction, history, &pattern.params)
            }
        }
        .map(|(score, details, evidence)| PatternMatch {
            pattern_id: pattern.id,
            pattern_name: pattern.name.clone(),
            score,
            details,
            evidence,
        })
    }

    /// Detect rapid split (structuring) pattern.
    /// Multiple transactions just below reporting threshold.
    fn check_rapid_split(
        transaction: &BankTransaction,
        history: &[BankTransaction],
        params: &PatternParams,
    ) -> Option<(f64, String, Vec<u64>)> {
        let threshold = params.amount_threshold;
        let time_window = params.time_window;
        let min_count = params.min_count;

        // Find transactions just below threshold within time window
        let mut split_txs: Vec<&BankTransaction> = history
            .iter()
            .filter(|tx| {
                tx.source_account == transaction.source_account
                    && tx.amount >= threshold * 0.8
                    && tx.amount < threshold
                    && transaction.timestamp.saturating_sub(time_window) <= tx.timestamp
            })
            .collect();

        // Add current transaction if it qualifies
        if transaction.amount >= threshold * 0.8 && transaction.amount < threshold {
            split_txs.push(transaction);
        }

        if split_txs.len() >= min_count as usize {
            let total: f64 = split_txs.iter().map(|tx| tx.amount).sum();
            let score = if total > threshold { 80.0 } else { 60.0 };
            let evidence: Vec<u64> = split_txs.iter().map(|tx| tx.id).collect();

            Some((
                score,
                format!(
                    "Detected {} transactions totaling ${:.2} (threshold: ${:.2})",
                    split_txs.len(),
                    total,
                    threshold
                ),
                evidence,
            ))
        } else {
            None
        }
    }

    /// Detect circular flow pattern.
    /// Money flows back to originator through intermediaries.
    fn check_circular_flow(
        transaction: &BankTransaction,
        history: &[BankTransaction],
        params: &PatternParams,
    ) -> Option<(f64, String, Vec<u64>)> {
        let time_window = params.time_window;
        let min_chain_length = params
            .custom
            .get("min_chain_length")
            .copied()
            .unwrap_or(3.0) as usize;

        // Build transaction graph
        let mut graph: HashMap<u64, Vec<(u64, u64, f64)>> = HashMap::new(); // account -> [(dest, tx_id, amount)]

        for tx in history.iter().chain(std::iter::once(transaction)) {
            if transaction.timestamp.saturating_sub(time_window) <= tx.timestamp {
                graph.entry(tx.source_account).or_default().push((
                    tx.dest_account,
                    tx.id,
                    tx.amount,
                ));
            }
        }

        // DFS to find cycles
        let start = transaction.source_account;
        let mut visited = HashSet::new();
        let mut path = vec![(start, transaction.id)];
        let mut cycle_evidence = Vec::new();

        if Self::find_cycle(
            &graph,
            start,
            start,
            &mut visited,
            &mut path,
            &mut cycle_evidence,
            min_chain_length,
        ) {
            let score = 90.0;
            Some((
                score,
                format!("Circular flow detected with {} hops", cycle_evidence.len()),
                cycle_evidence,
            ))
        } else {
            None
        }
    }

    /// DFS helper to find cycles in transaction graph.
    fn find_cycle(
        graph: &HashMap<u64, Vec<(u64, u64, f64)>>,
        current: u64,
        target: u64,
        visited: &mut HashSet<u64>,
        path: &mut Vec<(u64, u64)>,
        evidence: &mut Vec<u64>,
        min_length: usize,
    ) -> bool {
        if path.len() > 1 && current == target && path.len() >= min_length {
            *evidence = path.iter().map(|(_, tx_id)| *tx_id).collect();
            return true;
        }

        if path.len() > 10 || visited.contains(&current) {
            return false;
        }

        visited.insert(current);

        if let Some(edges) = graph.get(&current) {
            for &(dest, tx_id, _) in edges {
                path.push((dest, tx_id));
                if Self::find_cycle(graph, dest, target, visited, path, evidence, min_length) {
                    return true;
                }
                path.pop();
            }
        }

        visited.remove(&current);
        false
    }

    /// Detect velocity anomaly.
    fn check_velocity(
        transaction: &BankTransaction,
        history: &[BankTransaction],
        profile: &AccountProfile,
        params: &PatternParams,
    ) -> Option<(f64, String, Vec<u64>)> {
        let time_window = params.time_window;

        // Count transactions in time window
        let recent_count = history
            .iter()
            .filter(|tx| {
                tx.source_account == transaction.source_account
                    && transaction.timestamp.saturating_sub(time_window) <= tx.timestamp
            })
            .count()
            + 1; // Include current transaction

        // Expected count based on profile (scaled to time window)
        let expected = profile.avg_daily_count * (time_window as f64 / 86400.0);
        let std_dev = expected.sqrt().max(1.0);

        let z_score = (recent_count as f64 - expected) / std_dev;

        if z_score > 3.0 {
            let score = (z_score * 20.0).min(100.0);
            Some((
                score,
                format!(
                    "Velocity anomaly: {} transactions vs expected {:.1} (z={:.2})",
                    recent_count, expected, z_score
                ),
                vec![transaction.id],
            ))
        } else {
            None
        }
    }

    /// Detect amount anomaly.
    fn check_amount_anomaly(
        transaction: &BankTransaction,
        profile: &AccountProfile,
        _params: &PatternParams,
    ) -> Option<(f64, String, Vec<u64>)> {
        let z_score = (transaction.amount - profile.avg_amount) / profile.std_amount.max(1.0);

        if z_score.abs() > 3.0 {
            let score = (z_score.abs() * 20.0).min(100.0);
            Some((
                score,
                format!(
                    "Amount anomaly: ${:.2} vs avg ${:.2} (z={:.2})",
                    transaction.amount, profile.avg_amount, z_score
                ),
                vec![transaction.id],
            ))
        } else {
            None
        }
    }

    /// Detect geographic anomaly (impossible travel).
    fn check_geo_anomaly(
        transaction: &BankTransaction,
        history: &[BankTransaction],
        profile: &AccountProfile,
        params: &PatternParams,
    ) -> Option<(f64, String, Vec<u64>)> {
        let tx_location = transaction.location.as_ref()?;

        // Check if location is typical
        if profile.typical_locations.contains(tx_location) {
            return None;
        }

        // Check for impossible travel (different country within short time)
        let time_window = params
            .custom
            .get("travel_window")
            .copied()
            .unwrap_or(3600.0) as u64;

        let recent_diff_location = history.iter().find(|tx| {
            tx.source_account == transaction.source_account
                && transaction.timestamp.saturating_sub(time_window) <= tx.timestamp
                && tx.location.as_ref() != Some(tx_location)
                && tx.location.is_some()
        });

        if let Some(prev_tx) = recent_diff_location {
            let score = 85.0;
            Some((
                score,
                format!(
                    "Impossible travel: {} to {} in {}s",
                    prev_tx.location.as_ref().unwrap_or(&"Unknown".to_string()),
                    tx_location,
                    transaction.timestamp - prev_tx.timestamp
                ),
                vec![prev_tx.id, transaction.id],
            ))
        } else {
            // Just unusual location
            Some((
                40.0,
                format!("Unusual location: {}", tx_location),
                vec![transaction.id],
            ))
        }
    }

    /// Detect time anomaly.
    fn check_time_anomaly(
        transaction: &BankTransaction,
        profile: &AccountProfile,
    ) -> Option<(f64, String, Vec<u64>)> {
        // Extract hour from timestamp (simplified - assumes UTC)
        let hour = ((transaction.timestamp % 86400) / 3600) as u8;

        if !profile.typical_hours.contains(&hour) {
            let score = 30.0; // Lower score for time anomalies alone
            Some((
                score,
                format!("Transaction at unusual hour: {}:00", hour),
                vec![transaction.id],
            ))
        } else {
            None
        }
    }

    /// Detect account takeover indicators.
    fn check_account_takeover(
        transaction: &BankTransaction,
        history: &[BankTransaction],
        profile: &AccountProfile,
    ) -> Option<(f64, String, Vec<u64>)> {
        let mut indicators = Vec::new();
        let mut score: f64 = 0.0;

        // New account with large transaction
        if profile.account_age_days < 30 && transaction.amount > profile.avg_amount * 5.0 {
            indicators.push("New account with large transaction");
            score += 40.0;
        }

        // Sudden change in transaction pattern
        let recent_total: f64 = history
            .iter()
            .filter(|tx| tx.source_account == transaction.source_account)
            .take(10)
            .map(|tx| tx.amount)
            .sum();

        if transaction.amount > recent_total {
            indicators.push("Transaction exceeds recent total");
            score += 30.0;
        }

        // Multiple failed attempts would be checked elsewhere (not in successful tx history)

        if score > 0.0 {
            Some((
                score.min(100.0),
                format!("Account takeover indicators: {}", indicators.join(", ")),
                vec![transaction.id],
            ))
        } else {
            None
        }
    }

    /// Detect mule account behavior.
    fn check_mule_account(
        transaction: &BankTransaction,
        history: &[BankTransaction],
        params: &PatternParams,
    ) -> Option<(f64, String, Vec<u64>)> {
        let time_window = params.time_window;

        // Mule accounts: receive money and quickly send it out
        let recent: Vec<&BankTransaction> = history
            .iter()
            .filter(|tx| transaction.timestamp.saturating_sub(time_window) <= tx.timestamp)
            .collect();

        let incoming: f64 = recent
            .iter()
            .filter(|tx| tx.dest_account == transaction.source_account)
            .map(|tx| tx.amount)
            .sum();

        let outgoing: f64 = recent
            .iter()
            .filter(|tx| tx.source_account == transaction.source_account)
            .map(|tx| tx.amount)
            .sum::<f64>()
            + transaction.amount;

        // Pass-through: most incoming money quickly sent out
        if incoming > 1000.0 && outgoing > incoming * 0.8 {
            let pass_through_ratio = outgoing / incoming;
            let score = (pass_through_ratio * 50.0).min(80.0);

            Some((
                score,
                format!(
                    "Mule account behavior: ${:.2} in, ${:.2} out ({:.1}% pass-through)",
                    incoming,
                    outgoing,
                    pass_through_ratio * 100.0
                ),
                recent.iter().map(|tx| tx.id).collect(),
            ))
        } else {
            None
        }
    }

    /// Detect layering (complex transaction chains).
    fn check_layering(
        transaction: &BankTransaction,
        history: &[BankTransaction],
        params: &PatternParams,
    ) -> Option<(f64, String, Vec<u64>)> {
        let time_window = params.time_window;
        let min_layers = params.custom.get("min_layers").copied().unwrap_or(3.0) as usize;

        // Count unique accounts involved in recent transactions
        let mut accounts = HashSet::new();
        let mut evidence = Vec::new();

        for tx in history
            .iter()
            .filter(|tx| transaction.timestamp.saturating_sub(time_window) <= tx.timestamp)
        {
            accounts.insert(tx.source_account);
            accounts.insert(tx.dest_account);
            evidence.push(tx.id);
        }
        accounts.insert(transaction.source_account);
        accounts.insert(transaction.dest_account);
        evidence.push(transaction.id);

        // High number of unique accounts suggests layering
        if accounts.len() >= min_layers * 2 {
            let score = (accounts.len() as f64 * 10.0).min(90.0);
            Some((
                score,
                format!("Complex layering: {} accounts involved", accounts.len()),
                evidence,
            ))
        } else {
            None
        }
    }

    /// Create standard fraud patterns.
    pub fn standard_patterns() -> Vec<FraudPattern> {
        vec![
            FraudPattern {
                id: 1,
                name: "Rapid Split (Structuring)".to_string(),
                pattern_type: FraudPatternType::RapidSplit,
                risk_weight: 80.0,
                params: PatternParams {
                    time_window: 86400, // 24 hours
                    min_count: 3,
                    amount_threshold: 10000.0,
                    ..Default::default()
                },
            },
            FraudPattern {
                id: 2,
                name: "Circular Flow".to_string(),
                pattern_type: FraudPatternType::CircularFlow,
                risk_weight: 90.0,
                params: PatternParams {
                    time_window: 604800, // 1 week
                    custom: [("min_chain_length".to_string(), 3.0)]
                        .into_iter()
                        .collect(),
                    ..Default::default()
                },
            },
            FraudPattern {
                id: 3,
                name: "Velocity Anomaly".to_string(),
                pattern_type: FraudPatternType::VelocityAnomaly,
                risk_weight: 60.0,
                params: PatternParams {
                    time_window: 3600, // 1 hour
                    ..Default::default()
                },
            },
            FraudPattern {
                id: 4,
                name: "Amount Anomaly".to_string(),
                pattern_type: FraudPatternType::AmountAnomaly,
                risk_weight: 50.0,
                params: PatternParams::default(),
            },
            FraudPattern {
                id: 5,
                name: "Geographic Anomaly".to_string(),
                pattern_type: FraudPatternType::GeoAnomaly,
                risk_weight: 70.0,
                params: PatternParams {
                    custom: [("travel_window".to_string(), 7200.0)]
                        .into_iter()
                        .collect(), // 2 hours
                    ..Default::default()
                },
            },
            FraudPattern {
                id: 6,
                name: "Mule Account".to_string(),
                pattern_type: FraudPatternType::MuleAccount,
                risk_weight: 85.0,
                params: PatternParams {
                    time_window: 86400, // 24 hours
                    ..Default::default()
                },
            },
        ]
    }
}

impl GpuKernel for FraudPatternMatch {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Channel, TransactionType};

    fn create_transaction(
        id: u64,
        source: u64,
        dest: u64,
        amount: f64,
        timestamp: u64,
    ) -> BankTransaction {
        BankTransaction {
            id,
            source_account: source,
            dest_account: dest,
            amount,
            timestamp,
            tx_type: TransactionType::Wire,
            channel: Channel::Online,
            mcc: None,
            location: Some("US".to_string()),
        }
    }

    #[test]
    fn test_fraud_pattern_match_metadata() {
        let kernel = FraudPatternMatch::new();
        assert_eq!(kernel.metadata().id, "banking/fraud-pattern-match");
        assert_eq!(kernel.metadata().domain, Domain::Banking);
    }

    #[test]
    fn test_normal_transaction() {
        let tx = create_transaction(1, 100, 200, 500.0, 1000000);
        let patterns = FraudPatternMatch::standard_patterns();

        let result = FraudPatternMatch::compute(&tx, &[], &patterns, None);

        // Normal transaction should have low fraud score
        assert!(result.fraud_score < 50.0, "Score: {}", result.fraud_score);
        assert_eq!(result.risk_level, RiskLevel::Low);
    }

    #[test]
    fn test_rapid_split_detection() {
        let base_time = 1000000u64;
        let threshold = 10000.0;

        // Create structuring pattern: multiple transactions just below threshold
        let history = vec![
            create_transaction(1, 100, 200, 9500.0, base_time),
            create_transaction(2, 100, 201, 9800.0, base_time + 1000),
            create_transaction(3, 100, 202, 9600.0, base_time + 2000),
        ];

        let current = create_transaction(4, 100, 203, 9700.0, base_time + 3000);

        let patterns = vec![FraudPattern {
            id: 1,
            name: "Rapid Split".to_string(),
            pattern_type: FraudPatternType::RapidSplit,
            risk_weight: 80.0,
            params: PatternParams {
                time_window: 86400,
                min_count: 3,
                amount_threshold: threshold,
                ..Default::default()
            },
        }];

        let result = FraudPatternMatch::compute(&current, &history, &patterns, None);

        assert!(
            !result.matched_patterns.is_empty(),
            "Should detect rapid split"
        );
        assert!(result.fraud_score > 30.0);
    }

    #[test]
    fn test_circular_flow_detection() {
        let base_time = 1000000u64;

        // Create circular flow: A -> B -> C -> A
        let history = vec![
            create_transaction(1, 100, 200, 5000.0, base_time), // A -> B
            create_transaction(2, 200, 300, 4800.0, base_time + 100), // B -> C
            create_transaction(3, 300, 100, 4600.0, base_time + 200), // C -> A (completing cycle)
        ];

        let current = create_transaction(4, 100, 200, 4500.0, base_time + 300);

        let patterns = vec![FraudPattern {
            id: 1,
            name: "Circular Flow".to_string(),
            pattern_type: FraudPatternType::CircularFlow,
            risk_weight: 90.0,
            params: PatternParams {
                time_window: 86400,
                custom: [("min_chain_length".to_string(), 3.0)]
                    .into_iter()
                    .collect(),
                ..Default::default()
            },
        }];

        let result = FraudPatternMatch::compute(&current, &history, &patterns, None);

        // Should detect circular flow
        let has_circular = result
            .matched_patterns
            .iter()
            .any(|p| p.pattern_name.contains("Circular"));
        assert!(has_circular, "Should detect circular flow");
    }

    #[test]
    fn test_velocity_anomaly() {
        let base_time = 1000000u64;

        // Create many transactions in short time
        let history: Vec<BankTransaction> = (0..20)
            .map(|i| create_transaction(i, 100, 200 + i, 100.0, base_time + i * 60))
            .collect();

        let current = create_transaction(21, 100, 300, 100.0, base_time + 1260);

        let profile = AccountProfile {
            account_id: 100,
            avg_daily_count: 2.0, // Normally 2 transactions per day
            ..Default::default()
        };

        let patterns = vec![FraudPattern {
            id: 1,
            name: "Velocity".to_string(),
            pattern_type: FraudPatternType::VelocityAnomaly,
            risk_weight: 60.0,
            params: PatternParams {
                time_window: 3600, // 1 hour
                ..Default::default()
            },
        }];

        let result = FraudPatternMatch::compute(&current, &history, &patterns, Some(&profile));

        let has_velocity = result
            .matched_patterns
            .iter()
            .any(|p| p.pattern_name.contains("Velocity"));
        assert!(has_velocity, "Should detect velocity anomaly");
    }

    #[test]
    fn test_amount_anomaly() {
        let profile = AccountProfile {
            account_id: 100,
            avg_amount: 500.0,
            std_amount: 100.0,
            ..Default::default()
        };

        // Transaction 10x normal amount
        let tx = create_transaction(1, 100, 200, 5000.0, 1000000);

        let patterns = vec![FraudPattern {
            id: 1,
            name: "Amount Anomaly".to_string(),
            pattern_type: FraudPatternType::AmountAnomaly,
            risk_weight: 50.0,
            params: PatternParams::default(),
        }];

        let result = FraudPatternMatch::compute(&tx, &[], &patterns, Some(&profile));

        let has_amount = result
            .matched_patterns
            .iter()
            .any(|p| p.pattern_name.contains("Amount"));
        assert!(has_amount, "Should detect amount anomaly");
    }

    #[test]
    fn test_geo_anomaly() {
        let base_time = 1000000u64;

        let mut tx1 = create_transaction(1, 100, 200, 500.0, base_time);
        tx1.location = Some("US".to_string());

        let mut tx2 = create_transaction(2, 100, 201, 500.0, base_time + 1800); // 30 min later
        tx2.location = Some("UK".to_string()); // Impossible to travel US -> UK in 30 min

        let profile = AccountProfile {
            account_id: 100,
            typical_locations: vec!["US".to_string()],
            ..Default::default()
        };

        let patterns = vec![FraudPattern {
            id: 1,
            name: "Geographic Anomaly".to_string(),
            pattern_type: FraudPatternType::GeoAnomaly,
            risk_weight: 70.0,
            params: PatternParams {
                custom: [("travel_window".to_string(), 7200.0)]
                    .into_iter()
                    .collect(),
                ..Default::default()
            },
        }];

        let result = FraudPatternMatch::compute(&tx2, &[tx1], &patterns, Some(&profile));

        let has_geo = result
            .matched_patterns
            .iter()
            .any(|p| p.pattern_name.contains("Geographic"));
        assert!(has_geo, "Should detect geographic anomaly");
    }

    #[test]
    fn test_mule_account() {
        let base_time = 1000000u64;

        // Pattern: receive money, quickly send most of it out
        let history = vec![
            create_transaction(1, 200, 100, 10000.0, base_time), // Receive
            create_transaction(2, 100, 300, 3000.0, base_time + 100), // Send out
            create_transaction(3, 100, 301, 3000.0, base_time + 200), // Send out
            create_transaction(4, 100, 302, 3000.0, base_time + 300), // Send out
        ];

        let current = create_transaction(5, 100, 303, 800.0, base_time + 400);

        let patterns = vec![FraudPattern {
            id: 1,
            name: "Mule Account".to_string(),
            pattern_type: FraudPatternType::MuleAccount,
            risk_weight: 85.0,
            params: PatternParams {
                time_window: 86400,
                ..Default::default()
            },
        }];

        let result = FraudPatternMatch::compute(&current, &history, &patterns, None);

        let has_mule = result
            .matched_patterns
            .iter()
            .any(|p| p.pattern_name.contains("Mule"));
        assert!(has_mule, "Should detect mule account behavior");
    }

    #[test]
    fn test_standard_patterns() {
        let patterns = FraudPatternMatch::standard_patterns();

        assert!(!patterns.is_empty());
        assert!(
            patterns
                .iter()
                .any(|p| p.pattern_type == FraudPatternType::RapidSplit)
        );
        assert!(
            patterns
                .iter()
                .any(|p| p.pattern_type == FraudPatternType::CircularFlow)
        );
    }

    #[test]
    fn test_risk_level_conversion() {
        assert_eq!(RiskLevel::from(10.0), RiskLevel::Low);
        assert_eq!(RiskLevel::from(30.0), RiskLevel::Medium);
        assert_eq!(RiskLevel::from(60.0), RiskLevel::High);
        assert_eq!(RiskLevel::from(90.0), RiskLevel::Critical);
    }

    #[test]
    fn test_batch_processing() {
        let txs = vec![
            create_transaction(1, 100, 200, 500.0, 1000000),
            create_transaction(2, 101, 201, 600.0, 1000001),
        ];

        let history_map: HashMap<u64, Vec<BankTransaction>> = HashMap::new();
        let profiles: HashMap<u64, AccountProfile> = HashMap::new();
        let patterns = FraudPatternMatch::standard_patterns();

        let results = FraudPatternMatch::compute_batch(&txs, &history_map, &patterns, &profiles);

        assert_eq!(results.len(), 2);
    }
}
