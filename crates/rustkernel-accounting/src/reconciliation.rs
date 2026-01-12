//! GL reconciliation kernel.
//!
//! This module provides GL reconciliation for accounting:
//! - Match items between sources
//! - Identify exceptions
//! - Calculate variances

use crate::types::{
    ExceptionType, MatchType, MatchedPair, ReconciliationException, ReconciliationItem,
    ReconciliationResult, ReconciliationStats,
};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// GL Reconciliation Kernel
// ============================================================================

/// GL reconciliation kernel.
///
/// Reconciles items between general ledger and sub-ledgers.
#[derive(Debug, Clone)]
pub struct GLReconciliation {
    metadata: KernelMetadata,
}

impl Default for GLReconciliation {
    fn default() -> Self {
        Self::new()
    }
}

impl GLReconciliation {
    /// Create a new GL reconciliation kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("accounting/gl-reconciliation", Domain::Accounting)
                .with_description("General ledger reconciliation")
                .with_throughput(20_000)
                .with_latency_us(100.0),
        }
    }

    /// Reconcile items between two sources.
    pub fn reconcile(
        source_items: &[ReconciliationItem],
        target_items: &[ReconciliationItem],
        config: &ReconciliationConfig,
    ) -> ReconciliationResult {
        let mut matched_pairs = Vec::new();
        let mut unmatched = Vec::new();
        let mut exceptions = Vec::new();
        let mut total_variance = 0.0;

        let mut used_targets: Vec<bool> = vec![false; target_items.len()];

        // Group source items by account for efficient matching
        let source_by_account: HashMap<&str, Vec<(usize, &ReconciliationItem)>> = source_items
            .iter()
            .enumerate()
            .fold(HashMap::new(), |mut acc, (i, item)| {
                acc.entry(item.account_code.as_str())
                    .or_default()
                    .push((i, item));
                acc
            });

        let target_by_account: HashMap<&str, Vec<(usize, &ReconciliationItem)>> = target_items
            .iter()
            .enumerate()
            .fold(HashMap::new(), |mut acc, (i, item)| {
                acc.entry(item.account_code.as_str())
                    .or_default()
                    .push((i, item));
                acc
            });

        // Match items
        for (account, sources) in &source_by_account {
            if let Some(targets) = target_by_account.get(account) {
                for (_source_idx, source_item) in sources {
                    let best_match =
                        Self::find_best_match(source_item, targets, &used_targets, config);

                    match best_match {
                        Some((target_idx, confidence, variance, match_type)) => {
                            used_targets[target_idx] = true;
                            total_variance += variance.abs();

                            matched_pairs.push(MatchedPair {
                                source_id: source_item.id.clone(),
                                target_id: target_items[target_idx].id.clone(),
                                confidence,
                                variance,
                                match_type,
                            });

                            // Check for variance exception
                            if variance.abs() > config.variance_threshold {
                                exceptions.push(ReconciliationException {
                                    item_id: source_item.id.clone(),
                                    exception_type: ExceptionType::AmountVariance,
                                    description: format!(
                                        "Amount variance of {} exceeds threshold",
                                        variance
                                    ),
                                    suggested_action: Some("Review and adjust".to_string()),
                                });
                            }
                        }
                        None => {
                            unmatched.push(source_item.id.clone());
                            exceptions.push(ReconciliationException {
                                item_id: source_item.id.clone(),
                                exception_type: ExceptionType::MissingCounterpart,
                                description: "No matching item found in target".to_string(),
                                suggested_action: Some("Investigate missing item".to_string()),
                            });
                        }
                    }
                }
            } else {
                // No targets for this account
                for (_, source_item) in sources {
                    unmatched.push(source_item.id.clone());
                }
            }
        }

        // Add unmatched targets
        for (i, target) in target_items.iter().enumerate() {
            if !used_targets[i] {
                unmatched.push(target.id.clone());
                exceptions.push(ReconciliationException {
                    item_id: target.id.clone(),
                    exception_type: ExceptionType::MissingCounterpart,
                    description: "Target item has no matching source".to_string(),
                    suggested_action: Some("Investigate orphan item".to_string()),
                });
            }
        }

        let total_items = source_items.len() + target_items.len();
        let matched_count = matched_pairs.len() * 2;
        let match_rate = if total_items > 0 {
            matched_count as f64 / total_items as f64
        } else {
            0.0
        };

        let exception_count = exceptions.len();

        ReconciliationResult {
            matched_pairs,
            unmatched,
            exceptions,
            stats: ReconciliationStats {
                total_items,
                matched_count,
                unmatched_count: total_items - matched_count,
                exception_count,
                match_rate,
                total_variance,
            },
        }
    }

    /// Find best matching target for a source item.
    fn find_best_match(
        source: &ReconciliationItem,
        targets: &[(usize, &ReconciliationItem)],
        used_targets: &[bool],
        config: &ReconciliationConfig,
    ) -> Option<(usize, f64, f64, MatchType)> {
        let mut best: Option<(usize, f64, f64, MatchType)> = None;

        for &(target_idx, target) in targets {
            if used_targets[target_idx] {
                continue;
            }

            // Check currency match
            if source.currency != target.currency {
                continue;
            }

            // Calculate variance
            let variance = source.amount - target.amount;
            let abs_variance = variance.abs();
            let pct_variance = if source.amount.abs() > 0.0 {
                abs_variance / source.amount.abs()
            } else {
                0.0
            };

            // Determine match type and confidence
            let (match_type, confidence) = if abs_variance < 0.001 {
                (MatchType::Exact, 1.0)
            } else if abs_variance <= config.amount_tolerance
                || pct_variance <= config.percentage_tolerance
            {
                let conf = 1.0 - (pct_variance / config.percentage_tolerance).min(1.0);
                (MatchType::Tolerance, conf * 0.9)
            } else {
                continue; // No match
            };

            // Check date tolerance
            let date_diff = (source.date as i64 - target.date as i64).unsigned_abs();
            if date_diff > config.date_tolerance_days as u64 * 86400 {
                continue;
            }

            // Check reference match boost
            let ref_boost = if config.match_on_reference
                && !source.reference.is_empty()
                && source.reference == target.reference
            {
                0.1
            } else {
                0.0
            };

            let final_confidence = (confidence + ref_boost).min(1.0);

            // Keep best match
            if best.is_none() || final_confidence > best.as_ref().unwrap().1 {
                best = Some((target_idx, final_confidence, variance, match_type));
            }
        }

        best
    }

    /// Reconcile with many-to-one matching.
    pub fn reconcile_many_to_one(
        source_items: &[ReconciliationItem],
        target_items: &[ReconciliationItem],
        config: &ReconciliationConfig,
    ) -> ReconciliationResult {
        let mut matched_pairs = Vec::new();
        let mut unmatched = Vec::new();
        let mut exceptions = Vec::new();
        let mut total_variance = 0.0;

        // Group source items by account and sum
        let mut source_totals: HashMap<String, (f64, Vec<String>)> = HashMap::new();
        for item in source_items {
            let entry = source_totals
                .entry(item.account_code.clone())
                .or_insert((0.0, Vec::new()));
            entry.0 += item.amount;
            entry.1.push(item.id.clone());
        }

        // Match against targets
        for target in target_items {
            if let Some((source_total, source_ids)) = source_totals.get(&target.account_code) {
                let variance = *source_total - target.amount;

                if variance.abs() <= config.amount_tolerance
                    || (variance.abs() / target.amount.abs()) <= config.percentage_tolerance
                {
                    // Create many-to-one match
                    for source_id in source_ids {
                        matched_pairs.push(MatchedPair {
                            source_id: source_id.clone(),
                            target_id: target.id.clone(),
                            confidence: 0.9,
                            variance: variance / source_ids.len() as f64,
                            match_type: MatchType::ManyToOne,
                        });
                    }
                    total_variance += variance.abs();
                } else {
                    // Variance too large
                    unmatched.push(target.id.clone());
                    exceptions.push(ReconciliationException {
                        item_id: target.id.clone(),
                        exception_type: ExceptionType::AmountVariance,
                        description: format!("Sum variance of {} exceeds tolerance", variance),
                        suggested_action: None,
                    });
                }
            } else {
                unmatched.push(target.id.clone());
            }
        }

        let total_items = source_items.len() + target_items.len();
        let matched_count = matched_pairs.len();
        let exception_count = exceptions.len();

        ReconciliationResult {
            matched_pairs,
            unmatched,
            exceptions,
            stats: ReconciliationStats {
                total_items,
                matched_count,
                unmatched_count: total_items - matched_count,
                exception_count,
                match_rate: matched_count as f64 / total_items.max(1) as f64,
                total_variance,
            },
        }
    }

    /// Identify potential duplicates.
    pub fn find_duplicates(
        items: &[ReconciliationItem],
        config: &DuplicateConfig,
    ) -> Vec<DuplicateGroup> {
        let mut groups: Vec<DuplicateGroup> = Vec::new();

        for i in 0..items.len() {
            for j in (i + 1)..items.len() {
                let a = &items[i];
                let b = &items[j];

                let is_dup = Self::check_duplicate(a, b, config);

                if is_dup {
                    // Check if either is already in a group
                    let mut found_group = false;
                    for group in &mut groups {
                        if group.item_ids.contains(&a.id) || group.item_ids.contains(&b.id) {
                            if !group.item_ids.contains(&a.id) {
                                group.item_ids.push(a.id.clone());
                            }
                            if !group.item_ids.contains(&b.id) {
                                group.item_ids.push(b.id.clone());
                            }
                            found_group = true;
                            break;
                        }
                    }

                    if !found_group {
                        groups.push(DuplicateGroup {
                            item_ids: vec![a.id.clone(), b.id.clone()],
                            total_amount: a.amount + b.amount,
                            account_code: a.account_code.clone(),
                        });
                    }
                }
            }
        }

        groups
    }

    /// Check if two items are duplicates.
    fn check_duplicate(
        a: &ReconciliationItem,
        b: &ReconciliationItem,
        config: &DuplicateConfig,
    ) -> bool {
        // Same account
        if a.account_code != b.account_code {
            return false;
        }

        // Same or very close amount
        if (a.amount - b.amount).abs() > config.amount_threshold {
            return false;
        }

        // Same currency
        if a.currency != b.currency {
            return false;
        }

        // Within date range
        let date_diff = (a.date as i64 - b.date as i64).unsigned_abs();
        if date_diff > config.date_range_days as u64 * 86400 {
            return false;
        }

        // Check reference if configured
        if config.match_reference && !a.reference.is_empty() && a.reference == b.reference {
            return true;
        }

        // Check description similarity (simplified)
        if config.match_description && a.source == b.source {
            return true;
        }

        // Default: consider duplicate if all basic criteria match
        true
    }
}

impl GpuKernel for GLReconciliation {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Reconciliation configuration.
#[derive(Debug, Clone)]
pub struct ReconciliationConfig {
    /// Amount tolerance (absolute).
    pub amount_tolerance: f64,
    /// Percentage tolerance.
    pub percentage_tolerance: f64,
    /// Date tolerance in days.
    pub date_tolerance_days: u32,
    /// Match on reference field.
    pub match_on_reference: bool,
    /// Variance threshold for exceptions.
    pub variance_threshold: f64,
}

impl Default for ReconciliationConfig {
    fn default() -> Self {
        Self {
            amount_tolerance: 0.01,
            percentage_tolerance: 0.001,
            date_tolerance_days: 3,
            match_on_reference: true,
            variance_threshold: 1.0,
        }
    }
}

/// Duplicate detection configuration.
#[derive(Debug, Clone)]
pub struct DuplicateConfig {
    /// Amount threshold for duplicates.
    pub amount_threshold: f64,
    /// Date range in days.
    pub date_range_days: u32,
    /// Match on reference.
    pub match_reference: bool,
    /// Match on description.
    pub match_description: bool,
}

impl Default for DuplicateConfig {
    fn default() -> Self {
        Self {
            amount_threshold: 0.01,
            date_range_days: 7,
            match_reference: true,
            match_description: false,
        }
    }
}

/// Duplicate group.
#[derive(Debug, Clone)]
pub struct DuplicateGroup {
    /// Item IDs in this group.
    pub item_ids: Vec<String>,
    /// Total amount.
    pub total_amount: f64,
    /// Account code.
    pub account_code: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ReconciliationSource, ReconciliationStatus};

    fn create_test_source() -> Vec<ReconciliationItem> {
        vec![
            ReconciliationItem {
                id: "S1".to_string(),
                source: ReconciliationSource::GeneralLedger,
                account_code: "1000".to_string(),
                amount: 1000.0,
                currency: "USD".to_string(),
                date: 1700000000,
                reference: "REF001".to_string(),
                status: ReconciliationStatus::Unmatched,
                matched_with: None,
            },
            ReconciliationItem {
                id: "S2".to_string(),
                source: ReconciliationSource::GeneralLedger,
                account_code: "2000".to_string(),
                amount: 2500.0,
                currency: "USD".to_string(),
                date: 1700000000,
                reference: "REF002".to_string(),
                status: ReconciliationStatus::Unmatched,
                matched_with: None,
            },
        ]
    }

    fn create_test_target() -> Vec<ReconciliationItem> {
        vec![
            ReconciliationItem {
                id: "T1".to_string(),
                source: ReconciliationSource::SubLedger,
                account_code: "1000".to_string(),
                amount: 1000.0,
                currency: "USD".to_string(),
                date: 1700000000,
                reference: "REF001".to_string(),
                status: ReconciliationStatus::Unmatched,
                matched_with: None,
            },
            ReconciliationItem {
                id: "T2".to_string(),
                source: ReconciliationSource::SubLedger,
                account_code: "2000".to_string(),
                amount: 2500.5, // Slight variance
                currency: "USD".to_string(),
                date: 1700000000,
                reference: "REF002".to_string(),
                status: ReconciliationStatus::Unmatched,
                matched_with: None,
            },
        ]
    }

    #[test]
    fn test_reconciliation_metadata() {
        let kernel = GLReconciliation::new();
        assert_eq!(kernel.metadata().id, "accounting/gl-reconciliation");
        assert_eq!(kernel.metadata().domain, Domain::Accounting);
    }

    #[test]
    fn test_exact_match() {
        let source = create_test_source();
        let target = create_test_target();
        let config = ReconciliationConfig::default();

        let result = GLReconciliation::reconcile(&source, &target, &config);

        // First pair should be exact match
        let first_match = result
            .matched_pairs
            .iter()
            .find(|p| p.source_id == "S1")
            .unwrap();
        assert_eq!(first_match.match_type, MatchType::Exact);
        assert!((first_match.confidence - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_tolerance_match() {
        let source = create_test_source();
        let target = create_test_target();
        let config = ReconciliationConfig {
            amount_tolerance: 1.0,
            ..Default::default()
        };

        let result = GLReconciliation::reconcile(&source, &target, &config);

        // Second pair should be tolerance match
        let second_match = result
            .matched_pairs
            .iter()
            .find(|p| p.source_id == "S2")
            .unwrap();
        assert_eq!(second_match.match_type, MatchType::Tolerance);
        assert!((second_match.variance - (-0.5)).abs() < 0.01);
    }

    #[test]
    fn test_no_match() {
        let source = vec![ReconciliationItem {
            id: "S1".to_string(),
            source: ReconciliationSource::GeneralLedger,
            account_code: "9999".to_string(), // Different account
            amount: 1000.0,
            currency: "USD".to_string(),
            date: 1700000000,
            reference: "REF001".to_string(),
            status: ReconciliationStatus::Unmatched,
            matched_with: None,
        }];

        let target = create_test_target();
        let config = ReconciliationConfig::default();

        let result = GLReconciliation::reconcile(&source, &target, &config);

        assert!(result.matched_pairs.is_empty());
        assert!(!result.unmatched.is_empty());
    }

    #[test]
    fn test_currency_mismatch() {
        let source = vec![ReconciliationItem {
            id: "S1".to_string(),
            source: ReconciliationSource::GeneralLedger,
            account_code: "1000".to_string(),
            amount: 1000.0,
            currency: "EUR".to_string(), // Different currency
            date: 1700000000,
            reference: "REF001".to_string(),
            status: ReconciliationStatus::Unmatched,
            matched_with: None,
        }];

        let target = create_test_target();
        let config = ReconciliationConfig::default();

        let result = GLReconciliation::reconcile(&source, &target, &config);

        assert!(result.matched_pairs.is_empty());
    }

    #[test]
    fn test_variance_exception() {
        let source = create_test_source();
        let mut target = create_test_target();
        target[0].amount = 1002.0; // Variance > threshold

        let config = ReconciliationConfig {
            amount_tolerance: 5.0,
            variance_threshold: 1.0,
            ..Default::default()
        };

        let result = GLReconciliation::reconcile(&source, &target, &config);

        assert!(
            result
                .exceptions
                .iter()
                .any(|e| e.exception_type == ExceptionType::AmountVariance)
        );
    }

    #[test]
    fn test_many_to_one() {
        let source = vec![
            ReconciliationItem {
                id: "S1".to_string(),
                source: ReconciliationSource::GeneralLedger,
                account_code: "1000".to_string(),
                amount: 500.0,
                currency: "USD".to_string(),
                date: 1700000000,
                reference: "".to_string(),
                status: ReconciliationStatus::Unmatched,
                matched_with: None,
            },
            ReconciliationItem {
                id: "S2".to_string(),
                source: ReconciliationSource::GeneralLedger,
                account_code: "1000".to_string(),
                amount: 500.0,
                currency: "USD".to_string(),
                date: 1700000000,
                reference: "".to_string(),
                status: ReconciliationStatus::Unmatched,
                matched_with: None,
            },
        ];

        let target = vec![ReconciliationItem {
            id: "T1".to_string(),
            source: ReconciliationSource::SubLedger,
            account_code: "1000".to_string(),
            amount: 1000.0,
            currency: "USD".to_string(),
            date: 1700000000,
            reference: "".to_string(),
            status: ReconciliationStatus::Unmatched,
            matched_with: None,
        }];

        let config = ReconciliationConfig::default();
        let result = GLReconciliation::reconcile_many_to_one(&source, &target, &config);

        assert_eq!(result.matched_pairs.len(), 2);
        assert!(
            result
                .matched_pairs
                .iter()
                .all(|p| p.match_type == MatchType::ManyToOne)
        );
    }

    #[test]
    fn test_find_duplicates() {
        let items = vec![
            ReconciliationItem {
                id: "S1".to_string(),
                source: ReconciliationSource::GeneralLedger,
                account_code: "1000".to_string(),
                amount: 1000.0,
                currency: "USD".to_string(),
                date: 1700000000,
                reference: "REF001".to_string(),
                status: ReconciliationStatus::Unmatched,
                matched_with: None,
            },
            ReconciliationItem {
                id: "S2".to_string(),
                source: ReconciliationSource::GeneralLedger,
                account_code: "1000".to_string(),
                amount: 1000.0,
                currency: "USD".to_string(),
                date: 1700000000 + 86400, // Next day
                reference: "REF001".to_string(),
                status: ReconciliationStatus::Unmatched,
                matched_with: None,
            },
        ];

        let config = DuplicateConfig::default();
        let duplicates = GLReconciliation::find_duplicates(&items, &config);

        assert_eq!(duplicates.len(), 1);
        assert_eq!(duplicates[0].item_ids.len(), 2);
    }

    #[test]
    fn test_match_rate() {
        let source = create_test_source();
        let target = create_test_target();
        let config = ReconciliationConfig {
            amount_tolerance: 1.0,
            ..Default::default()
        };

        let result = GLReconciliation::reconcile(&source, &target, &config);

        // All 4 items should be matched (2 pairs)
        assert!((result.stats.match_rate - 1.0).abs() < 0.001);
    }
}
