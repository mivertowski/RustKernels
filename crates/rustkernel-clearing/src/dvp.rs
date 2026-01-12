//! DVP (Delivery vs Payment) matching kernel.
//!
//! This module provides DVP matching for clearing:
//! - Match delivery and payment instructions
//! - Identify discrepancies
//! - Calculate match confidence

use crate::types::{DVPInstruction, DVPMatchDetail, DVPMatchResult, DVPStatus};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// DVP Matching Kernel
// ============================================================================

/// DVP matching kernel.
///
/// Matches delivery instructions with corresponding payment instructions.
#[derive(Debug, Clone)]
pub struct DVPMatching {
    metadata: KernelMetadata,
}

impl Default for DVPMatching {
    fn default() -> Self {
        Self::new()
    }
}

impl DVPMatching {
    /// Create a new DVP matching kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("clearing/dvp-matching", Domain::Clearing)
                .with_description("Delivery vs payment matching")
                .with_throughput(50_000)
                .with_latency_us(100.0),
        }
    }

    /// Match DVP instructions.
    pub fn match_instructions(
        instructions: &[DVPInstruction],
        config: &DVPConfig,
    ) -> DVPMatchResult {
        let mut matched_pairs = Vec::new();
        let mut unmatched: Vec<u64> = Vec::new();
        let mut details = Vec::new();

        // Group instructions by security and settlement date
        let mut by_security: HashMap<String, Vec<&DVPInstruction>> = HashMap::new();

        for instr in instructions {
            if instr.status != DVPStatus::Pending {
                continue;
            }
            by_security
                .entry(format!("{}:{}", instr.security_id, instr.settlement_date))
                .or_default()
                .push(instr);
        }

        // For each security/date group, match deliverers with receivers
        for (_key, group) in by_security {
            let deliverers: Vec<_> = group
                .iter()
                .filter(|i| i.quantity < 0) // Negative = delivering
                .collect();
            let receivers: Vec<_> = group
                .iter()
                .filter(|i| i.quantity > 0) // Positive = receiving
                .collect();

            let mut used_receivers: Vec<bool> = vec![false; receivers.len()];

            for deliverer in &deliverers {
                let mut best_match: Option<(usize, f64, Vec<String>)> = None;

                for (j, receiver) in receivers.iter().enumerate() {
                    if used_receivers[j] {
                        continue;
                    }

                    let (confidence, differences) =
                        Self::calculate_match_score(deliverer, receiver, config);

                    if confidence >= config.min_confidence
                        && (best_match.is_none() || confidence > best_match.as_ref().unwrap().1)
                    {
                        best_match = Some((j, confidence, differences));
                    }
                }

                if let Some((j, confidence, differences)) = best_match {
                    used_receivers[j] = true;
                    matched_pairs.push((deliverer.id, receivers[j].id));
                    details.push(DVPMatchDetail {
                        delivery_id: deliverer.id,
                        payment_id: receivers[j].id,
                        confidence,
                        differences,
                    });
                } else {
                    unmatched.push(deliverer.id);
                }
            }

            // Add unmatched receivers
            for (j, receiver) in receivers.iter().enumerate() {
                if !used_receivers[j] {
                    unmatched.push(receiver.id);
                }
            }
        }

        // Add any instructions that weren't in a group (non-pending)
        for instr in instructions {
            if instr.status != DVPStatus::Pending {
                unmatched.push(instr.id);
            }
        }

        let total_pending = instructions
            .iter()
            .filter(|i| i.status == DVPStatus::Pending)
            .count();
        let matched_count = matched_pairs.len() * 2;
        let match_rate = if total_pending > 0 {
            matched_count as f64 / total_pending as f64
        } else {
            0.0
        };

        DVPMatchResult {
            matched_pairs,
            unmatched,
            match_rate,
            details,
        }
    }

    /// Calculate match score between a delivery and receive instruction.
    fn calculate_match_score(
        delivery: &DVPInstruction,
        receive: &DVPInstruction,
        config: &DVPConfig,
    ) -> (f64, Vec<String>) {
        let mut score = 1.0;
        let mut differences = Vec::new();

        // Check counterparties match
        if delivery.deliverer != receive.receiver {
            differences.push(format!(
                "Deliverer mismatch: {} vs {}",
                delivery.deliverer, receive.receiver
            ));
            if config.strict_counterparty {
                return (0.0, differences);
            }
            score *= 0.5;
        }

        if delivery.receiver != receive.deliverer {
            differences.push(format!(
                "Receiver mismatch: {} vs {}",
                delivery.receiver, receive.deliverer
            ));
            if config.strict_counterparty {
                return (0.0, differences);
            }
            score *= 0.5;
        }

        // Check quantity matches (absolute value)
        let delivery_qty = delivery.quantity.unsigned_abs();
        let receive_qty = receive.quantity.unsigned_abs();
        if delivery_qty != receive_qty {
            let qty_diff = (delivery_qty as f64 - receive_qty as f64).abs();
            let qty_pct = qty_diff / delivery_qty.max(receive_qty) as f64;
            differences.push(format!(
                "Quantity mismatch: {} vs {} ({:.2}%)",
                delivery_qty,
                receive_qty,
                qty_pct * 100.0
            ));
            if qty_pct > config.quantity_tolerance {
                return (0.0, differences);
            }
            score *= 1.0 - qty_pct;
        }

        // Check payment amount matches
        let delivery_amt = delivery.payment_amount.unsigned_abs();
        let receive_amt = receive.payment_amount.unsigned_abs();
        if delivery_amt != receive_amt {
            let amt_diff = (delivery_amt as f64 - receive_amt as f64).abs();
            let amt_pct = amt_diff / delivery_amt.max(receive_amt) as f64;
            differences.push(format!(
                "Payment mismatch: {} vs {} ({:.2}%)",
                delivery_amt,
                receive_amt,
                amt_pct * 100.0
            ));
            if amt_pct > config.amount_tolerance {
                return (0.0, differences);
            }
            score *= 1.0 - amt_pct;
        }

        // Check currency matches
        if delivery.currency != receive.currency {
            differences.push(format!(
                "Currency mismatch: {} vs {}",
                delivery.currency, receive.currency
            ));
            return (0.0, differences);
        }

        (score, differences)
    }

    /// Execute settlement for matched pairs.
    pub fn execute_settlement(
        instructions: &mut [DVPInstruction],
        matched_pairs: &[(u64, u64)],
    ) -> SettlementSummary {
        let mut securities_settled = 0i64;
        let mut payments_settled = 0i64;
        let mut settled_count = 0u64;

        for (delivery_id, receive_id) in matched_pairs {
            // Find indices first
            let delivery_idx = instructions.iter().position(|i| i.id == *delivery_id);
            let receive_idx = instructions.iter().position(|i| i.id == *receive_id);

            if let (Some(d_idx), Some(r_idx)) = (delivery_idx, receive_idx) {
                // Capture values before mutating
                let quantity = instructions[d_idx].quantity.unsigned_abs() as i64;
                let payment = instructions[d_idx].payment_amount.unsigned_abs() as i64;

                // Update statuses
                instructions[d_idx].status = DVPStatus::Settled;
                instructions[r_idx].status = DVPStatus::Settled;

                securities_settled += quantity;
                payments_settled += payment;
                settled_count += 2;
            }
        }

        SettlementSummary {
            pairs_settled: matched_pairs.len() as u64,
            instructions_settled: settled_count,
            total_securities: securities_settled,
            total_payments: payments_settled,
        }
    }
}

impl GpuKernel for DVPMatching {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// DVP matching configuration.
#[derive(Debug, Clone)]
pub struct DVPConfig {
    /// Minimum confidence for a match.
    pub min_confidence: f64,
    /// Quantity tolerance (fraction).
    pub quantity_tolerance: f64,
    /// Amount tolerance (fraction).
    pub amount_tolerance: f64,
    /// Strict counterparty matching.
    pub strict_counterparty: bool,
}

impl Default for DVPConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.8,
            quantity_tolerance: 0.01,
            amount_tolerance: 0.01,
            strict_counterparty: true,
        }
    }
}

/// Settlement summary.
#[derive(Debug, Clone)]
pub struct SettlementSummary {
    /// Number of pairs settled.
    pub pairs_settled: u64,
    /// Total instructions settled.
    pub instructions_settled: u64,
    /// Total securities settled.
    pub total_securities: i64,
    /// Total payments settled.
    pub total_payments: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_matching_pair() -> (DVPInstruction, DVPInstruction) {
        let delivery = DVPInstruction {
            id: 1,
            trade_id: 100,
            security_id: "AAPL".to_string(),
            deliverer: "PARTY_A".to_string(),
            receiver: "PARTY_B".to_string(),
            quantity: -100, // Delivering
            payment_amount: -15000,
            currency: "USD".to_string(),
            settlement_date: 1700172800,
            status: DVPStatus::Pending,
        };

        let receive = DVPInstruction {
            id: 2,
            trade_id: 100,
            security_id: "AAPL".to_string(),
            deliverer: "PARTY_B".to_string(),
            receiver: "PARTY_A".to_string(),
            quantity: 100, // Receiving
            payment_amount: 15000,
            currency: "USD".to_string(),
            settlement_date: 1700172800,
            status: DVPStatus::Pending,
        };

        (delivery, receive)
    }

    #[test]
    fn test_dvp_metadata() {
        let kernel = DVPMatching::new();
        assert_eq!(kernel.metadata().id, "clearing/dvp-matching");
        assert_eq!(kernel.metadata().domain, Domain::Clearing);
    }

    #[test]
    fn test_perfect_match() {
        let (delivery, receive) = create_matching_pair();
        let instructions = vec![delivery, receive];
        let config = DVPConfig::default();

        let result = DVPMatching::match_instructions(&instructions, &config);

        assert_eq!(result.matched_pairs.len(), 1);
        assert!(result.unmatched.is_empty());
        assert!((result.match_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_no_match_different_security() {
        let (mut delivery, receive) = create_matching_pair();
        delivery.security_id = "MSFT".to_string();

        let instructions = vec![delivery, receive];
        let config = DVPConfig::default();

        let result = DVPMatching::match_instructions(&instructions, &config);

        assert!(result.matched_pairs.is_empty());
        assert_eq!(result.unmatched.len(), 2);
    }

    #[test]
    fn test_no_match_different_settlement_date() {
        let (mut delivery, receive) = create_matching_pair();
        delivery.settlement_date = 1700259200; // Different date

        let instructions = vec![delivery, receive];
        let config = DVPConfig::default();

        let result = DVPMatching::match_instructions(&instructions, &config);

        assert!(result.matched_pairs.is_empty());
    }

    #[test]
    fn test_quantity_mismatch() {
        let (mut delivery, receive) = create_matching_pair();
        delivery.quantity = -99; // Slight mismatch

        let instructions = vec![delivery, receive];
        let config = DVPConfig::default();

        let result = DVPMatching::match_instructions(&instructions, &config);

        // Should still match due to tolerance
        assert_eq!(result.matched_pairs.len(), 1);
        assert!(result.details[0].confidence < 1.0);
    }

    #[test]
    fn test_quantity_mismatch_too_large() {
        let (mut delivery, receive) = create_matching_pair();
        delivery.quantity = -50; // 50% mismatch

        let instructions = vec![delivery, receive];
        let config = DVPConfig::default();

        let result = DVPMatching::match_instructions(&instructions, &config);

        // Should not match due to large difference
        assert!(result.matched_pairs.is_empty());
    }

    #[test]
    fn test_currency_mismatch() {
        let (mut delivery, receive) = create_matching_pair();
        delivery.currency = "EUR".to_string();

        let instructions = vec![delivery, receive];
        let config = DVPConfig::default();

        let result = DVPMatching::match_instructions(&instructions, &config);

        assert!(result.matched_pairs.is_empty());
    }

    #[test]
    fn test_multiple_pairs() {
        let (d1, r1) = create_matching_pair();
        let (mut d2, mut r2) = create_matching_pair();
        d2.id = 3;
        d2.trade_id = 101;
        r2.id = 4;
        r2.trade_id = 101;

        let instructions = vec![d1, r1, d2, r2];
        let config = DVPConfig::default();

        let result = DVPMatching::match_instructions(&instructions, &config);

        assert_eq!(result.matched_pairs.len(), 2);
        assert!(result.unmatched.is_empty());
    }

    #[test]
    fn test_skip_non_pending() {
        let (mut delivery, receive) = create_matching_pair();
        delivery.status = DVPStatus::Matched;

        let instructions = vec![delivery, receive];
        let config = DVPConfig::default();

        let result = DVPMatching::match_instructions(&instructions, &config);

        assert!(result.matched_pairs.is_empty());
        assert_eq!(result.unmatched.len(), 2);
    }

    #[test]
    fn test_execute_settlement() {
        let (delivery, receive) = create_matching_pair();
        let mut instructions = vec![delivery, receive];
        let matched_pairs = vec![(1, 2)];

        let summary = DVPMatching::execute_settlement(&mut instructions, &matched_pairs);

        assert_eq!(summary.pairs_settled, 1);
        assert_eq!(summary.instructions_settled, 2);
        assert_eq!(instructions[0].status, DVPStatus::Settled);
        assert_eq!(instructions[1].status, DVPStatus::Settled);
    }

    #[test]
    fn test_match_confidence() {
        let (delivery, receive) = create_matching_pair();
        let config = DVPConfig::default();

        let (confidence, differences) =
            DVPMatching::calculate_match_score(&delivery, &receive, &config);

        assert!((confidence - 1.0).abs() < 0.001);
        assert!(differences.is_empty());
    }
}
