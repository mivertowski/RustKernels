//! Transaction monitoring kernels.
//!
//! This module provides real-time transaction monitoring
//! with configurable rules and thresholds.

use crate::messages::{TransactionMonitoringInput, TransactionMonitoringOutput};
use crate::types::{
    Alert, MonitoringResult, MonitoringRule, RuleType, Severity, TimeWindow, Transaction,
};
use async_trait::async_trait;
use rustkernel_core::error::Result;
use rustkernel_core::traits::BatchKernel;
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;
use std::time::Instant;

// ============================================================================
// Transaction Monitoring Kernel
// ============================================================================

/// Transaction monitoring kernel.
///
/// Monitors transactions in real-time against configurable rules
/// and generates alerts when thresholds are exceeded.
#[derive(Debug, Clone)]
pub struct TransactionMonitoring {
    metadata: KernelMetadata,
}

impl Default for TransactionMonitoring {
    fn default() -> Self {
        Self::new()
    }
}

impl TransactionMonitoring {
    /// Create a new transaction monitoring kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("compliance/transaction-monitoring", Domain::Compliance)
                .with_description("Real-time transaction threshold monitoring")
                .with_throughput(500_000)
                .with_latency_us(1.0),
        }
    }

    /// Monitor transactions against rules.
    ///
    /// # Arguments
    /// * `transactions` - Transactions to analyze
    /// * `rules` - Monitoring rules to apply
    /// * `current_time` - Current timestamp for time window calculations
    pub fn compute(
        transactions: &[Transaction],
        rules: &[MonitoringRule],
        current_time: u64,
    ) -> MonitoringResult {
        if transactions.is_empty() || rules.is_empty() {
            return MonitoringResult {
                alerts: Vec::new(),
                entities_checked: 0,
                transactions_analyzed: 0,
            };
        }

        let mut alerts = Vec::new();
        let mut alert_id = 1u64;

        // Group transactions by entity (both source and dest)
        let mut entity_txs: HashMap<u64, Vec<&Transaction>> = HashMap::new();
        for tx in transactions {
            entity_txs.entry(tx.source_id).or_default().push(tx);
            entity_txs.entry(tx.dest_id).or_default().push(tx);
        }

        let entities_checked = entity_txs.len();

        // Apply each rule
        for rule in rules {
            match rule.rule_type {
                RuleType::SingleAmount => {
                    for tx in transactions {
                        if tx.amount >= rule.threshold {
                            alerts.push(Alert {
                                id: alert_id,
                                rule_id: rule.id,
                                entity_id: tx.source_id,
                                timestamp: current_time,
                                severity: rule.severity,
                                current_value: tx.amount,
                                threshold: rule.threshold,
                                transaction_ids: vec![tx.id],
                                message: format!(
                                    "Single transaction ${:.2} exceeds threshold ${:.2}",
                                    tx.amount, rule.threshold
                                ),
                            });
                            alert_id += 1;
                        }
                    }
                }

                RuleType::AggregateAmount => {
                    let window = TimeWindow::new(
                        current_time.saturating_sub(rule.window_seconds),
                        current_time,
                    );

                    for (entity_id, txs) in &entity_txs {
                        let window_txs: Vec<_> =
                            txs.iter().filter(|tx| window.contains(tx.timestamp)).collect();

                        let total: f64 = window_txs.iter().map(|tx| tx.amount).sum();

                        if total >= rule.threshold {
                            alerts.push(Alert {
                                id: alert_id,
                                rule_id: rule.id,
                                entity_id: *entity_id,
                                timestamp: current_time,
                                severity: rule.severity,
                                current_value: total,
                                threshold: rule.threshold,
                                transaction_ids: window_txs.iter().map(|tx| tx.id).collect(),
                                message: format!(
                                    "Aggregate amount ${:.2} over {} hours exceeds ${:.2}",
                                    total,
                                    rule.window_seconds / 3600,
                                    rule.threshold
                                ),
                            });
                            alert_id += 1;
                        }
                    }
                }

                RuleType::TransactionCount => {
                    let window = TimeWindow::new(
                        current_time.saturating_sub(rule.window_seconds),
                        current_time,
                    );

                    for (entity_id, txs) in &entity_txs {
                        let window_txs: Vec<_> =
                            txs.iter().filter(|tx| window.contains(tx.timestamp)).collect();

                        let count = window_txs.len() as f64;

                        if count >= rule.threshold {
                            alerts.push(Alert {
                                id: alert_id,
                                rule_id: rule.id,
                                entity_id: *entity_id,
                                timestamp: current_time,
                                severity: rule.severity,
                                current_value: count,
                                threshold: rule.threshold,
                                transaction_ids: window_txs.iter().map(|tx| tx.id).collect(),
                                message: format!(
                                    "{} transactions over {} hours exceeds threshold {}",
                                    count as u64,
                                    rule.window_seconds / 3600,
                                    rule.threshold as u64
                                ),
                            });
                            alert_id += 1;
                        }
                    }
                }

                RuleType::Velocity => {
                    let window = TimeWindow::new(
                        current_time.saturating_sub(rule.window_seconds),
                        current_time,
                    );
                    let hours = rule.window_seconds as f64 / 3600.0;

                    for (entity_id, txs) in &entity_txs {
                        let window_txs: Vec<_> =
                            txs.iter().filter(|tx| window.contains(tx.timestamp)).collect();

                        let total: f64 = window_txs.iter().map(|tx| tx.amount).sum();
                        let velocity = if hours > 0.0 { total / hours } else { total };

                        if velocity >= rule.threshold {
                            alerts.push(Alert {
                                id: alert_id,
                                rule_id: rule.id,
                                entity_id: *entity_id,
                                timestamp: current_time,
                                severity: rule.severity,
                                current_value: velocity,
                                threshold: rule.threshold,
                                transaction_ids: window_txs.iter().map(|tx| tx.id).collect(),
                                message: format!(
                                    "Velocity ${:.2}/hour exceeds threshold ${:.2}/hour",
                                    velocity, rule.threshold
                                ),
                            });
                            alert_id += 1;
                        }
                    }
                }

                RuleType::GeographicRisk => {
                    // For geographic risk, we would need country data on transactions
                    // This is a simplified implementation
                    // In practice, would check against high-risk country lists
                }
            }
        }

        // Sort alerts by severity (descending)
        alerts.sort_by(|a, b| b.severity.cmp(&a.severity));

        MonitoringResult {
            alerts,
            entities_checked,
            transactions_analyzed: transactions.len(),
        }
    }

    /// Create default monitoring rules.
    pub fn default_rules() -> Vec<MonitoringRule> {
        vec![
            // Large single transaction
            MonitoringRule {
                id: 1,
                name: "Large Single Transaction".to_string(),
                rule_type: RuleType::SingleAmount,
                threshold: 10_000.0,
                window_seconds: 0,
                severity: Severity::High,
            },
            // CTR threshold
            MonitoringRule {
                id: 2,
                name: "CTR Threshold".to_string(),
                rule_type: RuleType::AggregateAmount,
                threshold: 10_000.0,
                window_seconds: 86400, // 24 hours
                severity: Severity::High,
            },
            // High transaction count
            MonitoringRule {
                id: 3,
                name: "High Transaction Count".to_string(),
                rule_type: RuleType::TransactionCount,
                threshold: 50.0,
                window_seconds: 86400, // 24 hours
                severity: Severity::Medium,
            },
            // High velocity
            MonitoringRule {
                id: 4,
                name: "High Velocity".to_string(),
                rule_type: RuleType::Velocity,
                threshold: 5000.0, // $5000/hour
                window_seconds: 3600,
                severity: Severity::High,
            },
        ]
    }
}

impl GpuKernel for TransactionMonitoring {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[async_trait]
impl BatchKernel<TransactionMonitoringInput, TransactionMonitoringOutput> for TransactionMonitoring {
    async fn execute(&self, input: TransactionMonitoringInput) -> Result<TransactionMonitoringOutput> {
        let start = Instant::now();
        let result = Self::compute(&input.transactions, &input.rules, input.current_time);
        Ok(TransactionMonitoringOutput {
            result,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_transactions() -> Vec<Transaction> {
        vec![
            Transaction {
                id: 1,
                source_id: 100,
                dest_id: 200,
                amount: 5000.0,
                timestamp: 1000,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
            Transaction {
                id: 2,
                source_id: 100,
                dest_id: 201,
                amount: 4500.0,
                timestamp: 1100,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
            Transaction {
                id: 3,
                source_id: 100,
                dest_id: 202,
                amount: 3000.0,
                timestamp: 1200,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
        ]
    }

    #[test]
    fn test_monitoring_metadata() {
        let kernel = TransactionMonitoring::new();
        assert_eq!(kernel.metadata().id, "compliance/transaction-monitoring");
        assert_eq!(kernel.metadata().domain, Domain::Compliance);
    }

    #[test]
    fn test_single_amount_rule() {
        let txs = vec![Transaction {
            id: 1,
            source_id: 100,
            dest_id: 200,
            amount: 15000.0, // Above threshold
            timestamp: 1000,
            currency: "USD".to_string(),
            tx_type: "wire".to_string(),
        }];

        let rules = vec![MonitoringRule {
            id: 1,
            name: "Large Transaction".to_string(),
            rule_type: RuleType::SingleAmount,
            threshold: 10000.0,
            window_seconds: 0,
            severity: Severity::High,
        }];

        let result = TransactionMonitoring::compute(&txs, &rules, 2000);

        assert!(!result.alerts.is_empty());
        assert_eq!(result.alerts[0].current_value, 15000.0);
        assert_eq!(result.alerts[0].severity, Severity::High);
    }

    #[test]
    fn test_aggregate_amount_rule() {
        let txs = create_test_transactions();

        let rules = vec![MonitoringRule {
            id: 1,
            name: "Aggregate Amount".to_string(),
            rule_type: RuleType::AggregateAmount,
            threshold: 10000.0,
            window_seconds: 3600, // 1 hour
            severity: Severity::High,
        }];

        let result = TransactionMonitoring::compute(&txs, &rules, 1500);

        // Total for entity 100 = 12500, should trigger
        assert!(!result.alerts.is_empty());
        let entity_alert = result.alerts.iter().find(|a| a.entity_id == 100);
        assert!(entity_alert.is_some());
    }

    #[test]
    fn test_transaction_count_rule() {
        // Create many transactions
        let txs: Vec<Transaction> = (0..60)
            .map(|i| Transaction {
                id: i as u64,
                source_id: 100,
                dest_id: 200,
                amount: 100.0,
                timestamp: 1000 + i as u64,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            })
            .collect();

        let rules = vec![MonitoringRule {
            id: 1,
            name: "High Count".to_string(),
            rule_type: RuleType::TransactionCount,
            threshold: 50.0,
            window_seconds: 3600,
            severity: Severity::Medium,
        }];

        let result = TransactionMonitoring::compute(&txs, &rules, 2000);

        assert!(!result.alerts.is_empty());
        assert!(result.alerts[0].current_value >= 50.0);
    }

    #[test]
    fn test_velocity_rule() {
        // High velocity: $10000 in 1 hour = $10000/hour
        let txs = vec![
            Transaction {
                id: 1,
                source_id: 100,
                dest_id: 200,
                amount: 5000.0,
                timestamp: 1000,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
            Transaction {
                id: 2,
                source_id: 100,
                dest_id: 201,
                amount: 5000.0,
                timestamp: 2000,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
        ];

        let rules = vec![MonitoringRule {
            id: 1,
            name: "High Velocity".to_string(),
            rule_type: RuleType::Velocity,
            threshold: 5000.0, // $5000/hour
            window_seconds: 3600,
            severity: Severity::High,
        }];

        let result = TransactionMonitoring::compute(&txs, &rules, 4000);

        assert!(!result.alerts.is_empty());
    }

    #[test]
    fn test_no_alerts_below_threshold() {
        let txs = vec![Transaction {
            id: 1,
            source_id: 100,
            dest_id: 200,
            amount: 500.0, // Below threshold
            timestamp: 1000,
            currency: "USD".to_string(),
            tx_type: "wire".to_string(),
        }];

        let rules = vec![MonitoringRule {
            id: 1,
            name: "Large Transaction".to_string(),
            rule_type: RuleType::SingleAmount,
            threshold: 10000.0,
            window_seconds: 0,
            severity: Severity::High,
        }];

        let result = TransactionMonitoring::compute(&txs, &rules, 2000);

        assert!(result.alerts.is_empty());
    }

    #[test]
    fn test_default_rules() {
        let rules = TransactionMonitoring::default_rules();
        assert!(!rules.is_empty());
        assert!(rules.iter().any(|r| r.rule_type == RuleType::SingleAmount));
        assert!(rules.iter().any(|r| r.rule_type == RuleType::AggregateAmount));
    }

    #[test]
    fn test_empty_inputs() {
        let txs = create_test_transactions();
        let rules = TransactionMonitoring::default_rules();

        let result1 = TransactionMonitoring::compute(&[], &rules, 1000);
        assert!(result1.alerts.is_empty());

        let result2 = TransactionMonitoring::compute(&txs, &[], 1000);
        assert!(result2.alerts.is_empty());
    }
}
