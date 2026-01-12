//! Payment flow analysis kernel.
//!
//! Batch-mode kernel for analyzing payment flows and network metrics.

use crate::types::*;
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::{HashMap, HashSet};

// ============================================================================
// FlowAnalysis Kernel
// ============================================================================

/// Payment flow analysis kernel for network metrics and pattern detection.
///
/// Analyzes payment flows to identify patterns, calculate network metrics,
/// and detect anomalies in payment behavior.
#[derive(Debug, Clone)]
pub struct FlowAnalysis {
    metadata: KernelMetadata,
}

impl Default for FlowAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl FlowAnalysis {
    /// Create a new flow analysis kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("payments/flow-analysis", Domain::PaymentProcessing)
                .with_description("Payment flow network analysis and metrics")
                .with_throughput(100_000)
                .with_latency_us(50.0),
        }
    }

    /// Analyze payment flows.
    pub fn analyze(payments: &[Payment], config: &FlowAnalysisConfig) -> FlowAnalysisResult {
        // Build flow graph
        let flows = Self::build_flows(payments);

        // Calculate node metrics
        let node_metrics = Self::calculate_node_metrics(&flows, payments);

        // Calculate overall metrics
        let overall_metrics = Self::calculate_overall_metrics(payments, &node_metrics);

        // Detect anomalies
        let anomalies = if config.detect_anomalies {
            Self::detect_anomalies(&flows, &node_metrics, config)
        } else {
            Vec::new()
        };

        FlowAnalysisResult {
            flows,
            node_metrics,
            overall_metrics,
            anomalies,
        }
    }

    /// Build payment flows from transactions.
    fn build_flows(payments: &[Payment]) -> Vec<PaymentFlow> {
        let mut flow_map: HashMap<(String, String), (f64, usize)> = HashMap::new();

        for payment in payments {
            if payment.status == PaymentStatus::Completed || payment.status == PaymentStatus::Processing {
                let key = (payment.payer_account.clone(), payment.payee_account.clone());
                let entry = flow_map.entry(key).or_insert((0.0, 0));
                entry.0 += payment.amount;
                entry.1 += 1;
            }
        }

        flow_map
            .into_iter()
            .map(|((source, target), (volume, count))| PaymentFlow {
                source,
                target,
                volume,
                count,
                avg_amount: volume / count as f64,
            })
            .collect()
    }

    /// Calculate metrics for each node (account).
    fn calculate_node_metrics(
        flows: &[PaymentFlow],
        payments: &[Payment],
    ) -> HashMap<String, NodeMetrics> {
        let mut metrics: HashMap<String, NodeMetrics> = HashMap::new();

        // Initialize with all accounts seen
        let mut accounts: HashSet<String> = HashSet::new();
        for payment in payments {
            accounts.insert(payment.payer_account.clone());
            accounts.insert(payment.payee_account.clone());
        }

        for account in accounts {
            metrics.insert(
                account.clone(),
                NodeMetrics {
                    node_id: account,
                    total_inflow: 0.0,
                    total_outflow: 0.0,
                    net_flow: 0.0,
                    inbound_count: 0,
                    outbound_count: 0,
                    centrality: 0.0,
                },
            );
        }

        // Calculate in/out flows
        for flow in flows {
            if let Some(source_metrics) = metrics.get_mut(&flow.source) {
                source_metrics.total_outflow += flow.volume;
                source_metrics.outbound_count += flow.count;
            }
            if let Some(target_metrics) = metrics.get_mut(&flow.target) {
                target_metrics.total_inflow += flow.volume;
                target_metrics.inbound_count += flow.count;
            }
        }

        // Calculate net flow and basic centrality
        let total_flow: f64 = flows.iter().map(|f| f.volume).sum();
        for (_, node) in metrics.iter_mut() {
            node.net_flow = node.total_inflow - node.total_outflow;
            // Simple degree centrality based on flow volume
            if total_flow > 0.0 {
                node.centrality = (node.total_inflow + node.total_outflow) / (2.0 * total_flow);
            }
        }

        metrics
    }

    /// Calculate overall network metrics.
    fn calculate_overall_metrics(
        payments: &[Payment],
        node_metrics: &HashMap<String, NodeMetrics>,
    ) -> OverallMetrics {
        let completed: Vec<_> = payments
            .iter()
            .filter(|p| p.status == PaymentStatus::Completed || p.status == PaymentStatus::Processing)
            .collect();

        let total_volume: f64 = completed.iter().map(|p| p.amount).sum();
        let total_transactions = completed.len();

        let unique_payers: HashSet<_> = completed.iter().map(|p| &p.payer_account).collect();
        let unique_payees: HashSet<_> = completed.iter().map(|p| &p.payee_account).collect();

        let avg_transaction_size = if total_transactions > 0 {
            total_volume / total_transactions as f64
        } else {
            0.0
        };

        // Calculate peak hour
        let mut hour_volumes: HashMap<u32, f64> = HashMap::new();
        for payment in completed.iter() {
            // Extract hour from timestamp (simplified)
            let hour = ((payment.initiated_at / 3600) % 24) as u32;
            *hour_volumes.entry(hour).or_insert(0.0) += payment.amount;
        }
        let peak_hour = hour_volumes
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(h, _)| h);

        // Calculate network density
        let n = node_metrics.len();
        let actual_edges: HashSet<_> = completed
            .iter()
            .map(|p| (&p.payer_account, &p.payee_account))
            .collect();
        let max_edges = if n > 1 { n * (n - 1) } else { 1 };
        let network_density = actual_edges.len() as f64 / max_edges as f64;

        OverallMetrics {
            total_volume,
            total_transactions,
            unique_payers: unique_payers.len(),
            unique_payees: unique_payees.len(),
            avg_transaction_size,
            peak_hour,
            network_density,
        }
    }

    /// Detect anomalies in payment flows.
    fn detect_anomalies(
        flows: &[PaymentFlow],
        node_metrics: &HashMap<String, NodeMetrics>,
        config: &FlowAnalysisConfig,
    ) -> Vec<FlowAnomaly> {
        let mut anomalies = Vec::new();

        // Detect unusual volume
        let avg_volume: f64 = flows.iter().map(|f| f.volume).sum::<f64>() / flows.len().max(1) as f64;
        let std_volume = Self::calculate_std(&flows.iter().map(|f| f.volume).collect::<Vec<_>>());

        for flow in flows {
            if flow.volume > avg_volume + config.volume_threshold_std * std_volume {
                anomalies.push(FlowAnomaly {
                    anomaly_type: FlowAnomalyType::UnusualVolume,
                    entity: format!("{}->{}", flow.source, flow.target),
                    description: format!(
                        "Unusually high volume: ${:.2} (avg: ${:.2})",
                        flow.volume, avg_volume
                    ),
                    severity: ((flow.volume - avg_volume) / std_volume / 10.0).min(1.0),
                    timestamp: 0,
                });
            }
        }

        // Detect unusual frequency
        let avg_count: f64 = flows.iter().map(|f| f.count as f64).sum::<f64>() / flows.len().max(1) as f64;
        for flow in flows {
            if flow.count as f64 > avg_count * config.frequency_threshold_multiple {
                anomalies.push(FlowAnomaly {
                    anomaly_type: FlowAnomalyType::UnusualFrequency,
                    entity: format!("{}->{}", flow.source, flow.target),
                    description: format!(
                        "Unusually high frequency: {} transactions (avg: {:.1})",
                        flow.count, avg_count
                    ),
                    severity: ((flow.count as f64 / avg_count) / 5.0).min(1.0),
                    timestamp: 0,
                });
            }
        }

        // Detect circular flows
        if config.detect_circular_flows {
            let circular = Self::detect_circular_flows(flows);
            anomalies.extend(circular);
        }

        // Detect rapid movement
        if config.detect_rapid_movement {
            let rapid = Self::detect_rapid_movement(node_metrics);
            anomalies.extend(rapid);
        }

        anomalies
    }

    /// Detect circular payment flows.
    fn detect_circular_flows(flows: &[PaymentFlow]) -> Vec<FlowAnomaly> {
        let mut anomalies = Vec::new();

        // Build adjacency list
        let mut graph: HashMap<&str, HashSet<&str>> = HashMap::new();
        for flow in flows {
            graph
                .entry(&flow.source)
                .or_default()
                .insert(&flow.target);
        }

        // Check for direct cycles (A->B->A)
        for flow in flows {
            if let Some(targets) = graph.get(flow.target.as_str()) {
                if targets.contains(flow.source.as_str()) {
                    anomalies.push(FlowAnomaly {
                        anomaly_type: FlowAnomalyType::CircularFlow,
                        entity: format!("{}<->{}", flow.source, flow.target),
                        description: format!(
                            "Circular flow detected between {} and {}",
                            flow.source, flow.target
                        ),
                        severity: 0.7,
                        timestamp: 0,
                    });
                }
            }
        }

        // Deduplicate (A->B and B->A would create two entries)
        let mut seen: HashSet<String> = HashSet::new();
        anomalies.retain(|a| {
            let key = if a.entity.contains("<->") {
                let parts: Vec<&str> = a.entity.split("<->").collect();
                if parts.len() == 2 {
                    let mut sorted = vec![parts[0], parts[1]];
                    sorted.sort();
                    format!("{}<->{}", sorted[0], sorted[1])
                } else {
                    a.entity.clone()
                }
            } else {
                a.entity.clone()
            };
            seen.insert(key)
        });

        anomalies
    }

    /// Detect rapid money movement.
    fn detect_rapid_movement(
        node_metrics: &HashMap<String, NodeMetrics>,
    ) -> Vec<FlowAnomaly> {
        let mut anomalies = Vec::new();

        // Find accounts that are pass-through (high inflow and outflow, low net)
        for (node_id, metrics) in node_metrics {
            let total_flow = metrics.total_inflow + metrics.total_outflow;
            if total_flow > 0.0 {
                let pass_through_ratio =
                    metrics.total_inflow.min(metrics.total_outflow) / (total_flow / 2.0);

                // If >80% of money flows through (in ~= out), it's suspicious
                if pass_through_ratio > 0.8 && metrics.inbound_count >= 2 && metrics.outbound_count >= 2 {
                    anomalies.push(FlowAnomaly {
                        anomaly_type: FlowAnomalyType::RapidMovement,
                        entity: node_id.clone(),
                        description: format!(
                            "Pass-through account: ${:.2} in, ${:.2} out ({:.0}% pass-through)",
                            metrics.total_inflow,
                            metrics.total_outflow,
                            pass_through_ratio * 100.0
                        ),
                        severity: pass_through_ratio,
                        timestamp: 0,
                    });
                }
            }
        }

        anomalies
    }

    /// Calculate standard deviation.
    fn calculate_std(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    /// Get top flows by volume.
    pub fn top_flows_by_volume(payments: &[Payment], limit: usize) -> Vec<PaymentFlow> {
        let mut flows = Self::build_flows(payments);
        flows.sort_by(|a, b| b.volume.partial_cmp(&a.volume).unwrap_or(std::cmp::Ordering::Equal));
        flows.truncate(limit);
        flows
    }

    /// Get top flows by count.
    pub fn top_flows_by_count(payments: &[Payment], limit: usize) -> Vec<PaymentFlow> {
        let mut flows = Self::build_flows(payments);
        flows.sort_by(|a, b| b.count.cmp(&a.count));
        flows.truncate(limit);
        flows
    }

    /// Analyze flows for a specific account.
    pub fn analyze_account(payments: &[Payment], account_id: &str) -> AccountFlowAnalysis {
        let inbound: Vec<_> = payments
            .iter()
            .filter(|p| p.payee_account == account_id)
            .collect();
        let outbound: Vec<_> = payments
            .iter()
            .filter(|p| p.payer_account == account_id)
            .collect();

        let total_inbound: f64 = inbound.iter().map(|p| p.amount).sum();
        let total_outbound: f64 = outbound.iter().map(|p| p.amount).sum();

        let unique_sources: HashSet<_> = inbound.iter().map(|p| &p.payer_account).collect();
        let unique_destinations: HashSet<_> = outbound.iter().map(|p| &p.payee_account).collect();

        // Payment type breakdown
        let mut type_breakdown: HashMap<PaymentType, (usize, f64)> = HashMap::new();
        for payment in inbound.iter().chain(outbound.iter()) {
            let entry = type_breakdown.entry(payment.payment_type).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += payment.amount;
        }

        AccountFlowAnalysis {
            account_id: account_id.to_string(),
            total_inbound,
            total_outbound,
            net_flow: total_inbound - total_outbound,
            inbound_count: inbound.len(),
            outbound_count: outbound.len(),
            unique_sources: unique_sources.len(),
            unique_destinations: unique_destinations.len(),
            payment_type_breakdown: type_breakdown,
        }
    }
}

impl GpuKernel for FlowAnalysis {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Configuration Types
// ============================================================================

/// Flow analysis configuration.
#[derive(Debug, Clone)]
pub struct FlowAnalysisConfig {
    /// Detect anomalies.
    pub detect_anomalies: bool,
    /// Volume threshold (standard deviations).
    pub volume_threshold_std: f64,
    /// Frequency threshold (multiple of average).
    pub frequency_threshold_multiple: f64,
    /// Detect circular flows.
    pub detect_circular_flows: bool,
    /// Structuring detection threshold.
    pub structuring_threshold: f64,
    /// Detect rapid movement patterns.
    pub detect_rapid_movement: bool,
}

impl Default for FlowAnalysisConfig {
    fn default() -> Self {
        Self {
            detect_anomalies: true,
            volume_threshold_std: 3.0,
            frequency_threshold_multiple: 5.0,
            detect_circular_flows: true,
            structuring_threshold: 10000.0,
            detect_rapid_movement: true,
        }
    }
}

/// Account flow analysis result.
#[derive(Debug, Clone)]
pub struct AccountFlowAnalysis {
    /// Account ID.
    pub account_id: String,
    /// Total inbound volume.
    pub total_inbound: f64,
    /// Total outbound volume.
    pub total_outbound: f64,
    /// Net flow (inbound - outbound).
    pub net_flow: f64,
    /// Inbound transaction count.
    pub inbound_count: usize,
    /// Outbound transaction count.
    pub outbound_count: usize,
    /// Unique sources.
    pub unique_sources: usize,
    /// Unique destinations.
    pub unique_destinations: usize,
    /// Breakdown by payment type.
    pub payment_type_breakdown: HashMap<PaymentType, (usize, f64)>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_payment(
        id: &str,
        payer: &str,
        payee: &str,
        amount: f64,
        payment_type: PaymentType,
    ) -> Payment {
        Payment {
            id: id.to_string(),
            payer_account: payer.to_string(),
            payee_account: payee.to_string(),
            amount,
            currency: "USD".to_string(),
            payment_type,
            status: PaymentStatus::Completed,
            initiated_at: 1000,
            completed_at: Some(1001),
            reference: format!("REF-{}", id),
            priority: PaymentPriority::Normal,
            attributes: HashMap::new(),
        }
    }

    fn create_test_payments() -> Vec<Payment> {
        vec![
            create_test_payment("P001", "A", "B", 1000.0, PaymentType::ACH),
            create_test_payment("P002", "A", "B", 500.0, PaymentType::ACH),
            create_test_payment("P003", "B", "C", 800.0, PaymentType::Wire),
            create_test_payment("P004", "C", "A", 600.0, PaymentType::RealTime),
            create_test_payment("P005", "A", "C", 300.0, PaymentType::ACH),
        ]
    }

    #[test]
    fn test_build_flows() {
        let payments = create_test_payments();

        let flows = FlowAnalysis::build_flows(&payments);

        assert_eq!(flows.len(), 4); // A->B, B->C, C->A, A->C

        let ab_flow = flows.iter().find(|f| f.source == "A" && f.target == "B").unwrap();
        assert_eq!(ab_flow.volume, 1500.0);
        assert_eq!(ab_flow.count, 2);
        assert_eq!(ab_flow.avg_amount, 750.0);
    }

    #[test]
    fn test_calculate_node_metrics() {
        let payments = create_test_payments();
        let flows = FlowAnalysis::build_flows(&payments);

        let metrics = FlowAnalysis::calculate_node_metrics(&flows, &payments);

        assert_eq!(metrics.len(), 3); // A, B, C

        let a_metrics = metrics.get("A").unwrap();
        assert_eq!(a_metrics.total_outflow, 1800.0); // 1000 + 500 + 300
        assert_eq!(a_metrics.total_inflow, 600.0); // From C
        assert_eq!(a_metrics.net_flow, -1200.0);
    }

    #[test]
    fn test_overall_metrics() {
        let payments = create_test_payments();
        let config = FlowAnalysisConfig::default();

        let result = FlowAnalysis::analyze(&payments, &config);

        assert_eq!(result.overall_metrics.total_transactions, 5);
        assert_eq!(result.overall_metrics.total_volume, 3200.0);
        assert_eq!(result.overall_metrics.unique_payers, 3);
        assert_eq!(result.overall_metrics.unique_payees, 3);
        assert_eq!(result.overall_metrics.avg_transaction_size, 640.0);
    }

    #[test]
    fn test_detect_circular_flows() {
        let payments = vec![
            create_test_payment("P001", "A", "B", 1000.0, PaymentType::ACH),
            create_test_payment("P002", "B", "A", 900.0, PaymentType::ACH),
        ];
        let config = FlowAnalysisConfig::default();

        let result = FlowAnalysis::analyze(&payments, &config);

        let circular = result
            .anomalies
            .iter()
            .filter(|a| a.anomaly_type == FlowAnomalyType::CircularFlow)
            .count();
        assert!(circular > 0);
    }

    #[test]
    fn test_detect_rapid_movement() {
        // Account B is a pass-through: receives from A and C, sends to D and E
        let payments = vec![
            create_test_payment("P001", "A", "B", 10000.0, PaymentType::ACH),
            create_test_payment("P002", "C", "B", 10000.0, PaymentType::ACH),
            create_test_payment("P003", "B", "D", 9500.0, PaymentType::Wire),
            create_test_payment("P004", "B", "E", 10000.0, PaymentType::Wire),
        ];
        let config = FlowAnalysisConfig::default();

        let result = FlowAnalysis::analyze(&payments, &config);

        let rapid = result
            .anomalies
            .iter()
            .filter(|a| a.anomaly_type == FlowAnomalyType::RapidMovement)
            .count();
        assert!(rapid > 0);
    }

    #[test]
    fn test_top_flows_by_volume() {
        let payments = create_test_payments();

        let top = FlowAnalysis::top_flows_by_volume(&payments, 2);

        assert_eq!(top.len(), 2);
        assert_eq!(top[0].volume, 1500.0); // A->B
    }

    #[test]
    fn test_top_flows_by_count() {
        let payments = create_test_payments();

        let top = FlowAnalysis::top_flows_by_count(&payments, 2);

        assert_eq!(top.len(), 2);
        assert_eq!(top[0].count, 2); // A->B has 2 transactions
    }

    #[test]
    fn test_analyze_account() {
        let payments = create_test_payments();

        let analysis = FlowAnalysis::analyze_account(&payments, "A");

        assert_eq!(analysis.account_id, "A");
        assert_eq!(analysis.total_outbound, 1800.0);
        assert_eq!(analysis.total_inbound, 600.0);
        assert_eq!(analysis.outbound_count, 3);
        assert_eq!(analysis.inbound_count, 1);
        assert_eq!(analysis.unique_destinations, 2); // B and C
        assert_eq!(analysis.unique_sources, 1); // C
    }

    #[test]
    fn test_network_density() {
        // Sparse network: 3 nodes but only 2 edges
        let payments = vec![
            create_test_payment("P001", "A", "B", 1000.0, PaymentType::ACH),
            create_test_payment("P002", "B", "C", 500.0, PaymentType::ACH),
        ];
        let config = FlowAnalysisConfig::default();

        let result = FlowAnalysis::analyze(&payments, &config);

        // 3 nodes, 2 actual edges, max possible = 3*2 = 6
        assert!((result.overall_metrics.network_density - 2.0 / 6.0).abs() < 0.01);
    }

    #[test]
    fn test_unusual_volume_detection() {
        // Many flows with similar volume, one massive outlier
        let payments = vec![
            create_test_payment("P001", "A", "B", 100.0, PaymentType::ACH),
            create_test_payment("P002", "B", "C", 100.0, PaymentType::ACH),
            create_test_payment("P003", "C", "D", 100.0, PaymentType::ACH),
            create_test_payment("P004", "D", "E", 100.0, PaymentType::ACH),
            create_test_payment("P005", "E", "F", 100.0, PaymentType::ACH),
            create_test_payment("P006", "F", "G", 100.0, PaymentType::ACH),
            create_test_payment("P007", "G", "H", 100.0, PaymentType::ACH),
            create_test_payment("P008", "H", "I", 100.0, PaymentType::ACH),
            create_test_payment("P009", "I", "J", 100.0, PaymentType::ACH),
            create_test_payment("P010", "J", "K", 100000.0, PaymentType::Wire), // Outlier
        ];
        let config = FlowAnalysisConfig {
            volume_threshold_std: 2.0,
            ..Default::default()
        };

        let result = FlowAnalysis::analyze(&payments, &config);

        let unusual_volume = result
            .anomalies
            .iter()
            .filter(|a| a.anomaly_type == FlowAnomalyType::UnusualVolume)
            .count();
        assert!(unusual_volume > 0);
    }

    #[test]
    fn test_payment_type_breakdown() {
        let payments = vec![
            create_test_payment("P001", "A", "B", 1000.0, PaymentType::ACH),
            create_test_payment("P002", "A", "C", 500.0, PaymentType::ACH),
            create_test_payment("P003", "D", "A", 800.0, PaymentType::Wire),
        ];

        let analysis = FlowAnalysis::analyze_account(&payments, "A");

        // A has 2 ACH outbound and 1 Wire inbound
        assert!(analysis.payment_type_breakdown.contains_key(&PaymentType::ACH));
        assert!(analysis.payment_type_breakdown.contains_key(&PaymentType::Wire));
        assert_eq!(analysis.payment_type_breakdown[&PaymentType::ACH].0, 2);
    }

    #[test]
    fn test_empty_payments() {
        let payments: Vec<Payment> = vec![];
        let config = FlowAnalysisConfig::default();

        let result = FlowAnalysis::analyze(&payments, &config);

        assert!(result.flows.is_empty());
        assert!(result.node_metrics.is_empty());
        assert_eq!(result.overall_metrics.total_transactions, 0);
        assert_eq!(result.overall_metrics.total_volume, 0.0);
    }

    #[test]
    fn test_no_anomaly_detection() {
        let payments = create_test_payments();
        let config = FlowAnalysisConfig {
            detect_anomalies: false,
            ..Default::default()
        };

        let result = FlowAnalysis::analyze(&payments, &config);

        assert!(result.anomalies.is_empty());
    }
}
