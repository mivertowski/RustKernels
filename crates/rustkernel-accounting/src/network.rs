//! Intercompany network analysis kernel.
//!
//! This module provides intercompany network analysis for accounting:
//! - Analyze intercompany relationships
//! - Detect circular references
//! - Generate elimination entries

use crate::types::{
    CircularReference, EliminationEntry, EntityBalance, EntityRelationship, IntercompanyStatus,
    IntercompanyTransaction, IntercompanyType, NetworkAnalysisResult, NetworkStats,
};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Network Analysis Kernel
// ============================================================================

/// Intercompany network analysis kernel.
///
/// Analyzes intercompany transactions and relationships.
#[derive(Debug, Clone)]
pub struct NetworkAnalysis {
    metadata: KernelMetadata,
}

impl Default for NetworkAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkAnalysis {
    /// Create a new network analysis kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("accounting/network-analysis", Domain::Accounting)
                .with_description("Intercompany network analysis")
                .with_throughput(10_000)
                .with_latency_us(200.0),
        }
    }

    /// Analyze intercompany network.
    pub fn analyze(
        transactions: &[IntercompanyTransaction],
        config: &NetworkConfig,
    ) -> NetworkAnalysisResult {
        // Calculate entity balances
        let entity_balances = Self::calculate_entity_balances(transactions);

        // Calculate relationships
        let relationships = Self::calculate_relationships(transactions);

        // Find circular references
        let circular_refs = Self::find_circular_references(transactions, config);

        // Generate elimination entries
        let elimination_entries = Self::generate_eliminations(transactions, config);

        let entities: HashSet<_> = transactions
            .iter()
            .flat_map(|t| [t.from_entity.clone(), t.to_entity.clone()])
            .collect();

        let total_volume: f64 = transactions.iter().map(|t| t.amount).sum();

        NetworkAnalysisResult {
            entity_balances,
            relationships,
            circular_refs: circular_refs.clone(),
            elimination_entries: elimination_entries.clone(),
            stats: NetworkStats {
                total_entities: entities.len(),
                total_transactions: transactions.len(),
                total_volume,
                circular_count: circular_refs.len(),
                elimination_count: elimination_entries.len(),
            },
        }
    }

    /// Calculate entity balances.
    fn calculate_entity_balances(
        transactions: &[IntercompanyTransaction],
    ) -> HashMap<String, EntityBalance> {
        let mut balances: HashMap<String, EntityBalance> = HashMap::new();

        for txn in transactions {
            if txn.status == IntercompanyStatus::Eliminated {
                continue;
            }

            // From entity has a receivable
            let from_balance =
                balances
                    .entry(txn.from_entity.clone())
                    .or_insert_with(|| EntityBalance {
                        entity_id: txn.from_entity.clone(),
                        total_receivables: 0.0,
                        total_payables: 0.0,
                        net_position: 0.0,
                        counterparty_count: 0,
                    });
            from_balance.total_receivables += txn.amount;

            // To entity has a payable
            let to_balance =
                balances
                    .entry(txn.to_entity.clone())
                    .or_insert_with(|| EntityBalance {
                        entity_id: txn.to_entity.clone(),
                        total_receivables: 0.0,
                        total_payables: 0.0,
                        net_position: 0.0,
                        counterparty_count: 0,
                    });
            to_balance.total_payables += txn.amount;
        }

        // Calculate net positions and counterparty counts
        let counterparty_counts = Self::count_counterparties(transactions);
        for (entity_id, balance) in &mut balances {
            balance.net_position = balance.total_receivables - balance.total_payables;
            balance.counterparty_count = counterparty_counts.get(entity_id).copied().unwrap_or(0);
        }

        balances
    }

    /// Count counterparties for each entity.
    fn count_counterparties(transactions: &[IntercompanyTransaction]) -> HashMap<String, usize> {
        let mut counterparties: HashMap<String, HashSet<String>> = HashMap::new();

        for txn in transactions {
            counterparties
                .entry(txn.from_entity.clone())
                .or_default()
                .insert(txn.to_entity.clone());
            counterparties
                .entry(txn.to_entity.clone())
                .or_default()
                .insert(txn.from_entity.clone());
        }

        counterparties
            .into_iter()
            .map(|(k, v)| (k, v.len()))
            .collect()
    }

    /// Calculate entity relationships.
    fn calculate_relationships(
        transactions: &[IntercompanyTransaction],
    ) -> Vec<EntityRelationship> {
        let mut relationships: HashMap<(String, String), EntityRelationship> = HashMap::new();

        for txn in transactions {
            if txn.status == IntercompanyStatus::Eliminated {
                continue;
            }

            let key = if txn.from_entity < txn.to_entity {
                (txn.from_entity.clone(), txn.to_entity.clone())
            } else {
                (txn.to_entity.clone(), txn.from_entity.clone())
            };

            let rel = relationships
                .entry(key.clone())
                .or_insert_with(|| EntityRelationship {
                    from_entity: key.0.clone(),
                    to_entity: key.1.clone(),
                    total_volume: 0.0,
                    transaction_count: 0,
                    net_balance: 0.0,
                });

            rel.total_volume += txn.amount;
            rel.transaction_count += 1;

            // Adjust net balance based on direction
            if txn.from_entity == key.0 {
                rel.net_balance += txn.amount;
            } else {
                rel.net_balance -= txn.amount;
            }
        }

        relationships.into_values().collect()
    }

    /// Find circular references in the transaction network.
    fn find_circular_references(
        transactions: &[IntercompanyTransaction],
        config: &NetworkConfig,
    ) -> Vec<CircularReference> {
        let mut circular_refs = Vec::new();

        // Build adjacency list
        let mut graph: HashMap<String, Vec<(String, f64)>> = HashMap::new();
        for txn in transactions {
            if txn.status == IntercompanyStatus::Eliminated {
                continue;
            }
            graph
                .entry(txn.from_entity.clone())
                .or_default()
                .push((txn.to_entity.clone(), txn.amount));
        }

        // Find cycles using DFS
        let entities: HashSet<_> = graph.keys().cloned().collect();

        for start in &entities {
            let mut path = vec![start.clone()];
            let mut visited: HashSet<String> = HashSet::new();
            visited.insert(start.clone());

            Self::dfs_find_cycles(
                &graph,
                start,
                &mut path,
                &mut visited,
                &mut circular_refs,
                config.max_cycle_length,
            );
        }

        // Deduplicate cycles
        let mut seen: HashSet<Vec<String>> = HashSet::new();
        circular_refs.retain(|c| {
            let mut sorted = c.entities.clone();
            sorted.sort();
            seen.insert(sorted)
        });

        circular_refs
    }

    /// DFS to find cycles.
    fn dfs_find_cycles(
        graph: &HashMap<String, Vec<(String, f64)>>,
        current: &str,
        path: &mut Vec<String>,
        visited: &mut HashSet<String>,
        cycles: &mut Vec<CircularReference>,
        max_length: usize,
    ) {
        if path.len() > max_length {
            return;
        }

        if let Some(neighbors) = graph.get(current) {
            for (next, _amount) in neighbors {
                if *next == path[0] && path.len() >= 2 {
                    // Found a cycle
                    let total_amount: f64 = path
                        .windows(2)
                        .filter_map(|w| {
                            graph.get(&w[0]).and_then(|edges| {
                                edges
                                    .iter()
                                    .find(|(to, _)| to == &w[1])
                                    .map(|(_, amt)| *amt)
                            })
                        })
                        .sum();

                    cycles.push(CircularReference {
                        entities: path.clone(),
                        amount: total_amount,
                        consolidation_impact: total_amount * 0.5, // Simplified impact
                    });
                } else if !visited.contains(next) {
                    visited.insert(next.clone());
                    path.push(next.clone());
                    Self::dfs_find_cycles(graph, next, path, visited, cycles, max_length);
                    path.pop();
                    visited.remove(next);
                }
            }
        }
    }

    /// Generate elimination entries.
    fn generate_eliminations(
        transactions: &[IntercompanyTransaction],
        config: &NetworkConfig,
    ) -> Vec<EliminationEntry> {
        let mut eliminations = Vec::new();
        let mut entry_id = 1;

        // Group transactions that need elimination
        for txn in transactions {
            if txn.status != IntercompanyStatus::Confirmed {
                continue;
            }

            if txn.amount < config.min_elimination_amount {
                continue;
            }

            let (debit_account, credit_account) =
                Self::get_elimination_accounts(&txn.transaction_type);

            eliminations.push(EliminationEntry {
                id: format!("ELIM{:05}", entry_id),
                from_entity: txn.from_entity.clone(),
                to_entity: txn.to_entity.clone(),
                debit_account,
                credit_account,
                amount: txn.amount,
                currency: txn.currency.clone(),
            });

            entry_id += 1;
        }

        eliminations
    }

    /// Get elimination accounts based on transaction type.
    fn get_elimination_accounts(txn_type: &IntercompanyType) -> (String, String) {
        match txn_type {
            IntercompanyType::Trade => ("IC_PAYABLES".to_string(), "IC_RECEIVABLES".to_string()),
            IntercompanyType::Loan => (
                "IC_LOAN_PAYABLE".to_string(),
                "IC_LOAN_RECEIVABLE".to_string(),
            ),
            IntercompanyType::Dividend => (
                "DIVIDEND_INCOME".to_string(),
                "DIVIDEND_EXPENSE".to_string(),
            ),
            IntercompanyType::ManagementFee => (
                "MGMT_FEE_INCOME".to_string(),
                "MGMT_FEE_EXPENSE".to_string(),
            ),
            IntercompanyType::Royalty => {
                ("ROYALTY_INCOME".to_string(), "ROYALTY_EXPENSE".to_string())
            }
            IntercompanyType::Other => (
                "IC_OTHER_PAYABLE".to_string(),
                "IC_OTHER_RECEIVABLE".to_string(),
            ),
        }
    }

    /// Calculate netting opportunities.
    pub fn calculate_netting(transactions: &[IntercompanyTransaction]) -> Vec<NettingOpportunity> {
        let mut opportunities = Vec::new();

        // Find bilateral netting opportunities
        let mut bilateral: HashMap<(String, String), (f64, f64)> = HashMap::new();

        for txn in transactions {
            if txn.status == IntercompanyStatus::Eliminated {
                continue;
            }

            let key = if txn.from_entity < txn.to_entity {
                (txn.from_entity.clone(), txn.to_entity.clone())
            } else {
                (txn.to_entity.clone(), txn.from_entity.clone())
            };

            let entry = bilateral.entry(key.clone()).or_insert((0.0, 0.0));
            if txn.from_entity == key.0 {
                entry.0 += txn.amount;
            } else {
                entry.1 += txn.amount;
            }
        }

        for ((from, to), (amount_forward, amount_backward)) in bilateral {
            if amount_forward > 0.0 && amount_backward > 0.0 {
                let net_amount = (amount_forward - amount_backward).abs();
                let gross_reduction = amount_forward.min(amount_backward) * 2.0;

                opportunities.push(NettingOpportunity {
                    entities: vec![from, to],
                    gross_amount: amount_forward + amount_backward,
                    net_amount,
                    reduction: gross_reduction,
                });
            }
        }

        opportunities
    }
}

impl GpuKernel for NetworkAnalysis {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Network analysis configuration.
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Maximum cycle length to detect.
    pub max_cycle_length: usize,
    /// Minimum amount for elimination.
    pub min_elimination_amount: f64,
    /// Include disputed transactions.
    pub include_disputed: bool,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            max_cycle_length: 5,
            min_elimination_amount: 0.0,
            include_disputed: false,
        }
    }
}

/// Netting opportunity.
#[derive(Debug, Clone)]
pub struct NettingOpportunity {
    /// Entities involved.
    pub entities: Vec<String>,
    /// Gross amount.
    pub gross_amount: f64,
    /// Net amount.
    pub net_amount: f64,
    /// Reduction achieved.
    pub reduction: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_transactions() -> Vec<IntercompanyTransaction> {
        vec![
            IntercompanyTransaction {
                id: "T1".to_string(),
                from_entity: "CORP_A".to_string(),
                to_entity: "CORP_B".to_string(),
                amount: 1000.0,
                currency: "USD".to_string(),
                date: 1700000000,
                transaction_type: IntercompanyType::Trade,
                status: IntercompanyStatus::Confirmed,
            },
            IntercompanyTransaction {
                id: "T2".to_string(),
                from_entity: "CORP_B".to_string(),
                to_entity: "CORP_C".to_string(),
                amount: 500.0,
                currency: "USD".to_string(),
                date: 1700000000,
                transaction_type: IntercompanyType::Trade,
                status: IntercompanyStatus::Confirmed,
            },
            IntercompanyTransaction {
                id: "T3".to_string(),
                from_entity: "CORP_B".to_string(),
                to_entity: "CORP_A".to_string(),
                amount: 300.0,
                currency: "USD".to_string(),
                date: 1700000000,
                transaction_type: IntercompanyType::ManagementFee,
                status: IntercompanyStatus::Confirmed,
            },
        ]
    }

    #[test]
    fn test_network_metadata() {
        let kernel = NetworkAnalysis::new();
        assert_eq!(kernel.metadata().id, "accounting/network-analysis");
        assert_eq!(kernel.metadata().domain, Domain::Accounting);
    }

    #[test]
    fn test_entity_balances() {
        let transactions = create_test_transactions();
        let config = NetworkConfig::default();

        let result = NetworkAnalysis::analyze(&transactions, &config);

        let corp_a = result.entity_balances.get("CORP_A").unwrap();
        assert_eq!(corp_a.total_receivables, 1000.0);
        assert_eq!(corp_a.total_payables, 300.0);
        assert_eq!(corp_a.net_position, 700.0);

        let corp_b = result.entity_balances.get("CORP_B").unwrap();
        assert_eq!(corp_b.total_receivables, 800.0); // 500 + 300
        assert_eq!(corp_b.total_payables, 1000.0);
    }

    #[test]
    fn test_relationships() {
        let transactions = create_test_transactions();
        let config = NetworkConfig::default();

        let result = NetworkAnalysis::analyze(&transactions, &config);

        assert!(result.relationships.len() >= 2);

        // Find A-B relationship
        let ab_rel = result.relationships.iter().find(|r| {
            (r.from_entity == "CORP_A" && r.to_entity == "CORP_B")
                || (r.from_entity == "CORP_B" && r.to_entity == "CORP_A")
        });
        assert!(ab_rel.is_some());

        let rel = ab_rel.unwrap();
        assert_eq!(rel.total_volume, 1300.0); // 1000 + 300
        assert_eq!(rel.transaction_count, 2);
    }

    #[test]
    fn test_circular_reference() {
        let transactions = vec![
            IntercompanyTransaction {
                id: "T1".to_string(),
                from_entity: "A".to_string(),
                to_entity: "B".to_string(),
                amount: 100.0,
                currency: "USD".to_string(),
                date: 1700000000,
                transaction_type: IntercompanyType::Trade,
                status: IntercompanyStatus::Confirmed,
            },
            IntercompanyTransaction {
                id: "T2".to_string(),
                from_entity: "B".to_string(),
                to_entity: "C".to_string(),
                amount: 100.0,
                currency: "USD".to_string(),
                date: 1700000000,
                transaction_type: IntercompanyType::Trade,
                status: IntercompanyStatus::Confirmed,
            },
            IntercompanyTransaction {
                id: "T3".to_string(),
                from_entity: "C".to_string(),
                to_entity: "A".to_string(),
                amount: 100.0,
                currency: "USD".to_string(),
                date: 1700000000,
                transaction_type: IntercompanyType::Trade,
                status: IntercompanyStatus::Confirmed,
            },
        ];

        let config = NetworkConfig::default();
        let result = NetworkAnalysis::analyze(&transactions, &config);

        assert!(!result.circular_refs.is_empty());
        assert_eq!(result.circular_refs[0].entities.len(), 3);
    }

    #[test]
    fn test_elimination_entries() {
        let transactions = create_test_transactions();
        let config = NetworkConfig::default();

        let result = NetworkAnalysis::analyze(&transactions, &config);

        assert!(!result.elimination_entries.is_empty());

        // Check trade elimination
        let trade_elim = result
            .elimination_entries
            .iter()
            .find(|e| e.from_entity == "CORP_A" && e.to_entity == "CORP_B");
        assert!(trade_elim.is_some());
    }

    #[test]
    fn test_netting_opportunities() {
        let transactions = create_test_transactions();

        let netting = NetworkAnalysis::calculate_netting(&transactions);

        // Should find A-B bilateral netting opportunity
        let ab_netting = netting.iter().find(|n| {
            n.entities.contains(&"CORP_A".to_string()) && n.entities.contains(&"CORP_B".to_string())
        });
        assert!(ab_netting.is_some());

        let opportunity = ab_netting.unwrap();
        assert_eq!(opportunity.gross_amount, 1300.0);
        assert_eq!(opportunity.net_amount, 700.0);
        assert_eq!(opportunity.reduction, 600.0); // 300 * 2
    }

    #[test]
    fn test_network_stats() {
        let transactions = create_test_transactions();
        let config = NetworkConfig::default();

        let result = NetworkAnalysis::analyze(&transactions, &config);

        assert_eq!(result.stats.total_transactions, 3);
        assert_eq!(result.stats.total_volume, 1800.0);
        assert_eq!(result.stats.total_entities, 3);
    }

    #[test]
    fn test_excluded_eliminated() {
        let mut transactions = create_test_transactions();
        transactions[0].status = IntercompanyStatus::Eliminated;

        let config = NetworkConfig::default();
        let result = NetworkAnalysis::analyze(&transactions, &config);

        // Eliminated transaction should not affect balances
        let corp_a = result.entity_balances.get("CORP_A").unwrap();
        assert_eq!(corp_a.total_receivables, 0.0);
        assert_eq!(corp_a.total_payables, 300.0);
    }

    #[test]
    fn test_min_elimination_amount() {
        let transactions = create_test_transactions();
        let config = NetworkConfig {
            min_elimination_amount: 400.0, // Should exclude 300 transaction
            ..Default::default()
        };

        let result = NetworkAnalysis::analyze(&transactions, &config);

        // Should only have eliminations for amounts >= 400
        assert!(result.elimination_entries.iter().all(|e| e.amount >= 400.0));
    }
}
