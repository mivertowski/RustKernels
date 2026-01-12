//! Hypergraph construction kernel for financial audit.
//!
//! Constructs multi-way relationship hypergraphs from audit records.

use crate::types::*;
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::{HashMap, HashSet, VecDeque};

// ============================================================================
// HypergraphConstruction Kernel
// ============================================================================

/// Hypergraph construction kernel for multi-way relationships.
///
/// Constructs hypergraphs where edges can connect more than two nodes,
/// enabling analysis of complex multi-party relationships in financial data.
#[derive(Debug, Clone)]
pub struct HypergraphConstruction {
    metadata: KernelMetadata,
}

impl Default for HypergraphConstruction {
    fn default() -> Self {
        Self::new()
    }
}

impl HypergraphConstruction {
    /// Create a new hypergraph construction kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("audit/hypergraph", Domain::FinancialAudit)
                .with_description("Multi-way relationship hypergraph")
                .with_throughput(10_000)
                .with_latency_us(500.0),
        }
    }

    /// Construct hypergraph from audit records.
    pub fn construct(
        records: &[AuditRecord],
        config: &HypergraphConfig,
    ) -> HypergraphResult {
        // Build nodes
        let nodes = Self::build_nodes(records, config);

        // Build hyperedges
        let edges = Self::build_edges(records, &nodes, config);

        // Build node-to-edge mapping
        let node_edges = Self::build_node_edge_mapping(&edges);

        let hypergraph = Hypergraph {
            nodes,
            edges,
            node_edges,
        };

        // Calculate centrality
        let node_centrality = Self::calculate_centrality(&hypergraph);

        // Calculate edge weights
        let edge_weights = Self::calculate_edge_weights(&hypergraph);

        // Detect patterns
        let patterns = if config.detect_patterns {
            Self::detect_patterns(&hypergraph, &node_centrality)
        } else {
            Vec::new()
        };

        // Calculate statistics
        let stats = Self::calculate_stats(&hypergraph);

        HypergraphResult {
            hypergraph,
            node_centrality,
            edge_weights,
            patterns,
            stats,
        }
    }

    /// Build nodes from audit records.
    fn build_nodes(
        records: &[AuditRecord],
        config: &HypergraphConfig,
    ) -> HashMap<String, HypergraphNode> {
        let mut nodes = HashMap::new();

        for record in records {
            // Entity node
            if config.include_entity_nodes {
                let entity_id = format!("entity:{}", record.entity_id);
                nodes.entry(entity_id.clone()).or_insert_with(|| HypergraphNode {
                    id: entity_id,
                    node_type: NodeType::Entity,
                    attributes: HashMap::new(),
                });
            }

            // Account node
            if config.include_account_nodes {
                if let Some(account) = &record.account {
                    let account_id = format!("account:{}", account);
                    nodes.entry(account_id.clone()).or_insert_with(|| HypergraphNode {
                        id: account_id,
                        node_type: NodeType::Account,
                        attributes: HashMap::new(),
                    });
                }
            }

            // Transaction node (optional - each record as a node)
            if config.include_transaction_nodes {
                let tx_id = format!("tx:{}", record.id);
                nodes.entry(tx_id.clone()).or_insert_with(|| HypergraphNode {
                    id: tx_id,
                    node_type: NodeType::Transaction,
                    attributes: HashMap::new(),
                });
            }

            // Category node
            if config.include_category_nodes {
                let cat_id = format!("category:{}", record.category);
                nodes.entry(cat_id.clone()).or_insert_with(|| HypergraphNode {
                    id: cat_id,
                    node_type: NodeType::Category,
                    attributes: HashMap::new(),
                });
            }

            // Time period node (bucket by day)
            if config.include_time_nodes {
                let day = record.timestamp / 86400;
                let time_id = format!("day:{}", day);
                nodes.entry(time_id.clone()).or_insert_with(|| HypergraphNode {
                    id: time_id,
                    node_type: NodeType::TimePeriod,
                    attributes: HashMap::new(),
                });
            }
        }

        nodes
    }

    /// Build hyperedges from records.
    fn build_edges(
        records: &[AuditRecord],
        nodes: &HashMap<String, HypergraphNode>,
        config: &HypergraphConfig,
    ) -> Vec<Hyperedge> {
        let mut edges = Vec::new();
        let mut edge_id = 0;

        for record in records {
            let mut connected_nodes = Vec::new();

            // Collect all nodes related to this record
            if config.include_entity_nodes {
                let entity_id = format!("entity:{}", record.entity_id);
                if nodes.contains_key(&entity_id) {
                    connected_nodes.push(entity_id);
                }
            }

            if config.include_account_nodes {
                if let Some(account) = &record.account {
                    let account_id = format!("account:{}", account);
                    if nodes.contains_key(&account_id) {
                        connected_nodes.push(account_id);
                    }
                }
            }

            if config.include_transaction_nodes {
                let tx_id = format!("tx:{}", record.id);
                if nodes.contains_key(&tx_id) {
                    connected_nodes.push(tx_id);
                }
            }

            if config.include_category_nodes {
                let cat_id = format!("category:{}", record.category);
                if nodes.contains_key(&cat_id) {
                    connected_nodes.push(cat_id);
                }
            }

            if config.include_time_nodes {
                let day = record.timestamp / 86400;
                let time_id = format!("day:{}", day);
                if nodes.contains_key(&time_id) {
                    connected_nodes.push(time_id);
                }
            }

            // Create hyperedge if we have multiple nodes
            if connected_nodes.len() >= config.min_edge_size {
                edge_id += 1;
                edges.push(Hyperedge {
                    id: format!("edge:{}", edge_id),
                    edge_type: Self::determine_edge_type(record),
                    nodes: connected_nodes,
                    weight: record.amount.unwrap_or(1.0),
                    timestamp: Some(record.timestamp),
                    attributes: HashMap::new(),
                });
            }
        }

        edges
    }

    /// Determine hyperedge type based on record.
    fn determine_edge_type(record: &AuditRecord) -> HyperedgeType {
        match record.record_type {
            AuditRecordType::Payment | AuditRecordType::Transfer => HyperedgeType::Transaction,
            AuditRecordType::Invoice | AuditRecordType::Receipt => HyperedgeType::DocumentRef,
            AuditRecordType::JournalEntry | AuditRecordType::Adjustment => HyperedgeType::AccountRel,
            AuditRecordType::Expense | AuditRecordType::Revenue => HyperedgeType::CategoryMembership,
        }
    }

    /// Build node to edge mapping.
    fn build_node_edge_mapping(edges: &[Hyperedge]) -> HashMap<String, Vec<String>> {
        let mut mapping: HashMap<String, Vec<String>> = HashMap::new();

        for edge in edges {
            for node_id in &edge.nodes {
                mapping
                    .entry(node_id.clone())
                    .or_default()
                    .push(edge.id.clone());
            }
        }

        mapping
    }

    /// Calculate node centrality.
    fn calculate_centrality(hypergraph: &Hypergraph) -> HashMap<String, f64> {
        let mut centrality = HashMap::new();

        // Degree centrality based on number of edges containing each node
        let total_edges = hypergraph.edges.len() as f64;

        for (node_id, _) in &hypergraph.nodes {
            let edge_count = hypergraph.node_edges
                .get(node_id)
                .map(|e| e.len())
                .unwrap_or(0) as f64;

            let degree_centrality = if total_edges > 0.0 {
                edge_count / total_edges
            } else {
                0.0
            };

            centrality.insert(node_id.clone(), degree_centrality);
        }

        centrality
    }

    /// Calculate edge weights (normalized).
    fn calculate_edge_weights(hypergraph: &Hypergraph) -> HashMap<String, f64> {
        let max_weight = hypergraph.edges.iter()
            .map(|e| e.weight)
            .fold(0.0, f64::max);

        hypergraph.edges.iter()
            .map(|e| {
                let normalized = if max_weight > 0.0 {
                    e.weight / max_weight
                } else {
                    0.0
                };
                (e.id.clone(), normalized)
            })
            .collect()
    }

    /// Detect patterns in hypergraph.
    fn detect_patterns(
        hypergraph: &Hypergraph,
        centrality: &HashMap<String, f64>,
    ) -> Vec<HypergraphPattern> {
        let mut patterns = Vec::new();
        let mut pattern_id = 0;

        // Detect high centrality hubs
        let avg_centrality: f64 = centrality.values().sum::<f64>() / centrality.len().max(1) as f64;
        let std_centrality = Self::std_dev(&centrality.values().cloned().collect::<Vec<_>>());

        for (node_id, cent) in centrality {
            if *cent > avg_centrality + 2.0 * std_centrality {
                pattern_id += 1;
                patterns.push(HypergraphPattern {
                    id: format!("pattern:{}", pattern_id),
                    pattern_type: PatternType::HighCentralityHub,
                    nodes: vec![node_id.clone()],
                    edges: hypergraph.node_edges
                        .get(node_id)
                        .cloned()
                        .unwrap_or_default(),
                    confidence: (*cent - avg_centrality) / std_centrality / 3.0,
                    description: format!(
                        "High centrality hub detected: {} (centrality: {:.3})",
                        node_id, cent
                    ),
                });
            }
        }

        // Detect isolated components
        let components = Self::find_connected_components(hypergraph);
        if components.len() > 1 {
            for (i, component) in components.iter().enumerate() {
                if component.len() < hypergraph.nodes.len() / 10 {
                    pattern_id += 1;
                    patterns.push(HypergraphPattern {
                        id: format!("pattern:{}", pattern_id),
                        pattern_type: PatternType::IsolatedComponent,
                        nodes: component.clone(),
                        edges: Vec::new(),
                        confidence: 0.8,
                        description: format!(
                            "Isolated component {} with {} nodes",
                            i + 1, component.len()
                        ),
                    });
                }
            }
        }

        // Detect dense subgraphs (nodes with many shared edges)
        let dense = Self::find_dense_subgraphs(hypergraph);
        for (nodes, density) in dense {
            if density > 0.5 {
                pattern_id += 1;
                let related_edges: Vec<String> = nodes.iter()
                    .flat_map(|n| hypergraph.node_edges.get(n).cloned().unwrap_or_default())
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect();

                patterns.push(HypergraphPattern {
                    id: format!("pattern:{}", pattern_id),
                    pattern_type: PatternType::DenseSubgraph,
                    nodes,
                    edges: related_edges,
                    confidence: density,
                    description: format!("Dense subgraph with density {:.2}", density),
                });
            }
        }

        patterns
    }

    /// Find connected components using BFS.
    fn find_connected_components(hypergraph: &Hypergraph) -> Vec<Vec<String>> {
        let mut components = Vec::new();
        let mut visited: HashSet<&String> = HashSet::new();

        for node_id in hypergraph.nodes.keys() {
            if visited.contains(node_id) {
                continue;
            }

            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(node_id);

            while let Some(current) = queue.pop_front() {
                if visited.contains(current) {
                    continue;
                }
                visited.insert(current);
                component.push(current.clone());

                // Find neighbors through shared edges
                if let Some(edges) = hypergraph.node_edges.get(current) {
                    for edge_id in edges {
                        if let Some(edge) = hypergraph.edges.iter().find(|e| &e.id == edge_id) {
                            for neighbor in &edge.nodes {
                                if !visited.contains(neighbor) {
                                    queue.push_back(neighbor);
                                }
                            }
                        }
                    }
                }
            }

            if !component.is_empty() {
                components.push(component);
            }
        }

        components
    }

    /// Find dense subgraphs.
    fn find_dense_subgraphs(hypergraph: &Hypergraph) -> Vec<(Vec<String>, f64)> {
        let mut dense_subgraphs = Vec::new();

        // For each node type, check density
        let entity_nodes: Vec<&String> = hypergraph.nodes.iter()
            .filter(|(_, n)| n.node_type == NodeType::Entity)
            .map(|(id, _)| id)
            .collect();

        if entity_nodes.len() >= 3 {
            // Check pairwise edge sharing
            let mut shared_counts: HashMap<(&String, &String), usize> = HashMap::new();

            for node in &entity_nodes {
                if let Some(edges) = hypergraph.node_edges.get(*node) {
                    let edge_set: HashSet<&String> = edges.iter().collect();

                    for other in &entity_nodes {
                        if node >= other {
                            continue;
                        }
                        if let Some(other_edges) = hypergraph.node_edges.get(*other) {
                            let other_set: HashSet<&String> = other_edges.iter().collect();
                            let shared = edge_set.intersection(&other_set).count();
                            if shared > 0 {
                                shared_counts.insert((node, other), shared);
                            }
                        }
                    }
                }
            }

            // Group highly connected nodes
            let avg_shared = if !shared_counts.is_empty() {
                shared_counts.values().sum::<usize>() as f64 / shared_counts.len() as f64
            } else {
                0.0
            };

            if avg_shared > 1.0 {
                let high_shared: Vec<_> = shared_counts.iter()
                    .filter(|&(_, &count)| count as f64 > avg_shared * 1.5)
                    .collect();

                if !high_shared.is_empty() {
                    let mut nodes_in_dense: HashSet<String> = HashSet::new();
                    for ((a, b), _) in high_shared {
                        nodes_in_dense.insert((*a).clone());
                        nodes_in_dense.insert((*b).clone());
                    }

                    let nodes: Vec<String> = nodes_in_dense.into_iter().collect();
                    let max_edges = nodes.len() * (nodes.len() - 1) / 2;
                    let actual_edges = shared_counts.iter()
                        .filter(|((a, b), _)| nodes.contains(*a) && nodes.contains(*b))
                        .count();
                    let density = if max_edges > 0 {
                        actual_edges as f64 / max_edges as f64
                    } else {
                        0.0
                    };

                    dense_subgraphs.push((nodes, density));
                }
            }
        }

        dense_subgraphs
    }

    /// Calculate hypergraph statistics.
    fn calculate_stats(hypergraph: &Hypergraph) -> HypergraphStats {
        let node_count = hypergraph.nodes.len();
        let edge_count = hypergraph.edges.len();

        let avg_edge_size = if edge_count > 0 {
            hypergraph.edges.iter()
                .map(|e| e.nodes.len() as f64)
                .sum::<f64>() / edge_count as f64
        } else {
            0.0
        };

        let avg_node_degree = if node_count > 0 {
            hypergraph.node_edges.values()
                .map(|e| e.len() as f64)
                .sum::<f64>() / node_count as f64
        } else {
            0.0
        };

        let components = Self::find_connected_components(hypergraph);
        let component_count = components.len();

        // Density for hypergraph: actual edges / possible edges
        let max_possible_edges = if node_count > 1 {
            (2_usize.pow(node_count as u32) - node_count - 1) as f64
        } else {
            1.0
        };
        let density = (edge_count as f64 / max_possible_edges).min(1.0);

        HypergraphStats {
            node_count,
            edge_count,
            avg_edge_size,
            avg_node_degree,
            component_count,
            density,
        }
    }

    /// Calculate standard deviation.
    fn std_dev(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    /// Get nodes by type.
    pub fn get_nodes_by_type(result: &HypergraphResult, node_type: NodeType) -> Vec<&HypergraphNode> {
        result.hypergraph.nodes.values()
            .filter(|n| n.node_type == node_type)
            .collect()
    }

    /// Get edges containing a node.
    pub fn get_node_edges<'a>(result: &'a HypergraphResult, node_id: &str) -> Vec<&'a Hyperedge> {
        result.hypergraph.node_edges
            .get(node_id)
            .map(|edge_ids| {
                result.hypergraph.edges.iter()
                    .filter(|e| edge_ids.contains(&e.id))
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl GpuKernel for HypergraphConstruction {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Configuration Types
// ============================================================================

/// Hypergraph construction configuration.
#[derive(Debug, Clone)]
pub struct HypergraphConfig {
    /// Include entity nodes.
    pub include_entity_nodes: bool,
    /// Include account nodes.
    pub include_account_nodes: bool,
    /// Include transaction nodes.
    pub include_transaction_nodes: bool,
    /// Include category nodes.
    pub include_category_nodes: bool,
    /// Include time period nodes.
    pub include_time_nodes: bool,
    /// Minimum edge size.
    pub min_edge_size: usize,
    /// Detect patterns.
    pub detect_patterns: bool,
}

impl Default for HypergraphConfig {
    fn default() -> Self {
        Self {
            include_entity_nodes: true,
            include_account_nodes: true,
            include_transaction_nodes: false,
            include_category_nodes: true,
            include_time_nodes: false,
            min_edge_size: 2,
            detect_patterns: true,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_record(
        id: &str,
        entity_id: &str,
        record_type: AuditRecordType,
        amount: f64,
        timestamp: u64,
        category: &str,
    ) -> AuditRecord {
        AuditRecord {
            id: id.to_string(),
            record_type,
            entity_id: entity_id.to_string(),
            timestamp,
            amount: Some(amount),
            currency: Some("USD".to_string()),
            account: Some(format!("ACC-{}", entity_id)),
            counter_party: Some("CP001".to_string()),
            category: category.to_string(),
            attributes: HashMap::new(),
        }
    }

    fn create_test_records() -> Vec<AuditRecord> {
        vec![
            create_test_record("R001", "E001", AuditRecordType::Payment, 1000.0, 1000000, "Operating"),
            create_test_record("R002", "E001", AuditRecordType::Invoice, 1500.0, 1000100, "Operating"),
            create_test_record("R003", "E002", AuditRecordType::Payment, 500.0, 1000200, "Operating"),
            create_test_record("R004", "E002", AuditRecordType::Revenue, 10000.0, 1000300, "Sales"),
            create_test_record("R005", "E003", AuditRecordType::Expense, 3000.0, 1000400, "Operating"),
        ]
    }

    #[test]
    fn test_construct_hypergraph() {
        let records = create_test_records();
        let config = HypergraphConfig::default();

        let result = HypergraphConstruction::construct(&records, &config);

        assert!(!result.hypergraph.nodes.is_empty());
        assert!(!result.hypergraph.edges.is_empty());
    }

    #[test]
    fn test_node_types() {
        let records = create_test_records();
        let config = HypergraphConfig::default();

        let result = HypergraphConstruction::construct(&records, &config);

        let entity_nodes = HypergraphConstruction::get_nodes_by_type(&result, NodeType::Entity);
        assert_eq!(entity_nodes.len(), 3); // E001, E002, E003

        let category_nodes = HypergraphConstruction::get_nodes_by_type(&result, NodeType::Category);
        assert_eq!(category_nodes.len(), 2); // Operating, Sales
    }

    #[test]
    fn test_hyperedges() {
        let records = create_test_records();
        let config = HypergraphConfig::default();

        let result = HypergraphConstruction::construct(&records, &config);

        // Each record should create one hyperedge
        assert_eq!(result.hypergraph.edges.len(), 5);

        // Check edge connects multiple nodes
        for edge in &result.hypergraph.edges {
            assert!(edge.nodes.len() >= config.min_edge_size);
        }
    }

    #[test]
    fn test_node_centrality() {
        let records = create_test_records();
        let config = HypergraphConfig::default();

        let result = HypergraphConstruction::construct(&records, &config);

        // Operating category should have high centrality (appears in 4 records)
        let operating_cent = result.node_centrality.get("category:Operating").unwrap();
        let sales_cent = result.node_centrality.get("category:Sales").unwrap();
        assert!(operating_cent > sales_cent);
    }

    #[test]
    fn test_statistics() {
        let records = create_test_records();
        let config = HypergraphConfig::default();

        let result = HypergraphConstruction::construct(&records, &config);

        assert!(result.stats.node_count > 0);
        assert!(result.stats.edge_count > 0);
        assert!(result.stats.avg_edge_size > 0.0);
    }

    #[test]
    fn test_empty_records() {
        let records: Vec<AuditRecord> = vec![];
        let config = HypergraphConfig::default();

        let result = HypergraphConstruction::construct(&records, &config);

        assert!(result.hypergraph.nodes.is_empty());
        assert!(result.hypergraph.edges.is_empty());
    }

    #[test]
    fn test_selective_nodes() {
        let records = create_test_records();
        let config = HypergraphConfig {
            include_entity_nodes: true,
            include_account_nodes: false,
            include_transaction_nodes: false,
            include_category_nodes: false,
            include_time_nodes: false,
            min_edge_size: 1,
            detect_patterns: false,
        };

        let result = HypergraphConstruction::construct(&records, &config);

        // Should only have entity nodes
        for (_, node) in &result.hypergraph.nodes {
            assert_eq!(node.node_type, NodeType::Entity);
        }
    }

    #[test]
    fn test_get_node_edges() {
        let records = create_test_records();
        let config = HypergraphConfig::default();

        let result = HypergraphConstruction::construct(&records, &config);

        let e001_edges = HypergraphConstruction::get_node_edges(&result, "entity:E001");
        assert_eq!(e001_edges.len(), 2); // E001 has 2 records
    }

    #[test]
    fn test_connected_components() {
        let records = create_test_records();
        let config = HypergraphConfig::default();

        let result = HypergraphConstruction::construct(&records, &config);

        // All records share "Operating" category, so should be connected
        assert_eq!(result.stats.component_count, 1);
    }

    #[test]
    fn test_edge_weights() {
        let records = create_test_records();
        let config = HypergraphConfig::default();

        let result = HypergraphConstruction::construct(&records, &config);

        // Weights should be normalized 0-1
        for (_, weight) in &result.edge_weights {
            assert!(*weight >= 0.0 && *weight <= 1.0);
        }

        // At least one edge should have weight 1.0 (the max)
        assert!(result.edge_weights.values().any(|w| (*w - 1.0).abs() < 0.01));
    }

    #[test]
    fn test_min_edge_size() {
        let records = create_test_records();
        let config = HypergraphConfig {
            min_edge_size: 5, // Require 5 nodes per edge
            ..Default::default()
        };

        let result = HypergraphConstruction::construct(&records, &config);

        // With min_edge_size=5, most edges won't qualify
        for edge in &result.hypergraph.edges {
            assert!(edge.nodes.len() >= 5);
        }
    }
}
