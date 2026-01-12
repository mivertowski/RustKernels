//! Audit types for financial audit kernels.

use std::collections::HashMap;

// ============================================================================
// Audit Record Types
// ============================================================================

/// Audit record for feature extraction.
#[derive(Debug, Clone)]
pub struct AuditRecord {
    /// Record ID.
    pub id: String,
    /// Record type.
    pub record_type: AuditRecordType,
    /// Entity ID.
    pub entity_id: String,
    /// Timestamp.
    pub timestamp: u64,
    /// Amount (if applicable).
    pub amount: Option<f64>,
    /// Currency (if applicable).
    pub currency: Option<String>,
    /// Account (if applicable).
    pub account: Option<String>,
    /// Counter-party (if applicable).
    pub counter_party: Option<String>,
    /// Category.
    pub category: String,
    /// Attributes.
    pub attributes: HashMap<String, String>,
}

/// Audit record type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuditRecordType {
    /// Journal entry.
    JournalEntry,
    /// Invoice.
    Invoice,
    /// Payment.
    Payment,
    /// Receipt.
    Receipt,
    /// Adjustment.
    Adjustment,
    /// Transfer.
    Transfer,
    /// Expense.
    Expense,
    /// Revenue.
    Revenue,
}

// ============================================================================
// Feature Extraction Types
// ============================================================================

/// Feature vector for an entity.
#[derive(Debug, Clone)]
pub struct EntityFeatureVector {
    /// Entity ID.
    pub entity_id: String,
    /// Feature values.
    pub features: Vec<f64>,
    /// Feature names.
    pub feature_names: Vec<String>,
    /// Metadata.
    pub metadata: HashMap<String, String>,
}

/// Feature extraction result.
#[derive(Debug, Clone)]
pub struct FeatureExtractionResult {
    /// Feature vectors by entity.
    pub entity_features: Vec<EntityFeatureVector>,
    /// Global statistics.
    pub global_stats: FeatureStats,
    /// Anomaly scores.
    pub anomaly_scores: HashMap<String, f64>,
}

/// Feature statistics.
#[derive(Debug, Clone)]
pub struct FeatureStats {
    /// Number of entities.
    pub entity_count: usize,
    /// Number of records processed.
    pub record_count: usize,
    /// Feature means.
    pub means: Vec<f64>,
    /// Feature standard deviations.
    pub std_devs: Vec<f64>,
    /// Feature names.
    pub feature_names: Vec<String>,
}

// ============================================================================
// Hypergraph Types
// ============================================================================

/// Hypergraph node.
#[derive(Debug, Clone)]
pub struct HypergraphNode {
    /// Node ID.
    pub id: String,
    /// Node type.
    pub node_type: NodeType,
    /// Attributes.
    pub attributes: HashMap<String, String>,
}

/// Node type in the hypergraph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeType {
    /// Entity (company, person).
    Entity,
    /// Account.
    Account,
    /// Transaction.
    Transaction,
    /// Document.
    Document,
    /// Category.
    Category,
    /// Time period.
    TimePeriod,
}

/// Hyperedge connecting multiple nodes.
#[derive(Debug, Clone)]
pub struct Hyperedge {
    /// Edge ID.
    pub id: String,
    /// Edge type.
    pub edge_type: HyperedgeType,
    /// Connected node IDs.
    pub nodes: Vec<String>,
    /// Weight.
    pub weight: f64,
    /// Timestamp (if applicable).
    pub timestamp: Option<u64>,
    /// Attributes.
    pub attributes: HashMap<String, String>,
}

/// Hyperedge type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HyperedgeType {
    /// Transaction linking entities.
    Transaction,
    /// Document reference.
    DocumentRef,
    /// Account relationship.
    AccountRel,
    /// Temporal co-occurrence.
    Temporal,
    /// Category membership.
    CategoryMembership,
    /// Approval chain.
    ApprovalChain,
}

/// Hypergraph structure.
#[derive(Debug, Clone)]
pub struct Hypergraph {
    /// Nodes.
    pub nodes: HashMap<String, HypergraphNode>,
    /// Hyperedges.
    pub edges: Vec<Hyperedge>,
    /// Node to edges mapping.
    pub node_edges: HashMap<String, Vec<String>>,
}

/// Hypergraph analysis result.
#[derive(Debug, Clone)]
pub struct HypergraphResult {
    /// Constructed hypergraph.
    pub hypergraph: Hypergraph,
    /// Node centrality scores.
    pub node_centrality: HashMap<String, f64>,
    /// Edge weights.
    pub edge_weights: HashMap<String, f64>,
    /// Detected patterns.
    pub patterns: Vec<HypergraphPattern>,
    /// Statistics.
    pub stats: HypergraphStats,
}

/// Detected pattern in hypergraph.
#[derive(Debug, Clone)]
pub struct HypergraphPattern {
    /// Pattern ID.
    pub id: String,
    /// Pattern type.
    pub pattern_type: PatternType,
    /// Involved nodes.
    pub nodes: Vec<String>,
    /// Involved edges.
    pub edges: Vec<String>,
    /// Confidence score.
    pub confidence: f64,
    /// Description.
    pub description: String,
}

/// Pattern type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternType {
    /// Circular transaction pattern.
    CircularTransaction,
    /// Unusual connection.
    UnusualConnection,
    /// High centrality hub.
    HighCentralityHub,
    /// Isolated component.
    IsolatedComponent,
    /// Dense subgraph.
    DenseSubgraph,
    /// Temporal anomaly.
    TemporalAnomaly,
}

/// Hypergraph statistics.
#[derive(Debug, Clone)]
pub struct HypergraphStats {
    /// Number of nodes.
    pub node_count: usize,
    /// Number of hyperedges.
    pub edge_count: usize,
    /// Average hyperedge size.
    pub avg_edge_size: f64,
    /// Average node degree.
    pub avg_node_degree: f64,
    /// Number of connected components.
    pub component_count: usize,
    /// Network density.
    pub density: f64,
}
