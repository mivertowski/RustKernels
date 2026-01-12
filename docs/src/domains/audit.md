# Audit

**Crate**: `rustkernel-audit`
**Kernels**: 2
**Feature**: `audit`

Financial audit and forensic analysis kernels.

## Kernel Overview

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| FeatureExtraction | `audit/feature-extraction` | Batch, Ring | Extract audit-relevant features |
| HypergraphConstruction | `audit/hypergraph-construction` | Batch | Build multi-entity relationship graphs |

---

## Kernel Details

### FeatureExtraction

Extracts features from financial data for audit analysis and anomaly detection.

**ID**: `audit/feature-extraction`
**Modes**: Batch, Ring

#### Input

```rust
pub struct FeatureExtractionInput {
    pub transactions: Vec<AuditTransaction>,
    pub feature_config: FeatureConfig,
    pub entity_context: Option<EntityContext>,
}

pub struct AuditTransaction {
    pub id: String,
    pub timestamp: u64,
    pub amount: f64,
    pub account_from: String,
    pub account_to: String,
    pub entity_id: String,
    pub user_id: String,
    pub transaction_type: String,
    pub attributes: HashMap<String, String>,
}

pub struct FeatureConfig {
    pub temporal_features: bool,
    pub behavioral_features: bool,
    pub network_features: bool,
    pub benford_analysis: bool,
}
```

#### Output

```rust
pub struct FeatureExtractionOutput {
    /// Extracted features per transaction
    pub features: Vec<TransactionFeatures>,
    /// Aggregate features
    pub aggregate_features: AggregateFeatures,
    /// Benford's Law analysis
    pub benford_results: Option<BenfordResults>,
}

pub struct TransactionFeatures {
    pub transaction_id: String,
    /// Temporal features
    pub hour_of_day: u8,
    pub day_of_week: u8,
    pub is_weekend: bool,
    pub is_month_end: bool,
    /// Amount features
    pub amount_log: f64,
    pub round_amount_flag: bool,
    pub just_below_threshold: bool,
    /// Behavioral features
    pub velocity_1h: u32,
    pub velocity_24h: u32,
    pub deviation_from_mean: f64,
}

pub struct BenfordResults {
    pub first_digit_distribution: Vec<f64>,
    pub expected_distribution: Vec<f64>,
    pub chi_square_statistic: f64,
    pub p_value: f64,
    pub conformity_score: f64,
}
```

#### Example

```rust
use rustkernel::audit::feature_extraction::{FeatureExtraction, FeatureExtractionInput};

let kernel = FeatureExtraction::new();

let result = kernel.execute(FeatureExtractionInput {
    transactions: journal_entries,
    feature_config: FeatureConfig {
        temporal_features: true,
        behavioral_features: true,
        network_features: true,
        benford_analysis: true,
    },
    entity_context: None,
}).await?;

// Check Benford's Law conformity
if let Some(benford) = result.benford_results {
    if benford.conformity_score < 0.8 {
        println!("Warning: Data may not conform to Benford's Law");
        println!("Conformity: {:.1}%", benford.conformity_score * 100.0);
    }
}

// Find suspicious transactions
for feat in result.features {
    if feat.just_below_threshold && feat.is_weekend {
        println!("Suspicious: {} - just below threshold on weekend",
            feat.transaction_id
        );
    }
}
```

---

### HypergraphConstruction

Builds hypergraphs representing complex multi-entity relationships.

**ID**: `audit/hypergraph-construction`
**Modes**: Batch

A hypergraph allows edges to connect more than two nodes, capturing complex relationships like:
- A transaction involving multiple parties
- A document signed by multiple entities
- An event affecting multiple accounts

#### Input

```rust
pub struct HypergraphInput {
    pub events: Vec<AuditEvent>,
    pub entity_types: Vec<EntityType>,
    pub relationship_rules: Vec<RelationshipRule>,
}

pub struct AuditEvent {
    pub id: String,
    pub event_type: String,
    pub entities: Vec<EntityReference>,
    pub timestamp: u64,
    pub attributes: HashMap<String, String>,
}

pub struct EntityReference {
    pub entity_id: String,
    pub entity_type: EntityType,
    pub role: String,  // e.g., "sender", "approver", "beneficiary"
}
```

#### Output

```rust
pub struct HypergraphOutput {
    pub nodes: Vec<HypergraphNode>,
    pub hyperedges: Vec<Hyperedge>,
    pub metrics: HypergraphMetrics,
}

pub struct Hyperedge {
    pub id: String,
    pub nodes: Vec<String>,
    pub edge_type: String,
    pub weight: f64,
    pub attributes: HashMap<String, String>,
}

pub struct HypergraphMetrics {
    pub node_count: usize,
    pub hyperedge_count: usize,
    pub avg_hyperedge_size: f64,
    pub max_hyperedge_size: usize,
    pub connected_components: usize,
}
```

#### Example

```rust
use rustkernel::audit::hypergraph::{HypergraphConstruction, HypergraphInput};

let kernel = HypergraphConstruction::new();

let result = kernel.execute(HypergraphInput {
    events: audit_events,
    entity_types: vec![
        EntityType::User,
        EntityType::Account,
        EntityType::Document,
    ],
    relationship_rules: default_rules(),
}).await?;

// Find densely connected entity clusters
for component in result.connected_components() {
    if component.density > 0.8 {
        println!("Highly connected cluster: {:?}", component.entities);
    }
}
```

---

## Use Cases

### Internal Audit

- Journal entry testing
- Segregation of duties analysis
- Unusual transaction detection

### External Audit

- Substantive testing sample selection
- Benford's Law analysis
- Related party transaction identification

### Fraud Investigation

- Network analysis of parties
- Pattern detection across time
- Relationship mapping

### Regulatory Compliance

- SOX testing automation
- Audit trail analysis
- Control effectiveness testing
