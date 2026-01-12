# Compliance

**Crate**: `rustkernel-compliance`
**Kernels**: 11
**Feature**: `compliance` (included in default features)

Anti-money laundering (AML), Know Your Customer (KYC), and regulatory compliance kernels.

## Kernel Overview

### AML Pattern Detection (6)

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| CircularFlowRatio | `compliance/circular-flow` | Batch, Ring | Detect circular fund flows |
| ReciprocityFlowRatio | `compliance/reciprocity-flow` | Batch, Ring | Identify reciprocal transactions |
| RapidMovement | `compliance/rapid-movement` | Batch, Ring | Flag rapid fund movements |
| AMLPatternDetection | `compliance/aml-pattern` | Batch, Ring | Combined AML scoring |
| FlowReversalPattern | `compliance/flow-reversal` | Batch | Transaction reversal detection (wash trading) |
| FlowSplitRatio | `compliance/flow-split` | Batch | Structuring/smurfing detection |

### KYC/Screening (4)

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| KYCScoring | `compliance/kyc-scoring` | Batch, Ring | Customer risk scoring |
| EntityResolution | `compliance/entity-resolution` | Batch | Match entities across records |
| SanctionsScreening | `compliance/sanctions-screening` | Batch, Ring | Check against sanctions lists |
| PEPScreening | `compliance/pep-screening` | Batch, Ring | Politically Exposed Person screening |

### Monitoring (1)

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| TransactionMonitoring | `compliance/transaction-monitoring` | Batch | Real-time transaction analysis |

---

## Kernel Details

### AMLPatternDetection

Comprehensive AML scoring combining multiple detection methods.

**ID**: `compliance/aml-pattern-detection`
**Modes**: Batch, Ring

#### Input

```rust
pub struct AMLPatternInput {
    /// Transaction graph edges (from_account, to_account, amount)
    pub transactions: Vec<(String, String, f64)>,
    /// Time window in seconds
    pub time_window: u64,
    /// Minimum amount threshold
    pub min_amount: f64,
    /// Detection thresholds
    pub thresholds: AMLThresholds,
}

pub struct AMLThresholds {
    pub circular_flow_threshold: f64,
    pub reciprocity_threshold: f64,
    pub rapid_movement_threshold: f64,
    pub structuring_threshold: f64,
}
```

#### Output

```rust
pub struct AMLPatternOutput {
    /// Overall risk scores per account
    pub risk_scores: HashMap<String, f64>,
    /// Detected patterns
    pub patterns: Vec<DetectedPattern>,
    /// High-risk accounts
    pub flagged_accounts: Vec<String>,
}

pub struct DetectedPattern {
    pub pattern_type: PatternType,
    pub accounts: Vec<String>,
    pub confidence: f64,
    pub description: String,
}
```

#### Example

```rust
use rustkernel::compliance::aml::{AMLPatternDetection, AMLPatternInput};

let kernel = AMLPatternDetection::new();

let input = AMLPatternInput {
    transactions: vec![
        ("A".into(), "B".into(), 9500.0),
        ("B".into(), "C".into(), 9400.0),
        ("C".into(), "A".into(), 9300.0),  // Circular flow
    ],
    time_window: 86400,  // 24 hours
    min_amount: 1000.0,
    thresholds: AMLThresholds::default(),
};

let result = kernel.execute(input).await?;

for pattern in result.patterns {
    println!("Detected: {:?} with confidence {:.2}",
        pattern.pattern_type,
        pattern.confidence
    );
}
```

---

### SanctionsScreening

Screens entities against sanctions and watchlists.

**ID**: `compliance/sanctions-screening`
**Modes**: Batch, Ring

#### Input

```rust
pub struct SanctionsScreeningInput {
    /// Entities to screen
    pub entities: Vec<EntityInfo>,
    /// Sanctions list identifier
    pub list_ids: Vec<String>,
    /// Fuzzy matching threshold (0.0-1.0)
    pub match_threshold: f64,
}

pub struct EntityInfo {
    pub name: String,
    pub aliases: Vec<String>,
    pub country: Option<String>,
    pub date_of_birth: Option<String>,
}
```

#### Output

```rust
pub struct SanctionsScreeningOutput {
    /// Matches found
    pub matches: Vec<SanctionsMatch>,
    /// Number of entities screened
    pub entities_screened: u32,
    /// Processing time
    pub processing_time_ms: u64,
}

pub struct SanctionsMatch {
    pub entity_index: u32,
    pub list_id: String,
    pub matched_entry: String,
    pub match_score: f64,
    pub match_type: MatchType,
}
```

---

### KYCScoring

Computes customer risk scores for KYC compliance.

**ID**: `compliance/kyc-scoring`
**Modes**: Batch, Ring

#### Example

```rust
use rustkernel::compliance::kyc::{KYCScoring, KYCInput};

let kernel = KYCScoring::new();

let result = kernel.execute(KYCInput {
    customer_id: "CUST001".into(),
    transaction_volume: 150000.0,
    transaction_count: 45,
    countries: vec!["US".into(), "UK".into()],
    account_age_days: 365,
    verification_level: VerificationLevel::Enhanced,
}).await?;

println!("Risk score: {:.2}", result.risk_score);
println!("Risk level: {:?}", result.risk_level);
```

---

### CircularFlowRatio

Detects circular transaction patterns indicative of money laundering.

**ID**: `compliance/circular-flow-ratio`
**Modes**: Batch, Ring

The kernel analyzes transaction graphs to find cycles where money flows back to its origin:

```
A → B → C → A  (circular flow detected)
```

#### Output

```rust
pub struct CircularFlowOutput {
    /// Detected circular flows
    pub circular_flows: Vec<CircularFlow>,
    /// Overall circular flow ratio
    pub overall_ratio: f64,
}

pub struct CircularFlow {
    /// Accounts in the circular path
    pub path: Vec<String>,
    /// Total amount circulated
    pub amount: f64,
    /// Time span of the cycle
    pub time_span_seconds: u64,
}
```

---

### FlowReversalPattern

Detects transaction reversals (A→B followed by B→A) that may indicate wash trading, round-tripping, or layering.

**ID**: `compliance/flow-reversal`
**Modes**: Batch
**Throughput**: ~80,000 transactions/sec

#### Configuration

```rust
pub struct FlowReversalConfig {
    /// Maximum time window to consider reversals (seconds)
    pub max_window_seconds: u64,        // default: 86400 (24 hours)
    /// Time threshold for suspicious reversals (seconds)
    pub suspicious_window_seconds: u64,  // default: 3600 (1 hour)
    /// Time threshold for critical reversals (seconds)
    pub critical_window_seconds: u64,    // default: 300 (5 minutes)
    /// Minimum amount match ratio (0-1)
    pub min_amount_match_ratio: f64,     // default: 0.9
}
```

#### Output

```rust
pub struct FlowReversalResult {
    /// Detected reversal pairs
    pub reversals: Vec<FlowReversalPair>,
    /// Total reversal volume
    pub reversal_volume: f64,
    /// Reversal ratio (reversal volume / total volume)
    pub reversal_ratio: f64,
    /// Entities with multiple reversals
    pub repeat_offenders: Vec<(u64, u32)>,
    /// Overall risk score (0-100)
    pub risk_score: f64,
}

pub struct FlowReversalPair {
    pub original_tx_id: u64,
    pub reversal_tx_id: u64,
    pub entity_a: u64,
    pub entity_b: u64,
    pub original_amount: f64,
    pub reversal_amount: f64,
    pub time_delta: u64,
    pub amount_match_ratio: f64,
    pub risk_level: ReversalRiskLevel,  // Normal, Suspicious, High, Critical
}
```

#### Example

```rust
use rustkernel::compliance::aml::{FlowReversalPattern, FlowReversalConfig};

let result = FlowReversalPattern::compute(&transactions, &FlowReversalConfig {
    max_window_seconds: 86400,
    suspicious_window_seconds: 3600,
    critical_window_seconds: 300,
    min_amount_match_ratio: 0.9,
});

// Find critical reversals
for reversal in &result.reversals {
    if matches!(reversal.risk_level, ReversalRiskLevel::Critical) {
        println!("CRITICAL: {} -> {} reversed in {}s",
            reversal.entity_a, reversal.entity_b, reversal.time_delta);
    }
}
```

---

### FlowSplitRatio

Detects structuring (smurfing) patterns where transactions are split to avoid reporting thresholds.

**ID**: `compliance/flow-split`
**Modes**: Batch
**Throughput**: ~60,000 transactions/sec

#### Configuration

```rust
pub struct FlowSplitConfig {
    /// Reporting threshold to detect structuring around (e.g., $10,000)
    pub reporting_threshold: f64,   // default: 10_000.0 (BSA threshold)
    /// Time window to look for split transactions (seconds)
    pub window_seconds: u64,        // default: 86400 (24 hours)
    /// Minimum number of transactions to constitute a split
    pub min_split_count: usize,     // default: 3
}
```

#### Output

```rust
pub struct FlowSplitResult {
    /// Detected split patterns
    pub splits: Vec<FlowSplitPattern>,
    /// Entities with structuring patterns
    pub structuring_entities: Vec<u64>,
    /// Total amount in split patterns
    pub split_volume: f64,
    /// Split ratio (split volume / total volume)
    pub split_ratio: f64,
    /// Overall risk score (0-100)
    pub risk_score: f64,
}

pub struct FlowSplitPattern {
    pub source_entity: u64,
    pub dest_entities: Vec<u64>,
    pub transaction_ids: Vec<u64>,
    pub amounts: Vec<f64>,
    pub total_amount: f64,
    pub time_span: u64,
    pub estimated_threshold: f64,
    pub risk_level: SplitRiskLevel,  // Normal, Elevated, High, Critical
}
```

#### Example

```rust
use rustkernel::compliance::aml::{FlowSplitRatio, FlowSplitConfig};

let result = FlowSplitRatio::compute(&transactions, &FlowSplitConfig {
    reporting_threshold: 10_000.0,  // BSA threshold
    window_seconds: 86400,
    min_split_count: 3,
});

// Find structuring entities
for entity in &result.structuring_entities {
    println!("STRUCTURING ALERT: Entity {} flagged", entity);
}

// Analyze high-risk splits
for split in result.splits.iter().filter(|s|
    matches!(s.risk_level, SplitRiskLevel::High | SplitRiskLevel::Critical)
) {
    println!("Split detected: {} transactions totaling ${:.2}",
        split.transaction_ids.len(), split.total_amount);
}
```

---

## Ring Mode for Real-Time Compliance

Ring mode enables real-time transaction screening:

```rust
use rustkernel::compliance::aml::AMLPatternRing;

let ring = AMLPatternRing::new();

// Process streaming transactions
for tx in transaction_stream {
    // Sub-millisecond screening
    let alert = ring.screen_transaction(tx).await?;

    if alert.risk_score > threshold {
        notify_compliance_team(alert);
    }
}
```

## Integration Patterns

### Batch Processing (Daily/Weekly)

```rust
// Load all transactions for period
let transactions = load_transactions(start_date, end_date)?;

// Run comprehensive AML analysis
let aml_result = aml_kernel.execute(AMLPatternInput {
    transactions,
    ..Default::default()
}).await?;

// Generate compliance report
generate_sar_report(aml_result)?;
```

### Real-Time Screening

```rust
// Screen each transaction before processing
async fn process_transaction(tx: Transaction) -> Result<()> {
    let screening = sanctions_ring.screen(&tx.counterparty).await?;

    if screening.is_match() {
        return Err(TransactionBlocked::Sanctions);
    }

    // Continue with transaction processing
    process_payment(tx).await
}
```

## Regulatory Alignment

These kernels support requirements from:

- **FATF**: Financial Action Task Force recommendations
- **BSA/AML**: Bank Secrecy Act
- **EU AMLD**: EU Anti-Money Laundering Directives
- **OFAC**: Office of Foreign Assets Control sanctions
