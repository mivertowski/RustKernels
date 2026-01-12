# Accounting

**Crate**: `rustkernel-accounting`
**Kernels**: 9
**Feature**: `accounting`

Accounting network generation, reconciliation, and analysis kernels for financial close and audit.

## Kernel Overview

### Core Kernels (7)

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| ChartOfAccountsMapping | `accounting/coa-mapping` | Batch | Map between chart of accounts |
| JournalTransformation | `accounting/journal-transformation` | Batch, Ring | Transform journal entries |
| GLReconciliation | `accounting/gl-reconciliation` | Batch, Ring | General ledger reconciliation |
| NetworkAnalysis | `accounting/network-analysis` | Batch, Ring | Intercompany network analysis |
| TemporalCorrelation | `accounting/temporal-correlation` | Batch | Account correlation over time |
| NetworkGeneration | `accounting/network-generation` | Batch | Generate accounting networks |
| NetworkGenerationRing | `accounting/network-generation-ring` | Ring | Streaming network generation |

### Detection Kernels (2)

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| SuspenseAccountDetection | `accounting/suspense-detection` | Batch | Centrality-based suspense account detection |
| GaapViolationDetection | `accounting/gaap-violation` | Batch | GAAP prohibited flow pattern detection |

---

## Kernel Details

### NetworkGeneration

Transforms double-entry journal entries into directed accounting flow networks.

**ID**: `accounting/network-generation`
**Modes**: Batch
**Feature Article**: [Accounting Network Generation](../articles/accounting-network-generation.md)

#### Input

```rust
pub struct NetworkGenerationInput {
    /// Journal entries to process
    pub entries: Vec<JournalEntry>,
    /// Configuration options
    pub config: Option<NetworkGenerationConfig>,
}

pub struct JournalEntry {
    pub id: u64,
    pub date: u64,
    pub posting_date: u64,
    pub document_number: String,
    pub lines: Vec<JournalLine>,
    pub status: JournalStatus,
    pub source_system: String,
    pub description: String,
}

pub struct JournalLine {
    pub line_number: u32,
    pub account_code: String,
    pub debit: f64,
    pub credit: f64,
    pub currency: String,
    pub entity_id: String,
    pub cost_center: Option<String>,
    pub description: String,
}
```

#### Output

```rust
pub struct NetworkGenerationOutput {
    /// Generated accounting flows
    pub flows: Vec<AccountingFlow>,
    /// Network statistics
    pub stats: NetworkGenerationStats,
}

pub struct AccountingFlow {
    pub flow_id: String,
    pub entry_id: u64,
    pub from_account: String,
    pub to_account: String,
    pub amount: f64,
    pub method: SolvingMethod,
    pub confidence: f64,
    pub from_entity: String,
    pub to_entity: String,
    /// Account classification
    pub from_account_class: Option<AccountClass>,
    pub to_account_class: Option<AccountClass>,
    /// Detected transaction pattern
    pub pattern: Option<TransactionPattern>,
    /// VAT/tax indicators
    pub is_tax_flow: bool,
    pub vat_rate: Option<f64>,
}
```

#### Solving Methods

The kernel uses five methods with decreasing confidence:

| Method | Confidence | Description |
|--------|------------|-------------|
| Method A | 1.00 | Trivial 1-to-1 mapping |
| Method B | 0.95 | n-to-n bijective matching |
| Method C | 0.85 | n-to-m partition matching |
| Method D | 0.70 | Account aggregation |
| Method E | 0.50 | Entity decomposition |

#### Example

```rust
use rustkernel::accounting::network_generation::{NetworkGeneration, NetworkGenerationInput};

let kernel = NetworkGeneration::new();

let result = kernel.execute(NetworkGenerationInput {
    entries: journal_entries,
    config: Some(NetworkGenerationConfig {
        enable_pattern_matching: true,
        enable_vat_detection: true,
        ..Default::default()
    }),
}).await?;

println!("Generated {} flows", result.flows.len());
println!("Weighted confidence: {:.2}", result.stats.weighted_confidence);

// Analyze flows by pattern
let sales = result.flows.iter()
    .filter(|f| matches!(f.pattern, Some(TransactionPattern::SaleWithVat)))
    .count();
println!("Sales transactions: {}", sales);
```

---

### GLReconciliation

Reconciles general ledger balances across systems.

**ID**: `accounting/gl-reconciliation`
**Modes**: Batch, Ring

#### Output

```rust
pub struct ReconciliationOutput {
    pub matched_pairs: Vec<MatchedPair>,
    pub unmatched: Vec<String>,
    pub exceptions: Vec<ReconciliationException>,
    pub stats: ReconciliationStats,
}

pub struct ReconciliationStats {
    pub total_items: usize,
    pub matched_count: usize,
    pub match_rate: f64,
    pub total_variance: f64,
}
```

---

### NetworkAnalysis

Analyzes intercompany transaction networks for consolidation.

**ID**: `accounting/network-analysis`
**Modes**: Batch, Ring

#### Example

```rust
use rustkernel::accounting::network::{NetworkAnalysis, NetworkAnalysisInput};

let kernel = NetworkAnalysis::new();

let result = kernel.execute(NetworkAnalysisInput {
    transactions: intercompany_transactions,
    entities: group_entities,
    analysis_type: AnalysisType::Elimination,
}).await?;

println!("Elimination entries needed: {}", result.elimination_entries.len());
for entry in result.elimination_entries {
    println!("Eliminate: {} -> {} (${:.2})",
        entry.from_entity,
        entry.to_entity,
        entry.amount
    );
}
```

---

### TemporalCorrelation

Analyzes correlations between accounts over time to detect anomalies.

**ID**: `accounting/temporal-correlation`
**Modes**: Batch

#### Output

```rust
pub struct CorrelationOutput {
    pub correlations: Vec<AccountCorrelation>,
    pub anomalies: Vec<CorrelationAnomaly>,
    pub stats: CorrelationStats,
}

pub struct AccountCorrelation {
    pub account_a: String,
    pub account_b: String,
    pub coefficient: f64,
    pub p_value: f64,
}
```

---

### SuspenseAccountDetection

Identifies suspense accounts using centrality-based analysis on the account transaction graph.

**ID**: `accounting/suspense-detection`
**Modes**: Batch
**Throughput**: ~20,000 accounts/sec

Suspense accounts are detected based on:
- **High centrality**: Accounts that connect many other accounts
- **High turnover**: Large volume relative to balance
- **Short holding period**: Funds don't stay long
- **Balanced flows**: Equal in/out suggests clearing function
- **Zero end balance**: Period-end balance near zero
- **Naming patterns**: Contains "suspense", "clearing", "holding"

#### Configuration

```rust
pub struct SuspenseDetectionConfig {
    /// Minimum betweenness centrality to flag
    pub centrality_threshold: f64,        // default: 0.1
    /// Minimum turnover ratio (turnover/balance)
    pub turnover_ratio_threshold: f64,    // default: 10.0
    /// Maximum average holding period (days)
    pub holding_period_threshold: f64,    // default: 7.0
    /// Minimum balance ratio to consider balanced (0-1)
    pub balance_ratio_threshold: f64,     // default: 0.9
    /// Minimum counterparty count to flag
    pub counterparty_threshold: usize,    // default: 5
    /// Maximum balance to consider "zero"
    pub zero_balance_threshold: f64,      // default: 100.0
}
```

#### Output

```rust
pub struct SuspenseAccountResult {
    /// Detected suspense account candidates
    pub candidates: Vec<SuspenseAccountCandidate>,
    /// High-risk accounts
    pub high_risk_accounts: Vec<String>,
    /// Total accounts analyzed
    pub accounts_analyzed: usize,
    /// Overall risk score
    pub risk_score: f64,
}

pub struct SuspenseAccountCandidate {
    pub account_code: String,
    pub account_name: String,
    pub suspense_score: f64,           // 0-100
    pub centrality_score: f64,
    pub turnover_volume: f64,
    pub avg_holding_period: f64,
    pub counterparty_count: usize,
    pub balance_ratio: f64,
    pub risk_level: SuspenseRiskLevel, // Low, Medium, High, Critical
    pub indicators: Vec<SuspenseIndicator>,
}
```

#### Example

```rust
use rustkernel::accounting::detection::{SuspenseAccountDetection, SuspenseDetectionConfig};

let result = SuspenseAccountDetection::detect(&journal_entries, &SuspenseDetectionConfig {
    centrality_threshold: 0.1,
    holding_period_threshold: 7.0,
    ..Default::default()
});

// Review high-risk suspense accounts
for account in &result.high_risk_accounts {
    println!("HIGH RISK: Account {} flagged as suspense", account);
}

// Analyze candidates
for candidate in &result.candidates {
    println!("{}: score={:.1}, centrality={:.3}, indicators={:?}",
        candidate.account_code,
        candidate.suspense_score,
        candidate.centrality_score,
        candidate.indicators);
}
```

---

### GaapViolationDetection

Detects prohibited transaction patterns that violate GAAP principles.

**ID**: `accounting/gaap-violation`
**Modes**: Batch
**Throughput**: ~15,000 entries/sec

Detected violation types:
- **DirectRevenueExpense**: Direct transfer from revenue to expense without capital account
- **RevenueInflation**: Circular flows that may inflate revenue
- **ImproperAssetExpense**: Asset expensed without proper depreciation
- **SuspenseAccountMisuse**: Large amounts in suspense accounts
- **ImproperElimination**: Incorrect intercompany eliminations
- **ProhibitedRelatedParty**: Prohibited related-party transactions

#### Configuration

```rust
pub struct GaapDetectionConfig {
    /// Threshold for suspense account amounts
    pub suspense_amount_threshold: f64,   // default: 10_000.0
    /// Minimum amount for asset-to-expense flag
    pub asset_expense_threshold: f64,     // default: 5_000.0
    /// Minimum circular flow amount
    pub circular_flow_threshold: f64,     // default: 1_000.0
}
```

#### Output

```rust
pub struct GaapViolationResult {
    /// Detected violations
    pub violations: Vec<GaapViolation>,
    /// Total entries analyzed
    pub entries_analyzed: usize,
    /// Total amount at risk
    pub amount_at_risk: f64,
    /// Overall compliance score (0-100, higher is better)
    pub compliance_score: f64,
    /// Violation counts by type
    pub violation_counts: HashMap<String, usize>,
}

pub struct GaapViolation {
    pub id: String,
    pub violation_type: GaapViolationType,
    pub accounts: Vec<String>,
    pub entry_ids: Vec<u64>,
    pub amount: f64,
    pub description: String,
    pub severity: GaapViolationSeverity,  // Minor, Moderate, Major, Critical
    pub remediation: String,
}
```

#### Example

```rust
use rustkernel::accounting::detection::{GaapViolationDetection, GaapDetectionConfig};
use std::collections::HashMap;

// Map account codes to types
let mut account_types = HashMap::new();
account_types.insert("SALES_REVENUE".to_string(), AccountType::Revenue);
account_types.insert("SALARIES_EXPENSE".to_string(), AccountType::Expense);
account_types.insert("EQUIPMENT_ASSET".to_string(), AccountType::Asset);

let result = GaapViolationDetection::detect(
    &journal_entries,
    &account_types,
    &GaapDetectionConfig::default()
);

println!("Compliance score: {:.1}%", result.compliance_score);
println!("Amount at risk: ${:.2}", result.amount_at_risk);

// Review violations by severity
for violation in result.violations.iter()
    .filter(|v| matches!(v.severity, GaapViolationSeverity::Major | GaapViolationSeverity::Critical))
{
    println!("{}: {} - {}",
        violation.id,
        violation.description,
        violation.remediation);
}
```

---

## Enhanced Features

### Account Classification

Automatic classification of accounts:

- **Asset** (1xxx): Cash, receivables, inventory
- **Liability** (2xxx): Payables, debt, accruals
- **Equity** (3xxx): Capital, retained earnings
- **Revenue** (4xxx): Sales, service income
- **COGS** (5xxx): Cost of goods sold
- **Expense** (6xxx-7xxx): Operating expenses
- **Tax**: VAT, GST, withholding tax

### VAT Detection

Automatic detection of VAT patterns:

- EU standard rates (19-25%)
- Reduced rates (5-10%)
- GST/HST (Canada, Australia)

### Transaction Patterns

Recognition of common patterns:

- SimpleSale, SaleWithVat
- SimplePurchase, PurchaseWithVat
- Payment, Receipt
- Payroll, Depreciation
- Intercompany, CostAllocation

---

## Use Cases

- **Financial close**: Automate journal analysis
- **Audit**: Trace value flows, detect anomalies
- **Consolidation**: Identify elimination entries
- **Compliance**: VAT reporting, intercompany analysis
