# Accounting

**Crate**: `rustkernel-accounting`
**Kernels**: 7
**Feature**: `accounting`

Accounting network generation, reconciliation, and analysis kernels for financial close and audit.

## Kernel Overview

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| ChartOfAccountsMapping | `accounting/coa-mapping` | Batch | Map between chart of accounts |
| JournalTransformation | `accounting/journal-transformation` | Batch, Ring | Transform journal entries |
| GLReconciliation | `accounting/gl-reconciliation` | Batch, Ring | General ledger reconciliation |
| NetworkAnalysis | `accounting/network-analysis` | Batch, Ring | Intercompany network analysis |
| TemporalCorrelation | `accounting/temporal-correlation` | Batch | Account correlation over time |
| NetworkGeneration | `accounting/network-generation` | Batch | Generate accounting networks |
| NetworkGenerationRing | `accounting/network-generation-ring` | Ring | Streaming network generation |

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
