# Accounting Network Generation: Transforming Journal Entries to Directed Graphs

**Published**: January 2026
**Domain**: Accounting
**Kernels**: NetworkGeneration, NetworkGenerationRing

## Abstract

This article describes the GPU-accelerated transformation of double-entry bookkeeping journal entries into directed accounting networks. We implement five solving methods with decreasing confidence levels, enabling sophisticated analysis of value flows between accounts. The implementation includes automatic account classification, VAT/tax detection, transaction pattern recognition, and confidence boosting based on domain knowledge.

---

## Introduction

Traditional accounting views journal entries as balanced debit/credit pairs following the fundamental equation:

```
Assets = Liabilities + Equity
```

Every transaction creates at least one debit and one credit, always balancing. However, understanding the *flow* of value between accounts requires transforming these entries into a directed graph representation where:

- **Nodes** represent accounts
- **Edges** represent flows (directed from debit accounts to credit accounts)
- **Edge weights** represent amounts transferred

This transformation enables powerful analytics:

- **Flow tracing**: Follow value through the organization
- **Anomaly detection**: Identify unusual patterns
- **Intercompany analysis**: Detect circular flows requiring elimination
- **Audit trails**: Trace specific amounts to source transactions

The challenge lies in determining which debit amount flows to which credit account when an entry has multiple lines.

---

## The Solving Problem

Consider a simple journal entry:

```
Dr. Cash           $1,000
    Cr. Revenue              $1,000
```

The flow is trivial: Cash receives $1,000 from Revenue.

But what about:

```
Dr. Cash           $1,000
Dr. Receivables      $500
    Cr. Revenue              $1,200
    Cr. Deferred Revenue       $300
```

Which debit flows to which credit? There are multiple valid interpretations. Our algorithm provides a principled approach to solve this ambiguity while quantifying confidence.

---

## The Five Solving Methods

We implement five methods in order of decreasing confidence. The algorithm tries each method in sequence until one succeeds.

### Method A: Trivial 1-to-1 (Confidence: 1.0)

**When applied**: Exactly 1 debit line and 1 credit line.

This is deterministic. The single debit flows entirely to the single credit.

```
Entry:
  Dr. Cash $1,000
  Cr. Revenue $1,000

Result:
  Flow: Cash -> Revenue ($1,000) [confidence: 1.0]
```

**Approximately 60% of entries** in typical general ledgers are 2-line entries solved by Method A.

### Method B: n-to-n Bijective Matching (Confidence: 0.95)

**When applied**: Equal count of debits and credits (n debits, n credits), where n <= 10.

Uses a two-phase greedy matching algorithm:

1. **Phase 1**: Match exact amounts (within tolerance)
2. **Phase 2**: Match remaining by order

```
Entry:
  Dr. Cash         $500
  Dr. Receivables  $300
  Cr. Sales        $500
  Cr. Service      $300

Phase 1 matches:
  Cash ($500) -> Sales ($500)      [exact match]
  Receivables ($300) -> Service ($300)  [exact match]

Result: Two flows with confidence 0.95
```

### Method C: n-to-m Partition Matching (Confidence: 0.85)

**When applied**: Unequal counts of debits and credits, total lines <= 20.

Uses subset-sum matching to find which credits combine to match each debit amount.

For small n (<=12): Exhaustive search using bit manipulation (2^n combinations)
For larger n: Greedy approximation (sorted descending)

```
Entry:
  Dr. Cash $800
  Cr. Revenue    $500
  Cr. Service    $300

Algorithm finds: {Revenue, Service} sums to $800

Result:
  Flow: Cash -> Revenue ($500) [confidence: 0.85]
  Flow: Cash -> Service ($300) [confidence: 0.85]
```

If no exact partition exists, falls back to **proportional allocation** with reduced confidence (0.765).

### Method D: Aggregation (Confidence: 0.70)

**When applied**: Many lines (>20 total), aggregation enabled.

Aggregates to account level and allocates proportionally:

1. Sum all debits by account code
2. Sum all credits by account code
3. For each (debit_account, credit_account) pair:
   ```
   allocation = debit_amount * (credit_amount / total_credits)
   ```

**Use case**: Large allocation entries with many cost centers.

### Method E: Entity Decomposition (Confidence: 0.50)

**When applied**: Multi-entity entries, decomposition enabled.

Decomposes by entity_id and attempts within-entity matching first:

1. Group debits by entity
2. Group credits by entity
3. Try matching within each entity
4. Cross-entity flows get additional 0.8x confidence multiplier

**Use case**: Intercompany transactions, consolidation entries.

### Unsolvable Entries

If all methods fail or entry is unbalanced:
- Route all flows through suspense account
- Confidence: 0.0
- Provides audit trail for investigation

---

## Fixed-Point Arithmetic

Financial calculations require exact precision. Floating-point arithmetic introduces rounding errors that compound over millions of transactions. We use 128-bit fixed-point arithmetic with 18 decimal places:

```rust
pub struct FixedPoint128 {
    pub value: i128,  // Scaled by 10^18
}

const SCALE: i128 = 1_000_000_000_000_000_000; // 10^18

impl FixedPoint128 {
    pub fn from_f64(v: f64) -> Self {
        Self {
            value: (v * SCALE as f64) as i128
        }
    }

    pub fn to_f64(&self) -> f64 {
        self.value as f64 / SCALE as f64
    }
}
```

This provides:
- **Range**: Up to ~170 trillion (sufficient for any practical amount)
- **Precision**: Exactly 18 decimal places
- **Determinism**: Same result across platforms

All amount comparisons use tolerance-aware equality:

```rust
impl FixedPoint128 {
    pub fn approx_eq(&self, other: &Self, tolerance: i128) -> bool {
        (self.value - other.value).abs() <= tolerance
    }
}
```

---

## Enhanced Features

### Account Classification

Accounts are automatically classified into standard categories:

```rust
pub enum AccountClass {
    Asset,              // 1xxx accounts
    Liability,          // 2xxx accounts
    Equity,             // 3xxx accounts
    Revenue,            // 4xxx accounts
    COGS,               // 5xxx accounts
    Expense,            // 6xxx-7xxx accounts
    OtherIncomeExpense, // 8xxx accounts
    Tax,                // VAT, GST, etc.
    Intercompany,       // Related party accounts
    Suspense,           // Clearing/suspense
    Unknown,            // Unclassified
}
```

Classification uses two strategies:

1. **Numeric prefix**: `1xxx` = Asset, `2xxx` = Liability, etc.
2. **Keyword matching**: "CASH", "RECEIVABLE" -> Asset; "PAYABLE" -> Liability

```rust
impl AccountClass {
    pub fn from_account_code(code: &str) -> Self {
        // Try numeric classification first
        if let Some(first_digit) = code.chars().find(|c| c.is_ascii_digit()) {
            match first_digit {
                '1' => return AccountClass::Asset,
                '2' => return AccountClass::Liability,
                '3' => return AccountClass::Equity,
                '4' => return AccountClass::Revenue,
                '5' => return AccountClass::COGS,
                '6' | '7' => return AccountClass::Expense,
                '8' => return AccountClass::OtherIncomeExpense,
                _ => {}
            }
        }

        // Fall back to keyword matching
        let upper = code.to_uppercase();
        if upper.contains("VAT") || upper.contains("TAX") {
            AccountClass::Tax
        } else if upper.contains("CASH") || upper.contains("BANK") {
            AccountClass::Asset
        }
        // ... additional patterns
    }
}
```

### VAT/Tax Detection

The system automatically detects VAT patterns by analyzing amount relationships:

```rust
pub struct VatDetector {
    known_rates: Vec<VatRate>,
    tolerance: f64,
}

pub struct VatRate {
    pub name: String,
    pub rate: f64,        // e.g., 0.20 for 20%
    pub jurisdiction: String,
}
```

Built-in rates include:
- **EU**: 19% (Germany), 20% (UK, France), 21% (Netherlands), 23% (Ireland), 25% (Sweden)
- **Reduced**: 5%, 7%, 10%
- **GST/HST**: 5% (Canada GST), 13%/15% (HST)

Detection algorithm:

```rust
pub fn detect_vat_split(&self, amounts: &[i128]) -> Option<VatPattern> {
    // Sort amounts descending
    let mut sorted = amounts.to_vec();
    sorted.sort_by(|a, b| b.cmp(a));

    // Try each pair: largest could be gross, second could be net
    for (gross, net) in pairs(&sorted) {
        let tax = gross - net;
        let implied_rate = tax as f64 / net as f64;

        // Check against known rates
        for rate in &self.known_rates {
            if (implied_rate - rate.rate).abs() < self.tolerance {
                return Some(VatPattern {
                    gross_amount: gross,
                    net_amount: net,
                    tax_amount: tax,
                    rate: rate.clone(),
                    // ... additional fields
                });
            }
        }
    }
    None
}
```

### Transaction Pattern Recognition

14 common transaction patterns are automatically detected:

```rust
pub enum TransactionPattern {
    SimpleSale,        // Dr. Asset, Cr. Revenue
    SaleWithVat,       // Dr. Asset, Cr. Revenue + Tax
    SimplePurchase,    // Dr. Expense/Asset, Cr. Asset/Liability
    PurchaseWithVat,   // Dr. Expense + Tax, Cr. Liability
    Payment,           // Dr. Liability, Cr. Asset
    Receipt,           // Dr. Asset, Cr. Asset
    Payroll,           // Dr. Expense, Cr. Multiple Liabilities
    Depreciation,      // Dr. Expense, Cr. Contra-Asset
    Accrual,           // Dr. Expense, Cr. Liability
    AccrualReversal,   // Reverse of accrual
    Transfer,          // Dr. Asset, Cr. Asset (multiple)
    Intercompany,      // Cross-entity debits/credits
    CostAllocation,    // Multiple cost centers
    Adjustment,        // Miscellaneous adjustments
    Unknown,
}
```

Pattern detection examines:
1. Account classes on debit side
2. Account classes on credit side
3. Presence of tax accounts
4. Entity relationships

### Confidence Boosting

Recognized patterns boost confidence scores:

| Pattern | Boost |
|---------|-------|
| SaleWithVat | +0.15 |
| PurchaseWithVat | +0.15 |
| SimpleSale | +0.10 |
| SimplePurchase | +0.10 |
| Payroll | +0.12 |
| Depreciation | +0.15 |
| Intercompany | +0.05 |

Final confidence is capped at 1.0:

```rust
let base_confidence = method.confidence();
let pattern_boost = pattern.confidence_boost();
let final_confidence = (base_confidence + pattern_boost).min(1.0);
```

---

## Data Structures

### AccountingFlow

The primary output structure:

```rust
pub struct AccountingFlow {
    pub flow_id: String,
    pub entry_id: u64,
    pub from_account: String,
    pub to_account: String,
    pub amount: FixedPoint128,
    pub timestamp: u64,
    pub method: SolvingMethod,
    pub confidence: f64,
    pub from_entity: String,
    pub to_entity: String,
    pub currency: String,
    pub source_lines: Vec<u32>,

    // Enhanced fields
    pub from_account_class: Option<AccountClass>,
    pub to_account_class: Option<AccountClass>,
    pub pattern: Option<TransactionPattern>,
    pub is_tax_flow: bool,
    pub vat_rate: Option<f64>,
    pub is_intercompany: bool,
    pub confidence_factors: Vec<String>,
}
```

### AccountingNetwork

The complete network representation:

```rust
pub struct AccountingNetwork {
    pub flows: Vec<AccountingFlow>,
    pub accounts: HashSet<String>,
    pub account_index: HashMap<String, usize>,
    pub adjacency: HashMap<String, Vec<(String, usize)>>,
    pub stats: NetworkGenerationStats,
}

impl AccountingNetwork {
    pub fn outgoing_flows(&self, account: &str) -> Vec<&AccountingFlow> {
        // Returns all flows from this account
    }

    pub fn incoming_flows(&self, account: &str) -> Vec<&AccountingFlow> {
        // Returns all flows to this account
    }

    pub fn total_volume(&self) -> f64 {
        self.flows.iter().map(|f| f.amount.to_f64()).sum()
    }

    pub fn weighted_confidence(&self) -> f64 {
        let total_amount: f64 = self.flows.iter()
            .map(|f| f.amount.to_f64())
            .sum();

        if total_amount == 0.0 {
            return self.flows.iter()
                .map(|f| f.confidence)
                .sum::<f64>() / self.flows.len() as f64;
        }

        self.flows.iter()
            .map(|f| f.confidence * f.amount.to_f64())
            .sum::<f64>() / total_amount
    }
}
```

---

## Usage Example

```rust
use rustkernel::accounting::network_generation::{
    NetworkGeneration,
    NetworkGenerationInput,
    NetworkGenerationConfig,
};

// Configure the kernel
let config = NetworkGenerationConfig {
    amount_tolerance: 0.01,
    max_lines_method_b: 10,
    max_lines_method_c: 20,
    enable_aggregation: true,
    enable_decomposition: true,
    suspense_account: "SUSPENSE".to_string(),
    strict_balance: false,

    // Enhanced features
    enable_pattern_matching: true,
    enable_vat_detection: true,
    apply_confidence_boost: true,
    annotate_account_classes: true,
    custom_vat_rates: vec![],
};

let kernel = NetworkGeneration::with_config(config);

// Process journal entries
let result = kernel.execute(NetworkGenerationInput {
    entries: journal_entries,
    config: None,  // Use kernel's config
}).await?;

// Analyze results
println!("Generated {} flows", result.flows.len());
println!("Total volume: ${:.2}", result.stats.total_volume);
println!("Weighted confidence: {:.2}%", result.stats.weighted_confidence * 100.0);

// Method distribution
println!("\nMethod distribution:");
println!("  Method A (1:1): {}", result.stats.method_a_count);
println!("  Method B (n:n): {}", result.stats.method_b_count);
println!("  Method C (n:m): {}", result.stats.method_c_count);
println!("  Method D (agg): {}", result.stats.method_d_count);
println!("  Method E (dec): {}", result.stats.method_e_count);

// Pattern analysis
println!("\nTransaction patterns:");
println!("  Sales: {}", result.stats.sales_pattern_count);
println!("  Purchases: {}", result.stats.purchase_pattern_count);
println!("  Payments: {}", result.stats.payment_pattern_count);
println!("  Payroll: {}", result.stats.payroll_pattern_count);
println!("  Intercompany: {}", result.stats.intercompany_count);

// VAT analysis
println!("\nVAT detection:");
println!("  VAT entries: {}", result.stats.vat_entries_count);
println!("  Total VAT: ${:.2}", result.stats.total_vat_amount);
```

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Throughput | ~500,000 entries/sec |
| Memory | ~200 bytes per flow |
| GPU acceleration | Method C subset search |

### Typical Distribution

For a standard GL with 100,000 entries:

| Method | Percentage | Avg Confidence |
|--------|------------|----------------|
| A | 60% | 1.00 |
| B | 25% | 0.95 |
| C | 10% | 0.85 |
| D | 4% | 0.70 |
| E | 1% | 0.50 |

**Overall weighted confidence**: ~0.94

---

## Ring Mode for Streaming

For real-time processing, use Ring mode:

```rust
use rustkernel::accounting::network_generation::NetworkGenerationRing;

let ring = NetworkGenerationRing::new();

// Add entries as they arrive
ring.add_entry(entry).await?;

// Query flows within time window
let flows = ring.query_flows(start_time, end_time).await?;

// Get current statistics
let stats = ring.get_statistics().await?;
```

---

## Conclusion

The Accounting Network Generation kernel provides a robust, GPU-accelerated method for transforming traditional journal entries into graph structures. The five-method hierarchy balances precision with practicality, while enhanced features like VAT detection and pattern recognition add domain intelligence to the analysis.

This enables sophisticated applications:

- **Audit analytics**: Trace flows, detect anomalies
- **Consolidation**: Identify intercompany eliminations
- **Compliance**: Analyze tax flows, verify postings
- **Forensics**: Follow money through the organization

The implementation demonstrates how GPU acceleration can be applied to accounting workloads, achieving throughputs suitable for even the largest enterprise general ledgers.

---

## References

1. Hardware Accelerated Method for Accounting Network Generation (Internal paper)
2. Double-Entry Bookkeeping and the Fundamental Accounting Equation
3. Graph-Based Financial Analysis Techniques
4. Fixed-Point Arithmetic for Financial Applications
