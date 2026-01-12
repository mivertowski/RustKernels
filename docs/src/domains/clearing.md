# Clearing

**Crate**: `rustkernel-clearing`
**Kernels**: 5
**Feature**: `clearing`

Post-trade clearing, settlement, and netting kernels for financial market infrastructure.

## Kernel Overview

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| ClearingValidation | `clearing/validation` | Batch, Ring | Validate clearing eligibility |
| DVPMatching | `clearing/dvp-matching` | Batch, Ring | Delivery vs Payment matching |
| NettingCalculation | `clearing/netting-calculation` | Batch, Ring | Multilateral netting |
| SettlementExecution | `clearing/settlement-execution` | Batch | Execute settlement instructions |
| ZeroBalanceFrequency | `clearing/zero-balance-frequency` | Batch, Ring | Optimize netting efficiency |

---

## Kernel Details

### NettingCalculation

Calculates multilateral netting positions to minimize settlement volume.

**ID**: `clearing/netting-calculation`
**Modes**: Batch, Ring

#### Input

```rust
pub struct NettingInput {
    /// Trades to net
    pub trades: Vec<Trade>,
    /// Netting set definition
    pub netting_set: NettingSet,
    /// Currency for settlement
    pub settlement_currency: String,
}

pub struct Trade {
    pub id: String,
    pub buyer: String,
    pub seller: String,
    pub instrument: String,
    pub quantity: i64,
    pub price: f64,
    pub trade_date: u64,
    pub settlement_date: u64,
}

pub struct NettingSet {
    pub participants: Vec<String>,
    pub netting_type: NettingType,
}

pub enum NettingType {
    Bilateral,
    Multilateral,
    CCP,  // Central Counterparty
}
```

#### Output

```rust
pub struct NettingOutput {
    /// Net positions per participant
    pub positions: Vec<NetPosition>,
    /// Settlement instructions
    pub instructions: Vec<SettlementInstruction>,
    /// Netting statistics
    pub statistics: NettingStatistics,
}

pub struct NetPosition {
    pub participant: String,
    pub instrument: String,
    pub net_quantity: i64,
    pub net_value: f64,
}

pub struct NettingStatistics {
    pub gross_value: f64,
    pub net_value: f64,
    pub netting_efficiency: f64,  // (1 - net/gross) * 100
    pub trades_netted: u32,
}
```

#### Example

```rust
use rustkernel::clearing::netting::{NettingCalculation, NettingInput};

let kernel = NettingCalculation::new();

let result = kernel.execute(NettingInput {
    trades: vec![
        Trade { buyer: "A".into(), seller: "B".into(), quantity: 100, price: 50.0, .. },
        Trade { buyer: "B".into(), seller: "A".into(), quantity: 80, price: 51.0, .. },
        Trade { buyer: "A".into(), seller: "C".into(), quantity: 50, price: 49.0, .. },
    ],
    netting_set: NettingSet {
        participants: vec!["A".into(), "B".into(), "C".into()],
        netting_type: NettingType::Multilateral,
    },
    settlement_currency: "USD".into(),
}).await?;

println!("Netting efficiency: {:.1}%", result.statistics.netting_efficiency);
for pos in result.positions {
    println!("{}: {} units, ${:.2}", pos.participant, pos.net_quantity, pos.net_value);
}
```

---

### DVPMatching

Matches Delivery versus Payment instructions to ensure atomic settlement.

**ID**: `clearing/dvp-matching`
**Modes**: Batch, Ring

#### Output

```rust
pub struct DVPMatchOutput {
    pub matched_pairs: Vec<MatchedPair>,
    pub unmatched_deliveries: Vec<String>,
    pub unmatched_payments: Vec<String>,
    pub match_rate: f64,
}
```

---

### SettlementExecution

Executes settlement instructions with fail handling.

**ID**: `clearing/settlement-execution`
**Modes**: Batch

#### Example

```rust
use rustkernel::clearing::settlement::{SettlementExecution, SettlementInput};

let kernel = SettlementExecution::new();

let result = kernel.execute(SettlementInput {
    instructions: settlement_instructions,
    available_securities: securities_inventory,
    available_cash: cash_positions,
    fail_tolerance: FailTolerance::PartialAllowed,
}).await?;

println!("Settled: {}/{}", result.settled_count, result.total_count);
for fail in result.fails {
    println!("Failed: {} - {}", fail.instruction_id, fail.reason);
}
```

---

## Use Cases

- **CCP clearing**: Central counterparty netting and novation
- **Securities settlement**: DVP settlement for equities, bonds
- **FX settlement**: CLS-style payment-versus-payment
- **Derivatives clearing**: Margin calculation and variation margin
