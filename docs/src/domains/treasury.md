# Treasury

**Crate**: `rustkernel-treasury`
**Kernels**: 5
**Feature**: `treasury`

Treasury management kernels for cash flow, FX hedging, and liquidity optimization.

## Kernel Overview

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| CashFlowForecasting | `treasury/cash-flow-forecasting` | Batch, Ring | Predict future cash flows |
| CollateralOptimization | `treasury/collateral-optimization` | Batch | Optimize collateral allocation |
| FXHedging | `treasury/fx-hedging` | Batch, Ring | FX exposure and hedging |
| InterestRateRisk | `treasury/interest-rate-risk` | Batch, Ring | Duration, convexity, DV01 |
| LiquidityOptimization | `treasury/liquidity-optimization` | Batch | LCR/NSFR optimization |

---

## Kernel Details

### CashFlowForecasting

Forecasts cash positions across accounts and time horizons.

**ID**: `treasury/cash-flow-forecasting`
**Modes**: Batch, Ring

#### Input

```rust
pub struct CashFlowInput {
    /// Current positions
    pub positions: Vec<CashPosition>,
    /// Expected inflows
    pub inflows: Vec<CashFlow>,
    /// Expected outflows
    pub outflows: Vec<CashFlow>,
    /// Forecast horizon in days
    pub horizon_days: u32,
}

pub struct CashPosition {
    pub account_id: String,
    pub currency: String,
    pub balance: f64,
}

pub struct CashFlow {
    pub account_id: String,
    pub amount: f64,
    pub currency: String,
    pub expected_date: u64,
    pub probability: f64,
    pub category: FlowCategory,
}
```

#### Output

```rust
pub struct CashFlowOutput {
    /// Daily forecast per account/currency
    pub daily_forecast: Vec<DailyPosition>,
    /// Minimum/maximum projections
    pub min_projection: Vec<f64>,
    pub max_projection: Vec<f64>,
    /// Shortfall alerts
    pub shortfall_alerts: Vec<ShortfallAlert>,
}
```

---

### FXHedging

Analyzes FX exposures and recommends hedging strategies.

**ID**: `treasury/fx-hedging`
**Modes**: Batch, Ring

#### Example

```rust
use rustkernel::treasury::fx::{FXHedging, FXHedgingInput};

let kernel = FXHedging::new();

let result = kernel.execute(FXHedgingInput {
    exposures: vec![
        FXExposure {
            currency_pair: "EUR/USD".into(),
            amount: 1_000_000.0,
            direction: ExposureDirection::Long,
            maturity_days: 90,
        },
    ],
    hedging_instruments: available_instruments,
    risk_tolerance: RiskTolerance::Moderate,
    hedge_ratio_target: 0.80,
}).await?;

for recommendation in result.recommendations {
    println!("Hedge {} with {} {} forward",
        recommendation.exposure,
        recommendation.amount,
        recommendation.instrument
    );
}
```

---

### InterestRateRisk

Calculates interest rate risk metrics for fixed income portfolios.

**ID**: `treasury/interest-rate-risk`
**Modes**: Batch, Ring

#### Output

```rust
pub struct InterestRateRiskOutput {
    /// Modified duration
    pub duration: f64,
    /// Convexity
    pub convexity: f64,
    /// Dollar value of a basis point
    pub dv01: f64,
    /// Key rate durations
    pub key_rate_durations: HashMap<String, f64>,
    /// Scenario analysis
    pub scenario_pnl: HashMap<String, f64>,
}
```

---

### LiquidityOptimization

Optimizes liquidity positions for regulatory compliance (LCR, NSFR).

**ID**: `treasury/liquidity-optimization`
**Modes**: Batch

#### Example

```rust
use rustkernel::treasury::liquidity::{LiquidityOptimization, LiquidityInput};

let kernel = LiquidityOptimization::new();

let result = kernel.execute(LiquidityInput {
    assets: liquid_assets,
    liabilities: funding_sources,
    target_lcr: 1.10,  // 110% target
    target_nsfr: 1.05, // 105% target
    constraints: optimization_constraints,
}).await?;

println!("Current LCR: {:.1}%", result.current_lcr * 100.0);
println!("Optimized LCR: {:.1}%", result.optimized_lcr * 100.0);
for action in result.recommended_actions {
    println!("Action: {}", action.description);
}
```

---

## Use Cases

- **Cash management**: Forecast positions, optimize sweeps
- **FX treasury**: Manage currency exposures, hedge programs
- **ALM**: Asset-liability management, gap analysis
- **Regulatory**: LCR/NSFR compliance, stress testing
