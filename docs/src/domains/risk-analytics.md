# Risk Analytics

**Crate**: `rustkernel-risk`
**Kernels**: 4
**Feature**: `risk` (included in default features)

Financial risk calculation kernels for credit risk, market risk, and portfolio analysis.

## Kernel Overview

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| CreditRiskScoring | `risk/credit-risk-scoring` | Batch, Ring | PD/LGD/EAD calculations |
| MonteCarloVaR | `risk/monte-carlo-var` | Batch, Ring | Value-at-Risk simulation |
| PortfolioRiskAggregation | `risk/portfolio-risk-aggregation` | Batch, Ring | Portfolio-level risk metrics |
| StressTesting | `risk/stress-testing` | Batch | Scenario-based stress analysis |

---

## Kernel Details

### CreditRiskScoring

Calculates Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD).

**ID**: `risk/credit-risk-scoring`
**Modes**: Batch, Ring

#### Input

```rust
pub struct CreditRiskInput {
    /// Borrower information
    pub borrowers: Vec<BorrowerInfo>,
    /// Scoring model to use
    pub model: CreditModel,
    /// Time horizon in months
    pub horizon_months: u32,
}

pub struct BorrowerInfo {
    pub id: String,
    pub credit_score: u32,
    pub debt_to_income: f64,
    pub loan_amount: f64,
    pub collateral_value: Option<f64>,
    pub industry_code: String,
    pub years_in_business: u32,
}

pub enum CreditModel {
    Scorecard,  // Traditional scorecard
    IRB,        // Basel IRB approach
    Merton,     // Structural model
}
```

#### Output

```rust
pub struct CreditRiskOutput {
    /// Risk metrics per borrower
    pub risk_metrics: Vec<BorrowerRisk>,
    /// Portfolio-level metrics
    pub portfolio_metrics: PortfolioCreditMetrics,
}

pub struct BorrowerRisk {
    pub borrower_id: String,
    pub pd: f64,         // Probability of Default
    pub lgd: f64,        // Loss Given Default
    pub ead: f64,        // Exposure at Default
    pub expected_loss: f64,
    pub risk_weight: f64,  // For RWA calculation
}
```

#### Example

```rust
use rustkernel::risk::credit::{CreditRiskScoring, CreditRiskInput};

let kernel = CreditRiskScoring::new();

let result = kernel.execute(CreditRiskInput {
    borrowers: vec![
        BorrowerInfo {
            id: "B001".into(),
            credit_score: 720,
            debt_to_income: 0.35,
            loan_amount: 250_000.0,
            collateral_value: Some(300_000.0),
            industry_code: "REAL_ESTATE".into(),
            years_in_business: 5,
        },
    ],
    model: CreditModel::IRB,
    horizon_months: 12,
}).await?;

let risk = &result.risk_metrics[0];
println!("PD: {:.2}%, LGD: {:.2}%, EL: ${:.2}",
    risk.pd * 100.0,
    risk.lgd * 100.0,
    risk.expected_loss
);
```

---

### MonteCarloVaR

Calculates Value-at-Risk using Monte Carlo simulation.

**ID**: `risk/monte-carlo-var`
**Modes**: Batch, Ring
**Throughput**: ~1M simulations/sec

#### Input

```rust
pub struct VaRInput {
    /// Portfolio positions
    pub positions: Vec<Position>,
    /// Number of simulations
    pub n_simulations: u32,
    /// Time horizon in days
    pub horizon_days: u32,
    /// Confidence levels
    pub confidence_levels: Vec<f64>,
    /// Correlation matrix (flattened)
    pub correlations: Vec<f64>,
    /// Volatilities per asset
    pub volatilities: Vec<f64>,
}

pub struct Position {
    pub asset_id: String,
    pub quantity: f64,
    pub current_price: f64,
}
```

#### Output

```rust
pub struct VaROutput {
    /// VaR at each confidence level
    pub var_values: HashMap<String, f64>,
    /// Expected Shortfall (CVaR)
    pub expected_shortfall: HashMap<String, f64>,
    /// Simulated P&L distribution
    pub pnl_distribution: Vec<f64>,
    /// Component VaR by position
    pub component_var: Vec<(String, f64)>,
}
```

#### Example

```rust
use rustkernel::risk::market::{MonteCarloVaR, VaRInput};

let kernel = MonteCarloVaR::new();

let result = kernel.execute(VaRInput {
    positions: portfolio_positions,
    n_simulations: 100_000,
    horizon_days: 10,
    confidence_levels: vec![0.95, 0.99],
    correlations: correlation_matrix,
    volatilities: asset_volatilities,
}).await?;

println!("10-day VaR (99%): ${:.2}", result.var_values["0.99"]);
println!("Expected Shortfall (99%): ${:.2}", result.expected_shortfall["0.99"]);
```

---

### PortfolioRiskAggregation

Aggregates risk across multiple portfolios with diversification effects.

**ID**: `risk/portfolio-risk-aggregation`
**Modes**: Batch, Ring

#### Example

```rust
use rustkernel::risk::portfolio::{PortfolioRiskAggregation, AggregationInput};

let kernel = PortfolioRiskAggregation::new();

let result = kernel.execute(AggregationInput {
    portfolio_vars: vec![
        ("Equities".into(), 1_000_000.0),
        ("Fixed Income".into(), 500_000.0),
        ("Commodities".into(), 250_000.0),
    ],
    correlations: correlation_matrix,
    method: AggregationMethod::VarianceCovariance,
}).await?;

println!("Undiversified VaR: ${:.2}", result.undiversified_var);
println!("Diversified VaR: ${:.2}", result.diversified_var);
println!("Diversification benefit: ${:.2}", result.diversification_benefit);
```

---

### StressTesting

Evaluates portfolio impact under stress scenarios.

**ID**: `risk/stress-testing`
**Modes**: Batch

#### Input

```rust
pub struct StressTestInput {
    pub portfolio: Portfolio,
    pub scenarios: Vec<StressScenario>,
}

pub struct StressScenario {
    pub name: String,
    /// Shocks to risk factors
    pub shocks: HashMap<String, f64>,
    pub description: String,
}
```

#### Output

```rust
pub struct StressTestOutput {
    pub results: Vec<ScenarioResult>,
    pub worst_case: ScenarioResult,
}

pub struct ScenarioResult {
    pub scenario_name: String,
    pub pnl_impact: f64,
    pub pnl_impact_pct: f64,
    pub positions_affected: Vec<PositionImpact>,
}
```

---

## Ring Mode for Real-Time Risk

Ring mode enables streaming risk calculations:

```rust
use rustkernel::risk::market::MonteCarloVaRRing;

let ring = MonteCarloVaRRing::new();

// Real-time position updates
ring.update_position("AAPL", 100, 185.50).await?;
ring.update_position("GOOG", 50, 142.30).await?;

// Query current VaR (sub-millisecond)
let current_var = ring.query_var(0.99).await?;
println!("Current VaR: ${:.2}", current_var);

// Recalculate on market data updates
ring.recalculate().await?;
```

## Regulatory Applications

These kernels support:

- **Basel III/IV**: RWA calculation, capital adequacy
- **FRTB**: Fundamental Review of the Trading Book
- **CCAR/DFAST**: Fed stress testing requirements
- **Solvency II**: Insurance capital requirements
