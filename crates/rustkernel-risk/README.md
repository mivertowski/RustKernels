# rustkernel-risk

[![Crates.io](https://img.shields.io/crates/v/rustkernel-risk.svg)](https://crates.io/crates/rustkernel-risk)
[![Documentation](https://docs.rs/rustkernel-risk/badge.svg)](https://docs.rs/rustkernel-risk)
[![License](https://img.shields.io/crates/l/rustkernel-risk.svg)](LICENSE)

GPU-accelerated risk analytics kernels for credit, market, and portfolio risk.

## Kernels (5)

### Credit Risk (1 kernel)
- **CreditRiskScoring** - PD/LGD/EAD calculation and credit scoring

### Market Risk (3 kernels)
- **MonteCarloVaR** - Monte Carlo Value at Risk simulation
- **PortfolioRiskAggregation** - Correlation-adjusted portfolio VaR
- **RealTimeCorrelation** - Streaming correlation matrix updates

### Stress Testing (1 kernel)
- **StressTestScenario** - Scenario-based stress testing

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernel-risk = "0.1.0"
```

## Usage

```rust
use rustkernel_risk::prelude::*;

// Calculate VaR using Monte Carlo
let var = MonteCarloVaR::new();
let result = var.calculate(&portfolio, confidence, horizon, simulations);
```

## License

Apache-2.0
