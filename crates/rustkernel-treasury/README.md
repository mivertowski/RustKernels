# rustkernel-treasury

[![Crates.io](https://img.shields.io/crates/v/rustkernel-treasury.svg)](https://crates.io/crates/rustkernel-treasury)
[![Documentation](https://docs.rs/rustkernel-treasury/badge.svg)](https://docs.rs/rustkernel-treasury)
[![License](https://img.shields.io/crates/l/rustkernel-treasury.svg)](LICENSE)

GPU-accelerated treasury management kernels.

## Kernels (5)

- **CashFlowForecasting** - Multi-horizon cash flow projection
- **CollateralOptimization** - LP/QP optimization for collateral allocation
- **FXHedging** - Currency exposure management
- **InterestRateRisk** - Duration/convexity analysis
- **LiquidityOptimization** - LCR/NSFR optimization

## Features

- Cash flow forecasting across multiple time horizons
- Optimal collateral allocation using linear/quadratic programming
- FX exposure analysis and hedging recommendations
- Interest rate risk metrics (duration, convexity, DV01)
- Liquidity coverage ratio and net stable funding ratio optimization

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernel-treasury = "0.1.0"
```

## Usage

```rust
use rustkernel_treasury::prelude::*;

// Forecast cash flows
let forecast = CashFlowForecasting::new();
let projections = forecast.project(&cash_flows, horizons);
```

## License

Apache-2.0
