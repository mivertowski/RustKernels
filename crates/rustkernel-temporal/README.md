# rustkernel-temporal

[![Crates.io](https://img.shields.io/crates/v/rustkernel-temporal.svg)](https://crates.io/crates/rustkernel-temporal)
[![Documentation](https://docs.rs/rustkernel-temporal/badge.svg)](https://docs.rs/rustkernel-temporal)
[![License](https://img.shields.io/crates/l/rustkernel-temporal.svg)](LICENSE)

GPU-accelerated temporal analysis kernels for forecasting, decomposition, detection, and volatility modeling.

## Kernels (7)

### Forecasting (2 kernels)
- **ARIMAForecast** - ARIMA(p,d,q) model fitting and forecasting
- **ProphetDecomposition** - Prophet-style trend/seasonal/holiday decomposition

### Detection (2 kernels)
- **ChangePointDetection** - PELT/Binary segmentation/CUSUM
- **TimeSeriesAnomalyDetection** - Statistical threshold detection

### Volatility (2 kernels)
- **GARCHVolatility** - GARCH(1,1) volatility modeling
- **RealizedVolatility** - High-frequency realized volatility

### Seasonality (1 kernel)
- **SeasonalDecomposition** - STL decomposition

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernel-temporal = "0.1.0"
```

## Usage

```rust
use rustkernel_temporal::prelude::*;

// Forecast with ARIMA
let arima = ARIMAForecast::new();
let forecast = arima.fit_and_forecast(&series, p, d, q, horizon);
```

## License

Apache-2.0
