# Temporal Analysis

**Crate**: `rustkernel-temporal`
**Kernels**: 7
**Feature**: `temporal` (included in default features)

Time series analysis kernels for forecasting, anomaly detection, and pattern recognition.

## Kernel Overview

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| ARIMAForecast | `temporal/arima-forecast` | Batch, Ring | Auto-regressive forecasting |
| ProphetDecomposition | `temporal/prophet-decomposition` | Batch | Facebook Prophet-style decomposition |
| ChangePointDetection | `temporal/change-point-detection` | Batch, Ring | Structural break detection |
| TimeSeriesAnomalyDetection | `temporal/anomaly-detection` | Batch, Ring | Anomaly scoring |
| SeasonalDecomposition | `temporal/seasonal-decomposition` | Batch | STL decomposition |
| TrendExtraction | `temporal/trend-extraction` | Batch, Ring | Trend component isolation |
| VolatilityAnalysis | `temporal/volatility-analysis` | Batch, Ring | GARCH/EWMA volatility |

---

## Kernel Details

### ARIMAForecast

Auto-Regressive Integrated Moving Average forecasting.

**ID**: `temporal/arima-forecast`
**Modes**: Batch, Ring

#### Input

```rust
pub struct ARIMAInput {
    /// Time series values
    pub values: Vec<f64>,
    /// AR order (p)
    pub p: u32,
    /// Differencing order (d)
    pub d: u32,
    /// MA order (q)
    pub q: u32,
    /// Forecast horizon
    pub forecast_periods: u32,
}
```

#### Output

```rust
pub struct ARIMAOutput {
    /// Forecasted values
    pub forecast: Vec<f64>,
    /// Confidence intervals (lower, upper)
    pub confidence_intervals: Vec<(f64, f64)>,
    /// Fitted values
    pub fitted: Vec<f64>,
    /// Model coefficients
    pub coefficients: ARIMACoefficients,
}
```

#### Example

```rust
use rustkernel::temporal::forecasting::{ARIMAForecast, ARIMAInput};

let kernel = ARIMAForecast::new();

let input = ARIMAInput {
    values: historical_prices,
    p: 2,  // AR(2)
    d: 1,  // First differencing
    q: 1,  // MA(1)
    forecast_periods: 30,
};

let result = kernel.execute(input).await?;

println!("30-day forecast: {:?}", result.forecast);
```

---

### ChangePointDetection

Detects structural breaks in time series data.

**ID**: `temporal/change-point-detection`
**Modes**: Batch, Ring

#### Input

```rust
pub struct ChangePointInput {
    pub values: Vec<f64>,
    /// Detection method
    pub method: ChangePointMethod,
    /// Minimum segment length
    pub min_segment_length: u32,
    /// Penalty factor for number of change points
    pub penalty: f64,
}

pub enum ChangePointMethod {
    PELT,      // Pruned Exact Linear Time
    BinSeg,    // Binary Segmentation
    Window,    // Sliding window
}
```

#### Output

```rust
pub struct ChangePointOutput {
    /// Indices of detected change points
    pub change_points: Vec<u32>,
    /// Segment statistics
    pub segments: Vec<SegmentStats>,
    /// Overall detection confidence
    pub confidence: f64,
}
```

---

### VolatilityAnalysis

Estimates and forecasts volatility using GARCH/EWMA models.

**ID**: `temporal/volatility-analysis`
**Modes**: Batch, Ring

#### Example

```rust
use rustkernel::temporal::volatility::{VolatilityAnalysis, VolatilityInput};

let kernel = VolatilityAnalysis::new();

let result = kernel.execute(VolatilityInput {
    returns: daily_returns,
    model: VolatilityModel::GARCH { p: 1, q: 1 },
    forecast_periods: 10,
}).await?;

println!("Current volatility: {:.4}", result.current_volatility);
println!("VaR (95%): {:.4}", result.var_95);
```

---

### SeasonalDecomposition

Decomposes time series into trend, seasonal, and residual components.

**ID**: `temporal/seasonal-decomposition`
**Modes**: Batch

#### Input

```rust
pub struct SeasonalDecompInput {
    pub values: Vec<f64>,
    /// Seasonal period (e.g., 12 for monthly, 7 for daily)
    pub period: u32,
    /// Decomposition model
    pub model: DecompModel,
}

pub enum DecompModel {
    Additive,       // y = trend + seasonal + residual
    Multiplicative, // y = trend * seasonal * residual
}
```

#### Output

```rust
pub struct SeasonalDecompOutput {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
    /// Seasonal strength measure
    pub seasonal_strength: f64,
}
```

---

### TimeSeriesAnomalyDetection

Identifies anomalies in time series using multiple detection methods.

**ID**: `temporal/anomaly-detection`
**Modes**: Batch, Ring

#### Example

```rust
use rustkernel::temporal::detection::{TimeSeriesAnomalyDetection, AnomalyInput};

let kernel = TimeSeriesAnomalyDetection::new();

let result = kernel.execute(AnomalyInput {
    values: sensor_readings,
    method: AnomalyMethod::Twitter,  // Twitter's anomaly detection
    sensitivity: 0.05,
}).await?;

for anomaly in result.anomalies {
    println!("Anomaly at index {}: value={:.2}, score={:.2}",
        anomaly.index,
        anomaly.value,
        anomaly.score
    );
}
```

---

## Ring Mode for Streaming

Ring mode enables real-time time series processing:

```rust
use rustkernel::temporal::detection::AnomalyDetectionRing;

let ring = AnomalyDetectionRing::new();

// Process streaming data points
for (timestamp, value) in data_stream {
    let result = ring.process_point(timestamp, value).await?;

    if result.is_anomaly {
        alert_system.notify(timestamp, result.anomaly_score);
    }
}
```

## Use Cases

### Financial Time Series

- Stock price forecasting
- Volatility estimation for options pricing
- Regime change detection in markets

### Operational Monitoring

- Server metric anomaly detection
- IoT sensor analysis
- Capacity planning forecasts

### Business Analytics

- Sales seasonality analysis
- Demand forecasting
- Trend identification for KPIs
