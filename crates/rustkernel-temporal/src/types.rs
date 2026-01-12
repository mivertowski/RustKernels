//! Temporal analysis types and data structures.

// serde support available via feature flag

// ============================================================================
// Time Series Types
// ============================================================================

/// A time series data structure.
#[derive(Debug, Clone)]
pub struct TimeSeries {
    /// Values in the series.
    pub values: Vec<f64>,
    /// Timestamps (optional, defaults to sequential indices).
    pub timestamps: Option<Vec<u64>>,
    /// Frequency in seconds (optional).
    pub frequency: Option<u64>,
}

impl TimeSeries {
    /// Create a new time series from values.
    pub fn new(values: Vec<f64>) -> Self {
        Self {
            values,
            timestamps: None,
            frequency: None,
        }
    }

    /// Create a time series with timestamps.
    pub fn with_timestamps(values: Vec<f64>, timestamps: Vec<u64>) -> Self {
        Self {
            values,
            timestamps: Some(timestamps),
            frequency: None,
        }
    }

    /// Get the length of the series.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the series is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get the mean of the series.
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// Get the variance of the series.
    pub fn variance(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        self.values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (self.values.len() - 1) as f64
    }

    /// Get the standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

// ============================================================================
// Forecasting Types
// ============================================================================

/// ARIMA model parameters.
#[derive(Debug, Clone, Copy)]
pub struct ARIMAParams {
    /// AR order (p).
    pub p: usize,
    /// Differencing order (d).
    pub d: usize,
    /// MA order (q).
    pub q: usize,
}

impl ARIMAParams {
    /// Create new ARIMA parameters.
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self { p, d, q }
    }
}

/// Result of ARIMA fitting and forecasting.
#[derive(Debug, Clone)]
pub struct ARIMAResult {
    /// AR coefficients.
    pub ar_coefficients: Vec<f64>,
    /// MA coefficients.
    pub ma_coefficients: Vec<f64>,
    /// Intercept/constant term.
    pub intercept: f64,
    /// Fitted values.
    pub fitted: Vec<f64>,
    /// Residuals.
    pub residuals: Vec<f64>,
    /// Forecasted values.
    pub forecast: Vec<f64>,
    /// AIC (Akaike Information Criterion).
    pub aic: f64,
}

/// Prophet-style decomposition result.
#[derive(Debug, Clone)]
pub struct ProphetResult {
    /// Trend component.
    pub trend: Vec<f64>,
    /// Seasonal component (if present).
    pub seasonal: Option<Vec<f64>>,
    /// Holiday/event effects (if present).
    pub holidays: Option<Vec<f64>>,
    /// Residuals.
    pub residuals: Vec<f64>,
    /// Forecast values.
    pub forecast: Vec<f64>,
}

// ============================================================================
// Decomposition Types
// ============================================================================

/// Seasonal decomposition result (STL-like).
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    /// Trend component.
    pub trend: Vec<f64>,
    /// Seasonal component.
    pub seasonal: Vec<f64>,
    /// Residual component.
    pub residual: Vec<f64>,
    /// Original series length.
    pub n: usize,
    /// Seasonal period used.
    pub period: usize,
}

/// Type of trend extraction method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendMethod {
    /// Simple moving average.
    SimpleMovingAverage,
    /// Exponential moving average.
    ExponentialMovingAverage,
    /// Centered moving average.
    CenteredMovingAverage,
    /// Lowess smoothing.
    Lowess,
}

/// Trend extraction result.
#[derive(Debug, Clone)]
pub struct TrendResult {
    /// Extracted trend.
    pub trend: Vec<f64>,
    /// Detrended series.
    pub detrended: Vec<f64>,
    /// Method used.
    pub method: TrendMethod,
}

// ============================================================================
// Detection Types
// ============================================================================

/// Change point detection result.
#[derive(Debug, Clone)]
pub struct ChangePointResult {
    /// Indices of detected change points.
    pub change_points: Vec<usize>,
    /// Confidence scores for each change point (0-1).
    pub confidence: Vec<f64>,
    /// Segment means.
    pub segment_means: Vec<f64>,
    /// Segment variances.
    pub segment_variances: Vec<f64>,
    /// Total cost (for PELT).
    pub cost: f64,
}

/// Change point detection method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangePointMethod {
    /// PELT (Pruned Exact Linear Time).
    PELT,
    /// Binary segmentation.
    BinarySegmentation,
    /// CUSUM (Cumulative Sum).
    CUSUM,
}

/// Anomaly detection result for time series.
#[derive(Debug, Clone)]
pub struct TimeSeriesAnomalyResult {
    /// Anomaly scores per point.
    pub scores: Vec<f64>,
    /// Indices of detected anomalies.
    pub anomaly_indices: Vec<usize>,
    /// Expected values (for context).
    pub expected: Vec<f64>,
    /// Threshold used for detection.
    pub threshold: f64,
}

/// Method for time series anomaly detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyMethod {
    /// Z-score based.
    ZScore,
    /// IQR (Interquartile Range) based.
    IQR,
    /// Moving average deviation.
    MovingAverageDeviation,
    /// Seasonal hybrid ESD (Twitter's algorithm).
    SeasonalESD,
}

// ============================================================================
// Volatility Types
// ============================================================================

/// GARCH model parameters.
#[derive(Debug, Clone, Copy)]
pub struct GARCHParams {
    /// ARCH order (p).
    pub p: usize,
    /// GARCH order (q).
    pub q: usize,
}

impl GARCHParams {
    /// Create new GARCH parameters.
    pub fn new(p: usize, q: usize) -> Self {
        Self { p, q }
    }
}

/// Volatility analysis result.
#[derive(Debug, Clone)]
pub struct VolatilityResult {
    /// Estimated conditional variance series.
    pub variance: Vec<f64>,
    /// Estimated volatility (sqrt of variance).
    pub volatility: Vec<f64>,
    /// GARCH coefficients (omega, alpha, beta).
    pub coefficients: GARCHCoefficients,
    /// Forecasted volatility.
    pub forecast: Vec<f64>,
}

/// GARCH model coefficients.
#[derive(Debug, Clone)]
pub struct GARCHCoefficients {
    /// Omega (constant term).
    pub omega: f64,
    /// Alpha coefficients (ARCH terms).
    pub alpha: Vec<f64>,
    /// Beta coefficients (GARCH terms).
    pub beta: Vec<f64>,
}
