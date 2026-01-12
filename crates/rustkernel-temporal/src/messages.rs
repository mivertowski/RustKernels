//! Message types for temporal analysis kernels.
//!
//! Input/output message types for the `BatchKernel` trait implementations
//! and Ring kernel messages for K2K communication.

use rustkernel_derive::KernelMessage;
use serde::{Deserialize, Serialize};

use crate::types::{
    ARIMAParams, ARIMAResult, AnomalyMethod, ChangePointMethod, ChangePointResult,
    DecompositionResult, GARCHParams, ProphetResult, TimeSeries, TimeSeriesAnomalyResult,
    TrendMethod, TrendResult, VolatilityResult,
};

// ============================================================================
// ARIMA Forecast Messages
// ============================================================================

/// Input for ARIMA forecasting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ARIMAForecastInput {
    /// Input time series.
    pub series: TimeSeries,
    /// ARIMA(p,d,q) parameters.
    pub params: ARIMAParams,
    /// Forecast horizon.
    pub horizon: usize,
}

impl ARIMAForecastInput {
    /// Create a new ARIMA forecast input.
    pub fn new(series: TimeSeries, params: ARIMAParams, horizon: usize) -> Self {
        Self {
            series,
            params,
            horizon,
        }
    }

    /// Create with default ARIMA(1,1,1) parameters.
    pub fn with_defaults(series: TimeSeries, horizon: usize) -> Self {
        Self {
            series,
            params: ARIMAParams::new(1, 1, 1),
            horizon,
        }
    }
}

/// Output from ARIMA forecasting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ARIMAForecastOutput {
    /// ARIMA result with coefficients, fitted values, and forecasts.
    pub result: ARIMAResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Prophet Decomposition Messages
// ============================================================================

/// Input for Prophet-style decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProphetDecompositionInput {
    /// Input time series.
    pub series: TimeSeries,
    /// Seasonal period (e.g., 12 for monthly, 7 for daily).
    pub period: Option<usize>,
    /// Forecast horizon.
    pub horizon: usize,
}

impl ProphetDecompositionInput {
    /// Create a new Prophet decomposition input.
    pub fn new(series: TimeSeries, period: Option<usize>, horizon: usize) -> Self {
        Self {
            series,
            period,
            horizon,
        }
    }

    /// Create with a specified period.
    pub fn with_period(series: TimeSeries, period: usize, horizon: usize) -> Self {
        Self {
            series,
            period: Some(period),
            horizon,
        }
    }

    /// Create without seasonality.
    pub fn without_seasonality(series: TimeSeries, horizon: usize) -> Self {
        Self {
            series,
            period: None,
            horizon,
        }
    }
}

/// Output from Prophet decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProphetDecompositionOutput {
    /// Prophet result with trend, seasonal, and forecast components.
    pub result: ProphetResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Change Point Detection Messages
// ============================================================================

/// Input for change point detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangePointDetectionInput {
    /// Input time series.
    pub series: TimeSeries,
    /// Detection method (PELT, BinarySegmentation, CUSUM).
    pub method: ChangePointMethod,
    /// Penalty for adding change points.
    pub penalty: f64,
    /// Minimum segment length.
    pub min_segment: usize,
}

impl ChangePointDetectionInput {
    /// Create a new change point detection input.
    pub fn new(
        series: TimeSeries,
        method: ChangePointMethod,
        penalty: f64,
        min_segment: usize,
    ) -> Self {
        Self {
            series,
            method,
            penalty,
            min_segment,
        }
    }

    /// Create with PELT method and default parameters.
    pub fn pelt(series: TimeSeries, penalty: f64) -> Self {
        Self {
            series,
            method: ChangePointMethod::PELT,
            penalty,
            min_segment: 10,
        }
    }

    /// Create with Binary Segmentation method.
    pub fn binary_segmentation(series: TimeSeries, penalty: f64) -> Self {
        Self {
            series,
            method: ChangePointMethod::BinarySegmentation,
            penalty,
            min_segment: 10,
        }
    }

    /// Create with CUSUM method.
    pub fn cusum(series: TimeSeries, threshold: f64) -> Self {
        Self {
            series,
            method: ChangePointMethod::CUSUM,
            penalty: threshold,
            min_segment: 10,
        }
    }
}

/// Output from change point detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangePointDetectionOutput {
    /// Change point detection result.
    pub result: ChangePointResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Time Series Anomaly Detection Messages
// ============================================================================

/// Input for time series anomaly detection.
///
/// Ring message type_id: 2020 (TemporalAnalysis domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 2020, domain = "TemporalAnalysis")]
pub struct TimeSeriesAnomalyDetectionInput {
    /// Input time series.
    pub series: TimeSeries,
    /// Detection method.
    pub method: AnomalyMethod,
    /// Anomaly threshold.
    pub threshold: f64,
    /// Window size for moving statistics.
    pub window: Option<usize>,
}

impl TimeSeriesAnomalyDetectionInput {
    /// Create a new anomaly detection input.
    pub fn new(
        series: TimeSeries,
        method: AnomalyMethod,
        threshold: f64,
        window: Option<usize>,
    ) -> Self {
        Self {
            series,
            method,
            threshold,
            window,
        }
    }

    /// Create with Z-score method.
    pub fn zscore(series: TimeSeries, threshold: f64) -> Self {
        Self {
            series,
            method: AnomalyMethod::ZScore,
            threshold,
            window: None,
        }
    }

    /// Create with rolling Z-score.
    pub fn rolling_zscore(series: TimeSeries, threshold: f64, window: usize) -> Self {
        Self {
            series,
            method: AnomalyMethod::ZScore,
            threshold,
            window: Some(window),
        }
    }

    /// Create with IQR method.
    pub fn iqr(series: TimeSeries, multiplier: f64) -> Self {
        Self {
            series,
            method: AnomalyMethod::IQR,
            threshold: multiplier,
            window: None,
        }
    }

    /// Create with moving average deviation method.
    pub fn moving_average(series: TimeSeries, threshold: f64, window: usize) -> Self {
        Self {
            series,
            method: AnomalyMethod::MovingAverageDeviation,
            threshold,
            window: Some(window),
        }
    }

    /// Create with seasonal ESD method.
    pub fn seasonal_esd(series: TimeSeries, threshold: f64, period: usize) -> Self {
        Self {
            series,
            method: AnomalyMethod::SeasonalESD,
            threshold,
            window: Some(period),
        }
    }
}

/// Output from time series anomaly detection.
///
/// Ring message type_id: 2021 (TemporalAnalysis domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 2021, domain = "TemporalAnalysis")]
pub struct TimeSeriesAnomalyDetectionOutput {
    /// Anomaly detection result.
    pub result: TimeSeriesAnomalyResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Seasonal Decomposition Messages
// ============================================================================

/// Input for seasonal decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalDecompositionInput {
    /// Input time series.
    pub series: TimeSeries,
    /// Seasonal period.
    pub period: usize,
    /// Use robust (median-based) estimation.
    pub robust: bool,
}

impl SeasonalDecompositionInput {
    /// Create a new seasonal decomposition input.
    pub fn new(series: TimeSeries, period: usize, robust: bool) -> Self {
        Self {
            series,
            period,
            robust,
        }
    }

    /// Create with standard (mean-based) decomposition.
    pub fn standard(series: TimeSeries, period: usize) -> Self {
        Self {
            series,
            period,
            robust: false,
        }
    }

    /// Create with robust (median-based) decomposition.
    pub fn robust(series: TimeSeries, period: usize) -> Self {
        Self {
            series,
            period,
            robust: true,
        }
    }
}

/// Output from seasonal decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalDecompositionOutput {
    /// Decomposition result with trend, seasonal, and residual.
    pub result: DecompositionResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Trend Extraction Messages
// ============================================================================

/// Input for trend extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendExtractionInput {
    /// Input time series.
    pub series: TimeSeries,
    /// Trend extraction method.
    pub method: TrendMethod,
    /// Window size for moving average.
    pub window: usize,
}

impl TrendExtractionInput {
    /// Create a new trend extraction input.
    pub fn new(series: TimeSeries, method: TrendMethod, window: usize) -> Self {
        Self {
            series,
            method,
            window,
        }
    }

    /// Create with simple moving average.
    pub fn simple_ma(series: TimeSeries, window: usize) -> Self {
        Self {
            series,
            method: TrendMethod::SimpleMovingAverage,
            window,
        }
    }

    /// Create with exponential moving average.
    pub fn ema(series: TimeSeries, span: usize) -> Self {
        Self {
            series,
            method: TrendMethod::ExponentialMovingAverage,
            window: span,
        }
    }

    /// Create with centered moving average.
    pub fn centered_ma(series: TimeSeries, window: usize) -> Self {
        Self {
            series,
            method: TrendMethod::CenteredMovingAverage,
            window,
        }
    }

    /// Create with Lowess smoothing.
    pub fn lowess(series: TimeSeries, bandwidth: usize) -> Self {
        Self {
            series,
            method: TrendMethod::Lowess,
            window: bandwidth,
        }
    }
}

/// Output from trend extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendExtractionOutput {
    /// Trend extraction result.
    pub result: TrendResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Volatility Analysis Messages
// ============================================================================

/// Input for volatility analysis.
///
/// Ring message type_id: 2050 (TemporalAnalysis domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 2050, domain = "TemporalAnalysis")]
pub struct VolatilityAnalysisInput {
    /// Time series of returns.
    pub returns: TimeSeries,
    /// GARCH(p,q) parameters.
    pub params: GARCHParams,
    /// Forecast horizon.
    pub forecast_horizon: usize,
}

impl VolatilityAnalysisInput {
    /// Create a new volatility analysis input.
    pub fn new(returns: TimeSeries, params: GARCHParams, forecast_horizon: usize) -> Self {
        Self {
            returns,
            params,
            forecast_horizon,
        }
    }

    /// Create with GARCH(1,1) model.
    pub fn garch_1_1(returns: TimeSeries, forecast_horizon: usize) -> Self {
        Self {
            returns,
            params: GARCHParams::new(1, 1),
            forecast_horizon,
        }
    }

    /// Create with custom GARCH(p,q).
    pub fn garch(returns: TimeSeries, p: usize, q: usize, forecast_horizon: usize) -> Self {
        Self {
            returns,
            params: GARCHParams::new(p, q),
            forecast_horizon,
        }
    }
}

/// Output from volatility analysis.
///
/// Ring message type_id: 2051 (TemporalAnalysis domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 2051, domain = "TemporalAnalysis")]
pub struct VolatilityAnalysisOutput {
    /// Volatility result with variance, volatility, and forecasts.
    pub result: VolatilityResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

/// Input for EWMA volatility analysis.
///
/// Ring message type_id: 2052 (TemporalAnalysis domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 2052, domain = "TemporalAnalysis")]
pub struct EWMAVolatilityInput {
    /// Time series of returns.
    pub returns: TimeSeries,
    /// Decay factor (lambda), typically 0.94 for daily data.
    pub lambda: f64,
    /// Forecast horizon.
    pub forecast_horizon: usize,
}

impl EWMAVolatilityInput {
    /// Create a new EWMA volatility input.
    pub fn new(returns: TimeSeries, lambda: f64, forecast_horizon: usize) -> Self {
        Self {
            returns,
            lambda,
            forecast_horizon,
        }
    }

    /// Create with RiskMetrics lambda (0.94).
    pub fn riskmetrics(returns: TimeSeries, forecast_horizon: usize) -> Self {
        Self {
            returns,
            lambda: 0.94,
            forecast_horizon,
        }
    }
}

/// Output from EWMA volatility analysis.
///
/// Ring message type_id: 2053 (TemporalAnalysis domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 2053, domain = "TemporalAnalysis")]
pub struct EWMAVolatilityOutput {
    /// Volatility result.
    pub result: VolatilityResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}
