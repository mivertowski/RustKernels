//! Forecasting kernels.
//!
//! This module provides time series forecasting:
//! - ARIMA (AutoRegressive Integrated Moving Average)
//! - Prophet-style decomposition forecasting

use crate::types::{ARIMAParams, ARIMAResult, ProphetResult, TimeSeries};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

// ============================================================================
// ARIMA Forecast Kernel
// ============================================================================

/// ARIMA forecasting kernel.
///
/// Fits an ARIMA(p,d,q) model and generates forecasts.
#[derive(Debug, Clone)]
pub struct ARIMAForecast {
    metadata: KernelMetadata,
}

impl Default for ARIMAForecast {
    fn default() -> Self {
        Self::new()
    }
}

impl ARIMAForecast {
    /// Create a new ARIMA forecast kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("temporal/arima-forecast", Domain::TemporalAnalysis)
                .with_description("ARIMA model fitting and forecasting")
                .with_throughput(10_000)
                .with_latency_us(100.0),
        }
    }

    /// Fit ARIMA model and generate forecasts.
    ///
    /// # Arguments
    /// * `series` - Input time series
    /// * `params` - ARIMA(p,d,q) parameters
    /// * `horizon` - Number of steps to forecast
    pub fn compute(series: &TimeSeries, params: ARIMAParams, horizon: usize) -> ARIMAResult {
        if series.is_empty() {
            return ARIMAResult {
                ar_coefficients: Vec::new(),
                ma_coefficients: Vec::new(),
                intercept: 0.0,
                fitted: Vec::new(),
                residuals: Vec::new(),
                forecast: Vec::new(),
                aic: f64::INFINITY,
            };
        }

        // Difference the series d times
        let mut diff_series = series.values.clone();
        for _ in 0..params.d {
            diff_series = Self::difference(&diff_series);
        }

        if diff_series.len() < params.p.max(params.q) + 1 {
            return ARIMAResult {
                ar_coefficients: vec![0.0; params.p],
                ma_coefficients: vec![0.0; params.q],
                intercept: series.mean(),
                fitted: series.values.clone(),
                residuals: vec![0.0; series.len()],
                forecast: vec![series.mean(); horizon],
                aic: f64::INFINITY,
            };
        }

        // Fit AR coefficients using Yule-Walker equations (simplified)
        let ar_coefficients = if params.p > 0 {
            Self::fit_ar(&diff_series, params.p)
        } else {
            Vec::new()
        };

        // Calculate residuals from AR fit
        let ar_fitted = Self::apply_ar(&diff_series, &ar_coefficients);
        let residuals: Vec<f64> = diff_series
            .iter()
            .zip(ar_fitted.iter())
            .map(|(y, yhat)| y - yhat)
            .collect();

        // Fit MA coefficients (simplified - innovation algorithm)
        let ma_coefficients = if params.q > 0 {
            Self::fit_ma(&residuals, params.q)
        } else {
            Vec::new()
        };

        // Calculate intercept
        let intercept = diff_series.iter().sum::<f64>() / diff_series.len() as f64;

        // Generate fitted values
        let fitted = Self::generate_fitted(&diff_series, &ar_coefficients, &ma_coefficients, intercept);

        // Integrate back to original scale
        let fitted_integrated = Self::integrate(&fitted, &series.values, params.d);

        // Calculate residuals on original scale
        let final_residuals: Vec<f64> = series
            .values
            .iter()
            .zip(fitted_integrated.iter())
            .map(|(y, yhat)| y - yhat)
            .collect();

        // Generate forecasts
        let forecast = Self::forecast_ahead(
            &diff_series,
            &ar_coefficients,
            &ma_coefficients,
            intercept,
            horizon,
        );

        // Integrate forecasts
        let forecast_integrated = Self::integrate_forecast(&forecast, &series.values, params.d);

        // Calculate AIC
        let n = series.len() as f64;
        let k = (params.p + params.q + 1) as f64;
        let rss: f64 = final_residuals.iter().map(|r| r.powi(2)).sum();
        let aic = n * (rss / n).ln() + 2.0 * k;

        ARIMAResult {
            ar_coefficients,
            ma_coefficients,
            intercept,
            fitted: fitted_integrated,
            residuals: final_residuals,
            forecast: forecast_integrated,
            aic,
        }
    }

    /// Difference a series.
    fn difference(series: &[f64]) -> Vec<f64> {
        if series.len() < 2 {
            return Vec::new();
        }
        series.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Fit AR coefficients using Yule-Walker equations.
    fn fit_ar(series: &[f64], p: usize) -> Vec<f64> {
        let n = series.len();
        if n <= p {
            return vec![0.0; p];
        }

        // Calculate autocorrelations
        let mean: f64 = series.iter().sum::<f64>() / n as f64;
        let var: f64 = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        if var < 1e-10 {
            return vec![0.0; p];
        }

        let mut acf = vec![1.0; p + 1];
        for k in 1..=p {
            let cov: f64 = (0..n - k)
                .map(|i| (series[i] - mean) * (series[i + k] - mean))
                .sum::<f64>()
                / n as f64;
            acf[k] = cov / var;
        }

        // Solve Yule-Walker using Levinson-Durbin
        Self::levinson_durbin(&acf, p)
    }

    /// Levinson-Durbin algorithm for solving Yule-Walker equations.
    fn levinson_durbin(acf: &[f64], p: usize) -> Vec<f64> {
        let mut phi = vec![vec![0.0; p + 1]; p + 1];
        let mut sigma = vec![0.0; p + 1];

        sigma[0] = acf[0];

        for k in 1..=p {
            let mut num = acf[k];
            for j in 1..k {
                num -= phi[k - 1][j] * acf[k - j];
            }
            phi[k][k] = num / sigma[k - 1];

            for j in 1..k {
                phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
            }

            sigma[k] = sigma[k - 1] * (1.0 - phi[k][k].powi(2));
        }

        (1..=p).map(|j| phi[p][j]).collect()
    }

    /// Apply AR model to get fitted values.
    fn apply_ar(series: &[f64], coefficients: &[f64]) -> Vec<f64> {
        let n = series.len();
        let p = coefficients.len();
        let mut fitted = vec![0.0; n];

        for i in p..n {
            for (j, &coef) in coefficients.iter().enumerate() {
                fitted[i] += coef * series[i - j - 1];
            }
        }

        fitted
    }

    /// Fit MA coefficients (simplified).
    fn fit_ma(residuals: &[f64], q: usize) -> Vec<f64> {
        // Simplified: use autocorrelations of residuals
        let n = residuals.len();
        if n <= q {
            return vec![0.0; q];
        }

        let mean: f64 = residuals.iter().sum::<f64>() / n as f64;
        let var: f64 = residuals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        if var < 1e-10 {
            return vec![0.0; q];
        }

        let mut ma_coefs = Vec::with_capacity(q);
        for k in 1..=q {
            let cov: f64 = (0..n - k)
                .map(|i| (residuals[i] - mean) * (residuals[i + k] - mean))
                .sum::<f64>()
                / n as f64;
            ma_coefs.push(cov / var);
        }

        ma_coefs
    }

    /// Generate fitted values from ARMA model.
    fn generate_fitted(
        series: &[f64],
        ar_coefs: &[f64],
        ma_coefs: &[f64],
        intercept: f64,
    ) -> Vec<f64> {
        let n = series.len();
        let p = ar_coefs.len();
        let q = ma_coefs.len();
        let start = p.max(q);

        let mut fitted = vec![series.iter().sum::<f64>() / n as f64; n];
        let mut errors = vec![0.0; n];

        for i in start..n {
            let mut yhat = intercept;

            // AR terms
            for (j, &coef) in ar_coefs.iter().enumerate() {
                yhat += coef * series[i - j - 1];
            }

            // MA terms
            for (j, &coef) in ma_coefs.iter().enumerate() {
                if i > j {
                    yhat += coef * errors[i - j - 1];
                }
            }

            fitted[i] = yhat;
            errors[i] = series[i] - yhat;
        }

        fitted
    }

    /// Integrate differenced series back to original scale.
    fn integrate(diff_fitted: &[f64], original: &[f64], d: usize) -> Vec<f64> {
        if d == 0 || original.is_empty() {
            return diff_fitted.to_vec();
        }

        let mut result = diff_fitted.to_vec();

        for i in 0..d {
            let start_val = if i < original.len() {
                original[i]
            } else {
                0.0
            };

            let mut integrated = vec![start_val];
            for &diff in &result {
                integrated.push(integrated.last().unwrap() + diff);
            }
            result = integrated;
        }

        // Trim to original length
        result.truncate(original.len());
        result
    }

    /// Generate forecasts.
    fn forecast_ahead(
        series: &[f64],
        ar_coefs: &[f64],
        _ma_coefs: &[f64],
        intercept: f64,
        horizon: usize,
    ) -> Vec<f64> {
        let _p = ar_coefs.len();
        let mut forecasts = Vec::with_capacity(horizon);
        let mut extended = series.to_vec();

        for _ in 0..horizon {
            let mut yhat = intercept;

            // AR terms using most recent values
            for (j, &coef) in ar_coefs.iter().enumerate() {
                let idx = extended.len().saturating_sub(j + 1);
                yhat += coef * extended[idx];
            }

            // MA terms fade out as we forecast further ahead
            forecasts.push(yhat);
            extended.push(yhat);
        }

        forecasts
    }

    /// Integrate forecasts.
    fn integrate_forecast(forecasts: &[f64], original: &[f64], d: usize) -> Vec<f64> {
        if d == 0 || original.is_empty() {
            return forecasts.to_vec();
        }

        let mut result = forecasts.to_vec();
        let last_val = *original.last().unwrap_or(&0.0);

        for _ in 0..d {
            let mut integrated = vec![last_val];
            for &diff in &result {
                integrated.push(integrated.last().unwrap() + diff);
            }
            result = integrated[1..].to_vec(); // Skip the initial value
        }

        result
    }
}

impl GpuKernel for ARIMAForecast {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Prophet-style Decomposition Forecast Kernel
// ============================================================================

/// Prophet-style decomposition and forecasting kernel.
///
/// Decomposes time series into trend + seasonality and forecasts.
#[derive(Debug, Clone)]
pub struct ProphetDecomposition {
    metadata: KernelMetadata,
}

impl Default for ProphetDecomposition {
    fn default() -> Self {
        Self::new()
    }
}

impl ProphetDecomposition {
    /// Create a new Prophet decomposition kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("temporal/prophet-decomposition", Domain::TemporalAnalysis)
                .with_description("Prophet-style trend/seasonal decomposition")
                .with_throughput(5_000)
                .with_latency_us(200.0),
        }
    }

    /// Decompose and forecast time series.
    ///
    /// # Arguments
    /// * `series` - Input time series
    /// * `period` - Seasonal period (e.g., 12 for monthly, 7 for daily)
    /// * `horizon` - Forecast horizon
    pub fn compute(series: &TimeSeries, period: Option<usize>, horizon: usize) -> ProphetResult {
        if series.is_empty() {
            return ProphetResult {
                trend: Vec::new(),
                seasonal: None,
                holidays: None,
                residuals: Vec::new(),
                forecast: Vec::new(),
            };
        }

        let n = series.len();

        // Extract trend using centered moving average
        let window = period.unwrap_or(1);
        let trend = Self::extract_trend(&series.values, window);

        // Extract seasonality if period specified
        let seasonal = if let Some(p) = period {
            if p > 1 && n > p {
                Some(Self::extract_seasonal(&series.values, &trend, p))
            } else {
                None
            }
        } else {
            None
        };

        // Calculate residuals
        let residuals: Vec<f64> = series
            .values
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                let t = trend[i];
                let s = seasonal.as_ref().map(|s| s[i % s.len()]).unwrap_or(0.0);
                y - t - s
            })
            .collect();

        // Generate forecasts
        let forecast = Self::forecast(
            &trend,
            seasonal.as_ref(),
            &residuals,
            horizon,
        );

        ProphetResult {
            trend,
            seasonal,
            holidays: None,
            residuals,
            forecast,
        }
    }

    /// Extract trend using centered moving average.
    fn extract_trend(values: &[f64], window: usize) -> Vec<f64> {
        let n = values.len();
        let w = window.max(1);
        let half_w = w / 2;

        let mut trend = vec![0.0; n];

        for i in 0..n {
            let start = i.saturating_sub(half_w);
            let end = (i + half_w + 1).min(n);
            let count = end - start;

            trend[i] = values[start..end].iter().sum::<f64>() / count as f64;
        }

        trend
    }

    /// Extract seasonal component.
    fn extract_seasonal(values: &[f64], trend: &[f64], period: usize) -> Vec<f64> {
        let _n = values.len();

        // Detrend
        let detrended: Vec<f64> = values
            .iter()
            .zip(trend.iter())
            .map(|(v, t)| v - t)
            .collect();

        // Average by season
        let mut seasonal = vec![0.0; period];
        let mut counts = vec![0usize; period];

        for (i, &d) in detrended.iter().enumerate() {
            let s = i % period;
            seasonal[s] += d;
            counts[s] += 1;
        }

        for (s, &c) in counts.iter().enumerate() {
            if c > 0 {
                seasonal[s] /= c as f64;
            }
        }

        // Center seasonal (subtract mean)
        let mean: f64 = seasonal.iter().sum::<f64>() / period as f64;
        for s in &mut seasonal {
            *s -= mean;
        }

        seasonal
    }

    /// Generate forecasts.
    fn forecast(
        trend: &[f64],
        seasonal: Option<&Vec<f64>>,
        _residuals: &[f64],
        horizon: usize,
    ) -> Vec<f64> {
        let n = trend.len();
        if n < 2 {
            return vec![trend.last().copied().unwrap_or(0.0); horizon];
        }

        // Extrapolate trend linearly
        let slope = trend[n - 1] - trend[n - 2];
        let last_trend = trend[n - 1];

        let mut forecasts = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            let trend_forecast = last_trend + slope * h as f64;
            let seasonal_forecast = seasonal
                .map(|s| s[(n + h - 1) % s.len()])
                .unwrap_or(0.0);
            forecasts.push(trend_forecast + seasonal_forecast);
        }

        forecasts
    }
}

impl GpuKernel for ProphetDecomposition {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_trend_series() -> TimeSeries {
        // Linear trend: y = 10 + 2*t
        TimeSeries::new((0..50).map(|t| 10.0 + 2.0 * t as f64).collect())
    }

    fn create_seasonal_series() -> TimeSeries {
        // Trend + seasonal pattern
        let period = 12;
        let values: Vec<f64> = (0..60)
            .map(|t| {
                let trend = 100.0 + 0.5 * t as f64;
                let seasonal = 10.0 * ((2.0 * std::f64::consts::PI * t as f64 / period as f64).sin());
                trend + seasonal
            })
            .collect();
        TimeSeries::new(values)
    }

    #[test]
    fn test_arima_metadata() {
        let kernel = ARIMAForecast::new();
        assert_eq!(kernel.metadata().id, "temporal/arima-forecast");
        assert_eq!(kernel.metadata().domain, Domain::TemporalAnalysis);
    }

    #[test]
    fn test_arima_forecast_trend() {
        let series = create_trend_series();
        let params = ARIMAParams::new(1, 1, 0); // AR(1) with differencing
        let result = ARIMAForecast::compute(&series, params, 5);

        assert_eq!(result.forecast.len(), 5);
        assert!(!result.ar_coefficients.is_empty() || params.p == 0);

        // Forecasts should continue the trend
        let last = *series.values.last().unwrap();
        for f in &result.forecast {
            assert!(*f > last * 0.8); // Should be reasonably close to continuation
        }
    }

    #[test]
    fn test_prophet_metadata() {
        let kernel = ProphetDecomposition::new();
        assert_eq!(kernel.metadata().id, "temporal/prophet-decomposition");
    }

    #[test]
    fn test_prophet_decomposition() {
        let series = create_seasonal_series();
        let result = ProphetDecomposition::compute(&series, Some(12), 12);

        // Should have all components
        assert_eq!(result.trend.len(), series.len());
        assert!(result.seasonal.is_some());
        assert_eq!(result.seasonal.as_ref().unwrap().len(), 12);
        assert_eq!(result.forecast.len(), 12);
    }

    #[test]
    fn test_prophet_no_seasonality() {
        let series = create_trend_series();
        let result = ProphetDecomposition::compute(&series, None, 5);

        // No seasonal component
        assert!(result.seasonal.is_none());
        assert_eq!(result.forecast.len(), 5);
    }

    #[test]
    fn test_empty_series() {
        let empty = TimeSeries::new(Vec::new());

        let arima = ARIMAForecast::compute(&empty, ARIMAParams::new(1, 0, 0), 5);
        assert!(arima.forecast.is_empty());

        let prophet = ProphetDecomposition::compute(&empty, Some(12), 5);
        assert!(prophet.forecast.is_empty());
    }
}
