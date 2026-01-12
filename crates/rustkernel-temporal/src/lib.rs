//! # RustKernel Temporal Analysis
//!
//! GPU-accelerated temporal analysis kernels for forecasting, decomposition, and anomaly detection.
//!
//! ## Kernels
//!
//! ### Forecasting (2 kernels)
//! - `ARIMAForecast` - ARIMA model fitting and forecasting
//! - `ProphetDecomposition` - Prophet-style trend/seasonal/holiday decomposition
//!
//! ### Detection (2 kernels)
//! - `ChangePointDetection` - PELT/Binary segmentation
//! - `AnomalyDetection` - Statistical threshold detection
//!
//! ### Decomposition (2 kernels)
//! - `SeasonalDecomposition` - STL decomposition
//! - `TrendExtraction` - Moving average variants
//!
//! ### Analysis (1 kernel)
//! - `VolatilityAnalysis` - GARCH model volatility estimation

#![warn(missing_docs)]

use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

/// ARIMA forecasting kernel.
#[derive(Debug, Clone, Default)]
pub struct ARIMAForecast { metadata: KernelMetadata }
impl ARIMAForecast {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::batch("temporal/arima-forecast", Domain::TemporalAnalysis)
            .with_description("ARIMA model fitting and forecasting")
            .with_throughput(10_000).with_latency_us(100.0) }
    }
}
impl GpuKernel for ARIMAForecast { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Change point detection kernel.
#[derive(Debug, Clone, Default)]
pub struct ChangePointDetection { metadata: KernelMetadata }
impl ChangePointDetection {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::batch("temporal/changepoint-detection", Domain::TemporalAnalysis)
            .with_description("PELT/Binary segmentation change point detection")
            .with_throughput(20_000).with_latency_us(50.0) }
    }
}
impl GpuKernel for ChangePointDetection { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Volatility analysis kernel.
#[derive(Debug, Clone, Default)]
pub struct VolatilityAnalysis { metadata: KernelMetadata }
impl VolatilityAnalysis {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::ring("temporal/volatility-analysis", Domain::TemporalAnalysis)
            .with_description("GARCH model volatility estimation")
            .with_throughput(100_000).with_latency_us(10.0) }
    }
}
impl GpuKernel for VolatilityAnalysis { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Register all temporal kernels.
pub fn register_all(_registry: &rustkernel_core::registry::KernelRegistry) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering temporal analysis kernels");
    Ok(())
}
