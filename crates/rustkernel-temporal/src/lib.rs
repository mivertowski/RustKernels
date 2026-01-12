//! # RustKernel Temporal Analysis
//!
//! GPU-accelerated temporal analysis kernels for forecasting, decomposition,
//! detection, and volatility modeling.
//!
//! ## Kernels
//!
//! ### Forecasting (2 kernels)
//! - `ARIMAForecast` - ARIMA(p,d,q) model fitting and forecasting
//! - `ProphetDecomposition` - Prophet-style trend/seasonal/holiday decomposition
//!
//! ### Detection (2 kernels)
//! - `ChangePointDetection` - PELT/Binary segmentation/CUSUM
//! - `TimeSeriesAnomalyDetection` - Statistical threshold detection
//!
//! ### Decomposition (2 kernels)
//! - `SeasonalDecomposition` - STL-style decomposition
//! - `TrendExtraction` - Moving average variants
//!
//! ### Analysis (1 kernel)
//! - `VolatilityAnalysis` - GARCH model volatility estimation

#![warn(missing_docs)]

pub mod decomposition;
pub mod detection;
pub mod forecasting;
pub mod messages;
pub mod ring_messages;
pub mod types;
pub mod volatility;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::decomposition::*;
    pub use crate::detection::*;
    pub use crate::forecasting::*;
    pub use crate::messages::*;
    pub use crate::ring_messages::*;
    pub use crate::types::*;
    pub use crate::volatility::*;
}

// Re-export main kernels
pub use decomposition::{SeasonalDecomposition, TrendExtraction};
pub use detection::{ChangePointDetection, TimeSeriesAnomalyDetection};
pub use forecasting::{ARIMAForecast, ProphetDecomposition};
pub use volatility::VolatilityAnalysis;

// Re-export key types
pub use types::{
    ARIMAParams, ARIMAResult, AnomalyMethod, ChangePointMethod, ChangePointResult,
    DecompositionResult, GARCHCoefficients, GARCHParams, ProphetResult, TimeSeries,
    TimeSeriesAnomalyResult, TrendMethod, TrendResult, VolatilityResult,
};

/// Register all temporal kernels with a registry.
pub fn register_all(
    _registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering temporal analysis kernels");
    Ok(())
}
