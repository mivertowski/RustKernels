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
    registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    use rustkernel_core::traits::GpuKernel;

    tracing::info!("Registering temporal analysis kernels");

    // Forecasting kernels (2)
    registry.register_metadata(forecasting::ARIMAForecast::new().metadata().clone())?;
    registry.register_metadata(forecasting::ProphetDecomposition::new().metadata().clone())?;

    // Detection kernels (2)
    registry.register_metadata(detection::ChangePointDetection::new().metadata().clone())?;
    registry.register_metadata(
        detection::TimeSeriesAnomalyDetection::new()
            .metadata()
            .clone(),
    )?;

    // Decomposition kernels (2)
    registry.register_metadata(
        decomposition::SeasonalDecomposition::new()
            .metadata()
            .clone(),
    )?;
    registry.register_metadata(decomposition::TrendExtraction::new().metadata().clone())?;

    // Volatility kernel (1)
    registry.register_metadata(volatility::VolatilityAnalysis::new().metadata().clone())?;

    tracing::info!("Registered 7 temporal analysis kernels");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustkernel_core::registry::KernelRegistry;

    #[test]
    fn test_register_all() {
        let registry = KernelRegistry::new();
        register_all(&registry).expect("Failed to register temporal kernels");
        assert_eq!(registry.total_count(), 7);
    }
}
