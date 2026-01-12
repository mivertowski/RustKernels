//! Ring message types for Temporal Analysis domain kernels.
//!
//! These messages implement the `RingMessage` trait for GPU-native persistent
//! actor communication in volatility analysis and temporal operations.
//!
//! Type ID range: 400-499 (Temporal Analysis domain)
//!
//! ## Type ID Assignments
//! - 400-409: VolatilityAnalysis messages
//! - 410-419: Reserved for ARIMA
//! - 420-429: Reserved for ChangePointDetection
//! - 430-439: Reserved for SeasonalDecomposition

use ringkernel_core::message::{CorrelationId, MessageId};
use ringkernel_derive::RingMessage;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

// ============================================================================
// Volatility Analysis Ring Messages (400-409)
// ============================================================================

/// Ring message for updating volatility model with new return data.
///
/// Type ID: 400
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 400)]
#[archive(check_bytes)]
pub struct UpdateVolatilityRing {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Asset/instrument ID.
    pub asset_id: u64,
    /// Return value (fixed-point, 8 decimals).
    pub return_value: i64,
    /// Timestamp (nanoseconds since epoch).
    pub timestamp: u64,
}

impl UpdateVolatilityRing {
    /// Create a new volatility update message.
    pub fn new(asset_id: u64, return_value: f64, timestamp: u64) -> Self {
        Self {
            id: MessageId::generate(),
            correlation_id: CorrelationId::generate(),
            asset_id,
            return_value: (return_value * 100_000_000.0) as i64,
            timestamp,
        }
    }

    /// Get return value as f64.
    pub fn return_f64(&self) -> f64 {
        self.return_value as f64 / 100_000_000.0
    }
}

/// Response from volatility update.
///
/// Type ID: 401
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 401)]
#[archive(check_bytes)]
pub struct UpdateVolatilityResponse {
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Asset ID.
    pub asset_id: u64,
    /// Current volatility estimate (fixed-point, 8 decimals).
    pub current_volatility: i64,
    /// Current variance (fixed-point, 8 decimals).
    pub current_variance: i64,
    /// Number of observations in model.
    pub observation_count: u32,
}

impl UpdateVolatilityResponse {
    /// Get volatility as f64.
    pub fn volatility_f64(&self) -> f64 {
        self.current_volatility as f64 / 100_000_000.0
    }

    /// Get variance as f64.
    pub fn variance_f64(&self) -> f64 {
        self.current_variance as f64 / 100_000_000.0
    }
}

/// Ring message for querying current volatility forecast.
///
/// Type ID: 402
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 402)]
#[archive(check_bytes)]
pub struct QueryVolatilityRing {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Asset ID to query.
    pub asset_id: u64,
    /// Forecast horizon (number of periods).
    pub horizon: u32,
}

impl QueryVolatilityRing {
    /// Create a new volatility query message.
    pub fn new(asset_id: u64, horizon: u32) -> Self {
        Self {
            id: MessageId::generate(),
            correlation_id: CorrelationId::generate(),
            asset_id,
            horizon,
        }
    }
}

/// Response with volatility forecast.
///
/// Type ID: 403
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 403)]
#[archive(check_bytes)]
pub struct QueryVolatilityResponse {
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Asset ID.
    pub asset_id: u64,
    /// Current volatility (fixed-point).
    pub current_volatility: i64,
    /// Forecasted volatilities (up to 10 periods, fixed-point).
    pub forecast: [i64; 10],
    /// Number of valid forecast periods.
    pub forecast_count: u8,
    /// GARCH persistence (alpha + beta).
    pub persistence: i32, // Fixed-point, 4 decimals
}

impl QueryVolatilityResponse {
    /// Get forecast as Vec<f64>.
    pub fn forecast_f64(&self) -> Vec<f64> {
        self.forecast[..self.forecast_count as usize]
            .iter()
            .map(|&v| v as f64 / 100_000_000.0)
            .collect()
    }
}

/// Volatility spike alert.
///
/// Type ID: 404
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 404)]
#[archive(check_bytes)]
pub struct VolatilitySpikeAlert {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Asset ID.
    pub asset_id: u64,
    /// Current volatility (fixed-point).
    pub current_volatility: i64,
    /// Previous volatility (fixed-point).
    pub previous_volatility: i64,
    /// Spike ratio (current / previous, fixed-point 4 decimals).
    pub spike_ratio: i32,
    /// Timestamp.
    pub timestamp: u64,
    /// Alert severity (1-5).
    pub severity: u8,
}

// ============================================================================
// EWMA Volatility Ring Messages
// ============================================================================

/// Ring message for EWMA volatility update.
///
/// Type ID: 405
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 405)]
#[archive(check_bytes)]
pub struct UpdateEWMAVolatilityRing {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Asset ID.
    pub asset_id: u64,
    /// Return value (fixed-point).
    pub return_value: i64,
    /// Lambda decay factor (fixed-point, 4 decimals).
    /// Default: 9400 (0.94)
    pub lambda: u16,
    /// Timestamp.
    pub timestamp: u64,
}

impl UpdateEWMAVolatilityRing {
    /// Create a new EWMA update message with default lambda (0.94).
    pub fn new(asset_id: u64, return_value: f64, timestamp: u64) -> Self {
        Self {
            id: MessageId::generate(),
            correlation_id: CorrelationId::generate(),
            asset_id,
            return_value: (return_value * 100_000_000.0) as i64,
            lambda: 9400, // 0.94
            timestamp,
        }
    }

    /// Create with custom lambda.
    pub fn with_lambda(asset_id: u64, return_value: f64, lambda: f64, timestamp: u64) -> Self {
        Self {
            id: MessageId::generate(),
            correlation_id: CorrelationId::generate(),
            asset_id,
            return_value: (return_value * 100_000_000.0) as i64,
            lambda: (lambda * 10000.0) as u16,
            timestamp,
        }
    }

    /// Get lambda as f64.
    pub fn lambda_f64(&self) -> f64 {
        self.lambda as f64 / 10000.0
    }
}

/// Response from EWMA update.
///
/// Type ID: 406
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 406)]
#[archive(check_bytes)]
pub struct UpdateEWMAVolatilityResponse {
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Asset ID.
    pub asset_id: u64,
    /// Current EWMA variance (fixed-point).
    pub ewma_variance: i64,
    /// Current EWMA volatility (fixed-point).
    pub ewma_volatility: i64,
}

// ============================================================================
// Model Coefficients Messages
// ============================================================================

/// Ring message to set GARCH coefficients.
///
/// Type ID: 407
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 407)]
#[archive(check_bytes)]
pub struct SetGARCHCoefficientsRing {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Asset ID.
    pub asset_id: u64,
    /// Omega (constant term, fixed-point 8 decimals).
    pub omega: i64,
    /// Alpha (ARCH coefficient, fixed-point 4 decimals).
    pub alpha: i32,
    /// Beta (GARCH coefficient, fixed-point 4 decimals).
    pub beta: i32,
}

impl SetGARCHCoefficientsRing {
    /// Create a new set coefficients message.
    pub fn new(asset_id: u64, omega: f64, alpha: f64, beta: f64) -> Self {
        Self {
            id: MessageId::generate(),
            correlation_id: CorrelationId::generate(),
            asset_id,
            omega: (omega * 100_000_000.0) as i64,
            alpha: (alpha * 10000.0) as i32,
            beta: (beta * 10000.0) as i32,
        }
    }
}

/// Response from setting coefficients.
///
/// Type ID: 408
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 408)]
#[archive(check_bytes)]
pub struct SetGARCHCoefficientsResponse {
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Asset ID.
    pub asset_id: u64,
    /// Whether update succeeded.
    pub success: bool,
    /// Long-run variance implied by coefficients (fixed-point).
    pub long_run_variance: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_volatility_ring() {
        let msg = UpdateVolatilityRing::new(1, 0.015, 1234567890);
        assert_eq!(msg.asset_id, 1);
        assert_eq!(msg.return_value, 1_500_000); // 0.015 * 10^8
        assert!((msg.return_f64() - 0.015).abs() < 1e-10);
    }

    #[test]
    fn test_query_volatility_ring() {
        let msg = QueryVolatilityRing::new(42, 10);
        assert_eq!(msg.asset_id, 42);
        assert_eq!(msg.horizon, 10);
    }

    #[test]
    fn test_ewma_with_lambda() {
        let msg = UpdateEWMAVolatilityRing::with_lambda(1, 0.02, 0.97, 1234567890);
        assert_eq!(msg.lambda, 9700);
        assert!((msg.lambda_f64() - 0.97).abs() < 1e-4);
    }

    #[test]
    fn test_garch_coefficients() {
        let msg = SetGARCHCoefficientsRing::new(1, 0.00001, 0.1, 0.85);
        assert_eq!(msg.asset_id, 1);
        assert_eq!(msg.alpha, 1000); // 0.1 * 10000
        assert_eq!(msg.beta, 8500); // 0.85 * 10000
    }
}
