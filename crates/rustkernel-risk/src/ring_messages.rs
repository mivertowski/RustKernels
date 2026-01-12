//! Ring message types for Risk Analytics kernels.
//!
//! This module defines zero-copy Ring messages for GPU-native persistent actors.
//! Type IDs 600-699 are reserved for Risk Analytics domain.
//!
//! ## Type ID Allocation
//!
//! - 600-619: Monte Carlo VaR messages
//! - 620-639: Portfolio risk aggregation messages
//! - 640-659: Credit risk messages
//! - 660-679: K2K streaming coordination messages

use ringkernel_derive::RingMessage;
use rkyv::{Archive, Deserialize, Serialize};
use rustkernel_core::messages::MessageId;

// ============================================================================
// Monte Carlo VaR Ring Messages (600-619)
// ============================================================================

/// Update position for streaming VaR calculation.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 600)]
pub struct UpdatePositionRing {
    /// Message ID.
    pub id: MessageId,
    /// Asset ID.
    pub asset_id: u64,
    /// New position value (fixed-point: value * 100 for 2 decimal places).
    pub value_fp: i64,
    /// Expected return (fixed-point: value * 100_000_000).
    pub expected_return_fp: i64,
    /// Volatility (fixed-point: value * 100_000_000).
    pub volatility_fp: i64,
}

/// Position update response.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 601)]
pub struct UpdatePositionResponse {
    /// Original message ID.
    pub request_id: u64,
    /// Asset ID updated.
    pub asset_id: u64,
    /// Whether VaR needs recalculation.
    pub var_stale: bool,
}

/// Query current VaR value.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 602)]
pub struct QueryVaRRing {
    /// Message ID.
    pub id: MessageId,
    /// Confidence level (fixed-point: value * 100_000_000, e.g., 0.95 = 95_000_000).
    pub confidence_fp: i64,
    /// Holding period in days.
    pub holding_period: u32,
}

/// VaR query response.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 603)]
pub struct QueryVaRResponse {
    /// Original message ID.
    pub request_id: u64,
    /// Value at Risk (fixed-point: value * 100).
    pub var_fp: i64,
    /// Expected Shortfall (fixed-point: value * 100).
    pub es_fp: i64,
    /// Confidence level.
    pub confidence_fp: i64,
    /// Holding period.
    pub holding_period: u32,
    /// Whether this is a fresh calculation.
    pub is_fresh: bool,
}

/// Trigger VaR recalculation.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 604)]
pub struct RecalculateVaRRing {
    /// Message ID.
    pub id: MessageId,
    /// Number of simulations.
    pub n_simulations: u32,
    /// Confidence level (fixed-point).
    pub confidence_fp: i64,
    /// Holding period.
    pub holding_period: u32,
}

/// VaR recalculation response.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 605)]
pub struct RecalculateVaRResponse {
    /// Original message ID.
    pub request_id: u64,
    /// New VaR value (fixed-point).
    pub var_fp: i64,
    /// New ES value (fixed-point).
    pub es_fp: i64,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
    /// Number of simulations used.
    pub n_simulations: u32,
}

// ============================================================================
// K2K Streaming Coordination Messages (660-679)
// ============================================================================

/// K2K position batch update for distributed VaR.
///
/// Used when positions are partitioned across multiple workers.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 660)]
pub struct K2KPositionBatch {
    /// Message ID.
    pub id: MessageId,
    /// Source worker ID.
    pub source_worker: u64,
    /// Batch sequence number.
    pub batch_seq: u64,
    /// Number of positions in batch.
    pub position_count: u32,
    /// Packed asset IDs (up to 8).
    pub asset_ids: [u64; 8],
    /// Packed values (fixed-point, up to 8).
    pub values_fp: [i64; 8],
}

/// K2K partial VaR result from a worker.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 661)]
pub struct K2KPartialVaR {
    /// Message ID.
    pub id: MessageId,
    /// Worker ID.
    pub worker_id: u64,
    /// Correlation ID for the calculation request.
    pub correlation_id: u64,
    /// Partial VaR contribution (fixed-point).
    pub partial_var_fp: i64,
    /// Partial ES contribution (fixed-point).
    pub partial_es_fp: i64,
    /// Number of positions processed.
    pub positions_processed: u32,
    /// Covariance contribution term (fixed-point).
    pub cov_contribution_fp: i64,
}

/// K2K VaR aggregation request.
///
/// Sent to aggregator to combine partial VaR results.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 662)]
pub struct K2KVaRAggregation {
    /// Message ID.
    pub id: MessageId,
    /// Correlation ID.
    pub correlation_id: u64,
    /// Number of workers expected.
    pub expected_workers: u32,
    /// Workers that have reported.
    pub workers_reported: u32,
    /// Aggregated VaR so far (fixed-point).
    pub aggregated_var_fp: i64,
}

/// K2K VaR aggregation response.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 663)]
pub struct K2KVaRAggregationResponse {
    /// Original correlation ID.
    pub correlation_id: u64,
    /// All workers reported.
    pub complete: bool,
    /// Final aggregated VaR (fixed-point).
    pub final_var_fp: i64,
    /// Final aggregated ES (fixed-point).
    pub final_es_fp: i64,
    /// Diversification benefit (fixed-point).
    pub diversification_benefit_fp: i64,
}

/// K2K streaming market data update.
///
/// Broadcasts market data updates to all VaR workers.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 664)]
pub struct K2KMarketUpdate {
    /// Message ID.
    pub id: MessageId,
    /// Update timestamp (microseconds since epoch).
    pub timestamp_us: u64,
    /// Asset ID.
    pub asset_id: u64,
    /// New price (fixed-point: value * 100).
    pub price_fp: i64,
    /// Implied volatility change (fixed-point: delta * 100_000_000).
    pub vol_delta_fp: i64,
}

/// K2K market update acknowledgment.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 665)]
pub struct K2KMarketUpdateAck {
    /// Original message ID.
    pub request_id: u64,
    /// Worker ID acknowledging.
    pub worker_id: u64,
    /// Updated VaR impact estimate (fixed-point).
    pub var_impact_fp: i64,
}

/// K2K risk limit breach alert.
///
/// Sent when a position update causes VaR to breach limits.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, RingMessage)]
#[archive(check_bytes)]
#[message(type_id = 666)]
pub struct K2KRiskLimitAlert {
    /// Message ID.
    pub id: MessageId,
    /// Alert timestamp (microseconds since epoch).
    pub timestamp_us: u64,
    /// Alert severity: 1=warning, 2=breach, 3=critical.
    pub severity: u8,
    /// Current VaR (fixed-point).
    pub current_var_fp: i64,
    /// VaR limit (fixed-point).
    pub var_limit_fp: i64,
    /// Breach amount (fixed-point).
    pub breach_amount_fp: i64,
    /// Triggering asset ID.
    pub trigger_asset_id: u64,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert f64 to fixed-point i64 (8 decimal places).
#[inline]
pub fn to_fixed_point(value: f64) -> i64 {
    (value * 100_000_000.0) as i64
}

/// Convert fixed-point i64 to f64.
#[inline]
pub fn from_fixed_point(fp: i64) -> f64 {
    fp as f64 / 100_000_000.0
}

/// Convert value to fixed-point with 2 decimal places (for currency).
#[inline]
pub fn to_currency_fp(value: f64) -> i64 {
    (value * 100.0) as i64
}

/// Convert currency fixed-point to f64.
#[inline]
pub fn from_currency_fp(fp: i64) -> f64 {
    fp as f64 / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_conversion() {
        let value = 0.95;
        let fp = to_fixed_point(value);
        let back = from_fixed_point(fp);
        assert!((value - back).abs() < 1e-8);
    }

    #[test]
    fn test_currency_conversion() {
        // Use a value that can be exactly represented
        let value = 50000.50;
        let fp = to_currency_fp(value);
        let back = from_currency_fp(fp);
        assert!((value - back).abs() < 0.01);
        assert_eq!(fp, 5000050); // 50000.50 * 100 = 5000050
    }

    #[test]
    fn test_update_position_ring() {
        let msg = UpdatePositionRing {
            id: MessageId(1),
            asset_id: 100,
            value_fp: to_currency_fp(50000.0),
            expected_return_fp: to_fixed_point(0.08),
            volatility_fp: to_fixed_point(0.20),
        };
        assert_eq!(msg.asset_id, 100);
        assert!((from_currency_fp(msg.value_fp) - 50000.0).abs() < 0.01);
    }

    #[test]
    fn test_k2k_partial_var() {
        let msg = K2KPartialVaR {
            id: MessageId(2),
            worker_id: 1,
            correlation_id: 12345,
            partial_var_fp: to_currency_fp(10000.0),
            partial_es_fp: to_currency_fp(12000.0),
            positions_processed: 50,
            cov_contribution_fp: to_fixed_point(0.0015),
        };
        assert_eq!(msg.worker_id, 1);
        assert_eq!(msg.positions_processed, 50);
    }

    #[test]
    fn test_k2k_risk_limit_alert() {
        let msg = K2KRiskLimitAlert {
            id: MessageId(3),
            timestamp_us: 1234567890,
            severity: 2,
            current_var_fp: to_currency_fp(1_100_000.0),
            var_limit_fp: to_currency_fp(1_000_000.0),
            breach_amount_fp: to_currency_fp(100_000.0),
            trigger_asset_id: 42,
        };
        assert_eq!(msg.severity, 2);
        assert!((from_currency_fp(msg.breach_amount_fp) - 100_000.0).abs() < 0.01);
    }
}
