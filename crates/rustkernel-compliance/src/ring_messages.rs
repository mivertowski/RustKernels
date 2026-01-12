//! Ring message types for Compliance domain kernels.
//!
//! These messages implement the `RingMessage` trait for GPU-native persistent
//! actor communication in compliance/AML operations.
//!
//! Type ID range: 300-399 (Compliance domain)
//!
//! ## Type ID Assignments
//! - 300-309: TransactionMonitoring messages
//! - 310-319: CircularFlowRatio messages
//! - 320-329: AMLPatternDetection messages
//! - 330-339: Reserved for KYC
//! - 340-349: Reserved for Sanctions

use ringkernel_core::message::{CorrelationId, MessageId};
use ringkernel_derive::RingMessage;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

// ============================================================================
// Transaction Monitoring Ring Messages (300-309)
// ============================================================================

/// Ring message for real-time transaction monitoring.
///
/// Type ID: 300
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 300)]
#[archive(check_bytes)]
pub struct MonitorTransactionRing {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Transaction ID.
    pub tx_id: u64,
    /// Source entity ID.
    pub source_id: u64,
    /// Destination entity ID.
    pub dest_id: u64,
    /// Transaction amount (fixed-point, 8 decimals).
    pub amount: i64,
    /// Timestamp (nanoseconds since epoch).
    pub timestamp: u64,
    /// Transaction type code.
    pub tx_type: u8,
    /// Currency code (3-letter ISO).
    pub currency: [u8; 4],
}

impl MonitorTransactionRing {
    /// Create a new transaction monitoring message.
    pub fn new(
        tx_id: u64,
        source_id: u64,
        dest_id: u64,
        amount: f64,
        timestamp: u64,
        tx_type: u8,
        currency: &str,
    ) -> Self {
        let mut curr = [0u8; 4];
        let bytes = currency.as_bytes();
        let len = bytes.len().min(3);
        curr[..len].copy_from_slice(&bytes[..len]);

        Self {
            id: MessageId::generate(),
            correlation_id: CorrelationId::generate(),
            tx_id,
            source_id,
            dest_id,
            amount: (amount * 100_000_000.0) as i64,
            timestamp,
            tx_type,
            currency: curr,
        }
    }
}

/// Response from transaction monitoring.
///
/// Type ID: 301
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 301)]
#[archive(check_bytes)]
pub struct MonitorTransactionResponse {
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Transaction ID.
    pub tx_id: u64,
    /// Number of alerts generated.
    pub alert_count: u32,
    /// Risk score (0-100).
    pub risk_score: u8,
    /// Alert flags (bitmask).
    pub alert_flags: u64,
}

/// Alert emitted by transaction monitoring.
///
/// Type ID: 302
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 302)]
#[archive(check_bytes)]
pub struct TransactionAlert {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Transaction ID that triggered the alert.
    pub tx_id: u64,
    /// Alert type code.
    pub alert_type: u16,
    /// Alert severity (1-5).
    pub severity: u8,
    /// Related entity ID.
    pub entity_id: u64,
    /// Threshold that was exceeded.
    pub threshold: i64,
    /// Actual value.
    pub actual_value: i64,
    /// Timestamp.
    pub timestamp: u64,
}

// ============================================================================
// Circular Flow Ratio Ring Messages (310-319)
// ============================================================================

/// Ring message for adding an edge to the transaction graph.
///
/// Type ID: 310
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 310)]
#[archive(check_bytes)]
pub struct AddGraphEdgeRing {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Source entity ID.
    pub source_id: u64,
    /// Destination entity ID.
    pub dest_id: u64,
    /// Transaction amount.
    pub amount: i64,
    /// Timestamp.
    pub timestamp: u64,
}

impl AddGraphEdgeRing {
    /// Create a new add edge message.
    pub fn new(source_id: u64, dest_id: u64, amount: f64, timestamp: u64) -> Self {
        Self {
            id: MessageId::generate(),
            correlation_id: CorrelationId::generate(),
            source_id,
            dest_id,
            amount: (amount * 100_000_000.0) as i64,
            timestamp,
        }
    }
}

/// Response from adding a graph edge.
///
/// Type ID: 311
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 311)]
#[archive(check_bytes)]
pub struct AddGraphEdgeResponse {
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Whether a new cycle was detected.
    pub cycle_detected: bool,
    /// Number of entities in detected cycle (0 if none).
    pub cycle_size: u32,
    /// Current circular flow ratio for source entity.
    pub source_ratio: f32,
}

/// Query for circular flow ratio.
///
/// Type ID: 312
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 312)]
#[archive(check_bytes)]
pub struct QueryCircularRatioRing {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Entity ID to query.
    pub entity_id: u64,
}

/// Response with circular flow ratio.
///
/// Type ID: 313
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 313)]
#[archive(check_bytes)]
pub struct QueryCircularRatioResponse {
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Entity ID.
    pub entity_id: u64,
    /// Circular flow ratio (0.0-1.0).
    pub ratio: f32,
    /// Number of SCCs the entity belongs to.
    pub scc_count: u32,
    /// Total transaction volume in cycles.
    pub cycle_volume: i64,
}

/// Cycle detection alert.
///
/// Type ID: 314
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 314)]
#[archive(check_bytes)]
pub struct CycleDetectedAlert {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Cycle identifier.
    pub cycle_id: u64,
    /// Number of entities in cycle.
    pub cycle_size: u32,
    /// Total value flowing through cycle.
    pub total_value: i64,
    /// Timestamp of detection.
    pub timestamp: u64,
    /// Risk level (1-5).
    pub risk_level: u8,
}

// ============================================================================
// AML Pattern Detection Ring Messages (320-329)
// ============================================================================

/// Ring message for pattern matching on transaction stream.
///
/// Type ID: 320
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 320)]
#[archive(check_bytes)]
pub struct MatchPatternRing {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Transaction data (encoded).
    pub tx_id: u64,
    /// Source entity.
    pub source_id: u64,
    /// Destination entity.
    pub dest_id: u64,
    /// Amount.
    pub amount: i64,
    /// Transaction type.
    pub tx_type: u8,
    /// Timestamp.
    pub timestamp: u64,
}

/// Response from pattern matching.
///
/// Type ID: 321
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 321)]
#[archive(check_bytes)]
pub struct MatchPatternResponse {
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Transaction ID.
    pub tx_id: u64,
    /// Patterns matched (bitmask).
    pub patterns_matched: u64,
    /// Highest pattern score.
    pub max_score: f32,
    /// Number of patterns matched.
    pub match_count: u32,
}

/// AML pattern alert.
///
/// Type ID: 322
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 322)]
#[archive(check_bytes)]
pub struct AMLPatternAlert {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Pattern type.
    pub pattern_type: u16,
    /// Entity ID involved.
    pub entity_id: u64,
    /// Confidence score (0.0-1.0).
    pub confidence: f32,
    /// Evidence transaction IDs (up to 4).
    pub evidence_tx_ids: [u64; 4],
    /// Number of evidence transactions.
    pub evidence_count: u8,
    /// Timestamp.
    pub timestamp: u64,
}

// ============================================================================
// Common Ring Types
// ============================================================================

/// Alert severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
#[repr(u8)]
pub enum AlertSeverity {
    /// Low severity.
    Low = 1,
    /// Medium severity.
    Medium = 2,
    /// High severity.
    High = 3,
    /// Critical severity.
    Critical = 4,
    /// Emergency.
    Emergency = 5,
}

/// AML pattern types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
#[repr(u16)]
pub enum AMLPatternType {
    /// Structuring/smurfing.
    Structuring = 1,
    /// Layering.
    Layering = 2,
    /// Circular flow.
    CircularFlow = 3,
    /// Rapid movement.
    RapidMovement = 4,
    /// Unusual volume.
    UnusualVolume = 5,
    /// Geographic anomaly.
    GeographicAnomaly = 6,
    /// Timing anomaly.
    TimingAnomaly = 7,
    /// Counterparty risk.
    CounterpartyRisk = 8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_transaction_ring() {
        let msg = MonitorTransactionRing::new(1, 100, 200, 1000.0, 1234567890, 1, "USD");
        assert_eq!(msg.tx_id, 1);
        assert_eq!(msg.source_id, 100);
        assert_eq!(msg.dest_id, 200);
        assert_eq!(msg.amount, 100_000_000_000); // 1000 * 10^8
    }

    #[test]
    fn test_add_graph_edge_ring() {
        let msg = AddGraphEdgeRing::new(1, 2, 500.0, 1234567890);
        assert_eq!(msg.source_id, 1);
        assert_eq!(msg.dest_id, 2);
        assert_eq!(msg.amount, 50_000_000_000); // 500 * 10^8
    }
}
