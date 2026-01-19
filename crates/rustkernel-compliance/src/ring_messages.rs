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
// K2K Cross-Compliance Alert Coordination (350-369)
// ============================================================================

/// K2K alert broadcast - share alert across compliance kernels.
///
/// Type ID: 350
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 350)]
#[archive(check_bytes)]
pub struct K2KAlertBroadcast {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Source kernel ID (hashed).
    pub source_kernel: u64,
    /// Alert type: 1=transaction, 2=cycle, 3=pattern, 4=sanctions.
    pub alert_type: u8,
    /// Alert severity (1-5).
    pub severity: u8,
    /// Primary entity ID.
    pub entity_id: u64,
    /// Related entity ID (if applicable).
    pub related_entity_id: u64,
    /// Risk score (0-100).
    pub risk_score: u8,
    /// Alert timestamp.
    pub timestamp: u64,
    /// Additional context (encoded).
    pub context: u64,
}

/// K2K alert acknowledgment.
///
/// Type ID: 351
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 351)]
#[archive(check_bytes)]
pub struct K2KAlertAck {
    /// Original message ID.
    pub request_id: u64,
    /// Acknowledging kernel ID (hashed).
    pub kernel_id: u64,
    /// Whether this kernel will take action.
    pub will_action: bool,
    /// Correlation evidence found.
    pub correlation_found: bool,
}

/// K2K entity risk aggregation request.
///
/// Aggregates risk scores from multiple compliance kernels for an entity.
///
/// Type ID: 352
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 352)]
#[archive(check_bytes)]
pub struct K2KEntityRiskRequest {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Requesting kernel ID.
    pub source_kernel: u64,
    /// Entity ID to assess.
    pub entity_id: u64,
    /// Time window (microseconds) for aggregation.
    pub time_window_us: u64,
}

/// K2K entity risk response.
///
/// Type ID: 353
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 353)]
#[archive(check_bytes)]
pub struct K2KEntityRiskResponse {
    /// Original message ID.
    pub request_id: u64,
    /// Responding kernel ID.
    pub kernel_id: u64,
    /// Entity ID.
    pub entity_id: u64,
    /// Risk score from this kernel (0-100).
    pub risk_score: u8,
    /// Number of alerts in time window.
    pub alert_count: u32,
    /// Highest severity alert (1-5).
    pub max_severity: u8,
    /// Pattern types detected (bitmask).
    pub patterns_detected: u64,
}

/// K2K aggregated entity risk.
///
/// Final aggregated risk after collecting from all kernels.
///
/// Type ID: 354
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 354)]
#[archive(check_bytes)]
pub struct K2KAggregatedRisk {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Entity ID.
    pub entity_id: u64,
    /// Aggregated risk score (weighted average, 0-100).
    pub aggregated_score: u8,
    /// Number of kernels contributing.
    pub kernel_count: u8,
    /// Total alerts across all kernels.
    pub total_alerts: u32,
    /// Recommendation: 0=none, 1=monitor, 2=escalate, 3=block.
    pub recommendation: u8,
    /// Timestamp of aggregation.
    pub timestamp: u64,
}

/// K2K cross-kernel case creation.
///
/// When multiple kernels detect related suspicious activity, create a unified case.
///
/// Type ID: 355
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 355)]
#[archive(check_bytes)]
pub struct K2KCaseCreation {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Case ID.
    pub case_id: u64,
    /// Primary entity ID.
    pub entity_id: u64,
    /// Related entity IDs (up to 4).
    pub related_entities: [u64; 4],
    /// Number of related entities.
    pub related_count: u8,
    /// Contributing kernel IDs (up to 4).
    pub kernel_ids: [u64; 4],
    /// Number of contributing kernels.
    pub kernel_count: u8,
    /// Total risk score.
    pub total_risk: u8,
    /// Case priority: 1=low, 2=medium, 3=high, 4=critical.
    pub priority: u8,
    /// Creation timestamp.
    pub timestamp: u64,
}

/// K2K case update.
///
/// Type ID: 356
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 356)]
#[archive(check_bytes)]
pub struct K2KCaseUpdate {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Case ID.
    pub case_id: u64,
    /// Updating kernel ID.
    pub kernel_id: u64,
    /// Update type: 1=new_evidence, 2=risk_change, 3=status_change.
    pub update_type: u8,
    /// New risk score (if applicable).
    pub new_risk: u8,
    /// New evidence transaction IDs (up to 4).
    pub evidence: [u64; 4],
    /// Evidence count.
    pub evidence_count: u8,
    /// Update timestamp.
    pub timestamp: u64,
}

/// K2K sanctions match notification.
///
/// When SanctionsScreening finds a match, notify other kernels.
///
/// Type ID: 357
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 357)]
#[archive(check_bytes)]
pub struct K2KSanctionsMatch {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Entity ID with sanctions match.
    pub entity_id: u64,
    /// Match type: 1=exact, 2=fuzzy, 3=alias.
    pub match_type: u8,
    /// Match confidence (0-100).
    pub confidence: u8,
    /// Sanctions list code.
    pub list_code: u32,
    /// Entry ID on sanctions list.
    pub list_entry_id: u64,
    /// Detection timestamp.
    pub timestamp: u64,
}

/// K2K request to freeze entity across all kernels.
///
/// Type ID: 358
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 358)]
#[archive(check_bytes)]
pub struct K2KFreezeRequest {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Requesting kernel ID.
    pub source_kernel: u64,
    /// Entity ID to freeze.
    pub entity_id: u64,
    /// Reason code.
    pub reason: u16,
    /// Duration (0 = indefinite, otherwise seconds).
    pub duration_secs: u32,
    /// Request timestamp.
    pub timestamp: u64,
}

/// K2K freeze acknowledgment.
///
/// Type ID: 359
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 359)]
#[archive(check_bytes)]
pub struct K2KFreezeAck {
    /// Original message ID.
    pub request_id: u64,
    /// Acknowledging kernel ID.
    pub kernel_id: u64,
    /// Entity ID.
    pub entity_id: u64,
    /// Whether freeze was applied.
    pub frozen: bool,
    /// Transactions blocked since freeze.
    pub blocked_count: u32,
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

    // K2K Cross-Compliance Tests

    #[test]
    fn test_k2k_alert_broadcast() {
        let msg = K2KAlertBroadcast {
            id: MessageId(1),
            source_kernel: 12345,
            alert_type: 2, // cycle
            severity: 4,
            entity_id: 100,
            related_entity_id: 200,
            risk_score: 85,
            timestamp: 1234567890,
            context: 0,
        };
        assert_eq!(msg.alert_type, 2);
        assert_eq!(msg.severity, 4);
        assert_eq!(msg.risk_score, 85);
    }

    #[test]
    fn test_k2k_entity_risk_request() {
        let msg = K2KEntityRiskRequest {
            id: MessageId(2),
            source_kernel: 111,
            entity_id: 500,
            time_window_us: 3_600_000_000, // 1 hour
        };
        assert_eq!(msg.entity_id, 500);
        assert_eq!(msg.time_window_us, 3_600_000_000);
    }

    #[test]
    fn test_k2k_aggregated_risk() {
        let msg = K2KAggregatedRisk {
            id: MessageId(3),
            entity_id: 500,
            aggregated_score: 75,
            kernel_count: 4,
            total_alerts: 12,
            recommendation: 2, // escalate
            timestamp: 1234567890,
        };
        assert_eq!(msg.aggregated_score, 75);
        assert_eq!(msg.kernel_count, 4);
        assert_eq!(msg.recommendation, 2);
    }

    #[test]
    fn test_k2k_case_creation() {
        let msg = K2KCaseCreation {
            id: MessageId(4),
            case_id: 9001,
            entity_id: 100,
            related_entities: [200, 300, 0, 0],
            related_count: 2,
            kernel_ids: [111, 222, 333, 0],
            kernel_count: 3,
            total_risk: 90,
            priority: 3, // high
            timestamp: 1234567890,
        };
        assert_eq!(msg.case_id, 9001);
        assert_eq!(msg.related_count, 2);
        assert_eq!(msg.kernel_count, 3);
        assert_eq!(msg.priority, 3);
    }

    #[test]
    fn test_k2k_sanctions_match() {
        let msg = K2KSanctionsMatch {
            id: MessageId(5),
            entity_id: 999,
            match_type: 1, // exact
            confidence: 98,
            list_code: 1, // OFAC SDN
            list_entry_id: 12345,
            timestamp: 1234567890,
        };
        assert_eq!(msg.entity_id, 999);
        assert_eq!(msg.match_type, 1);
        assert_eq!(msg.confidence, 98);
    }

    #[test]
    fn test_k2k_freeze_request() {
        let msg = K2KFreezeRequest {
            id: MessageId(6),
            source_kernel: 111,
            entity_id: 999,
            reason: 1,        // sanctions match
            duration_secs: 0, // indefinite
            timestamp: 1234567890,
        };
        assert_eq!(msg.entity_id, 999);
        assert_eq!(msg.duration_secs, 0);
    }

    #[test]
    fn test_k2k_freeze_ack() {
        let msg = K2KFreezeAck {
            request_id: 6,
            kernel_id: 222,
            entity_id: 999,
            frozen: true,
            blocked_count: 3,
        };
        assert!(msg.frozen);
        assert_eq!(msg.blocked_count, 3);
    }
}
