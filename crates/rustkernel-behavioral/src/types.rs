//! Behavioral analytics types and data structures.

use std::collections::HashMap;

// ============================================================================
// Event Types
// ============================================================================

/// A user event for behavioral analysis.
#[derive(Debug, Clone)]
pub struct UserEvent {
    /// Event ID.
    pub id: u64,
    /// User ID.
    pub user_id: u64,
    /// Event type.
    pub event_type: String,
    /// Timestamp (Unix epoch seconds).
    pub timestamp: u64,
    /// Event attributes.
    pub attributes: HashMap<String, EventValue>,
    /// Session ID.
    pub session_id: Option<u64>,
    /// Device fingerprint.
    pub device_id: Option<String>,
    /// IP address.
    pub ip_address: Option<String>,
    /// Location (country code).
    pub location: Option<String>,
}

/// Event attribute value.
#[derive(Debug, Clone)]
pub enum EventValue {
    /// String value.
    String(String),
    /// Numeric value.
    Number(f64),
    /// Boolean value.
    Bool(bool),
    /// List of values.
    List(Vec<EventValue>),
}

// ============================================================================
// Profile Types
// ============================================================================

/// User behavioral profile.
#[derive(Debug, Clone)]
pub struct BehaviorProfile {
    /// User ID.
    pub user_id: u64,
    /// Feature vector.
    pub features: Vec<f64>,
    /// Feature names.
    pub feature_names: Vec<String>,
    /// Profile creation time.
    pub created_at: u64,
    /// Last update time.
    pub updated_at: u64,
    /// Number of events used to build profile.
    pub event_count: u64,
}

impl BehaviorProfile {
    /// Create a new empty profile.
    pub fn new(user_id: u64, feature_names: Vec<String>) -> Self {
        let n = feature_names.len();
        Self {
            user_id,
            features: vec![0.0; n],
            feature_names,
            created_at: 0,
            updated_at: 0,
            event_count: 0,
        }
    }

    /// Get a feature by name.
    pub fn get_feature(&self, name: &str) -> Option<f64> {
        self.feature_names
            .iter()
            .position(|n| n == name)
            .map(|i| self.features[i])
    }
}

/// Profiling result.
#[derive(Debug, Clone)]
pub struct ProfilingResult {
    /// User ID.
    pub user_id: u64,
    /// Extracted features.
    pub features: Vec<(String, f64)>,
    /// Profile stability score (0-1).
    pub stability: f64,
    /// Confidence in profile (0-1).
    pub confidence: f64,
}

// ============================================================================
// Anomaly Types
// ============================================================================

/// Anomaly detection result.
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    /// User ID.
    pub user_id: u64,
    /// Event ID that triggered anomaly.
    pub event_id: u64,
    /// Overall anomaly score (0-100).
    pub anomaly_score: f64,
    /// Is this an anomaly?
    pub is_anomaly: bool,
    /// Feature-level deviations.
    pub deviations: Vec<FeatureDeviation>,
    /// Anomaly type classification.
    pub anomaly_type: Option<AnomalyType>,
}

/// Feature-level deviation.
#[derive(Debug, Clone)]
pub struct FeatureDeviation {
    /// Feature name.
    pub feature_name: String,
    /// Expected value.
    pub expected: f64,
    /// Actual value.
    pub actual: f64,
    /// Z-score.
    pub z_score: f64,
    /// Contribution to anomaly score.
    pub contribution: f64,
}

/// Type of detected anomaly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyType {
    /// Time-based anomaly (unusual hours).
    Temporal,
    /// Location-based anomaly.
    Geographic,
    /// Device/access method anomaly.
    Device,
    /// Behavior pattern anomaly.
    Behavioral,
    /// Volume/frequency anomaly.
    Velocity,
    /// Multiple anomaly types.
    Mixed,
}

// ============================================================================
// Fraud Signature Types
// ============================================================================

/// A fraud signature pattern.
#[derive(Debug, Clone)]
pub struct FraudSignature {
    /// Signature ID.
    pub id: u32,
    /// Signature name.
    pub name: String,
    /// Pattern to match.
    pub pattern: SignaturePattern,
    /// Severity (0-100).
    pub severity: f64,
    /// Whether signature is active.
    pub active: bool,
}

/// Signature pattern definition.
#[derive(Debug, Clone)]
pub enum SignaturePattern {
    /// Sequence of event types.
    EventSequence(Vec<String>),
    /// Event with specific attributes.
    EventAttributes(String, HashMap<String, EventValue>),
    /// Time-based pattern (events within time window).
    TimeWindow {
        /// Events that must occur within the window.
        events: Vec<String>,
        /// Time window in seconds.
        window_secs: u64,
    },
    /// Count-based pattern.
    CountThreshold {
        /// Type of event to count.
        event_type: String,
        /// Minimum count threshold.
        count: u32,
        /// Time window in seconds.
        window_secs: u64,
    },
    /// Regex pattern on event data.
    Regex(String),
}

/// Signature match result.
#[derive(Debug, Clone)]
pub struct SignatureMatch {
    /// Signature ID.
    pub signature_id: u32,
    /// Signature name.
    pub signature_name: String,
    /// Match score (0-100).
    pub score: f64,
    /// Matched event IDs.
    pub matched_events: Vec<u64>,
    /// Match details.
    pub details: String,
}

// ============================================================================
// Causal Graph Types
// ============================================================================

/// A causal graph node.
#[derive(Debug, Clone)]
pub struct CausalNode {
    /// Node ID.
    pub id: u64,
    /// Event type this node represents.
    pub event_type: String,
    /// Node probability.
    pub probability: f64,
}

/// A causal graph edge.
#[derive(Debug, Clone)]
pub struct CausalEdge {
    /// Source node ID.
    pub source: u64,
    /// Target node ID.
    pub target: u64,
    /// Causal strength (0-1).
    pub strength: f64,
    /// Time lag (average seconds between events).
    pub lag: f64,
    /// Number of observations.
    pub count: u64,
}

/// Causal graph construction result.
#[derive(Debug, Clone)]
pub struct CausalGraphResult {
    /// Graph nodes.
    pub nodes: Vec<CausalNode>,
    /// Graph edges.
    pub edges: Vec<CausalEdge>,
    /// Root causes (nodes with high out-degree).
    pub root_causes: Vec<u64>,
    /// Effects (nodes with high in-degree).
    pub effects: Vec<u64>,
}

// ============================================================================
// Forensic Query Types
// ============================================================================

/// A forensic query definition.
#[derive(Debug, Clone)]
pub struct ForensicQuery {
    /// Query ID.
    pub id: u64,
    /// Query type.
    pub query_type: QueryType,
    /// Time range start.
    pub start_time: u64,
    /// Time range end.
    pub end_time: u64,
    /// User filter.
    pub user_ids: Option<Vec<u64>>,
    /// Event type filter.
    pub event_types: Option<Vec<String>>,
    /// Custom filters.
    pub filters: HashMap<String, String>,
}

/// Type of forensic query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// Search for specific pattern.
    PatternSearch,
    /// Timeline reconstruction.
    Timeline,
    /// User activity summary.
    ActivitySummary,
    /// Anomaly hunt.
    AnomalyHunt,
    /// Correlation analysis.
    Correlation,
}

/// Forensic query result.
#[derive(Debug, Clone)]
pub struct ForensicResult {
    /// Query ID.
    pub query_id: u64,
    /// Matched events.
    pub events: Vec<u64>,
    /// Match count.
    pub total_matches: u64,
    /// Summary statistics.
    pub summary: HashMap<String, f64>,
    /// Execution time (ms).
    pub execution_time_ms: u64,
}

// ============================================================================
// Event Correlation Types
// ============================================================================

/// Event correlation result.
#[derive(Debug, Clone)]
pub struct CorrelationResult {
    /// Primary event ID.
    pub event_id: u64,
    /// Correlated events.
    pub correlations: Vec<EventCorrelation>,
    /// Correlation clusters.
    pub clusters: Vec<CorrelationCluster>,
}

/// A single event correlation.
#[derive(Debug, Clone)]
pub struct EventCorrelation {
    /// Correlated event ID.
    pub correlated_event_id: u64,
    /// Correlation score (0-1).
    pub score: f64,
    /// Correlation type.
    pub correlation_type: CorrelationType,
    /// Time difference (seconds).
    pub time_diff: i64,
}

/// Type of correlation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CorrelationType {
    /// Temporal proximity.
    Temporal,
    /// Same user.
    User,
    /// Same session.
    Session,
    /// Same device.
    Device,
    /// Same location.
    Location,
    /// Causal relationship.
    Causal,
}

/// Cluster of correlated events.
#[derive(Debug, Clone)]
pub struct CorrelationCluster {
    /// Cluster ID.
    pub id: u64,
    /// Event IDs in cluster.
    pub event_ids: Vec<u64>,
    /// Cluster coherence score.
    pub coherence: f64,
    /// Dominant correlation type.
    pub dominant_type: CorrelationType,
}
