//! Compliance types and data structures.

use serde::{Deserialize, Serialize};

// ============================================================================
// Transaction Types
// ============================================================================

/// A financial transaction for compliance analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    /// Unique transaction ID.
    pub id: u64,
    /// Source account/entity ID.
    pub source_id: u64,
    /// Destination account/entity ID.
    pub dest_id: u64,
    /// Transaction amount.
    pub amount: f64,
    /// Timestamp (Unix epoch seconds).
    pub timestamp: u64,
    /// Currency code (e.g., "USD").
    pub currency: String,
    /// Transaction type (e.g., "wire", "ach", "internal").
    pub tx_type: String,
}

/// Time window for analysis.
#[derive(Debug, Clone, Copy)]
pub struct TimeWindow {
    /// Start timestamp (inclusive).
    pub start: u64,
    /// End timestamp (exclusive).
    pub end: u64,
}

impl TimeWindow {
    /// Create a new time window.
    pub fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }

    /// Check if a timestamp is within this window.
    pub fn contains(&self, timestamp: u64) -> bool {
        timestamp >= self.start && timestamp < self.end
    }

    /// Duration in seconds.
    pub fn duration(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }
}

// ============================================================================
// AML Types
// ============================================================================

/// Result of circular flow analysis.
#[derive(Debug, Clone)]
pub struct CircularFlowResult {
    /// Ratio of circular flow amount to total flow.
    pub circular_ratio: f64,
    /// Strongly connected components found.
    pub sccs: Vec<Vec<u64>>,
    /// Total amount in circular flows.
    pub circular_amount: f64,
    /// Total transaction amount analyzed.
    pub total_amount: f64,
}

/// Result of reciprocity analysis.
#[derive(Debug, Clone)]
pub struct ReciprocityResult {
    /// Ratio of reciprocal transactions.
    pub reciprocity_ratio: f64,
    /// Pairs of entities with reciprocal transactions.
    pub reciprocal_pairs: Vec<(u64, u64)>,
    /// Amount involved in reciprocal transactions.
    pub reciprocal_amount: f64,
}

/// Result of rapid movement (velocity) analysis.
#[derive(Debug, Clone)]
pub struct RapidMovementResult {
    /// Entities flagged for rapid movement.
    pub flagged_entities: Vec<u64>,
    /// Velocity metrics per entity (transactions per hour).
    pub velocity_metrics: Vec<(u64, f64)>,
    /// Amount moved rapidly.
    pub rapid_amount: f64,
}

/// AML pattern types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AMLPattern {
    /// Structuring (smurfing) - breaking large amounts into smaller ones.
    Structuring,
    /// Layering - moving money through multiple accounts.
    Layering,
    /// Rapid movement - quick in/out of accounts.
    RapidMovement,
    /// Round tripping - money returning to origin.
    RoundTripping,
    /// Funnel account - many-to-one pattern.
    FunnelAccount,
    /// Fan out - one-to-many pattern.
    FanOut,
}

/// Result of AML pattern detection.
#[derive(Debug, Clone)]
pub struct AMLPatternResult {
    /// Detected patterns with associated entities.
    pub patterns: Vec<(AMLPattern, Vec<u64>)>,
    /// Overall risk score (0-100).
    pub risk_score: f64,
    /// Details per pattern.
    pub pattern_details: Vec<PatternDetail>,
}

/// Details of a detected pattern.
#[derive(Debug, Clone)]
pub struct PatternDetail {
    /// Pattern type.
    pub pattern: AMLPattern,
    /// Involved entity IDs.
    pub entities: Vec<u64>,
    /// Amount involved.
    pub amount: f64,
    /// Confidence score (0-1).
    pub confidence: f64,
    /// Time span of the pattern.
    pub time_span: TimeWindow,
}

// ============================================================================
// KYC Types
// ============================================================================

/// KYC risk factors.
#[derive(Debug, Clone)]
pub struct KYCFactors {
    /// Customer ID.
    pub customer_id: u64,
    /// Country risk score (0-100).
    pub country_risk: f64,
    /// Industry risk score (0-100).
    pub industry_risk: f64,
    /// Product risk score (0-100).
    pub product_risk: f64,
    /// Transaction pattern risk (0-100).
    pub transaction_risk: f64,
    /// Documentation completeness (0-100).
    pub documentation_score: f64,
    /// Years as customer.
    pub tenure_years: f64,
}

/// Result of KYC scoring.
#[derive(Debug, Clone)]
pub struct KYCResult {
    /// Customer ID.
    pub customer_id: u64,
    /// Overall risk score (0-100).
    pub risk_score: f64,
    /// Risk tier (Low, Medium, High, Very High).
    pub risk_tier: RiskTier,
    /// Contributing factors.
    pub factor_contributions: Vec<(String, f64)>,
}

/// Risk tier classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskTier {
    /// Low risk (0-25).
    Low,
    /// Medium risk (25-50).
    Medium,
    /// High risk (50-75).
    High,
    /// Very high risk (75-100).
    VeryHigh,
}

impl From<f64> for RiskTier {
    fn from(score: f64) -> Self {
        match score {
            s if s < 25.0 => RiskTier::Low,
            s if s < 50.0 => RiskTier::Medium,
            s if s < 75.0 => RiskTier::High,
            _ => RiskTier::VeryHigh,
        }
    }
}

/// Entity for resolution/matching.
#[derive(Debug, Clone)]
pub struct Entity {
    /// Entity ID.
    pub id: u64,
    /// Entity name.
    pub name: String,
    /// Alternative names/aliases.
    pub aliases: Vec<String>,
    /// Date of birth (YYYYMMDD format) or incorporation date.
    pub date: Option<u32>,
    /// Country code.
    pub country: Option<String>,
    /// Entity type.
    pub entity_type: EntityType,
}

/// Type of entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityType {
    /// Individual person.
    Individual,
    /// Corporation/company.
    Corporation,
    /// Government entity.
    Government,
    /// Unknown.
    Unknown,
}

/// Result of entity resolution.
#[derive(Debug, Clone)]
pub struct EntityResolutionResult {
    /// Query entity ID.
    pub query_id: u64,
    /// Potential matches with scores.
    pub matches: Vec<EntityMatch>,
}

/// A potential entity match.
#[derive(Debug, Clone)]
pub struct EntityMatch {
    /// Matched entity ID.
    pub entity_id: u64,
    /// Overall match score (0-1).
    pub score: f64,
    /// Name similarity.
    pub name_score: f64,
    /// Date similarity.
    pub date_score: f64,
    /// Country match.
    pub country_match: bool,
}

// ============================================================================
// Sanctions Types
// ============================================================================

/// Sanctions list entry.
#[derive(Debug, Clone)]
pub struct SanctionsEntry {
    /// Entry ID.
    pub id: u64,
    /// Primary name.
    pub name: String,
    /// Alternative names.
    pub aliases: Vec<String>,
    /// List source (OFAC, UN, EU, etc.).
    pub source: String,
    /// Program (SDN, sectoral, etc.).
    pub program: String,
    /// Country.
    pub country: Option<String>,
    /// Date of birth.
    pub dob: Option<u32>,
}

/// Result of sanctions screening.
#[derive(Debug, Clone)]
pub struct SanctionsResult {
    /// Query entity name.
    pub query_name: String,
    /// Potential matches.
    pub matches: Vec<SanctionsMatch>,
    /// Overall hit status.
    pub is_hit: bool,
}

/// A sanctions list match.
#[derive(Debug, Clone)]
pub struct SanctionsMatch {
    /// Matched entry ID.
    pub entry_id: u64,
    /// Match score (0-1).
    pub score: f64,
    /// Matched name.
    pub matched_name: String,
    /// List source.
    pub source: String,
    /// Match reason.
    pub reason: String,
}

/// PEP (Politically Exposed Person) entry.
#[derive(Debug, Clone)]
pub struct PEPEntry {
    /// Entry ID.
    pub id: u64,
    /// Name.
    pub name: String,
    /// Position/title.
    pub position: String,
    /// Country.
    pub country: String,
    /// Level (1=head of state, 2=senior official, 3=family member).
    pub level: u8,
    /// Still active in position.
    pub active: bool,
}

/// Result of PEP screening.
#[derive(Debug, Clone)]
pub struct PEPResult {
    /// Query name.
    pub query_name: String,
    /// Potential matches.
    pub matches: Vec<PEPMatch>,
    /// Is a PEP hit.
    pub is_pep: bool,
}

/// A PEP match.
#[derive(Debug, Clone)]
pub struct PEPMatch {
    /// Entry ID.
    pub entry_id: u64,
    /// Match score.
    pub score: f64,
    /// Matched name.
    pub name: String,
    /// Position.
    pub position: String,
    /// Country.
    pub country: String,
    /// PEP level.
    pub level: u8,
}

// ============================================================================
// Transaction Monitoring Types
// ============================================================================

/// Transaction monitoring rule.
#[derive(Debug, Clone)]
pub struct MonitoringRule {
    /// Rule ID.
    pub id: u64,
    /// Rule name.
    pub name: String,
    /// Rule type.
    pub rule_type: RuleType,
    /// Threshold value.
    pub threshold: f64,
    /// Time window in seconds.
    pub window_seconds: u64,
    /// Severity level.
    pub severity: Severity,
}

/// Type of monitoring rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleType {
    /// Single transaction amount threshold.
    SingleAmount,
    /// Aggregate amount over time window.
    AggregateAmount,
    /// Transaction count threshold.
    TransactionCount,
    /// Velocity (amount per hour).
    Velocity,
    /// Geographic risk (cross-border).
    GeographicRisk,
}

/// Alert severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational.
    Info,
    /// Low severity.
    Low,
    /// Medium severity.
    Medium,
    /// High severity.
    High,
    /// Critical.
    Critical,
}

/// Generated alert from monitoring.
#[derive(Debug, Clone)]
pub struct Alert {
    /// Alert ID.
    pub id: u64,
    /// Rule that triggered the alert.
    pub rule_id: u64,
    /// Entity that triggered the alert.
    pub entity_id: u64,
    /// Alert timestamp.
    pub timestamp: u64,
    /// Severity.
    pub severity: Severity,
    /// Current value that triggered the alert.
    pub current_value: f64,
    /// Threshold that was exceeded.
    pub threshold: f64,
    /// Related transaction IDs.
    pub transaction_ids: Vec<u64>,
    /// Alert message.
    pub message: String,
}

/// Result of transaction monitoring.
#[derive(Debug, Clone)]
pub struct MonitoringResult {
    /// Generated alerts.
    pub alerts: Vec<Alert>,
    /// Entities monitored.
    pub entities_checked: usize,
    /// Transactions analyzed.
    pub transactions_analyzed: usize,
}
