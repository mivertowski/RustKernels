//! Banking types and data structures.

use std::collections::HashMap;

// ============================================================================
// Transaction Types
// ============================================================================

/// A financial transaction for fraud analysis.
#[derive(Debug, Clone)]
pub struct BankTransaction {
    /// Transaction ID.
    pub id: u64,
    /// Source account ID.
    pub source_account: u64,
    /// Destination account ID.
    pub dest_account: u64,
    /// Transaction amount.
    pub amount: f64,
    /// Timestamp (Unix epoch seconds).
    pub timestamp: u64,
    /// Transaction type.
    pub tx_type: TransactionType,
    /// Channel (online, branch, ATM, etc.).
    pub channel: Channel,
    /// Optional merchant category code.
    pub mcc: Option<u16>,
    /// Optional location (country code).
    pub location: Option<String>,
}

/// Transaction type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransactionType {
    /// Wire transfer.
    Wire,
    /// ACH transfer.
    ACH,
    /// Card payment.
    Card,
    /// Cash withdrawal.
    CashWithdrawal,
    /// Cash deposit.
    CashDeposit,
    /// Check.
    Check,
    /// Internal transfer.
    Internal,
    /// Cryptocurrency.
    Crypto,
}

/// Transaction channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Channel {
    /// Online banking.
    Online,
    /// Mobile app.
    Mobile,
    /// Branch.
    Branch,
    /// ATM.
    ATM,
    /// Phone.
    Phone,
    /// API.
    API,
}

// ============================================================================
// Fraud Pattern Types
// ============================================================================

/// A fraud pattern definition.
#[derive(Debug, Clone)]
pub struct FraudPattern {
    /// Pattern ID.
    pub id: u32,
    /// Pattern name.
    pub name: String,
    /// Pattern type.
    pub pattern_type: FraudPatternType,
    /// Risk score weight (0-100).
    pub risk_weight: f64,
    /// Pattern parameters.
    pub params: PatternParams,
}

/// Type of fraud pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FraudPatternType {
    /// Rapid succession of transactions (structuring).
    RapidSplit,
    /// Circular transaction flow.
    CircularFlow,
    /// Velocity anomaly (too many transactions).
    VelocityAnomaly,
    /// Amount anomaly (unusual amounts).
    AmountAnomaly,
    /// Geographic anomaly (impossible travel).
    GeoAnomaly,
    /// Time anomaly (unusual hours).
    TimeAnomaly,
    /// Account takeover indicators.
    AccountTakeover,
    /// Mule account behavior.
    MuleAccount,
    /// Layering (complex transaction chains).
    Layering,
}

/// Pattern detection parameters.
#[derive(Debug, Clone)]
pub struct PatternParams {
    /// Time window in seconds.
    pub time_window: u64,
    /// Minimum count threshold.
    pub min_count: u32,
    /// Amount threshold.
    pub amount_threshold: f64,
    /// Additional string patterns (for Aho-Corasick).
    pub string_patterns: Vec<String>,
    /// Custom parameters.
    pub custom: HashMap<String, f64>,
}

impl Default for PatternParams {
    fn default() -> Self {
        Self {
            time_window: 3600, // 1 hour
            min_count: 3,
            amount_threshold: 10000.0,
            string_patterns: Vec::new(),
            custom: HashMap::new(),
        }
    }
}

// ============================================================================
// Detection Result Types
// ============================================================================

/// Result of fraud pattern detection.
#[derive(Debug, Clone)]
pub struct FraudDetectionResult {
    /// Transaction ID that triggered detection.
    pub transaction_id: u64,
    /// Overall fraud score (0-100).
    pub fraud_score: f64,
    /// Matched patterns.
    pub matched_patterns: Vec<PatternMatch>,
    /// Risk level.
    pub risk_level: RiskLevel,
    /// Recommended action.
    pub recommended_action: RecommendedAction,
    /// Related transaction IDs.
    pub related_transactions: Vec<u64>,
}

/// A pattern match.
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// Pattern ID.
    pub pattern_id: u32,
    /// Pattern name.
    pub pattern_name: String,
    /// Match score (0-100).
    pub score: f64,
    /// Match details.
    pub details: String,
    /// Evidence transactions.
    pub evidence: Vec<u64>,
}

/// Risk level classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    /// Low risk - normal processing.
    Low,
    /// Medium risk - flag for review.
    Medium,
    /// High risk - hold for investigation.
    High,
    /// Critical - block immediately.
    Critical,
}

impl From<f64> for RiskLevel {
    fn from(score: f64) -> Self {
        match score {
            s if s < 25.0 => RiskLevel::Low,
            s if s < 50.0 => RiskLevel::Medium,
            s if s < 75.0 => RiskLevel::High,
            _ => RiskLevel::Critical,
        }
    }
}

/// Recommended action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendedAction {
    /// Allow transaction.
    Allow,
    /// Flag for review.
    Review,
    /// Hold pending investigation.
    Hold,
    /// Block transaction.
    Block,
    /// Block and alert.
    BlockAndAlert,
}

impl From<RiskLevel> for RecommendedAction {
    fn from(level: RiskLevel) -> Self {
        match level {
            RiskLevel::Low => RecommendedAction::Allow,
            RiskLevel::Medium => RecommendedAction::Review,
            RiskLevel::High => RecommendedAction::Hold,
            RiskLevel::Critical => RecommendedAction::BlockAndAlert,
        }
    }
}

// ============================================================================
// Account Profile Types
// ============================================================================

/// Account profile for behavioral baseline.
#[derive(Debug, Clone)]
pub struct AccountProfile {
    /// Account ID.
    pub account_id: u64,
    /// Average transaction amount.
    pub avg_amount: f64,
    /// Standard deviation of amounts.
    pub std_amount: f64,
    /// Average transactions per day.
    pub avg_daily_count: f64,
    /// Typical transaction hours (0-23).
    pub typical_hours: Vec<u8>,
    /// Typical locations.
    pub typical_locations: Vec<String>,
    /// Account age in days.
    pub account_age_days: u32,
    /// Total historical transaction count.
    pub total_transactions: u64,
}

impl Default for AccountProfile {
    fn default() -> Self {
        Self {
            account_id: 0,
            avg_amount: 500.0,
            std_amount: 200.0,
            avg_daily_count: 5.0,
            typical_hours: (9..18).collect(),
            typical_locations: vec!["US".to_string()],
            account_age_days: 365,
            total_transactions: 1000,
        }
    }
}
