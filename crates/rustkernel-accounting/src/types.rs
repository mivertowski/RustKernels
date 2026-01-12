//! Accounting types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Chart of Accounts Types
// ============================================================================

/// Account in chart of accounts.
#[derive(Debug, Clone)]
pub struct Account {
    /// Account code.
    pub code: String,
    /// Account name.
    pub name: String,
    /// Account type.
    pub account_type: AccountType,
    /// Parent account code (for hierarchy).
    pub parent_code: Option<String>,
    /// Is active.
    pub is_active: bool,
    /// Currency.
    pub currency: String,
    /// Entity ID.
    pub entity_id: String,
    /// Attributes.
    pub attributes: HashMap<String, String>,
}

/// Account type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccountType {
    /// Asset account.
    Asset,
    /// Liability account.
    Liability,
    /// Equity account.
    Equity,
    /// Revenue account.
    Revenue,
    /// Expense account.
    Expense,
    /// Contra account.
    Contra,
}

/// Account mapping rule.
#[derive(Debug, Clone)]
pub struct MappingRule {
    /// Rule ID.
    pub id: String,
    /// Source account pattern.
    pub source_pattern: String,
    /// Target account code.
    pub target_code: String,
    /// Entity filter (if any).
    pub entity_filter: Option<String>,
    /// Priority (higher = applied first).
    pub priority: u32,
    /// Transformation to apply.
    pub transformation: MappingTransformation,
}

/// Mapping transformation.
#[derive(Debug, Clone)]
pub enum MappingTransformation {
    /// Direct mapping.
    Direct,
    /// Proportional split.
    Split(Vec<(String, f64)>),
    /// Aggregation.
    Aggregate,
    /// Conditional mapping.
    Conditional {
        condition: String,
        if_true: String,
        if_false: String,
    },
}

/// Mapping result.
#[derive(Debug, Clone)]
pub struct MappingResult {
    /// Mapped accounts.
    pub mapped: Vec<MappedAccount>,
    /// Unmapped accounts.
    pub unmapped: Vec<String>,
    /// Mapping statistics.
    pub stats: MappingStats,
}

/// Mapped account.
#[derive(Debug, Clone)]
pub struct MappedAccount {
    /// Source account code.
    pub source_code: String,
    /// Target account code.
    pub target_code: String,
    /// Applied rule ID.
    pub rule_id: String,
    /// Amount (if split).
    pub amount_ratio: f64,
}

/// Mapping statistics.
#[derive(Debug, Clone)]
pub struct MappingStats {
    /// Total accounts.
    pub total_accounts: usize,
    /// Mapped count.
    pub mapped_count: usize,
    /// Unmapped count.
    pub unmapped_count: usize,
    /// Rules applied.
    pub rules_applied: usize,
    /// Mapping rate.
    pub mapping_rate: f64,
}

// ============================================================================
// Journal Types
// ============================================================================

/// Journal entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntry {
    /// Entry ID.
    pub id: u64,
    /// Entry date.
    pub date: u64,
    /// Posting date.
    pub posting_date: u64,
    /// Document number.
    pub document_number: String,
    /// Lines.
    pub lines: Vec<JournalLine>,
    /// Status.
    pub status: JournalStatus,
    /// Source system.
    pub source_system: String,
    /// Description.
    pub description: String,
}

/// Journal line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalLine {
    /// Line number.
    pub line_number: u32,
    /// Account code.
    pub account_code: String,
    /// Debit amount.
    pub debit: f64,
    /// Credit amount.
    pub credit: f64,
    /// Currency.
    pub currency: String,
    /// Entity ID.
    pub entity_id: String,
    /// Cost center.
    pub cost_center: Option<String>,
    /// Description.
    pub description: String,
}

/// Journal status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JournalStatus {
    /// Draft.
    Draft,
    /// Pending approval.
    Pending,
    /// Posted.
    Posted,
    /// Reversed.
    Reversed,
}

/// Journal transformation result.
#[derive(Debug, Clone)]
pub struct TransformationResult {
    /// Transformed entries.
    pub entries: Vec<JournalEntry>,
    /// Validation errors.
    pub errors: Vec<ValidationError>,
    /// Statistics.
    pub stats: TransformationStats,
}

/// Validation error.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Entry ID.
    pub entry_id: u64,
    /// Line number (if applicable).
    pub line_number: Option<u32>,
    /// Error code.
    pub code: String,
    /// Error message.
    pub message: String,
    /// Severity.
    pub severity: ErrorSeverity,
}

/// Error severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// Warning.
    Warning,
    /// Error.
    Error,
    /// Critical.
    Critical,
}

/// Transformation statistics.
#[derive(Debug, Clone)]
pub struct TransformationStats {
    /// Total entries.
    pub total_entries: usize,
    /// Transformed count.
    pub transformed_count: usize,
    /// Error count.
    pub error_count: usize,
    /// Total debit.
    pub total_debit: f64,
    /// Total credit.
    pub total_credit: f64,
}

// ============================================================================
// Reconciliation Types
// ============================================================================

/// Reconciliation item.
#[derive(Debug, Clone)]
pub struct ReconciliationItem {
    /// Item ID.
    pub id: String,
    /// Source.
    pub source: ReconciliationSource,
    /// Account code.
    pub account_code: String,
    /// Amount.
    pub amount: f64,
    /// Currency.
    pub currency: String,
    /// Date.
    pub date: u64,
    /// Reference.
    pub reference: String,
    /// Status.
    pub status: ReconciliationStatus,
    /// Matched item ID (if matched).
    pub matched_with: Option<String>,
}

/// Reconciliation source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconciliationSource {
    /// General ledger.
    GeneralLedger,
    /// Sub-ledger.
    SubLedger,
    /// Bank statement.
    BankStatement,
    /// External system.
    External,
}

/// Reconciliation status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconciliationStatus {
    /// Unmatched.
    Unmatched,
    /// Matched.
    Matched,
    /// Partially matched.
    PartiallyMatched,
    /// Exception.
    Exception,
}

/// Reconciliation result.
#[derive(Debug, Clone)]
pub struct ReconciliationResult {
    /// Matched pairs.
    pub matched_pairs: Vec<MatchedPair>,
    /// Unmatched items.
    pub unmatched: Vec<String>,
    /// Exceptions.
    pub exceptions: Vec<ReconciliationException>,
    /// Statistics.
    pub stats: ReconciliationStats,
}

/// Matched pair.
#[derive(Debug, Clone)]
pub struct MatchedPair {
    /// Source item ID.
    pub source_id: String,
    /// Target item ID.
    pub target_id: String,
    /// Match confidence.
    pub confidence: f64,
    /// Variance (if any).
    pub variance: f64,
    /// Match type.
    pub match_type: MatchType,
}

/// Match type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchType {
    /// Exact match.
    Exact,
    /// Tolerance match.
    Tolerance,
    /// Many to one.
    ManyToOne,
    /// One to many.
    OneToMany,
    /// Many to many.
    ManyToMany,
}

/// Reconciliation exception.
#[derive(Debug, Clone)]
pub struct ReconciliationException {
    /// Item ID.
    pub item_id: String,
    /// Exception type.
    pub exception_type: ExceptionType,
    /// Description.
    pub description: String,
    /// Suggested action.
    pub suggested_action: Option<String>,
}

/// Exception type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExceptionType {
    /// Amount variance.
    AmountVariance,
    /// Date variance.
    DateVariance,
    /// Missing counterpart.
    MissingCounterpart,
    /// Duplicate.
    Duplicate,
    /// Other.
    Other,
}

/// Reconciliation statistics.
#[derive(Debug, Clone)]
pub struct ReconciliationStats {
    /// Total items.
    pub total_items: usize,
    /// Matched count.
    pub matched_count: usize,
    /// Unmatched count.
    pub unmatched_count: usize,
    /// Exception count.
    pub exception_count: usize,
    /// Match rate.
    pub match_rate: f64,
    /// Total variance.
    pub total_variance: f64,
}

// ============================================================================
// Network Analysis Types
// ============================================================================

/// Intercompany transaction.
#[derive(Debug, Clone)]
pub struct IntercompanyTransaction {
    /// Transaction ID.
    pub id: String,
    /// From entity.
    pub from_entity: String,
    /// To entity.
    pub to_entity: String,
    /// Amount.
    pub amount: f64,
    /// Currency.
    pub currency: String,
    /// Date.
    pub date: u64,
    /// Transaction type.
    pub transaction_type: IntercompanyType,
    /// Status.
    pub status: IntercompanyStatus,
}

/// Intercompany transaction type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntercompanyType {
    /// Trade receivable/payable.
    Trade,
    /// Loan.
    Loan,
    /// Dividend.
    Dividend,
    /// Management fee.
    ManagementFee,
    /// Royalty.
    Royalty,
    /// Other.
    Other,
}

/// Intercompany status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntercompanyStatus {
    /// Open.
    Open,
    /// Confirmed.
    Confirmed,
    /// Eliminated.
    Eliminated,
    /// Disputed.
    Disputed,
}

/// Network analysis result.
#[derive(Debug, Clone)]
pub struct NetworkAnalysisResult {
    /// Entity balances.
    pub entity_balances: HashMap<String, EntityBalance>,
    /// Relationship strengths.
    pub relationships: Vec<EntityRelationship>,
    /// Circular references.
    pub circular_refs: Vec<CircularReference>,
    /// Elimination entries.
    pub elimination_entries: Vec<EliminationEntry>,
    /// Statistics.
    pub stats: NetworkStats,
}

/// Entity balance.
#[derive(Debug, Clone)]
pub struct EntityBalance {
    /// Entity ID.
    pub entity_id: String,
    /// Total intercompany receivables.
    pub total_receivables: f64,
    /// Total intercompany payables.
    pub total_payables: f64,
    /// Net position.
    pub net_position: f64,
    /// Counterparty count.
    pub counterparty_count: usize,
}

/// Entity relationship.
#[derive(Debug, Clone)]
pub struct EntityRelationship {
    /// From entity.
    pub from_entity: String,
    /// To entity.
    pub to_entity: String,
    /// Total volume.
    pub total_volume: f64,
    /// Transaction count.
    pub transaction_count: usize,
    /// Net balance.
    pub net_balance: f64,
}

/// Circular reference.
#[derive(Debug, Clone)]
pub struct CircularReference {
    /// Entities in the circle.
    pub entities: Vec<String>,
    /// Total amount.
    pub amount: f64,
    /// Impact on consolidation.
    pub consolidation_impact: f64,
}

/// Elimination entry.
#[derive(Debug, Clone)]
pub struct EliminationEntry {
    /// Entry ID.
    pub id: String,
    /// From entity.
    pub from_entity: String,
    /// To entity.
    pub to_entity: String,
    /// Debit account.
    pub debit_account: String,
    /// Credit account.
    pub credit_account: String,
    /// Amount.
    pub amount: f64,
    /// Currency.
    pub currency: String,
}

/// Network statistics.
#[derive(Debug, Clone)]
pub struct NetworkStats {
    /// Total entities.
    pub total_entities: usize,
    /// Total transactions.
    pub total_transactions: usize,
    /// Total volume.
    pub total_volume: f64,
    /// Circular reference count.
    pub circular_count: usize,
    /// Elimination count.
    pub elimination_count: usize,
}

// ============================================================================
// Temporal Correlation Types
// ============================================================================

/// Account time series.
#[derive(Debug, Clone)]
pub struct AccountTimeSeries {
    /// Account code.
    pub account_code: String,
    /// Data points.
    pub data_points: Vec<TimeSeriesPoint>,
    /// Frequency.
    pub frequency: TimeFrequency,
}

/// Time series point.
#[derive(Debug, Clone)]
pub struct TimeSeriesPoint {
    /// Date.
    pub date: u64,
    /// Balance.
    pub balance: f64,
    /// Period change.
    pub period_change: f64,
}

/// Time frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeFrequency {
    /// Daily.
    Daily,
    /// Weekly.
    Weekly,
    /// Monthly.
    Monthly,
    /// Quarterly.
    Quarterly,
    /// Annual.
    Annual,
}

/// Correlation result.
#[derive(Debug, Clone)]
pub struct CorrelationResult {
    /// Correlation matrix.
    pub correlations: Vec<AccountCorrelation>,
    /// Anomalies.
    pub anomalies: Vec<CorrelationAnomaly>,
    /// Statistics.
    pub stats: CorrelationStats,
}

/// Account correlation.
#[derive(Debug, Clone)]
pub struct AccountCorrelation {
    /// First account.
    pub account_a: String,
    /// Second account.
    pub account_b: String,
    /// Correlation coefficient.
    pub coefficient: f64,
    /// P-value.
    pub p_value: f64,
    /// Correlation type.
    pub correlation_type: CorrelationType,
}

/// Correlation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationType {
    /// Positive correlation.
    Positive,
    /// Negative correlation.
    Negative,
    /// No significant correlation.
    None,
}

/// Correlation anomaly.
#[derive(Debug, Clone)]
pub struct CorrelationAnomaly {
    /// Account code.
    pub account_code: String,
    /// Date.
    pub date: u64,
    /// Expected value.
    pub expected: f64,
    /// Actual value.
    pub actual: f64,
    /// Z-score.
    pub z_score: f64,
    /// Anomaly type.
    pub anomaly_type: AnomalyType,
}

/// Anomaly type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyType {
    /// Unexpectedly high.
    UnexpectedHigh,
    /// Unexpectedly low.
    UnexpectedLow,
    /// Pattern break.
    PatternBreak,
    /// Missing expected correlation.
    MissingCorrelation,
}

/// Correlation statistics.
#[derive(Debug, Clone)]
pub struct CorrelationStats {
    /// Total accounts analyzed.
    pub accounts_analyzed: usize,
    /// Significant correlations.
    pub significant_correlations: usize,
    /// Anomaly count.
    pub anomaly_count: usize,
    /// Average correlation strength.
    pub avg_correlation: f64,
}
