//! Payment processing types.

use std::collections::HashMap;

// ============================================================================
// Payment Types
// ============================================================================

/// Payment transaction.
#[derive(Debug, Clone)]
pub struct Payment {
    /// Payment ID.
    pub id: String,
    /// Payer account.
    pub payer_account: String,
    /// Payee account.
    pub payee_account: String,
    /// Amount.
    pub amount: f64,
    /// Currency.
    pub currency: String,
    /// Payment type.
    pub payment_type: PaymentType,
    /// Status.
    pub status: PaymentStatus,
    /// Initiated timestamp.
    pub initiated_at: u64,
    /// Completed timestamp.
    pub completed_at: Option<u64>,
    /// Reference.
    pub reference: String,
    /// Priority.
    pub priority: PaymentPriority,
    /// Attributes.
    pub attributes: HashMap<String, String>,
}

/// Payment type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PaymentType {
    /// ACH transfer.
    ACH,
    /// Wire transfer.
    Wire,
    /// Real-time payment.
    RealTime,
    /// Internal transfer.
    Internal,
    /// Check.
    Check,
    /// Card payment.
    Card,
}

/// Payment status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaymentStatus {
    /// Initiated.
    Initiated,
    /// Pending validation.
    Pending,
    /// Validated.
    Validated,
    /// In processing.
    Processing,
    /// Completed.
    Completed,
    /// Failed.
    Failed,
    /// Cancelled.
    Cancelled,
    /// Reversed.
    Reversed,
}

/// Payment priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PaymentPriority {
    /// Low priority.
    Low = 0,
    /// Normal priority.
    Normal = 1,
    /// High priority.
    High = 2,
    /// Urgent.
    Urgent = 3,
}

/// Payment processing result.
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    /// Processed payments.
    pub processed: Vec<String>,
    /// Failed payments.
    pub failed: Vec<(String, String)>,
    /// Pending payments.
    pub pending: Vec<String>,
    /// Statistics.
    pub stats: ProcessingStats,
}

/// Processing statistics.
#[derive(Debug, Clone)]
pub struct ProcessingStats {
    /// Total count.
    pub total_count: usize,
    /// Processed count.
    pub processed_count: usize,
    /// Failed count.
    pub failed_count: usize,
    /// Total amount processed.
    pub total_amount: f64,
    /// Average processing time (microseconds).
    pub avg_processing_time_us: f64,
}

// ============================================================================
// Flow Analysis Types
// ============================================================================

/// Payment flow.
#[derive(Debug, Clone)]
pub struct PaymentFlow {
    /// Source node.
    pub source: String,
    /// Target node.
    pub target: String,
    /// Total volume.
    pub volume: f64,
    /// Transaction count.
    pub count: usize,
    /// Average amount.
    pub avg_amount: f64,
}

/// Flow analysis result.
#[derive(Debug, Clone)]
pub struct FlowAnalysisResult {
    /// Payment flows.
    pub flows: Vec<PaymentFlow>,
    /// Node metrics.
    pub node_metrics: HashMap<String, NodeMetrics>,
    /// Overall metrics.
    pub overall_metrics: OverallMetrics,
    /// Anomalies detected.
    pub anomalies: Vec<FlowAnomaly>,
}

/// Node metrics.
#[derive(Debug, Clone)]
pub struct NodeMetrics {
    /// Node ID.
    pub node_id: String,
    /// Total inflow.
    pub total_inflow: f64,
    /// Total outflow.
    pub total_outflow: f64,
    /// Net flow.
    pub net_flow: f64,
    /// Inbound connections.
    pub inbound_count: usize,
    /// Outbound connections.
    pub outbound_count: usize,
    /// Centrality score.
    pub centrality: f64,
}

/// Overall flow metrics.
#[derive(Debug, Clone)]
pub struct OverallMetrics {
    /// Total volume.
    pub total_volume: f64,
    /// Total transactions.
    pub total_transactions: usize,
    /// Unique payers.
    pub unique_payers: usize,
    /// Unique payees.
    pub unique_payees: usize,
    /// Average transaction size.
    pub avg_transaction_size: f64,
    /// Peak hour.
    pub peak_hour: Option<u32>,
    /// Network density.
    pub network_density: f64,
}

/// Flow anomaly.
#[derive(Debug, Clone)]
pub struct FlowAnomaly {
    /// Anomaly type.
    pub anomaly_type: FlowAnomalyType,
    /// Related node or edge.
    pub entity: String,
    /// Description.
    pub description: String,
    /// Severity (0-1).
    pub severity: f64,
    /// Timestamp.
    pub timestamp: u64,
}

/// Flow anomaly type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowAnomalyType {
    /// Unusual volume.
    UnusualVolume,
    /// Unusual frequency.
    UnusualFrequency,
    /// New connection.
    NewConnection,
    /// Circular flow.
    CircularFlow,
    /// Rapid movement.
    RapidMovement,
    /// Structuring.
    Structuring,
}

// ============================================================================
// Account Types
// ============================================================================

/// Account for payment processing.
#[derive(Debug, Clone)]
pub struct PaymentAccount {
    /// Account ID.
    pub id: String,
    /// Account type.
    pub account_type: AccountType,
    /// Balance.
    pub balance: f64,
    /// Available balance.
    pub available_balance: f64,
    /// Currency.
    pub currency: String,
    /// Status.
    pub status: AccountStatus,
    /// Daily limit.
    pub daily_limit: Option<f64>,
    /// Daily used.
    pub daily_used: f64,
}

/// Account type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccountType {
    /// Checking account.
    Checking,
    /// Savings account.
    Savings,
    /// Operating account.
    Operating,
    /// Settlement account.
    Settlement,
    /// Escrow account.
    Escrow,
}

/// Account status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccountStatus {
    /// Active.
    Active,
    /// Frozen.
    Frozen,
    /// Closed.
    Closed,
    /// Pending.
    Pending,
}

// ============================================================================
// Validation Types
// ============================================================================

/// Validation result.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Is valid.
    pub is_valid: bool,
    /// Errors.
    pub errors: Vec<ValidationError>,
    /// Warnings.
    pub warnings: Vec<ValidationWarning>,
}

/// Validation error.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error code.
    pub code: String,
    /// Error message.
    pub message: String,
    /// Field (if applicable).
    pub field: Option<String>,
}

/// Validation warning.
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning code.
    pub code: String,
    /// Warning message.
    pub message: String,
}
