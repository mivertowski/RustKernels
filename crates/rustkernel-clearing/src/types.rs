//! Clearing and settlement types.

use std::collections::HashMap;

// ============================================================================
// Trade Types
// ============================================================================

/// A trade for clearing.
#[derive(Debug, Clone)]
pub struct Trade {
    /// Trade ID.
    pub id: u64,
    /// Security/instrument ID.
    pub security_id: String,
    /// Buyer party ID.
    pub buyer_id: String,
    /// Seller party ID.
    pub seller_id: String,
    /// Trade quantity.
    pub quantity: i64,
    /// Trade price (in cents/smallest unit).
    pub price: i64,
    /// Trade date (Unix timestamp).
    pub trade_date: u64,
    /// Settlement date (Unix timestamp).
    pub settlement_date: u64,
    /// Trade status.
    pub status: TradeStatus,
    /// Trade type.
    pub trade_type: TradeType,
    /// Additional attributes.
    pub attributes: HashMap<String, String>,
}

/// Trade status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TradeStatus {
    /// Pending validation.
    Pending,
    /// Validated and ready for settlement.
    Validated,
    /// Matched for DVP.
    Matched,
    /// Netted into position.
    Netted,
    /// Settled successfully.
    Settled,
    /// Failed validation.
    Failed,
    /// Cancelled.
    Cancelled,
}

/// Trade type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeType {
    /// Regular trade.
    Regular,
    /// Repo/reverse repo.
    Repo,
    /// Securities lending.
    SecLending,
    /// Corporate action.
    CorpAction,
    /// Transfer.
    Transfer,
}

impl Trade {
    /// Create a new trade.
    pub fn new(
        id: u64,
        security_id: String,
        buyer_id: String,
        seller_id: String,
        quantity: i64,
        price: i64,
        trade_date: u64,
        settlement_date: u64,
    ) -> Self {
        Self {
            id,
            security_id,
            buyer_id,
            seller_id,
            quantity,
            price,
            trade_date,
            settlement_date,
            status: TradeStatus::Pending,
            trade_type: TradeType::Regular,
            attributes: HashMap::new(),
        }
    }

    /// Get total value (quantity * price).
    pub fn value(&self) -> i64 {
        self.quantity.saturating_mul(self.price)
    }
}

// ============================================================================
// Validation Types
// ============================================================================

/// Trade validation result.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Trade ID.
    pub trade_id: u64,
    /// Is trade valid?
    pub is_valid: bool,
    /// Validation errors.
    pub errors: Vec<ValidationError>,
    /// Validation warnings.
    pub warnings: Vec<String>,
}

/// Validation error.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error code.
    pub code: String,
    /// Error message.
    pub message: String,
    /// Error severity.
    pub severity: ErrorSeverity,
}

/// Error severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// Blocks settlement.
    Critical,
    /// May block settlement.
    Warning,
    /// Informational only.
    Info,
}

/// Validation configuration.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Check counterparty eligibility.
    pub check_counterparty: bool,
    /// Check security eligibility.
    pub check_security: bool,
    /// Check position limits.
    pub check_limits: bool,
    /// Check settlement date validity.
    pub check_settlement_date: bool,
    /// Minimum settlement days from trade.
    pub min_settlement_days: u32,
    /// Maximum settlement days from trade.
    pub max_settlement_days: u32,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            check_counterparty: true,
            check_security: true,
            check_limits: true,
            check_settlement_date: true,
            min_settlement_days: 0,
            max_settlement_days: 30,
        }
    }
}

// ============================================================================
// DVP Types
// ============================================================================

/// DVP (Delivery vs Payment) instruction.
#[derive(Debug, Clone)]
pub struct DVPInstruction {
    /// Instruction ID.
    pub id: u64,
    /// Trade ID.
    pub trade_id: u64,
    /// Security ID.
    pub security_id: String,
    /// Delivering party.
    pub deliverer: String,
    /// Receiving party.
    pub receiver: String,
    /// Quantity to deliver.
    pub quantity: i64,
    /// Payment amount.
    pub payment_amount: i64,
    /// Payment currency.
    pub currency: String,
    /// Settlement date.
    pub settlement_date: u64,
    /// Instruction status.
    pub status: DVPStatus,
}

/// DVP status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DVPStatus {
    /// Awaiting match.
    Pending,
    /// Matched with counterparty instruction.
    Matched,
    /// Securities delivered.
    SecuritiesDelivered,
    /// Payment made.
    PaymentMade,
    /// Fully settled.
    Settled,
    /// Failed.
    Failed,
}

/// DVP matching result.
#[derive(Debug, Clone)]
pub struct DVPMatchResult {
    /// Matched instruction pairs.
    pub matched_pairs: Vec<(u64, u64)>,
    /// Unmatched instructions.
    pub unmatched: Vec<u64>,
    /// Match rate.
    pub match_rate: f64,
    /// Matching details.
    pub details: Vec<DVPMatchDetail>,
}

/// DVP match detail.
#[derive(Debug, Clone)]
pub struct DVPMatchDetail {
    /// Delivery instruction ID.
    pub delivery_id: u64,
    /// Payment instruction ID.
    pub payment_id: u64,
    /// Match confidence.
    pub confidence: f64,
    /// Matching differences (if any).
    pub differences: Vec<String>,
}

// ============================================================================
// Netting Types
// ============================================================================

/// Netting position.
#[derive(Debug, Clone)]
pub struct NetPosition {
    /// Party ID.
    pub party_id: String,
    /// Security ID.
    pub security_id: String,
    /// Net quantity (positive = receive, negative = deliver).
    pub net_quantity: i64,
    /// Net payment (positive = receive, negative = pay).
    pub net_payment: i64,
    /// Currency.
    pub currency: String,
    /// Contributing trades.
    pub trade_ids: Vec<u64>,
}

/// Netting result.
#[derive(Debug, Clone)]
pub struct NettingResult {
    /// Net positions per party/security.
    pub positions: Vec<NetPosition>,
    /// Gross trade count.
    pub gross_trade_count: u64,
    /// Net instruction count.
    pub net_instruction_count: u64,
    /// Netting efficiency (reduction ratio).
    pub efficiency: f64,
    /// Netting by party.
    pub party_summary: HashMap<String, PartySummary>,
}

/// Party netting summary.
#[derive(Debug, Clone, Default)]
pub struct PartySummary {
    /// Gross deliveries.
    pub gross_deliveries: i64,
    /// Gross receipts.
    pub gross_receipts: i64,
    /// Net position.
    pub net_position: i64,
    /// Gross payments.
    pub gross_payments: i64,
    /// Net payment.
    pub net_payment: i64,
    /// Trade count.
    pub trade_count: u64,
}

/// Netting configuration.
#[derive(Debug, Clone)]
pub struct NettingConfig {
    /// Net by security.
    pub net_by_security: bool,
    /// Net by settlement date.
    pub net_by_settlement_date: bool,
    /// Net by currency.
    pub net_by_currency: bool,
    /// Include failed trades.
    pub include_failed: bool,
}

impl Default for NettingConfig {
    fn default() -> Self {
        Self {
            net_by_security: true,
            net_by_settlement_date: true,
            net_by_currency: true,
            include_failed: false,
        }
    }
}

// ============================================================================
// Settlement Types
// ============================================================================

/// Settlement instruction.
#[derive(Debug, Clone)]
pub struct SettlementInstruction {
    /// Instruction ID.
    pub id: u64,
    /// Party ID.
    pub party_id: String,
    /// Security ID.
    pub security_id: String,
    /// Instruction type.
    pub instruction_type: InstructionType,
    /// Quantity.
    pub quantity: i64,
    /// Payment amount.
    pub payment_amount: i64,
    /// Currency.
    pub currency: String,
    /// Settlement date.
    pub settlement_date: u64,
    /// Status.
    pub status: SettlementStatus,
    /// Source trade IDs.
    pub source_trades: Vec<u64>,
}

/// Instruction type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstructionType {
    /// Deliver securities.
    Deliver,
    /// Receive securities.
    Receive,
    /// Make payment.
    Pay,
    /// Receive payment.
    Collect,
}

/// Settlement status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettlementStatus {
    /// Pending execution.
    Pending,
    /// In progress.
    InProgress,
    /// Partially settled.
    Partial,
    /// Fully settled.
    Settled,
    /// Failed.
    Failed,
    /// On hold.
    OnHold,
}

/// Settlement execution result.
#[derive(Debug, Clone)]
pub struct SettlementExecutionResult {
    /// Successfully settled instructions.
    pub settled: Vec<u64>,
    /// Failed instructions.
    pub failed: Vec<(u64, String)>,
    /// Pending instructions.
    pub pending: Vec<u64>,
    /// Settlement rate.
    pub settlement_rate: f64,
    /// Total value settled.
    pub value_settled: i64,
    /// Total value failed.
    pub value_failed: i64,
}

// ============================================================================
// Efficiency Metrics
// ============================================================================

/// Zero balance frequency metrics.
#[derive(Debug, Clone)]
pub struct ZeroBalanceMetrics {
    /// Party ID.
    pub party_id: String,
    /// Total settlement days analyzed.
    pub total_days: u32,
    /// Days with zero end-of-day balance.
    pub zero_balance_days: u32,
    /// Zero balance frequency (0-1).
    pub frequency: f64,
    /// Average end-of-day position.
    pub avg_eod_position: f64,
    /// Peak position.
    pub peak_position: i64,
    /// Intraday turnover.
    pub avg_intraday_turnover: f64,
}

/// Settlement efficiency result.
#[derive(Debug, Clone)]
pub struct SettlementEfficiency {
    /// Period analyzed (days).
    pub period_days: u32,
    /// Total instructions.
    pub total_instructions: u64,
    /// Settled on time.
    pub on_time_settlements: u64,
    /// Late settlements.
    pub late_settlements: u64,
    /// Failed settlements.
    pub failed_settlements: u64,
    /// On-time rate.
    pub on_time_rate: f64,
    /// Average settlement delay (seconds).
    pub avg_delay_seconds: f64,
    /// Zero balance metrics per party.
    pub party_metrics: Vec<ZeroBalanceMetrics>,
}
