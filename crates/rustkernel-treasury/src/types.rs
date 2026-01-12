//! Treasury management types.

use std::collections::HashMap;

// ============================================================================
// Cash Flow Types
// ============================================================================

/// A cash flow entry.
#[derive(Debug, Clone)]
pub struct CashFlow {
    /// Entry ID.
    pub id: u64,
    /// Date (Unix timestamp).
    pub date: u64,
    /// Amount (positive = inflow, negative = outflow).
    pub amount: f64,
    /// Currency.
    pub currency: String,
    /// Cash flow category.
    pub category: CashFlowCategory,
    /// Certainty level (0-1).
    pub certainty: f64,
    /// Description.
    pub description: String,
    /// Attributes.
    pub attributes: HashMap<String, String>,
}

/// Cash flow category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CashFlowCategory {
    /// Operating cash flows.
    Operating,
    /// Investment cash flows.
    Investing,
    /// Financing cash flows.
    Financing,
    /// Debt service.
    DebtService,
    /// Dividend payments.
    Dividend,
    /// Tax payments.
    Tax,
    /// Other.
    Other,
}

/// Cash flow forecast result.
#[derive(Debug, Clone)]
pub struct CashFlowForecast {
    /// Forecast horizon (days).
    pub horizon_days: u32,
    /// Daily forecasts.
    pub daily_forecasts: Vec<DailyForecast>,
    /// Total inflows.
    pub total_inflows: f64,
    /// Total outflows.
    pub total_outflows: f64,
    /// Net position at horizon.
    pub net_position: f64,
    /// Minimum balance during horizon.
    pub min_balance: f64,
    /// Maximum balance during horizon.
    pub max_balance: f64,
}

/// Daily forecast.
#[derive(Debug, Clone)]
pub struct DailyForecast {
    /// Date.
    pub date: u64,
    /// Expected inflows.
    pub inflows: f64,
    /// Expected outflows.
    pub outflows: f64,
    /// Net cash flow.
    pub net: f64,
    /// Cumulative balance.
    pub cumulative_balance: f64,
    /// Forecast uncertainty.
    pub uncertainty: f64,
}

// ============================================================================
// Collateral Types
// ============================================================================

/// Collateral asset.
#[derive(Debug, Clone)]
pub struct CollateralAsset {
    /// Asset ID.
    pub id: String,
    /// Asset type.
    pub asset_type: AssetType,
    /// Quantity held.
    pub quantity: f64,
    /// Market value.
    pub market_value: f64,
    /// Haircut (discount factor).
    pub haircut: f64,
    /// Eligible value after haircut.
    pub eligible_value: f64,
    /// Currency.
    pub currency: String,
    /// Is pledged.
    pub is_pledged: bool,
    /// Pledged to (counterparty).
    pub pledged_to: Option<String>,
}

/// Asset type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssetType {
    /// Cash.
    Cash,
    /// Government bonds.
    GovBond,
    /// Corporate bonds.
    CorpBond,
    /// Equities.
    Equity,
    /// Money market instruments.
    MoneyMarket,
    /// Other.
    Other,
}

/// Collateral requirement.
#[derive(Debug, Clone)]
pub struct CollateralRequirement {
    /// Counterparty ID.
    pub counterparty_id: String,
    /// Required amount.
    pub required_amount: f64,
    /// Currency.
    pub currency: String,
    /// Eligible asset types.
    pub eligible_types: Vec<AssetType>,
    /// Priority.
    pub priority: u32,
}

/// Collateral optimization result.
#[derive(Debug, Clone)]
pub struct CollateralOptimizationResult {
    /// Allocations.
    pub allocations: Vec<CollateralAllocation>,
    /// Total value allocated.
    pub total_allocated: f64,
    /// Excess collateral.
    pub excess: f64,
    /// Unmet requirements.
    pub shortfall: f64,
    /// Optimization score.
    pub score: f64,
}

/// Collateral allocation.
#[derive(Debug, Clone)]
pub struct CollateralAllocation {
    /// Asset ID.
    pub asset_id: String,
    /// Counterparty ID.
    pub counterparty_id: String,
    /// Allocated quantity.
    pub quantity: f64,
    /// Allocated value.
    pub value: f64,
}

// ============================================================================
// FX Types
// ============================================================================

/// Currency exposure.
#[derive(Debug, Clone)]
pub struct CurrencyExposure {
    /// Currency.
    pub currency: String,
    /// Net position.
    pub net_position: f64,
    /// Long positions.
    pub long_positions: f64,
    /// Short positions.
    pub short_positions: f64,
    /// Base currency equivalent.
    pub base_equivalent: f64,
}

/// FX rate.
#[derive(Debug, Clone)]
pub struct FXRate {
    /// Base currency.
    pub base: String,
    /// Quote currency.
    pub quote: String,
    /// Rate.
    pub rate: f64,
    /// Bid.
    pub bid: f64,
    /// Ask.
    pub ask: f64,
    /// Timestamp.
    pub timestamp: u64,
}

/// FX hedge.
#[derive(Debug, Clone)]
pub struct FXHedge {
    /// Hedge ID.
    pub id: u64,
    /// Currency pair.
    pub currency_pair: String,
    /// Notional amount.
    pub notional: f64,
    /// Hedge type.
    pub hedge_type: HedgeType,
    /// Strike rate (for options).
    pub strike: Option<f64>,
    /// Expiry date.
    pub expiry: u64,
    /// Cost.
    pub cost: f64,
}

/// Hedge type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HedgeType {
    /// Forward contract.
    Forward,
    /// Put option.
    Put,
    /// Call option.
    Call,
    /// Collar (put + call).
    Collar,
    /// Cross-currency swap.
    Swap,
}

/// FX hedging result.
#[derive(Debug, Clone)]
pub struct FXHedgingResult {
    /// Recommended hedges.
    pub hedges: Vec<FXHedge>,
    /// Net exposure after hedging.
    pub residual_exposure: f64,
    /// Hedge ratio.
    pub hedge_ratio: f64,
    /// Total hedge cost.
    pub total_cost: f64,
    /// VaR reduction.
    pub var_reduction: f64,
}

// ============================================================================
// Interest Rate Risk Types
// ============================================================================

/// Interest rate sensitive position.
#[derive(Debug, Clone)]
pub struct IRPosition {
    /// Position ID.
    pub id: String,
    /// Instrument type.
    pub instrument_type: IRInstrumentType,
    /// Notional.
    pub notional: f64,
    /// Current rate.
    pub rate: f64,
    /// Maturity date.
    pub maturity: u64,
    /// Next reset date (for floaters).
    pub next_reset: Option<u64>,
    /// Currency.
    pub currency: String,
}

/// IR instrument type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IRInstrumentType {
    /// Fixed rate bond.
    FixedBond,
    /// Floating rate note.
    FloatingNote,
    /// Interest rate swap.
    Swap,
    /// Fixed rate loan.
    FixedLoan,
    /// Floating rate loan.
    FloatingLoan,
    /// Deposit.
    Deposit,
}

/// Interest rate risk metrics.
#[derive(Debug, Clone)]
pub struct IRRiskMetrics {
    /// Total duration.
    pub duration: f64,
    /// Modified duration.
    pub modified_duration: f64,
    /// Convexity.
    pub convexity: f64,
    /// DV01 (dollar value of 1bp).
    pub dv01: f64,
    /// PV01 by currency.
    pub pv01_by_currency: HashMap<String, f64>,
    /// Gap analysis by time bucket.
    pub gap_by_bucket: Vec<GapBucket>,
}

/// Gap analysis bucket.
#[derive(Debug, Clone)]
pub struct GapBucket {
    /// Bucket name.
    pub bucket: String,
    /// Start days.
    pub start_days: u32,
    /// End days.
    pub end_days: u32,
    /// Rate sensitive assets.
    pub assets: f64,
    /// Rate sensitive liabilities.
    pub liabilities: f64,
    /// Gap (assets - liabilities).
    pub gap: f64,
    /// Cumulative gap.
    pub cumulative_gap: f64,
}

// ============================================================================
// Liquidity Types
// ============================================================================

/// Liquidity position.
#[derive(Debug, Clone)]
pub struct LiquidityPosition {
    /// Asset ID.
    pub id: String,
    /// Asset type.
    pub asset_type: LiquidityAssetType,
    /// Amount.
    pub amount: f64,
    /// Currency.
    pub currency: String,
    /// HQLA level (1, 2A, 2B, or None).
    pub hqla_level: Option<u8>,
    /// Haircut for LCR.
    pub lcr_haircut: f64,
    /// Days to liquidate.
    pub days_to_liquidate: u32,
}

/// Liquidity asset type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidityAssetType {
    /// Cash and central bank reserves.
    CashReserves,
    /// Level 1 HQLA.
    Level1HQLA,
    /// Level 2A HQLA.
    Level2AHQLA,
    /// Level 2B HQLA.
    Level2BHQLA,
    /// Other liquid assets.
    OtherLiquid,
    /// Illiquid assets.
    Illiquid,
}

/// Liquidity outflow.
#[derive(Debug, Clone)]
pub struct LiquidityOutflow {
    /// Category.
    pub category: OutflowCategory,
    /// Amount.
    pub amount: f64,
    /// Currency.
    pub currency: String,
    /// Runoff rate.
    pub runoff_rate: f64,
    /// Days to maturity.
    pub days_to_maturity: u32,
}

/// Outflow category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutflowCategory {
    /// Retail deposits.
    RetailDeposits,
    /// Wholesale funding.
    WholesaleFunding,
    /// Secured funding.
    SecuredFunding,
    /// Committed facilities.
    CommittedFacilities,
    /// Derivatives.
    Derivatives,
    /// Other.
    Other,
}

/// LCR calculation result.
#[derive(Debug, Clone)]
pub struct LCRResult {
    /// HQLA amount.
    pub hqla: f64,
    /// Net cash outflows.
    pub net_outflows: f64,
    /// LCR ratio.
    pub lcr_ratio: f64,
    /// Is compliant (>= 100%).
    pub is_compliant: bool,
    /// Buffer above minimum.
    pub buffer: f64,
    /// Breakdown by level.
    pub hqla_breakdown: HashMap<String, f64>,
}

/// NSFR calculation result.
#[derive(Debug, Clone)]
pub struct NSFRResult {
    /// Available stable funding.
    pub asf: f64,
    /// Required stable funding.
    pub rsf: f64,
    /// NSFR ratio.
    pub nsfr_ratio: f64,
    /// Is compliant (>= 100%).
    pub is_compliant: bool,
    /// Buffer above minimum.
    pub buffer: f64,
}

/// Liquidity optimization result.
#[derive(Debug, Clone)]
pub struct LiquidityOptimizationResult {
    /// LCR after optimization.
    pub lcr: LCRResult,
    /// NSFR after optimization.
    pub nsfr: NSFRResult,
    /// Recommended actions.
    pub actions: Vec<LiquidityAction>,
    /// Cost of actions.
    pub total_cost: f64,
    /// Improvement in LCR.
    pub lcr_improvement: f64,
}

/// Liquidity action.
#[derive(Debug, Clone)]
pub struct LiquidityAction {
    /// Action type.
    pub action_type: LiquidityActionType,
    /// Asset/liability ID.
    pub target_id: String,
    /// Amount.
    pub amount: f64,
    /// Impact on LCR.
    pub lcr_impact: f64,
    /// Cost.
    pub cost: f64,
}

/// Liquidity action type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidityActionType {
    /// Convert to HQLA.
    ConvertToHQLA,
    /// Extend funding maturity.
    ExtendFunding,
    /// Reduce outflow commitment.
    ReduceCommitment,
    /// Issue term funding.
    IssueTerm,
    /// Sell illiquid assets.
    SellIlliquid,
}
