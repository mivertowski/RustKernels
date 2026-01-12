//! Risk analytics types and data structures.

use serde::{Deserialize, Serialize};

// ============================================================================
// Portfolio Types
// ============================================================================

/// A portfolio of assets for risk analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    /// Asset identifiers.
    pub asset_ids: Vec<u64>,
    /// Position values (notional or market value).
    pub values: Vec<f64>,
    /// Expected returns per asset.
    pub expected_returns: Vec<f64>,
    /// Volatilities per asset.
    pub volatilities: Vec<f64>,
    /// Correlation matrix (flattened, row-major).
    pub correlation_matrix: Vec<f64>,
}

impl Portfolio {
    /// Create a new portfolio.
    pub fn new(
        asset_ids: Vec<u64>,
        values: Vec<f64>,
        expected_returns: Vec<f64>,
        volatilities: Vec<f64>,
        correlation_matrix: Vec<f64>,
    ) -> Self {
        Self {
            asset_ids,
            values,
            expected_returns,
            volatilities,
            correlation_matrix,
        }
    }

    /// Get the number of assets in the portfolio.
    pub fn n_assets(&self) -> usize {
        self.asset_ids.len()
    }

    /// Get total portfolio value.
    pub fn total_value(&self) -> f64 {
        self.values.iter().sum()
    }

    /// Get portfolio weights.
    pub fn weights(&self) -> Vec<f64> {
        let total = self.total_value();
        if total.abs() < 1e-10 {
            vec![0.0; self.n_assets()]
        } else {
            self.values.iter().map(|v| v / total).collect()
        }
    }

    /// Get correlation between two assets.
    pub fn correlation(&self, i: usize, j: usize) -> f64 {
        let n = self.n_assets();
        if i >= n || j >= n {
            return 0.0;
        }
        self.correlation_matrix[i * n + j]
    }
}

// ============================================================================
// Credit Risk Types
// ============================================================================

/// Credit exposure for a single obligor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditExposure {
    /// Obligor ID.
    pub obligor_id: u64,
    /// Exposure at Default (EAD).
    pub ead: f64,
    /// Probability of Default (PD).
    pub pd: f64,
    /// Loss Given Default (LGD).
    pub lgd: f64,
    /// Maturity in years.
    pub maturity: f64,
    /// Credit rating (1=best, higher=worse).
    pub rating: u8,
}

impl CreditExposure {
    /// Create a new credit exposure.
    pub fn new(obligor_id: u64, ead: f64, pd: f64, lgd: f64, maturity: f64, rating: u8) -> Self {
        Self {
            obligor_id,
            ead,
            pd,
            lgd,
            maturity,
            rating,
        }
    }

    /// Calculate expected loss.
    pub fn expected_loss(&self) -> f64 {
        self.pd * self.lgd * self.ead
    }
}

/// Credit scoring factors for an obligor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditFactors {
    /// Obligor ID.
    pub obligor_id: u64,
    /// Debt-to-income ratio.
    pub debt_to_income: f64,
    /// Loan-to-value ratio (for secured loans).
    pub loan_to_value: f64,
    /// Credit utilization (0-1).
    pub credit_utilization: f64,
    /// Payment history score (0-100).
    pub payment_history: f64,
    /// Time with current employer (years).
    pub employment_years: f64,
    /// Number of credit inquiries in last 12 months.
    pub recent_inquiries: u32,
    /// Number of delinquencies in history.
    pub delinquencies: u32,
    /// Years of credit history.
    pub credit_history_years: f64,
}

/// Result of credit risk scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditRiskResult {
    /// Obligor ID.
    pub obligor_id: u64,
    /// Probability of Default.
    pub pd: f64,
    /// Loss Given Default.
    pub lgd: f64,
    /// Expected Loss.
    pub expected_loss: f64,
    /// Risk-weighted assets (for capital calculation).
    pub rwa: f64,
    /// Credit score (internal rating).
    pub credit_score: f64,
    /// Factor contributions to score.
    pub factor_contributions: Vec<(String, f64)>,
}

// ============================================================================
// Market Risk Types
// ============================================================================

/// Value at Risk calculation parameters.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VaRParams {
    /// Confidence level (e.g., 0.95 for 95% VaR).
    pub confidence_level: f64,
    /// Holding period in days.
    pub holding_period: u32,
    /// Number of Monte Carlo simulations.
    pub n_simulations: u32,
}

impl Default for VaRParams {
    fn default() -> Self {
        Self {
            confidence_level: 0.99,
            holding_period: 10,
            n_simulations: 10_000,
        }
    }
}

impl VaRParams {
    /// Create new VaR parameters.
    pub fn new(confidence_level: f64, holding_period: u32, n_simulations: u32) -> Self {
        Self {
            confidence_level,
            holding_period,
            n_simulations,
        }
    }
}

/// Result of VaR calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaRResult {
    /// Value at Risk (loss at confidence level).
    pub var: f64,
    /// Expected Shortfall (Conditional VaR).
    pub expected_shortfall: f64,
    /// Confidence level used.
    pub confidence_level: f64,
    /// Holding period in days.
    pub holding_period: u32,
    /// Component VaR by asset (if available).
    pub component_var: Vec<f64>,
    /// Marginal VaR by asset (if available).
    pub marginal_var: Vec<f64>,
    /// P&L distribution percentiles.
    pub percentiles: Vec<(f64, f64)>,
}

/// Portfolio risk aggregation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioRiskResult {
    /// Total portfolio VaR.
    pub portfolio_var: f64,
    /// Portfolio Expected Shortfall.
    pub portfolio_es: f64,
    /// Undiversified VaR (sum of individual VaRs).
    pub undiversified_var: f64,
    /// Diversification benefit (undiversified - portfolio).
    pub diversification_benefit: f64,
    /// Individual asset VaRs.
    pub asset_vars: Vec<f64>,
    /// Risk contributions by asset.
    pub risk_contributions: Vec<f64>,
    /// Correlation-adjusted covariance matrix.
    pub covariance_matrix: Vec<f64>,
}

// ============================================================================
// Stress Testing Types
// ============================================================================

/// A stress scenario definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressScenario {
    /// Scenario name.
    pub name: String,
    /// Scenario description.
    pub description: String,
    /// Risk factor shocks (factor_name, shock_percentage).
    pub shocks: Vec<(String, f64)>,
    /// Scenario probability (for expected loss calculation).
    pub probability: f64,
}

impl StressScenario {
    /// Create a new stress scenario.
    pub fn new(name: &str, description: &str, shocks: Vec<(String, f64)>, probability: f64) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            shocks,
            probability,
        }
    }

    /// Create a simple equity shock scenario.
    pub fn equity_crash(shock_pct: f64) -> Self {
        Self::new(
            "Equity Crash",
            "Severe equity market decline",
            vec![("equity".to_string(), shock_pct)],
            0.01,
        )
    }

    /// Create an interest rate shock scenario.
    pub fn rate_shock(shock_bps: f64) -> Self {
        Self::new(
            "Rate Shock",
            "Parallel shift in yield curve",
            vec![("interest_rate".to_string(), shock_bps / 10000.0)],
            0.05,
        )
    }

    /// Create a credit spread widening scenario.
    pub fn credit_spread_widening(shock_bps: f64) -> Self {
        Self::new(
            "Credit Spread Widening",
            "Credit spreads widen across all sectors",
            vec![("credit_spread".to_string(), shock_bps / 10000.0)],
            0.03,
        )
    }
}

/// Result of stress testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResult {
    /// Scenario name.
    pub scenario_name: String,
    /// P&L impact (negative = loss).
    pub pnl_impact: f64,
    /// Impact as percentage of portfolio value.
    pub pnl_impact_pct: f64,
    /// Impact by asset.
    pub asset_impacts: Vec<(u64, f64)>,
    /// Impact by risk factor.
    pub factor_impacts: Vec<(String, f64)>,
    /// Post-stress portfolio value.
    pub post_stress_value: f64,
}

// ============================================================================
// Risk Factor Types
// ============================================================================

/// Market risk factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    /// Factor name.
    pub name: String,
    /// Current value.
    pub value: f64,
    /// Historical volatility (annualized).
    pub volatility: f64,
    /// Factor type.
    pub factor_type: RiskFactorType,
}

/// Type of risk factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskFactorType {
    /// Equity index or stock price.
    Equity,
    /// Interest rate.
    InterestRate,
    /// FX rate.
    ForeignExchange,
    /// Credit spread.
    CreditSpread,
    /// Commodity price.
    Commodity,
    /// Volatility (e.g., VIX).
    Volatility,
}

/// Sensitivity of a position to risk factors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sensitivity {
    /// Asset ID.
    pub asset_id: u64,
    /// Delta (first-order sensitivity).
    pub delta: f64,
    /// Gamma (second-order sensitivity).
    pub gamma: f64,
    /// Vega (volatility sensitivity).
    pub vega: f64,
    /// Theta (time decay).
    pub theta: f64,
    /// Rho (interest rate sensitivity).
    pub rho: f64,
}

impl Default for Sensitivity {
    fn default() -> Self {
        Self {
            asset_id: 0,
            delta: 1.0,
            gamma: 0.0,
            vega: 0.0,
            theta: 0.0,
            rho: 0.0,
        }
    }
}
