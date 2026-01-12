//! Message types for Risk Analytics kernels.
//!
//! This module defines input/output types for batch execution
//! of risk analytics kernels and Ring kernel messages for K2K communication.

use crate::types::{
    CreditExposure, CreditFactors, CreditRiskResult, Portfolio, PortfolioRiskResult, Sensitivity,
    StressScenario, StressTestResult, VaRParams, VaRResult,
};
use rustkernel_derive::KernelMessage;
use serde::{Deserialize, Serialize};

// ============================================================================
// Credit Risk Scoring Messages
// ============================================================================

/// Input for credit risk scoring.
///
/// Ring message type_id: 3000 (RiskAnalytics domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 3000, domain = "RiskAnalytics")]
pub struct CreditRiskScoringInput {
    /// Credit scoring factors for the obligor.
    pub factors: CreditFactors,
    /// Exposure at Default.
    pub ead: f64,
    /// Maturity in years.
    pub maturity: f64,
}

impl CreditRiskScoringInput {
    /// Create a new credit risk scoring input.
    pub fn new(factors: CreditFactors, ead: f64, maturity: f64) -> Self {
        Self {
            factors,
            ead,
            maturity,
        }
    }
}

/// Output from credit risk scoring.
///
/// Ring message type_id: 3001 (RiskAnalytics domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 3001, domain = "RiskAnalytics")]
pub struct CreditRiskScoringOutput {
    /// The credit risk result.
    pub result: CreditRiskResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

/// Input for batch credit risk scoring from exposures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditRiskBatchInput {
    /// List of credit exposures.
    pub exposures: Vec<CreditExposure>,
}

impl CreditRiskBatchInput {
    /// Create a new batch input from exposures.
    pub fn new(exposures: Vec<CreditExposure>) -> Self {
        Self { exposures }
    }
}

/// Output from batch credit risk scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditRiskBatchOutput {
    /// Credit risk results for each exposure.
    pub results: Vec<CreditRiskResult>,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Monte Carlo VaR Messages
// ============================================================================

/// Input for Monte Carlo VaR calculation.
///
/// Ring message type_id: 3010 (RiskAnalytics domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 3010, domain = "RiskAnalytics")]
pub struct MonteCarloVaRInput {
    /// Portfolio to analyze.
    pub portfolio: Portfolio,
    /// VaR calculation parameters.
    pub params: VaRParams,
}

impl MonteCarloVaRInput {
    /// Create a new Monte Carlo VaR input.
    pub fn new(portfolio: Portfolio, params: VaRParams) -> Self {
        Self { portfolio, params }
    }

    /// Create with default parameters.
    pub fn with_defaults(portfolio: Portfolio) -> Self {
        Self {
            portfolio,
            params: VaRParams::default(),
        }
    }
}

/// Output from Monte Carlo VaR calculation.
///
/// Ring message type_id: 3011 (RiskAnalytics domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 3011, domain = "RiskAnalytics")]
pub struct MonteCarloVaROutput {
    /// The VaR result.
    pub result: VaRResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Portfolio Risk Aggregation Messages
// ============================================================================

/// Input for portfolio risk aggregation.
///
/// Ring message type_id: 3020 (RiskAnalytics domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 3020, domain = "RiskAnalytics")]
pub struct PortfolioRiskAggregationInput {
    /// Portfolio to analyze.
    pub portfolio: Portfolio,
    /// Confidence level for VaR (e.g., 0.95).
    pub confidence_level: f64,
    /// Holding period in days.
    pub holding_period: u32,
}

impl PortfolioRiskAggregationInput {
    /// Create a new portfolio risk aggregation input.
    pub fn new(portfolio: Portfolio, confidence_level: f64, holding_period: u32) -> Self {
        Self {
            portfolio,
            confidence_level,
            holding_period,
        }
    }

    /// Create with standard parameters (99% confidence, 10-day holding).
    pub fn standard(portfolio: Portfolio) -> Self {
        Self {
            portfolio,
            confidence_level: 0.99,
            holding_period: 10,
        }
    }
}

/// Output from portfolio risk aggregation.
///
/// Ring message type_id: 3021 (RiskAnalytics domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 3021, domain = "RiskAnalytics")]
pub struct PortfolioRiskAggregationOutput {
    /// The portfolio risk result.
    pub result: PortfolioRiskResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Stress Testing Messages
// ============================================================================

/// Input for stress testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestingInput {
    /// Portfolio to stress test.
    pub portfolio: Portfolio,
    /// Stress scenario to apply.
    pub scenario: StressScenario,
    /// Optional sensitivities for non-linear effects.
    pub sensitivities: Option<Vec<Sensitivity>>,
}

impl StressTestingInput {
    /// Create a new stress testing input.
    pub fn new(portfolio: Portfolio, scenario: StressScenario) -> Self {
        Self {
            portfolio,
            scenario,
            sensitivities: None,
        }
    }

    /// Create with sensitivities.
    pub fn with_sensitivities(
        portfolio: Portfolio,
        scenario: StressScenario,
        sensitivities: Vec<Sensitivity>,
    ) -> Self {
        Self {
            portfolio,
            scenario,
            sensitivities: Some(sensitivities),
        }
    }
}

/// Output from stress testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestingOutput {
    /// The stress test result.
    pub result: StressTestResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

/// Input for batch stress testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestingBatchInput {
    /// Portfolio to stress test.
    pub portfolio: Portfolio,
    /// Stress scenarios to apply.
    pub scenarios: Vec<StressScenario>,
    /// Optional sensitivities for non-linear effects.
    pub sensitivities: Option<Vec<Sensitivity>>,
}

impl StressTestingBatchInput {
    /// Create a new batch stress testing input.
    pub fn new(portfolio: Portfolio, scenarios: Vec<StressScenario>) -> Self {
        Self {
            portfolio,
            scenarios,
            sensitivities: None,
        }
    }

    /// Create with sensitivities.
    pub fn with_sensitivities(
        portfolio: Portfolio,
        scenarios: Vec<StressScenario>,
        sensitivities: Vec<Sensitivity>,
    ) -> Self {
        Self {
            portfolio,
            scenarios,
            sensitivities: Some(sensitivities),
        }
    }

    /// Create with standard scenarios.
    pub fn standard_scenarios(portfolio: Portfolio) -> Self {
        use crate::stress::StressTesting;
        Self {
            portfolio,
            scenarios: StressTesting::standard_scenarios(),
            sensitivities: None,
        }
    }
}

/// Output from batch stress testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestingBatchOutput {
    /// Stress test results for each scenario.
    pub results: Vec<StressTestResult>,
    /// Worst-case scenario result.
    pub worst_case: Option<StressTestResult>,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_portfolio() -> Portfolio {
        Portfolio::new(
            vec![1, 2],
            vec![100_000.0, 50_000.0],
            vec![0.08, 0.10],
            vec![0.15, 0.20],
            vec![1.0, 0.5, 0.5, 1.0],
        )
    }

    fn create_test_factors() -> CreditFactors {
        CreditFactors {
            obligor_id: 1,
            debt_to_income: 0.30,
            loan_to_value: 0.70,
            credit_utilization: 0.25,
            payment_history: 85.0,
            employment_years: 5.0,
            recent_inquiries: 2,
            delinquencies: 0,
            credit_history_years: 8.0,
        }
    }

    #[test]
    fn test_credit_risk_scoring_input() {
        let factors = create_test_factors();
        let input = CreditRiskScoringInput::new(factors, 100_000.0, 5.0);
        assert_eq!(input.ead, 100_000.0);
        assert_eq!(input.maturity, 5.0);
    }

    #[test]
    fn test_monte_carlo_var_input() {
        let portfolio = create_test_portfolio();
        let input = MonteCarloVaRInput::with_defaults(portfolio);
        assert_eq!(input.params.confidence_level, 0.99);
    }

    #[test]
    fn test_portfolio_risk_aggregation_input() {
        let portfolio = create_test_portfolio();
        let input = PortfolioRiskAggregationInput::standard(portfolio);
        assert_eq!(input.confidence_level, 0.99);
        assert_eq!(input.holding_period, 10);
    }

    #[test]
    fn test_stress_testing_input() {
        let portfolio = create_test_portfolio();
        let scenario = StressScenario::equity_crash(-0.20);
        let input = StressTestingInput::new(portfolio, scenario);
        assert!(input.sensitivities.is_none());
    }
}
