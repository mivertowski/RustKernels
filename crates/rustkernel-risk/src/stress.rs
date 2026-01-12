//! Stress testing kernels.
//!
//! This module provides stress testing analytics:
//! - Scenario-based stress testing
//! - Historical stress scenarios
//! - Reverse stress testing

use crate::messages::{
    StressTestingBatchInput, StressTestingBatchOutput, StressTestingInput, StressTestingOutput,
};
use crate::types::{Portfolio, Sensitivity, StressScenario, StressTestResult};
use async_trait::async_trait;
use rustkernel_core::error::Result;
use rustkernel_core::traits::BatchKernel;
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::time::Instant;

// ============================================================================
// Stress Testing Kernel
// ============================================================================

/// Stress testing kernel.
///
/// Applies stress scenarios to portfolios and calculates P&L impacts.
#[derive(Debug, Clone)]
pub struct StressTesting {
    metadata: KernelMetadata,
}

impl Default for StressTesting {
    fn default() -> Self {
        Self::new()
    }
}

impl StressTesting {
    /// Create a new stress testing kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("risk/stress-testing", Domain::RiskAnalytics)
                .with_description("Scenario-based stress testing")
                .with_throughput(5_000)
                .with_latency_us(2000.0),
        }
    }

    /// Run a single stress scenario.
    ///
    /// # Arguments
    /// * `portfolio` - Portfolio to stress
    /// * `scenario` - Stress scenario to apply
    /// * `sensitivities` - Optional sensitivities for non-linear effects
    pub fn compute(
        portfolio: &Portfolio,
        scenario: &StressScenario,
        sensitivities: Option<&[Sensitivity]>,
    ) -> StressTestResult {
        if portfolio.n_assets() == 0 {
            return StressTestResult {
                scenario_name: scenario.name.clone(),
                pnl_impact: 0.0,
                pnl_impact_pct: 0.0,
                asset_impacts: Vec::new(),
                factor_impacts: Vec::new(),
                post_stress_value: 0.0,
            };
        }

        let total_value = portfolio.total_value();

        // Calculate impact by asset
        let mut asset_impacts = Vec::with_capacity(portfolio.n_assets());
        let mut total_pnl = 0.0;

        for (i, (&asset_id, &value)) in portfolio
            .asset_ids
            .iter()
            .zip(portfolio.values.iter())
            .enumerate()
        {
            let mut asset_pnl = 0.0;

            // Apply shocks based on sensitivities
            let sens = sensitivities
                .and_then(|s| s.get(i))
                .cloned()
                .unwrap_or_default();

            for (factor_name, shock) in &scenario.shocks {
                let factor_impact = Self::calculate_factor_impact(factor_name, *shock, value, &sens);
                asset_pnl += factor_impact;
            }

            asset_impacts.push((asset_id, asset_pnl));
            total_pnl += asset_pnl;
        }

        // Calculate factor-level impacts
        let factor_impacts: Vec<(String, f64)> = scenario
            .shocks
            .iter()
            .map(|(factor_name, shock)| {
                let mut factor_total = 0.0;
                for (i, &value) in portfolio.values.iter().enumerate() {
                    let sens = sensitivities
                        .and_then(|s| s.get(i))
                        .cloned()
                        .unwrap_or_default();
                    factor_total += Self::calculate_factor_impact(factor_name, *shock, value, &sens);
                }
                (factor_name.clone(), factor_total)
            })
            .collect();

        let pnl_impact_pct = if total_value.abs() > 1e-10 {
            total_pnl / total_value * 100.0
        } else {
            0.0
        };

        StressTestResult {
            scenario_name: scenario.name.clone(),
            pnl_impact: total_pnl,
            pnl_impact_pct,
            asset_impacts,
            factor_impacts,
            post_stress_value: total_value + total_pnl,
        }
    }

    /// Run multiple stress scenarios.
    pub fn compute_batch(
        portfolio: &Portfolio,
        scenarios: &[StressScenario],
        sensitivities: Option<&[Sensitivity]>,
    ) -> Vec<StressTestResult> {
        scenarios
            .iter()
            .map(|s| Self::compute(portfolio, s, sensitivities))
            .collect()
    }

    /// Calculate impact of a factor shock on an asset.
    fn calculate_factor_impact(
        factor_name: &str,
        shock: f64,
        value: f64,
        sens: &Sensitivity,
    ) -> f64 {
        match factor_name.to_lowercase().as_str() {
            "equity" | "stock" | "index" => {
                // Linear: delta * S * dS/S
                // Quadratic: + 0.5 * gamma * S^2 * (dS/S)^2
                let linear = sens.delta * value * shock;
                let quadratic = 0.5 * sens.gamma * value * shock.powi(2);
                linear + quadratic
            }
            "interest_rate" | "rate" | "ir" => {
                // Duration-based: rho * value * dR
                sens.rho * value * shock
            }
            "volatility" | "vol" | "vega" => {
                // Vega: vega * dVol
                sens.vega * shock
            }
            "fx" | "currency" => {
                // FX sensitivity similar to equity
                sens.delta * value * shock
            }
            "credit_spread" | "credit" => {
                // Credit spread sensitivity (negative impact for spread widening)
                -sens.delta * value * shock
            }
            "commodity" => {
                sens.delta * value * shock
            }
            _ => {
                // Default: linear sensitivity
                sens.delta * value * shock
            }
        }
    }

    /// Generate standard stress scenarios.
    pub fn standard_scenarios() -> Vec<StressScenario> {
        vec![
            StressScenario::new(
                "2008 Financial Crisis",
                "Equity -40%, Credit +300bps, Vol +100%",
                vec![
                    ("equity".to_string(), -0.40),
                    ("credit_spread".to_string(), 0.03),
                    ("volatility".to_string(), 1.0),
                ],
                0.01,
            ),
            StressScenario::new(
                "COVID-19 Crash",
                "Equity -30%, Rate -150bps, Vol +200%",
                vec![
                    ("equity".to_string(), -0.30),
                    ("interest_rate".to_string(), -0.015),
                    ("volatility".to_string(), 2.0),
                ],
                0.02,
            ),
            StressScenario::equity_crash(-0.20),
            StressScenario::rate_shock(200.0), // +200bps
            StressScenario::rate_shock(-100.0), // -100bps
            StressScenario::credit_spread_widening(100.0),
            StressScenario::new(
                "Stagflation",
                "Equity -15%, Rates +300bps, Commodity +30%",
                vec![
                    ("equity".to_string(), -0.15),
                    ("interest_rate".to_string(), 0.03),
                    ("commodity".to_string(), 0.30),
                ],
                0.03,
            ),
            StressScenario::new(
                "Flight to Quality",
                "Equity -25%, Rate -200bps, Credit +150bps",
                vec![
                    ("equity".to_string(), -0.25),
                    ("interest_rate".to_string(), -0.02),
                    ("credit_spread".to_string(), 0.015),
                ],
                0.02,
            ),
        ]
    }

    /// Find worst-case scenario from a set.
    pub fn worst_case(
        portfolio: &Portfolio,
        scenarios: &[StressScenario],
        sensitivities: Option<&[Sensitivity]>,
    ) -> Option<StressTestResult> {
        let results = Self::compute_batch(portfolio, scenarios, sensitivities);
        results.into_iter().min_by(|a, b| {
            a.pnl_impact
                .partial_cmp(&b.pnl_impact)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Calculate expected stress loss (probability-weighted).
    pub fn expected_stress_loss(
        portfolio: &Portfolio,
        scenarios: &[StressScenario],
        sensitivities: Option<&[Sensitivity]>,
    ) -> f64 {
        let results = Self::compute_batch(portfolio, scenarios, sensitivities);

        results
            .iter()
            .zip(scenarios.iter())
            .map(|(result, scenario)| result.pnl_impact.min(0.0) * scenario.probability)
            .sum()
    }
}

impl GpuKernel for StressTesting {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[async_trait]
impl BatchKernel<StressTestingInput, StressTestingOutput> for StressTesting {
    async fn execute(&self, input: StressTestingInput) -> Result<StressTestingOutput> {
        let start = Instant::now();
        let result = Self::compute(
            &input.portfolio,
            &input.scenario,
            input.sensitivities.as_deref(),
        );
        Ok(StressTestingOutput {
            result,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

#[async_trait]
impl BatchKernel<StressTestingBatchInput, StressTestingBatchOutput> for StressTesting {
    async fn execute(&self, input: StressTestingBatchInput) -> Result<StressTestingBatchOutput> {
        let start = Instant::now();
        let results = Self::compute_batch(
            &input.portfolio,
            &input.scenarios,
            input.sensitivities.as_deref(),
        );
        let worst_case = Self::worst_case(
            &input.portfolio,
            &input.scenarios,
            input.sensitivities.as_deref(),
        );
        Ok(StressTestingBatchOutput {
            results,
            worst_case,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_equity_portfolio() -> Portfolio {
        Portfolio::new(
            vec![1, 2, 3],
            vec![100_000.0, 50_000.0, 25_000.0],
            vec![0.08, 0.10, 0.06],
            vec![0.20, 0.25, 0.15],
            vec![
                1.0, 0.6, 0.4,
                0.6, 1.0, 0.5,
                0.4, 0.5, 1.0,
            ],
        )
    }

    fn create_sensitivities() -> Vec<Sensitivity> {
        vec![
            Sensitivity {
                asset_id: 1,
                delta: 1.0,
                gamma: 0.05,
                vega: 1000.0,
                theta: -50.0,
                rho: -200.0,
            },
            Sensitivity {
                asset_id: 2,
                delta: 1.2,
                gamma: 0.08,
                vega: 800.0,
                theta: -40.0,
                rho: -150.0,
            },
            Sensitivity {
                asset_id: 3,
                delta: 0.8,
                gamma: 0.03,
                vega: 500.0,
                theta: -25.0,
                rho: -100.0,
            },
        ]
    }

    #[test]
    fn test_stress_testing_metadata() {
        let kernel = StressTesting::new();
        assert_eq!(kernel.metadata().id, "risk/stress-testing");
        assert_eq!(kernel.metadata().domain, Domain::RiskAnalytics);
    }

    #[test]
    fn test_equity_crash_scenario() {
        let portfolio = create_equity_portfolio();
        let scenario = StressScenario::equity_crash(-0.30);

        let result = StressTesting::compute(&portfolio, &scenario, None);

        assert_eq!(result.scenario_name, "Equity Crash");

        // With 30% equity crash and delta=1, should lose ~30% of portfolio
        let expected_loss = -0.30 * portfolio.total_value();
        let tolerance = 0.01 * portfolio.total_value();
        assert!(
            (result.pnl_impact - expected_loss).abs() < tolerance,
            "Expected ~{}%, got {}%",
            -30.0,
            result.pnl_impact_pct
        );
    }

    #[test]
    fn test_stress_with_sensitivities() {
        let portfolio = create_equity_portfolio();
        let sensitivities = create_sensitivities();
        let scenario = StressScenario::equity_crash(-0.20);

        let result_no_sens = StressTesting::compute(&portfolio, &scenario, None);
        let result_with_sens = StressTesting::compute(&portfolio, &scenario, Some(&sensitivities));

        // With gamma, the result should differ due to convexity
        assert!(
            result_no_sens.pnl_impact != result_with_sens.pnl_impact,
            "Gamma should affect result"
        );
    }

    #[test]
    fn test_rate_shock_scenario() {
        let portfolio = create_equity_portfolio();
        let sensitivities = create_sensitivities();
        let scenario = StressScenario::rate_shock(200.0); // +200bps

        let result = StressTesting::compute(&portfolio, &scenario, Some(&sensitivities));

        // With negative rho (bond-like), rate increase should cause losses
        assert!(
            result.pnl_impact < 0.0,
            "Rate increase should cause loss with negative rho"
        );
    }

    #[test]
    fn test_batch_stress() {
        let portfolio = create_equity_portfolio();
        let scenarios = vec![
            StressScenario::equity_crash(-0.10),
            StressScenario::equity_crash(-0.20),
            StressScenario::equity_crash(-0.30),
        ];

        let results = StressTesting::compute_batch(&portfolio, &scenarios, None);

        assert_eq!(results.len(), 3);

        // Larger shocks should cause larger losses
        assert!(results[0].pnl_impact > results[1].pnl_impact);
        assert!(results[1].pnl_impact > results[2].pnl_impact);
    }

    #[test]
    fn test_standard_scenarios() {
        let scenarios = StressTesting::standard_scenarios();

        assert!(!scenarios.is_empty());
        assert!(scenarios
            .iter()
            .any(|s| s.name.contains("2008") || s.name.contains("Financial")));
        assert!(scenarios.iter().any(|s| s.name.contains("COVID")));
    }

    #[test]
    fn test_worst_case() {
        let portfolio = create_equity_portfolio();
        let scenarios = vec![
            StressScenario::equity_crash(-0.10),
            StressScenario::equity_crash(-0.40),
            StressScenario::equity_crash(-0.20),
        ];

        let worst = StressTesting::worst_case(&portfolio, &scenarios, None);

        assert!(worst.is_some());
        assert!(
            worst.as_ref().unwrap().pnl_impact_pct < -35.0,
            "Worst case should be -40% scenario"
        );
    }

    #[test]
    fn test_expected_stress_loss() {
        let portfolio = create_equity_portfolio();
        let scenarios = vec![
            StressScenario::new("Mild", "Mild downturn", vec![("equity".to_string(), -0.10)], 0.10),
            StressScenario::new(
                "Severe",
                "Severe crash",
                vec![("equity".to_string(), -0.40)],
                0.01,
            ),
        ];

        let expected_loss = StressTesting::expected_stress_loss(&portfolio, &scenarios, None);

        // Expected loss = P(mild) * Loss(mild) + P(severe) * Loss(severe)
        // = 0.10 * (-10% * 175k) + 0.01 * (-40% * 175k)
        // = 0.10 * -17500 + 0.01 * -70000
        // = -1750 + -700 = -2450
        let manual_expected = 0.10 * (-0.10 * 175_000.0) + 0.01 * (-0.40 * 175_000.0);

        assert!(
            (expected_loss - manual_expected).abs() < 100.0,
            "Expected stress loss calculation: got {}, expected {}",
            expected_loss,
            manual_expected
        );
    }

    #[test]
    fn test_asset_impacts() {
        let portfolio = create_equity_portfolio();
        let scenario = StressScenario::equity_crash(-0.25);

        let result = StressTesting::compute(&portfolio, &scenario, None);

        assert_eq!(result.asset_impacts.len(), 3);

        // Each asset impact should be proportional to value
        for (i, (asset_id, impact)) in result.asset_impacts.iter().enumerate() {
            assert_eq!(*asset_id, portfolio.asset_ids[i]);
            let expected = -0.25 * portfolio.values[i];
            assert!(
                (impact - expected).abs() < 1.0,
                "Asset {} impact: {} vs expected {}",
                asset_id,
                impact,
                expected
            );
        }
    }

    #[test]
    fn test_factor_impacts() {
        let portfolio = create_equity_portfolio();
        let scenario = StressScenario::new(
            "Multi-factor",
            "Multiple shocks",
            vec![
                ("equity".to_string(), -0.20),
                ("volatility".to_string(), 0.50),
            ],
            0.05,
        );
        let sensitivities = create_sensitivities();

        let result = StressTesting::compute(&portfolio, &scenario, Some(&sensitivities));

        assert_eq!(result.factor_impacts.len(), 2);
        assert!(result
            .factor_impacts
            .iter()
            .any(|(name, _)| name == "equity"));
        assert!(result
            .factor_impacts
            .iter()
            .any(|(name, _)| name == "volatility"));
    }

    #[test]
    fn test_empty_portfolio() {
        let empty = Portfolio::new(Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let scenario = StressScenario::equity_crash(-0.30);

        let result = StressTesting::compute(&empty, &scenario, None);

        assert_eq!(result.pnl_impact, 0.0);
        assert_eq!(result.post_stress_value, 0.0);
    }

    #[test]
    fn test_post_stress_value() {
        let portfolio = create_equity_portfolio();
        let scenario = StressScenario::equity_crash(-0.20);

        let result = StressTesting::compute(&portfolio, &scenario, None);

        let expected_post_stress = portfolio.total_value() * 0.80; // 80% remaining
        assert!(
            (result.post_stress_value - expected_post_stress).abs() < 100.0,
            "Post-stress value: {} vs expected {}",
            result.post_stress_value,
            expected_post_stress
        );
    }
}
