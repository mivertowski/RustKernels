//! Cash flow forecasting kernel.
//!
//! This module provides cash flow forecasting for treasury:
//! - Multi-horizon cash flow projections
//! - Certainty-weighted aggregation
//! - Min/max balance tracking

use crate::types::{CashFlow, CashFlowCategory, CashFlowForecast, DailyForecast};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// Cash Flow Forecasting Kernel
// ============================================================================

/// Cash flow forecasting kernel.
///
/// Projects cash flows across multiple time horizons with certainty weighting.
#[derive(Debug, Clone)]
pub struct CashFlowForecasting {
    metadata: KernelMetadata,
}

impl Default for CashFlowForecasting {
    fn default() -> Self {
        Self::new()
    }
}

impl CashFlowForecasting {
    /// Create a new cash flow forecasting kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch(
                "treasury/cashflow-forecast",
                Domain::TreasuryManagement,
            )
            .with_description("Multi-horizon cash flow forecasting")
            .with_throughput(10_000)
            .with_latency_us(500.0),
        }
    }

    /// Generate cash flow forecast.
    pub fn forecast(cash_flows: &[CashFlow], config: &ForecastConfig) -> CashFlowForecast {
        let start_date = config.start_date;
        let end_date = start_date + (config.horizon_days as u64 * 86400);

        // Group cash flows by date
        let mut by_date: HashMap<u64, Vec<&CashFlow>> = HashMap::new();
        for cf in cash_flows {
            if cf.date >= start_date && cf.date < end_date {
                // Normalize to day boundary
                let day = (cf.date - start_date) / 86400;
                let day_start = start_date + day * 86400;
                by_date.entry(day_start).or_default().push(cf);
            }
        }

        // Generate daily forecasts
        let mut daily_forecasts = Vec::with_capacity(config.horizon_days as usize);
        let mut cumulative_balance = config.opening_balance;
        let mut min_balance = cumulative_balance;
        let mut max_balance = cumulative_balance;
        let mut total_inflows = 0.0;
        let mut total_outflows = 0.0;

        for day in 0..config.horizon_days {
            let day_date = start_date + (day as u64 * 86400);
            let day_flows = by_date.get(&day_date);

            let (inflows, outflows, uncertainty) = if let Some(flows) = day_flows {
                Self::aggregate_flows(flows, config)
            } else {
                (0.0, 0.0, 0.0)
            };

            let net = inflows - outflows;
            cumulative_balance += net;

            min_balance = min_balance.min(cumulative_balance);
            max_balance = max_balance.max(cumulative_balance);
            total_inflows += inflows;
            total_outflows += outflows;

            daily_forecasts.push(DailyForecast {
                date: day_date,
                inflows,
                outflows,
                net,
                cumulative_balance,
                uncertainty,
            });
        }

        CashFlowForecast {
            horizon_days: config.horizon_days,
            daily_forecasts,
            total_inflows,
            total_outflows,
            net_position: cumulative_balance,
            min_balance,
            max_balance,
        }
    }

    /// Aggregate flows for a single day with certainty weighting.
    fn aggregate_flows(flows: &[&CashFlow], config: &ForecastConfig) -> (f64, f64, f64) {
        let mut inflows = 0.0;
        let mut outflows = 0.0;
        let mut total_certainty = 0.0;
        let mut count = 0;

        for flow in flows {
            let weighted_amount = if config.use_certainty_weighting {
                flow.amount * flow.certainty
            } else {
                flow.amount
            };

            if weighted_amount > 0.0 {
                inflows += weighted_amount;
            } else {
                outflows += weighted_amount.abs();
            }

            total_certainty += flow.certainty;
            count += 1;
        }

        let avg_uncertainty = if count > 0 {
            1.0 - (total_certainty / count as f64)
        } else {
            0.0
        };

        (inflows, outflows, avg_uncertainty)
    }

    /// Forecast by category.
    pub fn forecast_by_category(
        cash_flows: &[CashFlow],
        config: &ForecastConfig,
    ) -> HashMap<CashFlowCategory, CashFlowForecast> {
        let mut by_category: HashMap<CashFlowCategory, Vec<CashFlow>> = HashMap::new();

        for cf in cash_flows {
            by_category.entry(cf.category).or_default().push(cf.clone());
        }

        by_category
            .into_iter()
            .map(|(category, flows)| {
                let forecast = Self::forecast(&flows, config);
                (category, forecast)
            })
            .collect()
    }

    /// Calculate stress scenario forecast.
    pub fn stress_forecast(
        cash_flows: &[CashFlow],
        config: &ForecastConfig,
        stress: &StressScenario,
    ) -> CashFlowForecast {
        // Apply stress factors to cash flows
        let stressed_flows: Vec<CashFlow> = cash_flows
            .iter()
            .map(|cf| {
                let mut stressed = cf.clone();

                // Apply category-specific stress
                let factor = stress
                    .category_factors
                    .get(&cf.category)
                    .copied()
                    .unwrap_or(1.0);

                if cf.amount > 0.0 {
                    // Reduce inflows
                    stressed.amount *= factor * stress.inflow_haircut;
                } else {
                    // Increase outflows
                    stressed.amount *= factor * stress.outflow_multiplier;
                }

                // Reduce certainty under stress
                stressed.certainty *= stress.certainty_reduction;

                stressed
            })
            .collect();

        Self::forecast(&stressed_flows, config)
    }

    /// Identify funding gaps.
    pub fn identify_gaps(
        forecast: &CashFlowForecast,
        min_balance_threshold: f64,
    ) -> Vec<FundingGap> {
        let mut gaps = Vec::new();
        let mut in_gap = false;
        let mut gap_start = 0u64;
        let mut gap_max_shortfall = 0.0;

        for daily in &forecast.daily_forecasts {
            if daily.cumulative_balance < min_balance_threshold {
                let shortfall = min_balance_threshold - daily.cumulative_balance;

                if !in_gap {
                    in_gap = true;
                    gap_start = daily.date;
                    gap_max_shortfall = shortfall;
                } else {
                    gap_max_shortfall = gap_max_shortfall.max(shortfall);
                }
            } else if in_gap {
                // Gap ended
                gaps.push(FundingGap {
                    start_date: gap_start,
                    end_date: daily.date,
                    max_shortfall: gap_max_shortfall,
                    duration_days: ((daily.date - gap_start) / 86400) as u32,
                });
                in_gap = false;
            }
        }

        // Handle gap extending to end of horizon
        if in_gap {
            if let Some(last) = forecast.daily_forecasts.last() {
                gaps.push(FundingGap {
                    start_date: gap_start,
                    end_date: last.date + 86400,
                    max_shortfall: gap_max_shortfall,
                    duration_days: ((last.date + 86400 - gap_start) / 86400) as u32,
                });
            }
        }

        gaps
    }
}

impl GpuKernel for CashFlowForecasting {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Forecast configuration.
#[derive(Debug, Clone)]
pub struct ForecastConfig {
    /// Start date (Unix timestamp).
    pub start_date: u64,
    /// Forecast horizon in days.
    pub horizon_days: u32,
    /// Opening balance.
    pub opening_balance: f64,
    /// Use certainty weighting.
    pub use_certainty_weighting: bool,
    /// Base currency.
    pub base_currency: String,
}

impl Default for ForecastConfig {
    fn default() -> Self {
        Self {
            start_date: 0,
            horizon_days: 30,
            opening_balance: 0.0,
            use_certainty_weighting: true,
            base_currency: "USD".to_string(),
        }
    }
}

/// Stress scenario for forecasting.
#[derive(Debug, Clone)]
pub struct StressScenario {
    /// Name of the scenario.
    pub name: String,
    /// Haircut on inflows (e.g., 0.8 = 20% reduction).
    pub inflow_haircut: f64,
    /// Multiplier on outflows (e.g., 1.2 = 20% increase).
    pub outflow_multiplier: f64,
    /// Reduction in certainty (e.g., 0.5 = halve certainty).
    pub certainty_reduction: f64,
    /// Category-specific stress factors.
    pub category_factors: HashMap<CashFlowCategory, f64>,
}

impl Default for StressScenario {
    fn default() -> Self {
        Self {
            name: "Base Stress".to_string(),
            inflow_haircut: 0.8,
            outflow_multiplier: 1.2,
            certainty_reduction: 0.8,
            category_factors: HashMap::new(),
        }
    }
}

/// Funding gap identified in forecast.
#[derive(Debug, Clone)]
pub struct FundingGap {
    /// Start date of gap.
    pub start_date: u64,
    /// End date of gap.
    pub end_date: u64,
    /// Maximum shortfall during gap.
    pub max_shortfall: f64,
    /// Duration in days.
    pub duration_days: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_flows() -> Vec<CashFlow> {
        vec![
            CashFlow {
                id: 1,
                date: 86400,     // Day 1
                amount: 10000.0, // Inflow
                currency: "USD".to_string(),
                category: CashFlowCategory::Operating,
                certainty: 1.0,
                description: "Sales".to_string(),
                attributes: HashMap::new(),
            },
            CashFlow {
                id: 2,
                date: 86400,     // Day 1
                amount: -5000.0, // Outflow
                currency: "USD".to_string(),
                category: CashFlowCategory::Operating,
                certainty: 1.0,
                description: "Expenses".to_string(),
                attributes: HashMap::new(),
            },
            CashFlow {
                id: 3,
                date: 172800,    // Day 2
                amount: -8000.0, // Outflow
                currency: "USD".to_string(),
                category: CashFlowCategory::DebtService,
                certainty: 0.9,
                description: "Loan payment".to_string(),
                attributes: HashMap::new(),
            },
        ]
    }

    #[test]
    fn test_cashflow_metadata() {
        let kernel = CashFlowForecasting::new();
        assert_eq!(kernel.metadata().id, "treasury/cashflow-forecast");
        assert_eq!(kernel.metadata().domain, Domain::TreasuryManagement);
    }

    #[test]
    fn test_basic_forecast() {
        let flows = create_test_flows();
        let config = ForecastConfig {
            start_date: 0,
            horizon_days: 5,
            opening_balance: 10000.0,
            use_certainty_weighting: false,
            ..Default::default()
        };

        let forecast = CashFlowForecasting::forecast(&flows, &config);

        assert_eq!(forecast.horizon_days, 5);
        assert_eq!(forecast.daily_forecasts.len(), 5);
        assert_eq!(forecast.total_inflows, 10000.0);
        assert_eq!(forecast.total_outflows, 13000.0);
    }

    #[test]
    fn test_certainty_weighting() {
        let flows = vec![CashFlow {
            id: 1,
            date: 86400,
            amount: 10000.0,
            currency: "USD".to_string(),
            category: CashFlowCategory::Operating,
            certainty: 0.5, // 50% certainty
            description: "Expected payment".to_string(),
            attributes: HashMap::new(),
        }];

        let config = ForecastConfig {
            start_date: 0,
            horizon_days: 3,
            opening_balance: 0.0,
            use_certainty_weighting: true,
            ..Default::default()
        };

        let forecast = CashFlowForecasting::forecast(&flows, &config);

        // Should be 5000 due to 50% certainty weighting
        assert!((forecast.total_inflows - 5000.0).abs() < 0.01);
    }

    #[test]
    fn test_min_max_balance() {
        let flows = create_test_flows();
        let config = ForecastConfig {
            start_date: 0,
            horizon_days: 5,
            opening_balance: 5000.0,
            use_certainty_weighting: false,
            ..Default::default()
        };

        let forecast = CashFlowForecasting::forecast(&flows, &config);

        // Day 0: 5000
        // Day 1: 5000 + 10000 - 5000 = 10000 (max)
        // Day 2: 10000 - 8000 = 2000 (min)
        assert_eq!(forecast.min_balance, 2000.0);
        assert_eq!(forecast.max_balance, 10000.0);
    }

    #[test]
    fn test_forecast_by_category() {
        let flows = create_test_flows();
        let config = ForecastConfig {
            start_date: 0,
            horizon_days: 5,
            opening_balance: 0.0,
            use_certainty_weighting: false,
            ..Default::default()
        };

        let by_cat = CashFlowForecasting::forecast_by_category(&flows, &config);

        assert!(by_cat.contains_key(&CashFlowCategory::Operating));
        assert!(by_cat.contains_key(&CashFlowCategory::DebtService));

        let operating = by_cat.get(&CashFlowCategory::Operating).unwrap();
        assert_eq!(operating.total_inflows, 10000.0);
        assert_eq!(operating.total_outflows, 5000.0);
    }

    #[test]
    fn test_stress_forecast() {
        let flows = create_test_flows();
        let config = ForecastConfig {
            start_date: 0,
            horizon_days: 5,
            opening_balance: 10000.0,
            use_certainty_weighting: false,
            ..Default::default()
        };

        let stress = StressScenario {
            name: "Severe".to_string(),
            inflow_haircut: 0.5,     // 50% reduction
            outflow_multiplier: 1.5, // 50% increase
            certainty_reduction: 0.5,
            category_factors: HashMap::new(),
        };

        let normal = CashFlowForecasting::forecast(&flows, &config);
        let stressed = CashFlowForecasting::stress_forecast(&flows, &config, &stress);

        // Stressed inflows should be lower
        assert!(stressed.total_inflows < normal.total_inflows);
        // Stressed outflows should be higher
        assert!(stressed.total_outflows > normal.total_outflows);
    }

    #[test]
    fn test_identify_gaps() {
        let flows = vec![
            CashFlow {
                id: 1,
                date: 86400,
                amount: -20000.0,
                currency: "USD".to_string(),
                category: CashFlowCategory::Operating,
                certainty: 1.0,
                description: "Large payment".to_string(),
                attributes: HashMap::new(),
            },
            CashFlow {
                id: 2,
                date: 259200, // Day 3
                amount: 25000.0,
                currency: "USD".to_string(),
                category: CashFlowCategory::Operating,
                certainty: 1.0,
                description: "Funding received".to_string(),
                attributes: HashMap::new(),
            },
        ];

        let config = ForecastConfig {
            start_date: 0,
            horizon_days: 5,
            opening_balance: 10000.0,
            use_certainty_weighting: false,
            ..Default::default()
        };

        let forecast = CashFlowForecasting::forecast(&flows, &config);
        let gaps = CashFlowForecasting::identify_gaps(&forecast, 5000.0);

        // Should have a gap from day 1-3
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].max_shortfall, 15000.0); // 5000 - (-10000) = 15000 shortfall
    }

    #[test]
    fn test_empty_flows() {
        let flows: Vec<CashFlow> = vec![];
        let config = ForecastConfig {
            start_date: 0,
            horizon_days: 5,
            opening_balance: 10000.0,
            use_certainty_weighting: false,
            ..Default::default()
        };

        let forecast = CashFlowForecasting::forecast(&flows, &config);

        assert_eq!(forecast.total_inflows, 0.0);
        assert_eq!(forecast.total_outflows, 0.0);
        assert_eq!(forecast.net_position, 10000.0);
    }
}
