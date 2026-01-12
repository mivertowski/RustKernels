//! FX hedging kernel.
//!
//! This module provides FX hedging optimization for treasury:
//! - Currency exposure calculation
//! - Hedge recommendation generation
//! - Cost-benefit analysis of hedging strategies

use crate::types::{CurrencyExposure, FXHedge, FXHedgingResult, FXRate, HedgeType};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// FX Hedging Kernel
// ============================================================================

/// FX hedging kernel.
///
/// Calculates currency exposures and recommends hedging strategies.
#[derive(Debug, Clone)]
pub struct FXHedging {
    metadata: KernelMetadata,
}

impl Default for FXHedging {
    fn default() -> Self {
        Self::new()
    }
}

impl FXHedging {
    /// Create a new FX hedging kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("treasury/fx-hedging", Domain::TreasuryManagement)
                .with_description("FX exposure and hedging optimization")
                .with_throughput(10_000)
                .with_latency_us(500.0),
        }
    }

    /// Calculate currency exposures from positions.
    pub fn calculate_exposures(
        positions: &[FXPosition],
        rates: &[FXRate],
        base_currency: &str,
    ) -> Vec<CurrencyExposure> {
        // Build rate lookup
        let rate_map: HashMap<(String, String), f64> = rates
            .iter()
            .map(|r| ((r.base.clone(), r.quote.clone()), r.rate))
            .collect();

        // Aggregate positions by currency
        let mut by_currency: HashMap<String, (f64, f64)> = HashMap::new();

        for pos in positions {
            if pos.currency == base_currency {
                continue; // Skip base currency
            }

            let entry = by_currency.entry(pos.currency.clone()).or_default();
            if pos.amount > 0.0 {
                entry.0 += pos.amount; // Long
            } else {
                entry.1 += pos.amount.abs(); // Short
            }
        }

        // Convert to exposures
        by_currency
            .into_iter()
            .map(|(currency, (long, short))| {
                let rate = rate_map
                    .get(&(currency.clone(), base_currency.to_string()))
                    .copied()
                    .or_else(|| {
                        rate_map
                            .get(&(base_currency.to_string(), currency.clone()))
                            .map(|r| 1.0 / r)
                    })
                    .unwrap_or(1.0);

                CurrencyExposure {
                    currency,
                    net_position: long - short,
                    long_positions: long,
                    short_positions: short,
                    base_equivalent: (long - short) * rate,
                }
            })
            .collect()
    }

    /// Generate hedging recommendations.
    pub fn recommend_hedges(
        exposures: &[CurrencyExposure],
        rates: &[FXRate],
        config: &HedgingConfig,
    ) -> FXHedgingResult {
        let mut hedges = Vec::new();
        let mut total_cost = 0.0;
        let mut residual_exposure = 0.0;
        let mut total_exposure = 0.0;

        // Build rate lookup
        let rate_map: HashMap<String, &FXRate> = rates
            .iter()
            .map(|r| (format!("{}{}", r.base, r.quote), r))
            .collect();

        for exposure in exposures {
            let abs_exposure = exposure.net_position.abs();
            total_exposure += abs_exposure;

            // Skip if below threshold
            if abs_exposure < config.min_hedge_amount {
                residual_exposure += abs_exposure;
                continue;
            }

            // Determine hedge amount based on target ratio
            let hedge_amount = abs_exposure * config.target_hedge_ratio;

            // Find applicable rate
            let pair = format!("{}{}", exposure.currency, config.base_currency);
            let rate = rate_map.get(&pair);

            let hedge = match config.preferred_instrument {
                PreferredInstrument::Forward => {
                    let cost = Self::calculate_forward_cost(hedge_amount, rate, config);
                    FXHedge {
                        id: hedges.len() as u64 + 1,
                        currency_pair: pair,
                        notional: hedge_amount,
                        hedge_type: HedgeType::Forward,
                        strike: rate.map(|r| r.rate),
                        expiry: config.hedge_horizon_days as u64 * 86400,
                        cost,
                    }
                }
                PreferredInstrument::Option => {
                    let (cost, strike) = Self::calculate_option_cost(
                        hedge_amount,
                        rate,
                        exposure.net_position < 0.0,
                        config,
                    );
                    FXHedge {
                        id: hedges.len() as u64 + 1,
                        currency_pair: pair,
                        notional: hedge_amount,
                        hedge_type: if exposure.net_position < 0.0 {
                            HedgeType::Call
                        } else {
                            HedgeType::Put
                        },
                        strike: Some(strike),
                        expiry: config.hedge_horizon_days as u64 * 86400,
                        cost,
                    }
                }
                PreferredInstrument::Collar => {
                    let (cost, strike) = Self::calculate_collar_cost(hedge_amount, rate, config);
                    FXHedge {
                        id: hedges.len() as u64 + 1,
                        currency_pair: pair,
                        notional: hedge_amount,
                        hedge_type: HedgeType::Collar,
                        strike: Some(strike),
                        expiry: config.hedge_horizon_days as u64 * 86400,
                        cost,
                    }
                }
            };

            total_cost += hedge.cost;
            residual_exposure += abs_exposure - hedge_amount;
            hedges.push(hedge);
        }

        let hedge_ratio = if total_exposure > 0.0 {
            1.0 - (residual_exposure / total_exposure)
        } else {
            0.0
        };

        // Estimate VaR reduction (simplified model)
        let var_reduction = Self::estimate_var_reduction(&hedges, exposures, config);

        FXHedgingResult {
            hedges,
            residual_exposure,
            hedge_ratio,
            total_cost,
            var_reduction,
        }
    }

    /// Calculate forward contract cost.
    fn calculate_forward_cost(
        notional: f64,
        rate: Option<&&FXRate>,
        config: &HedgingConfig,
    ) -> f64 {
        // Forward cost = notional * (bid-ask spread + interest rate differential)
        let spread = rate.map(|r| r.ask - r.bid).unwrap_or(0.01);
        let ir_diff = config.interest_rate_differential;
        let days = config.hedge_horizon_days as f64;

        notional * (spread + ir_diff * days / 365.0)
    }

    /// Calculate option cost (premium).
    fn calculate_option_cost(
        notional: f64,
        rate: Option<&&FXRate>,
        is_call: bool,
        config: &HedgingConfig,
    ) -> (f64, f64) {
        let spot = rate.map(|r| r.rate).unwrap_or(1.0);
        let volatility = config.implied_volatility;
        let days = config.hedge_horizon_days as f64;
        let time = days / 365.0;

        // OTM strike based on configuration
        let strike = if is_call {
            spot * (1.0 + config.option_otm_offset)
        } else {
            spot * (1.0 - config.option_otm_offset)
        };

        // Full Black-Scholes formula
        // Assuming risk-free rate of 2% for FX options
        let r = 0.02;
        let premium =
            Self::black_scholes(spot, strike, time, r, volatility, is_call) * notional / spot;

        (premium, strike)
    }

    /// Black-Scholes option pricing formula.
    ///
    /// # Arguments
    /// * `spot` - Current spot price
    /// * `strike` - Strike price
    /// * `time` - Time to expiration in years
    /// * `rate` - Risk-free interest rate
    /// * `vol` - Implied volatility
    /// * `is_call` - True for call, false for put
    fn black_scholes(spot: f64, strike: f64, time: f64, rate: f64, vol: f64, is_call: bool) -> f64 {
        if time <= 0.0 || vol <= 0.0 {
            // Expired or invalid - return intrinsic value
            return if is_call {
                (spot - strike).max(0.0)
            } else {
                (strike - spot).max(0.0)
            };
        }

        let sqrt_t = time.sqrt();
        let d1 = ((spot / strike).ln() + (rate + vol * vol / 2.0) * time) / (vol * sqrt_t);
        let d2 = d1 - vol * sqrt_t;

        let discount = (-rate * time).exp();

        if is_call {
            spot * Self::norm_cdf(d1) - strike * discount * Self::norm_cdf(d2)
        } else {
            strike * discount * Self::norm_cdf(-d2) - spot * Self::norm_cdf(-d1)
        }
    }

    /// Standard normal CDF approximation (Abramowitz and Stegun).
    fn norm_cdf(x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x_abs = x.abs();

        let t = 1.0 / (1.0 + p * x_abs);
        let y = 1.0
            - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x_abs * x_abs / 2.0).exp();

        0.5 * (1.0 + sign * y)
    }

    /// Calculate collar cost.
    fn calculate_collar_cost(
        notional: f64,
        rate: Option<&&FXRate>,
        config: &HedgingConfig,
    ) -> (f64, f64) {
        let _spot = rate.map(|r| r.rate).unwrap_or(1.0);

        // Zero-cost collar approximation
        // Buy put, sell call at symmetric strikes
        let (put_cost, put_strike) = Self::calculate_option_cost(notional, rate, false, config);
        let (call_premium, _) = Self::calculate_option_cost(notional, rate, true, config);

        // Net cost (usually close to zero for symmetric collar)
        let net_cost = (put_cost - call_premium * 0.9).max(0.0);

        (net_cost, put_strike)
    }

    /// Estimate VaR reduction from hedging using delta-gamma VaR model.
    ///
    /// This calculates VaR reduction considering:
    /// - Linear hedge delta for forwards
    /// - Option delta for option hedges
    /// - Gamma effects for large moves
    fn estimate_var_reduction(
        hedges: &[FXHedge],
        exposures: &[CurrencyExposure],
        config: &HedgingConfig,
    ) -> f64 {
        if exposures.is_empty() {
            return 0.0;
        }

        let confidence_factor = 1.645; // 95% confidence (z-score)
        let days = config.hedge_horizon_days as f64;
        let time = days / 365.0;
        let volatility = config.implied_volatility;
        let sqrt_time = (days / 252.0).sqrt();

        // Calculate unhedged VaR
        let unhedged_var: f64 = exposures
            .iter()
            .map(|e| e.base_equivalent.abs() * volatility * sqrt_time * confidence_factor)
            .sum();

        // Calculate delta-equivalent hedge coverage
        let mut total_hedge_delta = 0.0;

        for hedge in hedges {
            use crate::types::HedgeType;
            let delta = match hedge.hedge_type {
                HedgeType::Forward | HedgeType::Swap => 1.0, // Forward/swap has delta = 1
                HedgeType::Call => {
                    // Call option delta using N(d1)
                    let d1 =
                        (0.02 + volatility * volatility / 2.0) * time / (volatility * time.sqrt());
                    Self::norm_cdf(d1)
                }
                HedgeType::Put => {
                    // Put option delta = N(d1) - 1
                    let d1 =
                        (0.02 + volatility * volatility / 2.0) * time / (volatility * time.sqrt());
                    Self::norm_cdf(d1) - 1.0
                }
                HedgeType::Collar => {
                    // Collar has partial delta (long put + short call effectively)
                    // Net delta depends on strikes but typically 0.5-0.8
                    0.7
                }
            };
            total_hedge_delta += hedge.notional * delta.abs();
        }

        let total_exposure: f64 = exposures.iter().map(|e| e.net_position.abs()).sum();

        // Calculate hedge effectiveness using delta
        let hedge_effectiveness = if total_exposure > 0.0 {
            (total_hedge_delta / total_exposure).min(1.0)
        } else {
            0.0
        };

        // Include gamma adjustment for options (reduces VaR benefit for large moves)
        let has_options = hedges.iter().any(|h| {
            use crate::types::HedgeType;
            matches!(
                h.hedge_type,
                HedgeType::Put | HedgeType::Call | HedgeType::Collar
            )
        });
        let gamma_adjustment = if has_options {
            // Gamma reduces hedge effectiveness for larger moves
            // At 95% confidence, expected move is ~1.645 std devs
            // Gamma effect reduces delta linearly with move size
            let expected_move = confidence_factor * volatility * sqrt_time;
            1.0 - expected_move * 0.3 // Approximate gamma decay
        } else {
            1.0
        };

        // Basis risk adjustment (hedges may not perfectly track exposure)
        let basis_risk_factor = 0.95;

        // Final VaR reduction
        unhedged_var * hedge_effectiveness * gamma_adjustment * basis_risk_factor
    }

    /// Calculate net exposure after hedges.
    pub fn net_exposure_after_hedges(
        exposures: &[CurrencyExposure],
        hedges: &[FXHedge],
    ) -> HashMap<String, f64> {
        let mut net: HashMap<String, f64> = HashMap::new();

        // Add exposures
        for exp in exposures {
            *net.entry(exp.currency.clone()).or_default() += exp.net_position;
        }

        // Subtract hedges
        for hedge in hedges {
            // Extract currency from pair (first 3 chars typically)
            let currency = if hedge.currency_pair.len() >= 3 {
                &hedge.currency_pair[0..3]
            } else {
                &hedge.currency_pair
            };

            *net.entry(currency.to_string()).or_default() -= hedge.notional;
        }

        net
    }
}

impl GpuKernel for FXHedging {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// FX position.
#[derive(Debug, Clone)]
pub struct FXPosition {
    /// Position ID.
    pub id: String,
    /// Currency.
    pub currency: String,
    /// Amount (positive = long, negative = short).
    pub amount: f64,
    /// Maturity date.
    pub maturity: Option<u64>,
    /// Source (e.g., trade, receivable, payable).
    pub source: String,
}

/// Hedging configuration.
#[derive(Debug, Clone)]
pub struct HedgingConfig {
    /// Base currency.
    pub base_currency: String,
    /// Target hedge ratio (0-1).
    pub target_hedge_ratio: f64,
    /// Minimum hedge amount.
    pub min_hedge_amount: f64,
    /// Hedge horizon in days.
    pub hedge_horizon_days: u32,
    /// Preferred hedging instrument.
    pub preferred_instrument: PreferredInstrument,
    /// Interest rate differential.
    pub interest_rate_differential: f64,
    /// Implied volatility.
    pub implied_volatility: f64,
    /// OTM offset for options.
    pub option_otm_offset: f64,
}

impl Default for HedgingConfig {
    fn default() -> Self {
        Self {
            base_currency: "USD".to_string(),
            target_hedge_ratio: 0.8,
            min_hedge_amount: 10_000.0,
            hedge_horizon_days: 90,
            preferred_instrument: PreferredInstrument::Forward,
            interest_rate_differential: 0.02,
            implied_volatility: 0.10,
            option_otm_offset: 0.05,
        }
    }
}

/// Preferred hedging instrument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreferredInstrument {
    /// Forward contract.
    Forward,
    /// Option (put or call).
    Option,
    /// Collar (put + call).
    Collar,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_positions() -> Vec<FXPosition> {
        vec![
            FXPosition {
                id: "P1".to_string(),
                currency: "EUR".to_string(),
                amount: 1_000_000.0,
                maturity: Some(7776000), // 90 days
                source: "Receivable".to_string(),
            },
            FXPosition {
                id: "P2".to_string(),
                currency: "EUR".to_string(),
                amount: -500_000.0,
                maturity: Some(7776000),
                source: "Payable".to_string(),
            },
            FXPosition {
                id: "P3".to_string(),
                currency: "GBP".to_string(),
                amount: 300_000.0,
                maturity: Some(7776000),
                source: "Receivable".to_string(),
            },
        ]
    }

    fn create_test_rates() -> Vec<FXRate> {
        vec![
            FXRate {
                base: "EUR".to_string(),
                quote: "USD".to_string(),
                rate: 1.10,
                bid: 1.0995,
                ask: 1.1005,
                timestamp: 1700000000,
            },
            FXRate {
                base: "GBP".to_string(),
                quote: "USD".to_string(),
                rate: 1.25,
                bid: 1.2490,
                ask: 1.2510,
                timestamp: 1700000000,
            },
        ]
    }

    #[test]
    fn test_fx_metadata() {
        let kernel = FXHedging::new();
        assert_eq!(kernel.metadata().id, "treasury/fx-hedging");
        assert_eq!(kernel.metadata().domain, Domain::TreasuryManagement);
    }

    #[test]
    fn test_calculate_exposures() {
        let positions = create_test_positions();
        let rates = create_test_rates();

        let exposures = FXHedging::calculate_exposures(&positions, &rates, "USD");

        assert_eq!(exposures.len(), 2); // EUR and GBP

        let eur_exp = exposures.iter().find(|e| e.currency == "EUR").unwrap();
        assert_eq!(eur_exp.long_positions, 1_000_000.0);
        assert_eq!(eur_exp.short_positions, 500_000.0);
        assert_eq!(eur_exp.net_position, 500_000.0);

        let gbp_exp = exposures.iter().find(|e| e.currency == "GBP").unwrap();
        assert_eq!(gbp_exp.net_position, 300_000.0);
    }

    #[test]
    fn test_recommend_hedges_forward() {
        let positions = create_test_positions();
        let rates = create_test_rates();
        let exposures = FXHedging::calculate_exposures(&positions, &rates, "USD");

        let config = HedgingConfig {
            preferred_instrument: PreferredInstrument::Forward,
            target_hedge_ratio: 0.8,
            ..Default::default()
        };

        let result = FXHedging::recommend_hedges(&exposures, &rates, &config);

        assert!(!result.hedges.is_empty());
        assert!(result.hedge_ratio > 0.0);
        assert!(result.total_cost > 0.0);

        // All hedges should be forwards
        assert!(
            result
                .hedges
                .iter()
                .all(|h| h.hedge_type == HedgeType::Forward)
        );
    }

    #[test]
    fn test_recommend_hedges_option() {
        let positions = create_test_positions();
        let rates = create_test_rates();
        let exposures = FXHedging::calculate_exposures(&positions, &rates, "USD");

        let config = HedgingConfig {
            preferred_instrument: PreferredInstrument::Option,
            target_hedge_ratio: 0.8,
            ..Default::default()
        };

        let result = FXHedging::recommend_hedges(&exposures, &rates, &config);

        assert!(!result.hedges.is_empty());

        // Should have put options for long exposures
        let eur_hedge = result
            .hedges
            .iter()
            .find(|h| h.currency_pair.starts_with("EUR"));
        assert!(eur_hedge.is_some());
        assert_eq!(eur_hedge.unwrap().hedge_type, HedgeType::Put);
    }

    #[test]
    fn test_min_hedge_threshold() {
        let positions = vec![FXPosition {
            id: "P1".to_string(),
            currency: "EUR".to_string(),
            amount: 5_000.0, // Below threshold
            maturity: None,
            source: "Receivable".to_string(),
        }];
        let rates = create_test_rates();
        let exposures = FXHedging::calculate_exposures(&positions, &rates, "USD");

        let config = HedgingConfig {
            min_hedge_amount: 10_000.0,
            ..Default::default()
        };

        let result = FXHedging::recommend_hedges(&exposures, &rates, &config);

        // Should not recommend hedge for small exposure
        assert!(result.hedges.is_empty());
        assert!(result.residual_exposure > 0.0);
    }

    #[test]
    fn test_hedge_ratio_calculation() {
        let positions = create_test_positions();
        let rates = create_test_rates();
        let exposures = FXHedging::calculate_exposures(&positions, &rates, "USD");

        let config = HedgingConfig {
            target_hedge_ratio: 1.0, // Full hedge
            min_hedge_amount: 0.0,
            ..Default::default()
        };

        let result = FXHedging::recommend_hedges(&exposures, &rates, &config);

        // Should be close to 100% hedged
        assert!(result.hedge_ratio > 0.9);
    }

    #[test]
    fn test_var_reduction() {
        let positions = create_test_positions();
        let rates = create_test_rates();
        let exposures = FXHedging::calculate_exposures(&positions, &rates, "USD");

        let config = HedgingConfig::default();
        let result = FXHedging::recommend_hedges(&exposures, &rates, &config);

        // VaR reduction should be positive if hedges exist
        if !result.hedges.is_empty() {
            assert!(result.var_reduction > 0.0);
        }
    }

    #[test]
    fn test_net_exposure_after_hedges() {
        let exposures = vec![CurrencyExposure {
            currency: "EUR".to_string(),
            net_position: 1_000_000.0,
            long_positions: 1_000_000.0,
            short_positions: 0.0,
            base_equivalent: 1_100_000.0,
        }];

        let hedges = vec![FXHedge {
            id: 1,
            currency_pair: "EURUSD".to_string(),
            notional: 800_000.0,
            hedge_type: HedgeType::Forward,
            strike: Some(1.10),
            expiry: 7776000,
            cost: 1000.0,
        }];

        let net = FXHedging::net_exposure_after_hedges(&exposures, &hedges);

        assert_eq!(net.get("EUR"), Some(&200_000.0));
    }

    #[test]
    fn test_collar_hedging() {
        let positions = create_test_positions();
        let rates = create_test_rates();
        let exposures = FXHedging::calculate_exposures(&positions, &rates, "USD");

        let config = HedgingConfig {
            preferred_instrument: PreferredInstrument::Collar,
            ..Default::default()
        };

        let result = FXHedging::recommend_hedges(&exposures, &rates, &config);

        // Should have collar hedges
        assert!(
            result
                .hedges
                .iter()
                .all(|h| h.hedge_type == HedgeType::Collar)
        );

        // Collar cost should typically be low (zero-cost structure)
        assert!(result.total_cost < result.hedges.iter().map(|h| h.notional * 0.05).sum::<f64>());
    }

    #[test]
    fn test_empty_positions() {
        let positions: Vec<FXPosition> = vec![];
        let rates = create_test_rates();

        let exposures = FXHedging::calculate_exposures(&positions, &rates, "USD");
        assert!(exposures.is_empty());

        let result = FXHedging::recommend_hedges(&exposures, &rates, &HedgingConfig::default());
        assert!(result.hedges.is_empty());
        assert_eq!(result.hedge_ratio, 0.0);
    }
}
