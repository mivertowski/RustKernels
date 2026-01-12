//! Interest rate risk kernel.
//!
//! This module provides interest rate risk analysis for treasury:
//! - Duration and convexity calculation
//! - DV01/PV01 sensitivity measures
//! - Gap analysis by time buckets

use crate::types::{GapBucket, IRInstrumentType, IRPosition, IRRiskMetrics};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// Interest Rate Risk Kernel
// ============================================================================

/// Interest rate risk kernel.
///
/// Calculates duration, convexity, and gap analysis for IR-sensitive positions.
#[derive(Debug, Clone)]
pub struct InterestRateRisk {
    metadata: KernelMetadata,
}

impl Default for InterestRateRisk {
    fn default() -> Self {
        Self::new()
    }
}

impl InterestRateRisk {
    /// Create a new interest rate risk kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("treasury/ir-risk", Domain::TreasuryManagement)
                .with_description("Interest rate risk analysis")
                .with_throughput(10_000)
                .with_latency_us(500.0),
        }
    }

    /// Calculate interest rate risk metrics.
    pub fn calculate_metrics(positions: &[IRPosition], config: &IRRiskConfig) -> IRRiskMetrics {
        let current_date = config.valuation_date;

        let mut total_pv = 0.0;
        let mut weighted_duration = 0.0;
        let mut weighted_convexity = 0.0;
        let mut pv01_by_currency: HashMap<String, f64> = HashMap::new();

        for pos in positions {
            let time_to_maturity = (pos.maturity - current_date) as f64 / (365.0 * 86400.0);
            if time_to_maturity <= 0.0 {
                continue;
            }

            let pv = Self::calculate_present_value(pos, config);
            let duration = Self::calculate_duration(pos, time_to_maturity, config);
            let convexity = Self::calculate_convexity(pos, time_to_maturity, config);

            total_pv += pv;
            weighted_duration += duration * pv;
            weighted_convexity += convexity * pv;

            // PV01 calculation: pv * modified_duration * 0.0001
            let mod_dur = duration / (1.0 + pos.rate / config.compounding_frequency as f64);
            let pv01 = pv * mod_dur * 0.0001;
            *pv01_by_currency.entry(pos.currency.clone()).or_default() += pv01;
        }

        // Normalize weighted metrics
        let duration = if total_pv != 0.0 {
            weighted_duration / total_pv
        } else {
            0.0
        };

        let convexity = if total_pv != 0.0 {
            weighted_convexity / total_pv
        } else {
            0.0
        };

        let modified_duration =
            duration / (1.0 + config.market_rate / config.compounding_frequency as f64);
        let dv01 = total_pv * modified_duration * 0.0001;

        // Gap analysis
        let gap_by_bucket = Self::gap_analysis(positions, config);

        IRRiskMetrics {
            duration,
            modified_duration,
            convexity,
            dv01,
            pv01_by_currency,
            gap_by_bucket,
        }
    }

    /// Calculate present value of a position.
    fn calculate_present_value(pos: &IRPosition, config: &IRRiskConfig) -> f64 {
        // Simplified PV calculation
        let current_date = config.valuation_date;
        let time_to_maturity = (pos.maturity - current_date) as f64 / (365.0 * 86400.0);

        if time_to_maturity <= 0.0 {
            return pos.notional;
        }

        // Discount factor
        let discount = (1.0 + config.market_rate / config.compounding_frequency as f64)
            .powf(-time_to_maturity * config.compounding_frequency as f64);

        pos.notional * discount
    }

    /// Calculate Macaulay duration.
    fn calculate_duration(pos: &IRPosition, ttm: f64, config: &IRRiskConfig) -> f64 {
        match pos.instrument_type {
            IRInstrumentType::FixedBond | IRInstrumentType::FixedLoan => {
                // Approximate bond duration
                let coupon_rate = pos.rate;
                let yield_rate = config.market_rate;
                let n = (ttm * config.compounding_frequency as f64).ceil() as i32;

                if n <= 0 || yield_rate <= 0.0 {
                    return ttm;
                }

                let y = yield_rate / config.compounding_frequency as f64;
                let c = coupon_rate / config.compounding_frequency as f64;

                // Duration formula for bond
                let mut numerator = 0.0;
                let mut denominator = 0.0;

                for t in 1..=n {
                    let df = 1.0 / (1.0 + y).powi(t);
                    let cf = if t == n { c + 1.0 } else { c };
                    numerator += (t as f64) * cf * df;
                    denominator += cf * df;
                }

                if denominator > 0.0 {
                    numerator / denominator / config.compounding_frequency as f64
                } else {
                    ttm
                }
            }
            IRInstrumentType::FloatingNote | IRInstrumentType::FloatingLoan => {
                // Floating rate: duration to next reset
                if let Some(next_reset) = pos.next_reset {
                    let reset_ttm = (next_reset - config.valuation_date) as f64 / (365.0 * 86400.0);
                    reset_ttm.max(0.0)
                } else {
                    0.25 // Default quarterly reset assumption
                }
            }
            IRInstrumentType::Swap => {
                // Interest rate swap duration
                // Pay fixed: Duration = -Duration_fixed + Duration_float ≈ -Duration_fixed
                // Receive fixed: Duration = Duration_fixed - Duration_float ≈ Duration_fixed
                // For receive-fixed (typical): fixed leg duration - floating leg duration
                // Floating leg duration ≈ time to next reset (typically ~0.25 for quarterly)
                let float_duration = 0.25; // Quarterly reset assumed

                // Fixed leg duration approximation (zero-coupon equivalent)
                // For an at-par swap, fixed leg duration ≈ (1 - e^(-y*T)) / y where y is swap rate
                let swap_rate = pos.rate.max(0.01);
                let fixed_duration = if swap_rate > 0.0001 {
                    (1.0 - (-swap_rate * ttm).exp()) / swap_rate
                } else {
                    ttm
                };

                // Net duration (receive fixed positive, pay fixed negative encoded in notional sign)
                (fixed_duration - float_duration).abs()
            }
            IRInstrumentType::Deposit => {
                // Deposit duration = time to maturity
                ttm
            }
        }
    }

    /// Calculate convexity.
    fn calculate_convexity(pos: &IRPosition, ttm: f64, config: &IRRiskConfig) -> f64 {
        match pos.instrument_type {
            IRInstrumentType::FixedBond | IRInstrumentType::FixedLoan => {
                // Approximate convexity
                let coupon_rate = pos.rate;
                let yield_rate = config.market_rate;
                let n = (ttm * config.compounding_frequency as f64).ceil() as i32;

                if n <= 0 || yield_rate <= 0.0 {
                    return ttm * ttm;
                }

                let y = yield_rate / config.compounding_frequency as f64;
                let c = coupon_rate / config.compounding_frequency as f64;

                let mut numerator = 0.0;
                let mut denominator = 0.0;

                for t in 1..=n {
                    let df = 1.0 / (1.0 + y).powi(t);
                    let cf = if t == n { c + 1.0 } else { c };
                    numerator += (t as f64) * ((t + 1) as f64) * cf * df;
                    denominator += cf * df;
                }

                if denominator > 0.0 {
                    numerator
                        / denominator
                        / (1.0 + y).powi(2)
                        / (config.compounding_frequency as f64).powi(2)
                } else {
                    ttm * ttm
                }
            }
            IRInstrumentType::FloatingNote | IRInstrumentType::FloatingLoan => {
                // Very low convexity for floaters
                0.01
            }
            IRInstrumentType::Swap | IRInstrumentType::Deposit => {
                // Simplified convexity
                ttm * ttm * 0.5
            }
        }
    }

    /// Perform gap analysis.
    pub fn gap_analysis(positions: &[IRPosition], config: &IRRiskConfig) -> Vec<GapBucket> {
        let buckets = &config.gap_buckets;
        let current_date = config.valuation_date;

        let mut results: Vec<GapBucket> = buckets
            .iter()
            .map(|b| GapBucket {
                bucket: b.name.clone(),
                start_days: b.start_days,
                end_days: b.end_days,
                assets: 0.0,
                liabilities: 0.0,
                gap: 0.0,
                cumulative_gap: 0.0,
            })
            .collect();

        // Classify positions into buckets
        for pos in positions {
            let maturity_date = match pos.instrument_type {
                IRInstrumentType::FloatingNote | IRInstrumentType::FloatingLoan => {
                    // Use next reset date for floating
                    pos.next_reset.unwrap_or(pos.maturity)
                }
                _ => pos.maturity,
            };

            let days_to_maturity = if maturity_date > current_date {
                ((maturity_date - current_date) / 86400) as u32
            } else {
                0
            };

            // Find appropriate bucket
            for bucket in &mut results {
                if days_to_maturity >= bucket.start_days && days_to_maturity < bucket.end_days {
                    if pos.notional > 0.0 {
                        bucket.assets += pos.notional;
                    } else {
                        bucket.liabilities += pos.notional.abs();
                    }
                    break;
                }
            }
        }

        // Calculate gaps and cumulative gaps
        let mut cumulative = 0.0;
        for bucket in &mut results {
            bucket.gap = bucket.assets - bucket.liabilities;
            cumulative += bucket.gap;
            bucket.cumulative_gap = cumulative;
        }

        results
    }

    /// Calculate sensitivity to parallel shift.
    pub fn parallel_shift_sensitivity(
        positions: &[IRPosition],
        config: &IRRiskConfig,
        shift_bps: f64,
    ) -> ShiftSensitivity {
        let shift = shift_bps / 10000.0;

        // Calculate PV before shift
        let pv_before: f64 = positions
            .iter()
            .map(|p| Self::calculate_present_value(p, config))
            .sum();

        // Calculate PV after shift
        let mut shifted_config = config.clone();
        shifted_config.market_rate += shift;
        let pv_after: f64 = positions
            .iter()
            .map(|p| Self::calculate_present_value(p, &shifted_config))
            .sum();

        let pv_change = pv_after - pv_before;
        let pct_change = if pv_before != 0.0 {
            pv_change / pv_before
        } else {
            0.0
        };

        ShiftSensitivity {
            shift_bps,
            pv_before,
            pv_after,
            pv_change,
            pct_change,
        }
    }

    /// Calculate key rate duration (sensitivity to specific tenor).
    pub fn key_rate_durations(
        positions: &[IRPosition],
        config: &IRRiskConfig,
        tenors: &[u32],
    ) -> HashMap<u32, f64> {
        let mut krd: HashMap<u32, f64> = HashMap::new();
        let shift_bps = 1.0;

        for &tenor in tenors {
            // Filter positions maturing around this tenor
            let tenor_positions: Vec<_> = positions
                .iter()
                .filter(|p| {
                    let days = ((p.maturity - config.valuation_date) / 86400) as u32;
                    let tenor_days = tenor * 365;
                    days >= tenor_days.saturating_sub(180) && days <= tenor_days + 180
                })
                .cloned()
                .collect();

            if tenor_positions.is_empty() {
                krd.insert(tenor, 0.0);
                continue;
            }

            let sensitivity = Self::parallel_shift_sensitivity(&tenor_positions, config, shift_bps);
            let krd_value = -sensitivity.pct_change * 10000.0; // Duration per 100bp
            krd.insert(tenor, krd_value);
        }

        krd
    }
}

impl GpuKernel for InterestRateRisk {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Interest rate risk configuration.
#[derive(Debug, Clone)]
pub struct IRRiskConfig {
    /// Valuation date (Unix timestamp).
    pub valuation_date: u64,
    /// Current market rate.
    pub market_rate: f64,
    /// Compounding frequency per year.
    pub compounding_frequency: u32,
    /// Gap analysis buckets.
    pub gap_buckets: Vec<GapBucketDef>,
}

impl Default for IRRiskConfig {
    fn default() -> Self {
        Self {
            valuation_date: 0,
            market_rate: 0.05,
            compounding_frequency: 2,
            gap_buckets: vec![
                GapBucketDef {
                    name: "0-30d".to_string(),
                    start_days: 0,
                    end_days: 30,
                },
                GapBucketDef {
                    name: "30-90d".to_string(),
                    start_days: 30,
                    end_days: 90,
                },
                GapBucketDef {
                    name: "90-180d".to_string(),
                    start_days: 90,
                    end_days: 180,
                },
                GapBucketDef {
                    name: "180d-1y".to_string(),
                    start_days: 180,
                    end_days: 365,
                },
                GapBucketDef {
                    name: "1-2y".to_string(),
                    start_days: 365,
                    end_days: 730,
                },
                GapBucketDef {
                    name: "2-5y".to_string(),
                    start_days: 730,
                    end_days: 1825,
                },
                GapBucketDef {
                    name: ">5y".to_string(),
                    start_days: 1825,
                    end_days: u32::MAX,
                },
            ],
        }
    }
}

/// Gap bucket definition.
#[derive(Debug, Clone)]
pub struct GapBucketDef {
    /// Bucket name.
    pub name: String,
    /// Start days.
    pub start_days: u32,
    /// End days.
    pub end_days: u32,
}

/// Parallel shift sensitivity result.
#[derive(Debug, Clone)]
pub struct ShiftSensitivity {
    /// Shift amount in basis points.
    pub shift_bps: f64,
    /// PV before shift.
    pub pv_before: f64,
    /// PV after shift.
    pub pv_after: f64,
    /// PV change.
    pub pv_change: f64,
    /// Percentage change.
    pub pct_change: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_positions() -> Vec<IRPosition> {
        let base_date: u64 = 1700000000;
        vec![
            IRPosition {
                id: "BOND_001".to_string(),
                instrument_type: IRInstrumentType::FixedBond,
                notional: 1_000_000.0,
                rate: 0.05,
                maturity: base_date + 2 * 365 * 86400, // 2 years
                next_reset: None,
                currency: "USD".to_string(),
            },
            IRPosition {
                id: "FRN_001".to_string(),
                instrument_type: IRInstrumentType::FloatingNote,
                notional: 500_000.0,
                rate: 0.04,
                maturity: base_date + 3 * 365 * 86400, // 3 years
                next_reset: Some(base_date + 90 * 86400), // 90 days
                currency: "USD".to_string(),
            },
            IRPosition {
                id: "LOAN_001".to_string(),
                instrument_type: IRInstrumentType::FixedLoan,
                notional: -750_000.0, // Liability
                rate: 0.06,
                maturity: base_date + 5 * 365 * 86400, // 5 years
                next_reset: None,
                currency: "USD".to_string(),
            },
        ]
    }

    #[test]
    fn test_ir_metadata() {
        let kernel = InterestRateRisk::new();
        assert_eq!(kernel.metadata().id, "treasury/ir-risk");
        assert_eq!(kernel.metadata().domain, Domain::TreasuryManagement);
    }

    #[test]
    fn test_calculate_metrics() {
        // Use only asset positions for this test
        let base_date: u64 = 1700000000;
        let positions = vec![
            IRPosition {
                id: "BOND_001".to_string(),
                instrument_type: IRInstrumentType::FixedBond,
                notional: 1_000_000.0,
                rate: 0.05,
                maturity: base_date + 2 * 365 * 86400,
                next_reset: None,
                currency: "USD".to_string(),
            },
            IRPosition {
                id: "FRN_001".to_string(),
                instrument_type: IRInstrumentType::FloatingNote,
                notional: 500_000.0,
                rate: 0.04,
                maturity: base_date + 3 * 365 * 86400,
                next_reset: Some(base_date + 90 * 86400),
                currency: "USD".to_string(),
            },
        ];

        let config = IRRiskConfig {
            valuation_date: base_date,
            market_rate: 0.05,
            ..Default::default()
        };

        let metrics = InterestRateRisk::calculate_metrics(&positions, &config);

        // Duration should be positive and reasonable for asset-only portfolio
        assert!(metrics.duration > 0.0);
        assert!(metrics.duration < 10.0);

        // Modified duration should be less than Macaulay duration
        assert!(metrics.modified_duration < metrics.duration);

        // DV01 should be positive for asset portfolio
        assert!(metrics.dv01 > 0.0);

        // Should have PV01 for USD
        assert!(metrics.pv01_by_currency.contains_key("USD"));
    }

    #[test]
    fn test_duration_fixed_vs_floating() {
        let base_date: u64 = 1700000000;
        let fixed_bond = IRPosition {
            id: "FIXED".to_string(),
            instrument_type: IRInstrumentType::FixedBond,
            notional: 1_000_000.0,
            rate: 0.05,
            maturity: base_date + 5 * 365 * 86400,
            next_reset: None,
            currency: "USD".to_string(),
        };

        let floating = IRPosition {
            id: "FLOAT".to_string(),
            instrument_type: IRInstrumentType::FloatingNote,
            notional: 1_000_000.0,
            rate: 0.05,
            maturity: base_date + 5 * 365 * 86400,
            next_reset: Some(base_date + 90 * 86400),
            currency: "USD".to_string(),
        };

        let config = IRRiskConfig {
            valuation_date: base_date,
            ..Default::default()
        };

        let fixed_metrics = InterestRateRisk::calculate_metrics(&[fixed_bond], &config);
        let float_metrics = InterestRateRisk::calculate_metrics(&[floating], &config);

        // Fixed bond should have higher duration than floater
        assert!(fixed_metrics.duration > float_metrics.duration);
    }

    #[test]
    fn test_gap_analysis() {
        let positions = create_test_positions();
        let config = IRRiskConfig {
            valuation_date: 1700000000,
            ..Default::default()
        };

        let gaps = InterestRateRisk::gap_analysis(&positions, &config);

        assert!(!gaps.is_empty());

        // All buckets should be present
        assert_eq!(gaps.len(), config.gap_buckets.len());

        // Cumulative gap should be sum of individual gaps
        let total_gap: f64 = gaps.iter().map(|g| g.gap).sum();
        let final_cumulative = gaps.last().unwrap().cumulative_gap;
        assert!((total_gap - final_cumulative).abs() < 0.01);
    }

    #[test]
    fn test_parallel_shift_sensitivity() {
        let positions = create_test_positions();
        let config = IRRiskConfig {
            valuation_date: 1700000000,
            market_rate: 0.05,
            ..Default::default()
        };

        let sensitivity = InterestRateRisk::parallel_shift_sensitivity(&positions, &config, 100.0);

        // PV should decrease when rates increase (for net asset positions)
        // This depends on the mix of assets/liabilities
        assert!(sensitivity.pv_before != 0.0);
        assert!(sensitivity.pv_after != 0.0);
    }

    #[test]
    fn test_key_rate_durations() {
        let positions = create_test_positions();
        let config = IRRiskConfig {
            valuation_date: 1700000000,
            ..Default::default()
        };

        let tenors = vec![1, 2, 5, 10];
        let krd = InterestRateRisk::key_rate_durations(&positions, &config, &tenors);

        assert_eq!(krd.len(), tenors.len());

        // Should have entry for each tenor
        for tenor in &tenors {
            assert!(krd.contains_key(tenor));
        }
    }

    #[test]
    fn test_convexity_positive() {
        let base_date: u64 = 1700000000;
        let bond = IRPosition {
            id: "BOND".to_string(),
            instrument_type: IRInstrumentType::FixedBond,
            notional: 1_000_000.0,
            rate: 0.05,
            maturity: base_date + 10 * 365 * 86400, // 10 years
            next_reset: None,
            currency: "USD".to_string(),
        };

        let config = IRRiskConfig {
            valuation_date: base_date,
            ..Default::default()
        };

        let metrics = InterestRateRisk::calculate_metrics(&[bond], &config);

        // Convexity should be positive for standard bonds
        assert!(metrics.convexity > 0.0);
    }

    #[test]
    fn test_empty_positions() {
        let positions: Vec<IRPosition> = vec![];
        let config = IRRiskConfig::default();

        let metrics = InterestRateRisk::calculate_metrics(&positions, &config);

        assert_eq!(metrics.duration, 0.0);
        assert_eq!(metrics.modified_duration, 0.0);
        assert_eq!(metrics.dv01, 0.0);
    }

    #[test]
    fn test_pv01_by_currency() {
        let base_date: u64 = 1700000000;
        let positions = vec![
            IRPosition {
                id: "USD_BOND".to_string(),
                instrument_type: IRInstrumentType::FixedBond,
                notional: 1_000_000.0,
                rate: 0.05,
                maturity: base_date + 2 * 365 * 86400,
                next_reset: None,
                currency: "USD".to_string(),
            },
            IRPosition {
                id: "EUR_BOND".to_string(),
                instrument_type: IRInstrumentType::FixedBond,
                notional: 500_000.0,
                rate: 0.04,
                maturity: base_date + 3 * 365 * 86400,
                next_reset: None,
                currency: "EUR".to_string(),
            },
        ];

        let config = IRRiskConfig {
            valuation_date: base_date,
            ..Default::default()
        };

        let metrics = InterestRateRisk::calculate_metrics(&positions, &config);

        assert!(metrics.pv01_by_currency.contains_key("USD"));
        assert!(metrics.pv01_by_currency.contains_key("EUR"));
    }
}
