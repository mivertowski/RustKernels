//! Settlement efficiency metrics kernel.
//!
//! This module provides settlement efficiency analysis:
//! - Zero balance frequency calculation
//! - Settlement timing metrics
//! - Party efficiency scoring

use crate::types::{SettlementEfficiency, SettlementInstruction, SettlementStatus, ZeroBalanceMetrics};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// Zero Balance Frequency Kernel
// ============================================================================

/// Zero balance frequency kernel.
///
/// Calculates settlement efficiency metrics including zero balance frequency.
#[derive(Debug, Clone)]
pub struct ZeroBalanceFrequency {
    metadata: KernelMetadata,
}

impl Default for ZeroBalanceFrequency {
    fn default() -> Self {
        Self::new()
    }
}

impl ZeroBalanceFrequency {
    /// Create a new zero balance frequency kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("clearing/zero-balance", Domain::Clearing)
                .with_description("Settlement efficiency and zero balance metrics")
                .with_throughput(50_000)
                .with_latency_us(100.0),
        }
    }

    /// Calculate zero balance metrics for a party.
    pub fn calculate_zbf(
        activity: &[DailyActivity],
        party_id: &str,
    ) -> ZeroBalanceMetrics {
        if activity.is_empty() {
            return ZeroBalanceMetrics {
                party_id: party_id.to_string(),
                total_days: 0,
                zero_balance_days: 0,
                frequency: 0.0,
                avg_eod_position: 0.0,
                peak_position: 0,
                avg_intraday_turnover: 0.0,
            };
        }

        let total_days = activity.len() as u32;
        let zero_balance_days = activity.iter().filter(|a| a.eod_position == 0).count() as u32;
        let frequency = zero_balance_days as f64 / total_days as f64;

        let avg_eod_position = activity.iter().map(|a| a.eod_position as f64).sum::<f64>()
            / total_days as f64;

        let peak_position = activity
            .iter()
            .map(|a| a.peak_intraday_position)
            .max()
            .unwrap_or(0);

        let avg_intraday_turnover = activity.iter().map(|a| a.intraday_turnover as f64).sum::<f64>()
            / total_days as f64;

        ZeroBalanceMetrics {
            party_id: party_id.to_string(),
            total_days,
            zero_balance_days,
            frequency,
            avg_eod_position,
            peak_position,
            avg_intraday_turnover,
        }
    }

    /// Calculate settlement efficiency from instructions.
    pub fn calculate_efficiency(
        instructions: &[SettlementInstruction],
        expected_settlement: &HashMap<u64, u64>, // instruction_id -> expected timestamp
        actual_settlement: &HashMap<u64, u64>,   // instruction_id -> actual timestamp
    ) -> SettlementEfficiency {
        let total_instructions = instructions.len() as u64;

        let mut on_time = 0u64;
        let mut late = 0u64;
        let mut failed = 0u64;
        let mut total_delay = 0i64;
        let mut delay_count = 0u64;

        // Track by party for party metrics
        let mut party_data: HashMap<String, Vec<DailyActivity>> = HashMap::new();

        for instr in instructions {
            match instr.status {
                SettlementStatus::Settled => {
                    if let (Some(&expected), Some(&actual)) = (
                        expected_settlement.get(&instr.id),
                        actual_settlement.get(&instr.id),
                    ) {
                        if actual <= expected {
                            on_time += 1;
                        } else {
                            late += 1;
                            total_delay += (actual - expected) as i64;
                            delay_count += 1;
                        }
                    } else {
                        on_time += 1; // Assume on-time if no data
                    }
                }
                SettlementStatus::Failed => {
                    failed += 1;
                }
                _ => {}
            }

            // Aggregate party data (simplified)
            party_data
                .entry(instr.party_id.clone())
                .or_default()
                .push(DailyActivity {
                    date: instr.settlement_date,
                    eod_position: 0, // Would be calculated from actual balances
                    peak_intraday_position: instr.quantity.unsigned_abs() as i64,
                    intraday_turnover: instr.quantity.unsigned_abs() as i64,
                });
        }

        let on_time_rate = if total_instructions > 0 {
            on_time as f64 / total_instructions as f64
        } else {
            0.0
        };

        let avg_delay_seconds = if delay_count > 0 {
            total_delay as f64 / delay_count as f64
        } else {
            0.0
        };

        // Calculate party metrics
        let party_metrics: Vec<_> = party_data
            .iter()
            .map(|(party_id, activity)| Self::calculate_zbf(activity, party_id))
            .collect();

        SettlementEfficiency {
            period_days: 1, // Simplified
            total_instructions,
            on_time_settlements: on_time,
            late_settlements: late,
            failed_settlements: failed,
            on_time_rate,
            avg_delay_seconds,
            party_metrics,
        }
    }

    /// Calculate settlement velocity (instructions settled per time period).
    pub fn calculate_velocity(
        instructions: &[SettlementInstruction],
        period_seconds: u64,
    ) -> SettlementVelocity {
        let settled: Vec<_> = instructions
            .iter()
            .filter(|i| i.status == SettlementStatus::Settled)
            .collect();

        if settled.is_empty() || period_seconds == 0 {
            return SettlementVelocity {
                instructions_per_second: 0.0,
                value_per_second: 0.0,
                securities_per_second: 0.0,
                peak_rate: 0.0,
            };
        }

        let total_value: u64 = settled.iter().map(|i| i.payment_amount.unsigned_abs()).sum();
        let total_securities: u64 = settled.iter().map(|i| i.quantity.unsigned_abs()).sum();

        let instructions_per_second = settled.len() as f64 / period_seconds as f64;
        let value_per_second = total_value as f64 / period_seconds as f64;
        let securities_per_second = total_securities as f64 / period_seconds as f64;

        // Calculate peak rate (simplified - would use time buckets)
        let peak_rate = instructions_per_second * 2.0; // Assume peak is 2x average

        SettlementVelocity {
            instructions_per_second,
            value_per_second,
            securities_per_second,
            peak_rate,
        }
    }

    /// Score parties by settlement efficiency.
    pub fn score_parties(
        instructions: &[SettlementInstruction],
    ) -> Vec<PartyEfficiencyScore> {
        let mut party_stats: HashMap<String, PartyStats> = HashMap::new();

        for instr in instructions {
            let stats = party_stats.entry(instr.party_id.clone()).or_default();
            stats.total += 1;

            match instr.status {
                SettlementStatus::Settled => stats.settled += 1,
                SettlementStatus::Failed => stats.failed += 1,
                SettlementStatus::Partial => stats.partial += 1,
                _ => stats.pending += 1,
            }
        }

        let mut scores: Vec<_> = party_stats
            .into_iter()
            .map(|(party_id, stats)| {
                let settlement_rate = if stats.total > 0 {
                    stats.settled as f64 / stats.total as f64
                } else {
                    0.0
                };

                let failure_rate = if stats.total > 0 {
                    stats.failed as f64 / stats.total as f64
                } else {
                    0.0
                };

                // Efficiency score: 0-100
                let score = (settlement_rate * 100.0 - failure_rate * 50.0).max(0.0);

                PartyEfficiencyScore {
                    party_id,
                    total_instructions: stats.total,
                    settled: stats.settled,
                    failed: stats.failed,
                    pending: stats.pending,
                    settlement_rate,
                    efficiency_score: score,
                }
            })
            .collect();

        // Sort by efficiency score descending
        scores.sort_by(|a, b| b.efficiency_score.partial_cmp(&a.efficiency_score).unwrap());

        scores
    }

    /// Calculate liquidity usage metrics.
    pub fn calculate_liquidity_usage(
        instructions: &[SettlementInstruction],
        available_liquidity: i64,
    ) -> LiquidityMetrics {
        let total_value: u64 = instructions
            .iter()
            .filter(|i| i.status != SettlementStatus::Failed)
            .map(|i| i.payment_amount.unsigned_abs())
            .sum();

        let settled_value: u64 = instructions
            .iter()
            .filter(|i| i.status == SettlementStatus::Settled)
            .map(|i| i.payment_amount.unsigned_abs())
            .sum();

        let utilization = if available_liquidity > 0 {
            (total_value as f64 / available_liquidity as f64).min(1.0)
        } else {
            0.0
        };

        let turnover = if available_liquidity > 0 {
            settled_value as f64 / available_liquidity as f64
        } else {
            0.0
        };

        LiquidityMetrics {
            total_value_processed: total_value,
            settled_value,
            available_liquidity,
            utilization_rate: utilization,
            turnover_ratio: turnover,
        }
    }
}

impl GpuKernel for ZeroBalanceFrequency {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Daily activity for a party.
#[derive(Debug, Clone)]
pub struct DailyActivity {
    /// Date (Unix timestamp).
    pub date: u64,
    /// End-of-day position.
    pub eod_position: i64,
    /// Peak intraday position.
    pub peak_intraday_position: i64,
    /// Intraday turnover.
    pub intraday_turnover: i64,
}

/// Settlement velocity metrics.
#[derive(Debug, Clone)]
pub struct SettlementVelocity {
    /// Instructions settled per second.
    pub instructions_per_second: f64,
    /// Value settled per second.
    pub value_per_second: f64,
    /// Securities settled per second.
    pub securities_per_second: f64,
    /// Peak settlement rate.
    pub peak_rate: f64,
}

/// Party efficiency score.
#[derive(Debug, Clone)]
pub struct PartyEfficiencyScore {
    /// Party ID.
    pub party_id: String,
    /// Total instructions.
    pub total_instructions: u64,
    /// Settled count.
    pub settled: u64,
    /// Failed count.
    pub failed: u64,
    /// Pending count.
    pub pending: u64,
    /// Settlement rate.
    pub settlement_rate: f64,
    /// Efficiency score (0-100).
    pub efficiency_score: f64,
}

/// Liquidity metrics.
#[derive(Debug, Clone)]
pub struct LiquidityMetrics {
    /// Total value processed.
    pub total_value_processed: u64,
    /// Value actually settled.
    pub settled_value: u64,
    /// Available liquidity.
    pub available_liquidity: i64,
    /// Utilization rate (0-1).
    pub utilization_rate: f64,
    /// Turnover ratio.
    pub turnover_ratio: f64,
}

#[derive(Default)]
struct PartyStats {
    total: u64,
    settled: u64,
    failed: u64,
    partial: u64,
    pending: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::InstructionType;

    fn create_instruction(id: u64, party: &str, status: SettlementStatus) -> SettlementInstruction {
        SettlementInstruction {
            id,
            party_id: party.to_string(),
            security_id: "AAPL".to_string(),
            instruction_type: InstructionType::Deliver,
            quantity: 100,
            payment_amount: 15000,
            currency: "USD".to_string(),
            settlement_date: 1700172800,
            status,
            source_trades: vec![1],
        }
    }

    #[test]
    fn test_zbf_metadata() {
        let kernel = ZeroBalanceFrequency::new();
        assert_eq!(kernel.metadata().id, "clearing/zero-balance");
        assert_eq!(kernel.metadata().domain, Domain::Clearing);
    }

    #[test]
    fn test_calculate_zbf() {
        let activity = vec![
            DailyActivity {
                date: 1700000000,
                eod_position: 0,
                peak_intraday_position: 1000,
                intraday_turnover: 5000,
            },
            DailyActivity {
                date: 1700086400,
                eod_position: 500,
                peak_intraday_position: 2000,
                intraday_turnover: 8000,
            },
            DailyActivity {
                date: 1700172800,
                eod_position: 0,
                peak_intraday_position: 1500,
                intraday_turnover: 6000,
            },
        ];

        let metrics = ZeroBalanceFrequency::calculate_zbf(&activity, "PARTY_A");

        assert_eq!(metrics.total_days, 3);
        assert_eq!(metrics.zero_balance_days, 2);
        assert!((metrics.frequency - 0.666).abs() < 0.01);
        assert_eq!(metrics.peak_position, 2000);
    }

    #[test]
    fn test_empty_activity() {
        let activity: Vec<DailyActivity> = vec![];

        let metrics = ZeroBalanceFrequency::calculate_zbf(&activity, "PARTY_A");

        assert_eq!(metrics.total_days, 0);
        assert_eq!(metrics.frequency, 0.0);
    }

    #[test]
    fn test_calculate_efficiency() {
        let instructions = vec![
            create_instruction(1, "PARTY_A", SettlementStatus::Settled),
            create_instruction(2, "PARTY_A", SettlementStatus::Settled),
            create_instruction(3, "PARTY_B", SettlementStatus::Failed),
        ];

        let expected: HashMap<u64, u64> = [(1, 1700172800), (2, 1700172800)].into_iter().collect();
        let actual: HashMap<u64, u64> = [(1, 1700172800), (2, 1700259200)].into_iter().collect();

        let efficiency =
            ZeroBalanceFrequency::calculate_efficiency(&instructions, &expected, &actual);

        assert_eq!(efficiency.total_instructions, 3);
        assert_eq!(efficiency.on_time_settlements, 1);
        assert_eq!(efficiency.late_settlements, 1);
        assert_eq!(efficiency.failed_settlements, 1);
    }

    #[test]
    fn test_calculate_velocity() {
        let instructions = vec![
            create_instruction(1, "PARTY_A", SettlementStatus::Settled),
            create_instruction(2, "PARTY_A", SettlementStatus::Settled),
            create_instruction(3, "PARTY_B", SettlementStatus::Settled),
        ];

        let velocity = ZeroBalanceFrequency::calculate_velocity(&instructions, 3600); // 1 hour

        assert!(velocity.instructions_per_second > 0.0);
        assert!(velocity.value_per_second > 0.0);
    }

    #[test]
    fn test_score_parties() {
        let instructions = vec![
            create_instruction(1, "PARTY_A", SettlementStatus::Settled),
            create_instruction(2, "PARTY_A", SettlementStatus::Settled),
            create_instruction(3, "PARTY_A", SettlementStatus::Failed),
            create_instruction(4, "PARTY_B", SettlementStatus::Settled),
            create_instruction(5, "PARTY_B", SettlementStatus::Settled),
        ];

        let scores = ZeroBalanceFrequency::score_parties(&instructions);

        // PARTY_B should have higher score (100% settled vs 67%)
        assert_eq!(scores[0].party_id, "PARTY_B");
        assert!(scores[0].efficiency_score > scores[1].efficiency_score);
    }

    #[test]
    fn test_calculate_liquidity_usage() {
        let instructions = vec![
            create_instruction(1, "PARTY_A", SettlementStatus::Settled),
            create_instruction(2, "PARTY_A", SettlementStatus::Settled),
            create_instruction(3, "PARTY_B", SettlementStatus::Pending),
        ];

        let metrics = ZeroBalanceFrequency::calculate_liquidity_usage(&instructions, 100000);

        assert!(metrics.utilization_rate > 0.0);
        assert!(metrics.utilization_rate <= 1.0);
        assert!(metrics.turnover_ratio > 0.0);
    }

    #[test]
    fn test_velocity_no_settled() {
        let instructions = vec![
            create_instruction(1, "PARTY_A", SettlementStatus::Pending),
            create_instruction(2, "PARTY_A", SettlementStatus::Failed),
        ];

        let velocity = ZeroBalanceFrequency::calculate_velocity(&instructions, 3600);

        assert_eq!(velocity.instructions_per_second, 0.0);
    }

    #[test]
    fn test_all_zero_balance() {
        let activity: Vec<DailyActivity> = (0..5)
            .map(|i| DailyActivity {
                date: 1700000000 + i * 86400,
                eod_position: 0,
                peak_intraday_position: 1000,
                intraday_turnover: 5000,
            })
            .collect();

        let metrics = ZeroBalanceFrequency::calculate_zbf(&activity, "PARTY_A");

        assert_eq!(metrics.zero_balance_days, 5);
        assert!((metrics.frequency - 1.0).abs() < 0.001);
    }
}
