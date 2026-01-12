//! Settlement execution kernel.
//!
//! This module provides settlement execution for clearing:
//! - Execute settlement instructions
//! - Track settlement status
//! - Handle partial settlements

use crate::types::{
    InstructionType, SettlementExecutionResult, SettlementInstruction, SettlementStatus,
};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// Settlement Execution Kernel
// ============================================================================

/// Settlement execution kernel.
///
/// Executes settlement instructions and tracks their status.
#[derive(Debug, Clone)]
pub struct SettlementExecution {
    metadata: KernelMetadata,
}

impl Default for SettlementExecution {
    fn default() -> Self {
        Self::new()
    }
}

impl SettlementExecution {
    /// Create a new settlement execution kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("clearing/settlement", Domain::Clearing)
                .with_description("Settlement execution and tracking")
                .with_throughput(20_000)
                .with_latency_us(200.0),
        }
    }

    /// Execute settlement instructions.
    pub fn execute(
        instructions: &mut [SettlementInstruction],
        context: &SettlementContext,
        config: &SettlementConfig,
    ) -> SettlementExecutionResult {
        let mut settled = Vec::new();
        let mut failed = Vec::new();
        let mut pending = Vec::new();
        let mut value_settled = 0i64;
        let mut value_failed = 0i64;

        for instruction in instructions.iter_mut() {
            // Skip already processed
            if matches!(
                instruction.status,
                SettlementStatus::Settled | SettlementStatus::Failed
            ) {
                if instruction.status == SettlementStatus::Settled {
                    settled.push(instruction.id);
                    value_settled += instruction.payment_amount.unsigned_abs() as i64;
                }
                continue;
            }

            // Check eligibility
            let eligibility = Self::check_eligibility(instruction, context, config);

            match eligibility {
                EligibilityResult::Eligible => {
                    // Execute settlement
                    match Self::execute_instruction(instruction, context) {
                        Ok(()) => {
                            instruction.status = SettlementStatus::Settled;
                            settled.push(instruction.id);
                            value_settled += instruction.payment_amount.unsigned_abs() as i64;
                        }
                        Err(reason) => {
                            if config.fail_on_error {
                                instruction.status = SettlementStatus::Failed;
                                failed.push((instruction.id, reason));
                                value_failed += instruction.payment_amount.unsigned_abs() as i64;
                            } else {
                                instruction.status = SettlementStatus::Pending;
                                pending.push(instruction.id);
                            }
                        }
                    }
                }
                EligibilityResult::InsufficientBalance(reason) => {
                    if config.allow_partial && matches!(instruction.instruction_type, InstructionType::Deliver | InstructionType::Pay) {
                        // Try partial settlement
                        instruction.status = SettlementStatus::Partial;
                        pending.push(instruction.id);
                    } else {
                        instruction.status = SettlementStatus::Failed;
                        failed.push((instruction.id, reason));
                        value_failed += instruction.payment_amount.unsigned_abs() as i64;
                    }
                }
                EligibilityResult::Hold(reason) => {
                    instruction.status = SettlementStatus::OnHold;
                    pending.push(instruction.id);
                    if config.fail_on_hold {
                        failed.push((instruction.id, reason));
                    }
                }
                EligibilityResult::Ineligible(reason) => {
                    instruction.status = SettlementStatus::Failed;
                    failed.push((instruction.id, reason));
                    value_failed += instruction.payment_amount.unsigned_abs() as i64;
                }
            }
        }

        let total = settled.len() + failed.len() + pending.len();
        let settlement_rate = if total > 0 {
            settled.len() as f64 / total as f64
        } else {
            0.0
        };

        SettlementExecutionResult {
            settled,
            failed,
            pending,
            settlement_rate,
            value_settled,
            value_failed,
        }
    }

    /// Check instruction eligibility.
    fn check_eligibility(
        instruction: &SettlementInstruction,
        context: &SettlementContext,
        _config: &SettlementConfig,
    ) -> EligibilityResult {
        // Check party eligibility
        if !context.eligible_parties.contains(&instruction.party_id) {
            return EligibilityResult::Ineligible(format!(
                "Party {} not eligible for settlement",
                instruction.party_id
            ));
        }

        // Check if on hold
        if context.parties_on_hold.contains(&instruction.party_id) {
            return EligibilityResult::Hold(format!(
                "Party {} is on hold",
                instruction.party_id
            ));
        }

        // Check balances for deliveries/payments
        match instruction.instruction_type {
            InstructionType::Deliver => {
                let key = (instruction.party_id.clone(), instruction.security_id.clone());
                let balance = context.security_balances.get(&key).copied().unwrap_or(0);
                if balance < instruction.quantity.unsigned_abs() as i64 {
                    return EligibilityResult::InsufficientBalance(format!(
                        "Insufficient securities: need {}, have {}",
                        instruction.quantity.unsigned_abs(),
                        balance
                    ));
                }
            }
            InstructionType::Pay => {
                let balance = context.cash_balances.get(&instruction.party_id).copied().unwrap_or(0);
                let required = instruction.payment_amount.unsigned_abs() as i64;
                if balance < required {
                    return EligibilityResult::InsufficientBalance(format!(
                        "Insufficient cash: need {}, have {}",
                        required,
                        balance
                    ));
                }
            }
            InstructionType::Receive | InstructionType::Collect => {
                // No balance check needed for receives
            }
        }

        EligibilityResult::Eligible
    }

    /// Execute a single instruction.
    fn execute_instruction(
        instruction: &SettlementInstruction,
        _context: &SettlementContext,
    ) -> Result<(), String> {
        // In a real implementation, this would:
        // 1. Update security/cash balances
        // 2. Record the transaction
        // 3. Notify counterparties

        // For now, just validate and "execute"
        match instruction.instruction_type {
            InstructionType::Deliver | InstructionType::Receive => {
                if instruction.quantity == 0 {
                    return Err("Cannot settle zero quantity".to_string());
                }
            }
            InstructionType::Pay | InstructionType::Collect => {
                if instruction.payment_amount == 0 {
                    return Err("Cannot settle zero payment".to_string());
                }
            }
        }

        Ok(())
    }

    /// Get settlement statistics by party.
    pub fn stats_by_party(
        instructions: &[SettlementInstruction],
    ) -> HashMap<String, PartySettlementStats> {
        let mut stats: HashMap<String, PartySettlementStats> = HashMap::new();

        for instr in instructions {
            let stat = stats.entry(instr.party_id.clone()).or_default();

            stat.total_instructions += 1;

            match instr.status {
                SettlementStatus::Settled => stat.settled += 1,
                SettlementStatus::Failed => stat.failed += 1,
                SettlementStatus::Pending | SettlementStatus::InProgress => stat.pending += 1,
                SettlementStatus::Partial => stat.partial += 1,
                SettlementStatus::OnHold => stat.on_hold += 1,
            }

            match instr.instruction_type {
                InstructionType::Deliver | InstructionType::Receive => {
                    stat.securities_volume += instr.quantity.unsigned_abs() as i64;
                }
                InstructionType::Pay | InstructionType::Collect => {
                    stat.cash_volume += instr.payment_amount.unsigned_abs() as i64;
                }
            }
        }

        stats
    }

    /// Prioritize instructions for settlement.
    pub fn prioritize(instructions: &mut [SettlementInstruction], priority: SettlementPriority) {
        match priority {
            SettlementPriority::ValueDescending => {
                instructions.sort_by(|a, b| {
                    b.payment_amount.unsigned_abs().cmp(&a.payment_amount.unsigned_abs())
                });
            }
            SettlementPriority::ValueAscending => {
                instructions.sort_by(|a, b| {
                    a.payment_amount.unsigned_abs().cmp(&b.payment_amount.unsigned_abs())
                });
            }
            SettlementPriority::DateFirst => {
                instructions.sort_by_key(|i| i.settlement_date);
            }
            SettlementPriority::Fifo => {
                instructions.sort_by_key(|i| i.id);
            }
        }
    }
}

impl GpuKernel for SettlementExecution {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Settlement context.
#[derive(Debug, Clone, Default)]
pub struct SettlementContext {
    /// Eligible parties.
    pub eligible_parties: std::collections::HashSet<String>,
    /// Parties on hold.
    pub parties_on_hold: std::collections::HashSet<String>,
    /// Security balances: (party, security) -> quantity.
    pub security_balances: HashMap<(String, String), i64>,
    /// Cash balances: party -> amount.
    pub cash_balances: HashMap<String, i64>,
}

/// Settlement configuration.
#[derive(Debug, Clone)]
pub struct SettlementConfig {
    /// Fail instruction on error.
    pub fail_on_error: bool,
    /// Fail instruction if party is on hold.
    pub fail_on_hold: bool,
    /// Allow partial settlements.
    pub allow_partial: bool,
    /// Settlement window (seconds from settlement date).
    pub settlement_window_seconds: u64,
}

impl Default for SettlementConfig {
    fn default() -> Self {
        Self {
            fail_on_error: true,
            fail_on_hold: false,
            allow_partial: true,
            settlement_window_seconds: 86400, // 24 hours
        }
    }
}

/// Eligibility check result.
enum EligibilityResult {
    Eligible,
    InsufficientBalance(String),
    Hold(String),
    Ineligible(String),
}

/// Party settlement statistics.
#[derive(Debug, Clone, Default)]
pub struct PartySettlementStats {
    /// Total instructions.
    pub total_instructions: u64,
    /// Settled count.
    pub settled: u64,
    /// Failed count.
    pub failed: u64,
    /// Pending count.
    pub pending: u64,
    /// Partial count.
    pub partial: u64,
    /// On hold count.
    pub on_hold: u64,
    /// Securities volume.
    pub securities_volume: i64,
    /// Cash volume.
    pub cash_volume: i64,
}

/// Settlement priority.
#[derive(Debug, Clone, Copy)]
pub enum SettlementPriority {
    /// Highest value first.
    ValueDescending,
    /// Lowest value first.
    ValueAscending,
    /// Earliest date first.
    DateFirst,
    /// First in, first out.
    Fifo,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn create_context() -> SettlementContext {
        let mut ctx = SettlementContext::default();
        ctx.eligible_parties.insert("PARTY_A".to_string());
        ctx.eligible_parties.insert("PARTY_B".to_string());
        ctx.security_balances.insert(("PARTY_A".to_string(), "AAPL".to_string()), 1000);
        ctx.cash_balances.insert("PARTY_A".to_string(), 1_000_000);
        ctx.cash_balances.insert("PARTY_B".to_string(), 500_000);
        ctx
    }

    fn create_instruction(id: u64, party: &str, instr_type: InstructionType) -> SettlementInstruction {
        SettlementInstruction {
            id,
            party_id: party.to_string(),
            security_id: "AAPL".to_string(),
            instruction_type: instr_type,
            quantity: 100,
            payment_amount: 15000,
            currency: "USD".to_string(),
            settlement_date: 1700172800,
            status: SettlementStatus::Pending,
            source_trades: vec![1],
        }
    }

    #[test]
    fn test_settlement_metadata() {
        let kernel = SettlementExecution::new();
        assert_eq!(kernel.metadata().id, "clearing/settlement");
        assert_eq!(kernel.metadata().domain, Domain::Clearing);
    }

    #[test]
    fn test_successful_settlement() {
        let mut instructions = vec![
            create_instruction(1, "PARTY_A", InstructionType::Deliver),
            create_instruction(2, "PARTY_B", InstructionType::Receive),
        ];

        let context = create_context();
        let config = SettlementConfig::default();

        let result = SettlementExecution::execute(&mut instructions, &context, &config);

        assert_eq!(result.settled.len(), 2);
        assert!(result.failed.is_empty());
        assert!((result.settlement_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_insufficient_balance() {
        let mut instructions = vec![
            create_instruction(1, "PARTY_A", InstructionType::Deliver),
        ];
        instructions[0].quantity = 10000; // More than balance

        let context = create_context();
        let config = SettlementConfig::default();

        let result = SettlementExecution::execute(&mut instructions, &context, &config);

        // With allow_partial, should be pending
        assert!(result.settled.is_empty());
        assert_eq!(result.pending.len(), 1);
    }

    #[test]
    fn test_ineligible_party() {
        let mut instructions = vec![
            create_instruction(1, "UNKNOWN", InstructionType::Deliver),
        ];

        let context = create_context();
        let config = SettlementConfig::default();

        let result = SettlementExecution::execute(&mut instructions, &context, &config);

        assert!(result.settled.is_empty());
        assert_eq!(result.failed.len(), 1);
    }

    #[test]
    fn test_party_on_hold() {
        let mut instructions = vec![
            create_instruction(1, "PARTY_A", InstructionType::Deliver),
        ];

        let mut context = create_context();
        context.parties_on_hold.insert("PARTY_A".to_string());

        let config = SettlementConfig::default();

        let result = SettlementExecution::execute(&mut instructions, &context, &config);

        // Party on hold -> pending (unless fail_on_hold)
        assert_eq!(result.pending.len(), 1);
        assert_eq!(instructions[0].status, SettlementStatus::OnHold);
    }

    #[test]
    fn test_stats_by_party() {
        let instructions = vec![
            {
                let mut i = create_instruction(1, "PARTY_A", InstructionType::Deliver);
                i.status = SettlementStatus::Settled;
                i
            },
            {
                let mut i = create_instruction(2, "PARTY_A", InstructionType::Deliver);
                i.status = SettlementStatus::Failed;
                i
            },
            {
                let mut i = create_instruction(3, "PARTY_B", InstructionType::Receive);
                i.status = SettlementStatus::Settled;
                i
            },
        ];

        let stats = SettlementExecution::stats_by_party(&instructions);

        let a_stats = stats.get("PARTY_A").unwrap();
        assert_eq!(a_stats.total_instructions, 2);
        assert_eq!(a_stats.settled, 1);
        assert_eq!(a_stats.failed, 1);

        let b_stats = stats.get("PARTY_B").unwrap();
        assert_eq!(b_stats.settled, 1);
    }

    #[test]
    fn test_prioritize_value_desc() {
        let mut instructions = vec![
            create_instruction(1, "PARTY_A", InstructionType::Pay),
            {
                let mut i = create_instruction(2, "PARTY_A", InstructionType::Pay);
                i.payment_amount = 50000;
                i
            },
            create_instruction(3, "PARTY_A", InstructionType::Pay),
        ];

        SettlementExecution::prioritize(&mut instructions, SettlementPriority::ValueDescending);

        assert_eq!(instructions[0].id, 2); // Highest value first
    }

    #[test]
    fn test_prioritize_fifo() {
        let mut instructions = vec![
            create_instruction(3, "PARTY_A", InstructionType::Pay),
            create_instruction(1, "PARTY_A", InstructionType::Pay),
            create_instruction(2, "PARTY_A", InstructionType::Pay),
        ];

        SettlementExecution::prioritize(&mut instructions, SettlementPriority::Fifo);

        assert_eq!(instructions[0].id, 1);
        assert_eq!(instructions[1].id, 2);
        assert_eq!(instructions[2].id, 3);
    }

    #[test]
    fn test_zero_quantity_rejected() {
        let mut instructions = vec![
            {
                let mut i = create_instruction(1, "PARTY_A", InstructionType::Deliver);
                i.quantity = 0;
                i
            },
        ];

        let context = create_context();
        let config = SettlementConfig::default();

        let result = SettlementExecution::execute(&mut instructions, &context, &config);

        assert!(result.settled.is_empty());
        assert_eq!(result.failed.len(), 1);
    }
}
