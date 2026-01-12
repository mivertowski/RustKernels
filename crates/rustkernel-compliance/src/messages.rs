//! Ring message types for compliance kernels.
//!
//! This module defines request/response message types for GPU-native
//! persistent actor communication for compliance algorithms.

use crate::types::{
    AMLPatternResult, CircularFlowResult, Entity, EntityResolutionResult, KYCFactors, KYCResult,
    MonitoringResult, MonitoringRule, PEPEntry, PEPResult, RapidMovementResult, ReciprocityResult,
    SanctionsEntry, SanctionsResult, TimeWindow, Transaction,
};
use serde::{Deserialize, Serialize};

// ============================================================================
// AML Messages
// ============================================================================

/// Circular flow ratio input for batch execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircularFlowInput {
    /// Transactions to analyze.
    pub transactions: Vec<Transaction>,
    /// Minimum amount for a cycle to be flagged.
    pub min_amount: f64,
}

impl CircularFlowInput {
    /// Create a new circular flow input.
    pub fn new(transactions: Vec<Transaction>, min_amount: f64) -> Self {
        Self {
            transactions,
            min_amount,
        }
    }
}

/// Circular flow output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircularFlowOutput {
    /// The analysis result.
    pub result: CircularFlowResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

/// Reciprocity flow input for batch execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReciprocityFlowInput {
    /// Transactions to analyze.
    pub transactions: Vec<Transaction>,
    /// Time window for analysis (optional).
    pub window: Option<TimeWindow>,
    /// Minimum amount to consider.
    pub min_amount: f64,
}

impl ReciprocityFlowInput {
    /// Create a new reciprocity flow input.
    pub fn new(transactions: Vec<Transaction>, min_amount: f64) -> Self {
        Self {
            transactions,
            window: None,
            min_amount,
        }
    }

    /// Set time window.
    pub fn with_window(mut self, window: TimeWindow) -> Self {
        self.window = Some(window);
        self
    }
}

/// Reciprocity flow output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReciprocityFlowOutput {
    /// The analysis result.
    pub result: ReciprocityResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

/// Rapid movement input for batch execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RapidMovementInput {
    /// Transactions to analyze.
    pub transactions: Vec<Transaction>,
    /// Time window in hours.
    pub window_hours: f64,
    /// Velocity threshold (transactions per hour).
    pub velocity_threshold: f64,
    /// Minimum amount threshold.
    pub amount_threshold: f64,
}

impl RapidMovementInput {
    /// Create a new rapid movement input.
    pub fn new(
        transactions: Vec<Transaction>,
        window_hours: f64,
        velocity_threshold: f64,
        amount_threshold: f64,
    ) -> Self {
        Self {
            transactions,
            window_hours,
            velocity_threshold,
            amount_threshold,
        }
    }
}

/// Rapid movement output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RapidMovementOutput {
    /// The analysis result.
    pub result: RapidMovementResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

/// AML pattern detection input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AMLPatternInput {
    /// Transactions to analyze.
    pub transactions: Vec<Transaction>,
    /// Structuring threshold amount.
    pub structuring_threshold: f64,
    /// Structuring window in hours.
    pub structuring_window_hours: f64,
}

impl AMLPatternInput {
    /// Create a new AML pattern input.
    pub fn new(transactions: Vec<Transaction>) -> Self {
        Self {
            transactions,
            structuring_threshold: 10_000.0,
            structuring_window_hours: 24.0,
        }
    }

    /// Set structuring threshold.
    pub fn with_structuring_threshold(mut self, threshold: f64) -> Self {
        self.structuring_threshold = threshold;
        self
    }
}

/// AML pattern detection output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AMLPatternOutput {
    /// The detection result.
    pub result: AMLPatternResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// KYC Messages
// ============================================================================

/// KYC scoring input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KYCScoringInput {
    /// KYC factors for scoring.
    pub factors: KYCFactors,
}

impl KYCScoringInput {
    /// Create a new KYC scoring input.
    pub fn new(factors: KYCFactors) -> Self {
        Self { factors }
    }
}

/// KYC scoring output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KYCScoringOutput {
    /// The scoring result.
    pub result: KYCResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

/// Entity resolution input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityResolutionInput {
    /// Query entity.
    pub query: Entity,
    /// Candidate entities.
    pub candidates: Vec<Entity>,
    /// Minimum match score threshold.
    pub min_score: f64,
    /// Maximum number of matches to return.
    pub max_matches: usize,
}

impl EntityResolutionInput {
    /// Create a new entity resolution input.
    pub fn new(query: Entity, candidates: Vec<Entity>) -> Self {
        Self {
            query,
            candidates,
            min_score: 0.7,
            max_matches: 10,
        }
    }
}

/// Entity resolution output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityResolutionOutput {
    /// The resolution result.
    pub result: EntityResolutionResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Sanctions Messages
// ============================================================================

/// Sanctions screening input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanctionsScreeningInput {
    /// Name to screen.
    pub name: String,
    /// Sanctions list to screen against.
    pub sanctions_list: Vec<SanctionsEntry>,
    /// Minimum match score threshold.
    pub min_score: f64,
    /// Maximum number of matches to return.
    pub max_matches: usize,
}

impl SanctionsScreeningInput {
    /// Create a new sanctions screening input.
    pub fn new(name: String, sanctions_list: Vec<SanctionsEntry>) -> Self {
        Self {
            name,
            sanctions_list,
            min_score: 0.7,
            max_matches: 10,
        }
    }
}

/// Sanctions screening output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanctionsScreeningOutput {
    /// The screening result.
    pub result: SanctionsResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

/// PEP screening input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PEPScreeningInput {
    /// Name to screen.
    pub name: String,
    /// PEP list to screen against.
    pub pep_list: Vec<PEPEntry>,
    /// Minimum match score threshold.
    pub min_score: f64,
    /// Maximum number of matches to return.
    pub max_matches: usize,
}

impl PEPScreeningInput {
    /// Create a new PEP screening input.
    pub fn new(name: String, pep_list: Vec<PEPEntry>) -> Self {
        Self {
            name,
            pep_list,
            min_score: 0.7,
            max_matches: 10,
        }
    }
}

/// PEP screening output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PEPScreeningOutput {
    /// The screening result.
    pub result: PEPResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Transaction Monitoring Messages
// ============================================================================

/// Transaction monitoring input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionMonitoringInput {
    /// Transactions to monitor.
    pub transactions: Vec<Transaction>,
    /// Monitoring rules to apply.
    pub rules: Vec<MonitoringRule>,
    /// Current timestamp for time window calculations.
    pub current_time: u64,
}

impl TransactionMonitoringInput {
    /// Create a new transaction monitoring input.
    pub fn new(
        transactions: Vec<Transaction>,
        rules: Vec<MonitoringRule>,
        current_time: u64,
    ) -> Self {
        Self {
            transactions,
            rules,
            current_time,
        }
    }
}

/// Transaction monitoring output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionMonitoringOutput {
    /// The monitoring result.
    pub result: MonitoringResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_flow_input_builder() {
        let input = CircularFlowInput::new(vec![], 100.0);
        assert_eq!(input.min_amount, 100.0);
    }

    #[test]
    fn test_aml_pattern_input_builder() {
        let input = AMLPatternInput::new(vec![]).with_structuring_threshold(5000.0);
        assert_eq!(input.structuring_threshold, 5000.0);
    }
}
