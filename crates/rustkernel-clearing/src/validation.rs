//! Trade validation kernel.
//!
//! This module provides trade validation for clearing:
//! - Counterparty eligibility checks
//! - Security eligibility checks
//! - Settlement date validation
//! - Position limit checks

use crate::types::{
    ErrorSeverity, Trade, TradeStatus, ValidationConfig, ValidationError, ValidationResult,
};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Clearing Validation Kernel
// ============================================================================

/// Clearing validation kernel.
///
/// Validates trades before they enter the clearing process.
#[derive(Debug, Clone)]
pub struct ClearingValidation {
    metadata: KernelMetadata,
}

impl Default for ClearingValidation {
    fn default() -> Self {
        Self::new()
    }
}

impl ClearingValidation {
    /// Create a new clearing validation kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("clearing/validation", Domain::Clearing)
                .with_description("Trade validation for clearing")
                .with_throughput(100_000)
                .with_latency_us(50.0),
        }
    }

    /// Validate a single trade.
    pub fn validate(
        trade: &Trade,
        config: &ValidationConfig,
        context: &ValidationContext,
    ) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check trade status
        if trade.status != TradeStatus::Pending {
            errors.push(ValidationError {
                code: "INVALID_STATUS".to_string(),
                message: format!("Trade status {:?} not valid for clearing", trade.status),
                severity: ErrorSeverity::Critical,
            });
        }

        // Check quantity
        if trade.quantity == 0 {
            errors.push(ValidationError {
                code: "ZERO_QUANTITY".to_string(),
                message: "Trade quantity cannot be zero".to_string(),
                severity: ErrorSeverity::Critical,
            });
        }

        // Check price
        if trade.price <= 0 {
            errors.push(ValidationError {
                code: "INVALID_PRICE".to_string(),
                message: "Trade price must be positive".to_string(),
                severity: ErrorSeverity::Critical,
            });
        }

        // Check counterparty eligibility
        if config.check_counterparty {
            if !context.eligible_parties.contains(&trade.buyer_id) {
                errors.push(ValidationError {
                    code: "BUYER_NOT_ELIGIBLE".to_string(),
                    message: format!("Buyer {} not eligible for clearing", trade.buyer_id),
                    severity: ErrorSeverity::Critical,
                });
            }
            if !context.eligible_parties.contains(&trade.seller_id) {
                errors.push(ValidationError {
                    code: "SELLER_NOT_ELIGIBLE".to_string(),
                    message: format!("Seller {} not eligible for clearing", trade.seller_id),
                    severity: ErrorSeverity::Critical,
                });
            }
            if trade.buyer_id == trade.seller_id {
                warnings.push("Buyer and seller are the same party".to_string());
            }
        }

        // Check security eligibility
        if config.check_security && !context.eligible_securities.contains(&trade.security_id) {
            errors.push(ValidationError {
                code: "SECURITY_NOT_ELIGIBLE".to_string(),
                message: format!("Security {} not eligible for clearing", trade.security_id),
                severity: ErrorSeverity::Critical,
            });
        }

        // Check settlement date
        if config.check_settlement_date {
            if trade.settlement_date < trade.trade_date {
                errors.push(ValidationError {
                    code: "INVALID_SETTLEMENT_DATE".to_string(),
                    message: "Settlement date cannot be before trade date".to_string(),
                    severity: ErrorSeverity::Critical,
                });
            }

            let days_to_settle = (trade.settlement_date.saturating_sub(trade.trade_date)) / 86400;
            if days_to_settle < config.min_settlement_days as u64 {
                errors.push(ValidationError {
                    code: "SETTLEMENT_TOO_SOON".to_string(),
                    message: format!(
                        "Settlement {} days from trade, minimum is {}",
                        days_to_settle, config.min_settlement_days
                    ),
                    severity: ErrorSeverity::Critical,
                });
            }
            if days_to_settle > config.max_settlement_days as u64 {
                errors.push(ValidationError {
                    code: "SETTLEMENT_TOO_FAR".to_string(),
                    message: format!(
                        "Settlement {} days from trade, maximum is {}",
                        days_to_settle, config.max_settlement_days
                    ),
                    severity: ErrorSeverity::Critical,
                });
            }
        }

        // Check position limits
        if config.check_limits {
            if let Some(&limit) = context.position_limits.get(&trade.security_id) {
                if trade.quantity.unsigned_abs() > limit {
                    errors.push(ValidationError {
                        code: "EXCEEDS_POSITION_LIMIT".to_string(),
                        message: format!(
                            "Quantity {} exceeds limit {} for security {}",
                            trade.quantity, limit, trade.security_id
                        ),
                        severity: ErrorSeverity::Critical,
                    });
                }
            }
        }

        ValidationResult {
            trade_id: trade.id,
            is_valid: errors.iter().all(|e| e.severity != ErrorSeverity::Critical),
            errors,
            warnings,
        }
    }

    /// Validate a batch of trades.
    pub fn validate_batch(
        trades: &[Trade],
        config: &ValidationConfig,
        context: &ValidationContext,
    ) -> Vec<ValidationResult> {
        trades
            .iter()
            .map(|trade| Self::validate(trade, config, context))
            .collect()
    }

    /// Get validation statistics.
    pub fn get_stats(results: &[ValidationResult]) -> ValidationStats {
        let total = results.len() as u64;
        let valid = results.iter().filter(|r| r.is_valid).count() as u64;
        let invalid = total - valid;

        let mut error_counts: HashMap<String, u64> = HashMap::new();
        for result in results {
            for error in &result.errors {
                *error_counts.entry(error.code.clone()).or_insert(0) += 1;
            }
        }

        ValidationStats {
            total_trades: total,
            valid_trades: valid,
            invalid_trades: invalid,
            validation_rate: if total > 0 {
                valid as f64 / total as f64
            } else {
                0.0
            },
            error_counts,
        }
    }
}

impl GpuKernel for ClearingValidation {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Validation context with reference data.
#[derive(Debug, Clone, Default)]
pub struct ValidationContext {
    /// Eligible clearing parties.
    pub eligible_parties: HashSet<String>,
    /// Eligible securities.
    pub eligible_securities: HashSet<String>,
    /// Position limits per security.
    pub position_limits: HashMap<String, u64>,
    /// Current date (for date validation).
    pub current_date: u64,
}

/// Validation statistics.
#[derive(Debug, Clone)]
pub struct ValidationStats {
    /// Total trades validated.
    pub total_trades: u64,
    /// Valid trades.
    pub valid_trades: u64,
    /// Invalid trades.
    pub invalid_trades: u64,
    /// Validation success rate.
    pub validation_rate: f64,
    /// Error counts by code.
    pub error_counts: HashMap<String, u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_valid_trade() -> Trade {
        Trade::new(
            1,
            "AAPL".to_string(),
            "BUYER001".to_string(),
            "SELLER001".to_string(),
            100,
            15000, // $150.00
            1700000000,
            1700172800, // T+2
        )
    }

    fn create_context() -> ValidationContext {
        let mut ctx = ValidationContext::default();
        ctx.eligible_parties.insert("BUYER001".to_string());
        ctx.eligible_parties.insert("SELLER001".to_string());
        ctx.eligible_securities.insert("AAPL".to_string());
        ctx.position_limits.insert("AAPL".to_string(), 10000);
        ctx.current_date = 1700000000;
        ctx
    }

    #[test]
    fn test_validation_metadata() {
        let kernel = ClearingValidation::new();
        assert_eq!(kernel.metadata().id, "clearing/validation");
        assert_eq!(kernel.metadata().domain, Domain::Clearing);
    }

    #[test]
    fn test_valid_trade() {
        let trade = create_valid_trade();
        let config = ValidationConfig::default();
        let context = create_context();

        let result = ClearingValidation::validate(&trade, &config, &context);

        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_zero_quantity() {
        let mut trade = create_valid_trade();
        trade.quantity = 0;

        let config = ValidationConfig::default();
        let context = create_context();

        let result = ClearingValidation::validate(&trade, &config, &context);

        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.code == "ZERO_QUANTITY"));
    }

    #[test]
    fn test_invalid_price() {
        let mut trade = create_valid_trade();
        trade.price = -100;

        let config = ValidationConfig::default();
        let context = create_context();

        let result = ClearingValidation::validate(&trade, &config, &context);

        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.code == "INVALID_PRICE"));
    }

    #[test]
    fn test_ineligible_buyer() {
        let mut trade = create_valid_trade();
        trade.buyer_id = "UNKNOWN".to_string();

        let config = ValidationConfig::default();
        let context = create_context();

        let result = ClearingValidation::validate(&trade, &config, &context);

        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.code == "BUYER_NOT_ELIGIBLE"));
    }

    #[test]
    fn test_ineligible_security() {
        let mut trade = create_valid_trade();
        trade.security_id = "UNKNOWN".to_string();

        let config = ValidationConfig::default();
        let context = create_context();

        let result = ClearingValidation::validate(&trade, &config, &context);

        assert!(!result.is_valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.code == "SECURITY_NOT_ELIGIBLE")
        );
    }

    #[test]
    fn test_settlement_before_trade() {
        let mut trade = create_valid_trade();
        trade.settlement_date = trade.trade_date - 86400; // 1 day before

        let config = ValidationConfig::default();
        let context = create_context();

        let result = ClearingValidation::validate(&trade, &config, &context);

        assert!(!result.is_valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.code == "INVALID_SETTLEMENT_DATE")
        );
    }

    #[test]
    fn test_exceeds_position_limit() {
        let mut trade = create_valid_trade();
        trade.quantity = 50000; // Exceeds 10000 limit

        let config = ValidationConfig::default();
        let context = create_context();

        let result = ClearingValidation::validate(&trade, &config, &context);

        assert!(!result.is_valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.code == "EXCEEDS_POSITION_LIMIT")
        );
    }

    #[test]
    fn test_batch_validation() {
        let trades = vec![create_valid_trade(), {
            let mut t = create_valid_trade();
            t.id = 2;
            t.quantity = 0;
            t
        }];

        let config = ValidationConfig::default();
        let context = create_context();

        let results = ClearingValidation::validate_batch(&trades, &config, &context);

        assert_eq!(results.len(), 2);
        assert!(results[0].is_valid);
        assert!(!results[1].is_valid);
    }

    #[test]
    fn test_validation_stats() {
        let trades = vec![
            create_valid_trade(),
            {
                let mut t = create_valid_trade();
                t.id = 2;
                t.quantity = 0;
                t
            },
            {
                let mut t = create_valid_trade();
                t.id = 3;
                t
            },
        ];

        let config = ValidationConfig::default();
        let context = create_context();

        let results = ClearingValidation::validate_batch(&trades, &config, &context);
        let stats = ClearingValidation::get_stats(&results);

        assert_eq!(stats.total_trades, 3);
        assert_eq!(stats.valid_trades, 2);
        assert_eq!(stats.invalid_trades, 1);
        assert!((stats.validation_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_skip_counterparty_check() {
        let mut trade = create_valid_trade();
        trade.buyer_id = "UNKNOWN".to_string();

        let config = ValidationConfig {
            check_counterparty: false,
            ..ValidationConfig::default()
        };

        let context = create_context();

        let result = ClearingValidation::validate(&trade, &config, &context);

        // Should be valid since counterparty check is disabled
        assert!(result.is_valid);
    }
}
