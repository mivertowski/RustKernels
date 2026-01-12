//! Payment processing kernel.
//!
//! Ring-mode kernel for real-time payment transaction execution.

use crate::types::*;
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// PaymentProcessing Kernel
// ============================================================================

/// Payment processing kernel for real-time transaction execution.
///
/// Handles payment validation, routing, and execution across multiple
/// payment rails (ACH, Wire, RealTime, Internal, Check, Card).
#[derive(Debug, Clone)]
pub struct PaymentProcessing {
    metadata: KernelMetadata,
}

impl Default for PaymentProcessing {
    fn default() -> Self {
        Self::new()
    }
}

impl PaymentProcessing {
    /// Create a new payment processing kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("payments/processing", Domain::PaymentProcessing)
                .with_description("Real-time payment transaction execution")
                .with_throughput(50_000)
                .with_latency_us(100.0),
        }
    }

    /// Process a batch of payments.
    pub fn process_payments(
        payments: &[Payment],
        accounts: &HashMap<String, PaymentAccount>,
        config: &ProcessingConfig,
    ) -> ProcessingResult {
        let total_count = payments.len();

        let mut processed = Vec::new();
        let mut failed = Vec::new();
        let mut pending = Vec::new();
        let mut total_amount = 0.0;
        let mut processing_times = Vec::new();

        for payment in payments {
            let payment_start = std::time::Instant::now();

            match Self::process_single_payment(payment, accounts, config) {
                PaymentOutcome::Processed => {
                    processed.push(payment.id.clone());
                    total_amount += payment.amount;
                }
                PaymentOutcome::Failed(reason) => {
                    failed.push((payment.id.clone(), reason));
                }
                PaymentOutcome::Pending => {
                    pending.push(payment.id.clone());
                }
            }

            processing_times.push(payment_start.elapsed().as_micros() as f64);
        }

        let processed_count = processed.len();
        let failed_count = failed.len();
        let avg_processing_time_us = if !processing_times.is_empty() {
            processing_times.iter().sum::<f64>() / processing_times.len() as f64
        } else {
            0.0
        };

        ProcessingResult {
            processed,
            failed,
            pending,
            stats: ProcessingStats {
                total_count,
                processed_count,
                failed_count,
                total_amount,
                avg_processing_time_us,
            },
        }
    }

    /// Process a single payment transaction.
    fn process_single_payment(
        payment: &Payment,
        accounts: &HashMap<String, PaymentAccount>,
        config: &ProcessingConfig,
    ) -> PaymentOutcome {
        // 1. Validate payment
        if let Err(reason) = Self::validate_payment(payment, accounts, config) {
            return PaymentOutcome::Failed(reason);
        }

        // 2. Check payment type routing
        if !Self::is_payment_type_enabled(payment.payment_type, config) {
            return PaymentOutcome::Failed("Payment type not enabled".to_string());
        }

        // 3. Check processing windows for non-real-time payments
        if !Self::is_within_processing_window(payment, config) {
            return PaymentOutcome::Pending;
        }

        // 4. Apply fraud checks
        if config.fraud_check_enabled {
            if let Err(reason) = Self::fraud_check(payment, config) {
                return PaymentOutcome::Failed(reason);
            }
        }

        // 5. Route payment
        match payment.payment_type {
            PaymentType::RealTime | PaymentType::Internal => PaymentOutcome::Processed,
            PaymentType::Wire if payment.priority >= PaymentPriority::High => {
                PaymentOutcome::Processed
            }
            _ => {
                // Batch processing for ACH, Check, low-priority Wire
                if config.batch_mode {
                    PaymentOutcome::Pending
                } else {
                    PaymentOutcome::Processed
                }
            }
        }
    }

    /// Validate a payment.
    fn validate_payment(
        payment: &Payment,
        accounts: &HashMap<String, PaymentAccount>,
        config: &ProcessingConfig,
    ) -> std::result::Result<(), String> {
        // Check amount
        if payment.amount <= 0.0 {
            return Err("Invalid amount: must be positive".to_string());
        }

        // Check payer account
        let payer = accounts
            .get(&payment.payer_account)
            .ok_or_else(|| "Payer account not found".to_string())?;

        // Check account status
        if payer.status != AccountStatus::Active {
            return Err("Payer account is not active".to_string());
        }

        // Check sufficient balance
        if payer.available_balance < payment.amount {
            return Err("Insufficient funds".to_string());
        }

        // Check daily limit
        if let Some(limit) = payer.daily_limit {
            if payer.daily_used + payment.amount > limit {
                return Err("Daily limit exceeded".to_string());
            }
        }

        // Check payee account exists
        if !accounts.contains_key(&payment.payee_account) {
            return Err("Payee account not found".to_string());
        }

        // Check currency match
        if payer.currency != payment.currency {
            return Err("Currency mismatch".to_string());
        }

        // Check amount limits
        if let Some(min) = config.min_amount {
            if payment.amount < min {
                return Err(format!("Amount below minimum: {}", min));
            }
        }

        if let Some(max) = config.max_amount {
            if payment.amount > max {
                return Err(format!("Amount exceeds maximum: {}", max));
            }
        }

        Ok(())
    }

    /// Check if payment type is enabled.
    fn is_payment_type_enabled(payment_type: PaymentType, config: &ProcessingConfig) -> bool {
        config.enabled_payment_types.contains(&payment_type)
    }

    /// Check if payment is within processing window.
    fn is_within_processing_window(payment: &Payment, config: &ProcessingConfig) -> bool {
        // Real-time and internal payments always process
        if matches!(
            payment.payment_type,
            PaymentType::RealTime | PaymentType::Internal
        ) {
            return true;
        }

        // If process_outside_hours is true, always allow processing
        if config.process_outside_hours {
            return true;
        }

        // Get current time - use payment submission timestamp if available,
        // otherwise use current system time
        let timestamp = payment
            .attributes
            .get("submission_time")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0)
            });

        // Extract hour and day of week from timestamp
        // UTC-based calculation, adjusted by timezone offset if configured
        let adjusted_timestamp = timestamp as i64 + config.timezone_offset_seconds;
        let seconds_in_day = 86400i64;
        let seconds_since_epoch_start = adjusted_timestamp;

        // Unix epoch (1970-01-01) was a Thursday (day 4)
        // Calculate day of week (0 = Sunday, 6 = Saturday)
        let days_since_epoch = seconds_since_epoch_start / seconds_in_day;
        let day_of_week = ((days_since_epoch + 4) % 7) as u8;

        // Calculate hour of day (0-23)
        let seconds_into_day = (seconds_since_epoch_start % seconds_in_day) as u32;
        let hour = (seconds_into_day / 3600) as u8;

        // Check if it's a weekday (Monday=1 through Friday=5)
        let is_weekday = (1..=5).contains(&day_of_week);

        // Check if within business hours
        let is_business_hours =
            hour >= config.business_hours_start && hour < config.business_hours_end;

        // Must be a weekday within business hours
        is_weekday && is_business_hours
    }

    /// Run fraud checks on a payment.
    fn fraud_check(
        payment: &Payment,
        config: &ProcessingConfig,
    ) -> std::result::Result<(), String> {
        // Check velocity limits
        if let Some(velocity_limit) = config.velocity_limit {
            // In a real implementation, this would check recent transaction count
            if payment.amount > velocity_limit * 10.0 {
                return Err("Velocity check failed".to_string());
            }
        }

        // Check for suspicious patterns - large real-time payments need review
        if payment.payment_type == PaymentType::RealTime {
            if let Some(threshold) = config.large_payment_threshold {
                if payment.amount > threshold {
                    return Err("Large payment requires manual review".to_string());
                }
            }
        }

        Ok(())
    }

    /// Validate a payment without processing.
    pub fn validate_only(
        payment: &Payment,
        accounts: &HashMap<String, PaymentAccount>,
        config: &ProcessingConfig,
    ) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Run standard validation
        if let Err(msg) = Self::validate_payment(payment, accounts, config) {
            errors.push(ValidationError {
                code: "VALIDATION_ERROR".to_string(),
                message: msg,
                field: None,
            });
        }

        // Check for warnings
        if payment.amount > 5000.0 {
            warnings.push(ValidationWarning {
                code: "LARGE_AMOUNT".to_string(),
                message: "Payment amount is above reporting threshold".to_string(),
            });
        }

        if payment.payment_type == PaymentType::Check {
            warnings.push(ValidationWarning {
                code: "SLOW_PAYMENT".to_string(),
                message: "Check payments may take 3-5 business days".to_string(),
            });
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
        }
    }

    /// Get payment routing information.
    pub fn get_routing(payment: &Payment) -> PaymentRouting {
        let (rail, estimated_settlement) = match payment.payment_type {
            PaymentType::RealTime => ("RTP".to_string(), 0),
            PaymentType::Wire => (
                "FEDWIRE".to_string(),
                if payment.priority >= PaymentPriority::High {
                    0
                } else {
                    1
                },
            ),
            PaymentType::ACH => ("ACH".to_string(), 2),
            PaymentType::Internal => ("INTERNAL".to_string(), 0),
            PaymentType::Check => ("CHECK".to_string(), 5),
            PaymentType::Card => ("CARD_NETWORK".to_string(), 1),
        };

        let requires_approval =
            payment.amount > 10000.0 || payment.priority >= PaymentPriority::Urgent;

        PaymentRouting {
            payment_id: payment.id.clone(),
            rail,
            estimated_settlement_days: estimated_settlement,
            requires_approval,
            fees: Self::calculate_fees(payment),
        }
    }

    /// Calculate payment fees.
    fn calculate_fees(payment: &Payment) -> f64 {
        match payment.payment_type {
            PaymentType::RealTime => 0.50, // Flat fee
            PaymentType::Wire => {
                if payment.priority >= PaymentPriority::Urgent {
                    25.0
                } else {
                    15.0
                }
            }
            PaymentType::ACH => payment.amount * 0.001, // 0.1%
            PaymentType::Internal => 0.0,
            PaymentType::Check => 1.0,
            PaymentType::Card => payment.amount * 0.029 + 0.30, // 2.9% + $0.30
        }
    }

    /// Process payments by priority.
    pub fn process_by_priority(
        payments: &[Payment],
        accounts: &HashMap<String, PaymentAccount>,
        config: &ProcessingConfig,
    ) -> Vec<ProcessingResult> {
        // Group payments by priority
        let mut priority_groups: HashMap<PaymentPriority, Vec<&Payment>> = HashMap::new();
        for payment in payments {
            priority_groups
                .entry(payment.priority)
                .or_default()
                .push(payment);
        }

        // Process in priority order (highest first)
        let mut results = Vec::new();
        let priorities = [
            PaymentPriority::Urgent,
            PaymentPriority::High,
            PaymentPriority::Normal,
            PaymentPriority::Low,
        ];

        for priority in priorities {
            if let Some(group) = priority_groups.get(&priority) {
                let payments_vec: Vec<Payment> = group.iter().map(|p| (*p).clone()).collect();
                let result = Self::process_payments(&payments_vec, accounts, config);
                results.push(result);
            }
        }

        results
    }
}

impl GpuKernel for PaymentProcessing {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Helper Types
// ============================================================================

/// Payment processing outcome.
enum PaymentOutcome {
    Processed,
    Failed(String),
    Pending,
}

/// Payment processing configuration.
#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    /// Enabled payment types.
    pub enabled_payment_types: Vec<PaymentType>,
    /// Minimum payment amount.
    pub min_amount: Option<f64>,
    /// Maximum payment amount.
    pub max_amount: Option<f64>,
    /// Enable fraud checks.
    pub fraud_check_enabled: bool,
    /// Velocity limit (transactions per hour).
    pub velocity_limit: Option<f64>,
    /// Large payment threshold for manual review.
    pub large_payment_threshold: Option<f64>,
    /// Process outside business hours.
    pub process_outside_hours: bool,
    /// Batch mode (queue non-urgent payments).
    pub batch_mode: bool,
    /// Business hours start (hour of day, 0-23).
    pub business_hours_start: u8,
    /// Business hours end (hour of day, 0-23).
    pub business_hours_end: u8,
    /// Timezone offset in seconds from UTC (e.g., -18000 for EST/UTC-5).
    pub timezone_offset_seconds: i64,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            enabled_payment_types: vec![
                PaymentType::ACH,
                PaymentType::Wire,
                PaymentType::RealTime,
                PaymentType::Internal,
                PaymentType::Check,
                PaymentType::Card,
            ],
            min_amount: Some(0.01),
            max_amount: Some(1_000_000.0),
            fraud_check_enabled: true,
            velocity_limit: Some(100.0),
            large_payment_threshold: Some(50_000.0),
            process_outside_hours: true,
            batch_mode: false,
            business_hours_start: 9,    // 9 AM
            business_hours_end: 17,     // 5 PM
            timezone_offset_seconds: 0, // UTC by default
        }
    }
}

/// Payment routing information.
#[derive(Debug, Clone)]
pub struct PaymentRouting {
    /// Payment ID.
    pub payment_id: String,
    /// Payment rail.
    pub rail: String,
    /// Estimated settlement time in days.
    pub estimated_settlement_days: u32,
    /// Requires manual approval.
    pub requires_approval: bool,
    /// Calculated fees.
    pub fees: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_accounts() -> HashMap<String, PaymentAccount> {
        let mut accounts = HashMap::new();
        accounts.insert(
            "ACC001".to_string(),
            PaymentAccount {
                id: "ACC001".to_string(),
                account_type: AccountType::Checking,
                balance: 10000.0,
                available_balance: 9500.0,
                currency: "USD".to_string(),
                status: AccountStatus::Active,
                daily_limit: Some(25000.0),
                daily_used: 1000.0,
            },
        );
        accounts.insert(
            "ACC002".to_string(),
            PaymentAccount {
                id: "ACC002".to_string(),
                account_type: AccountType::Checking,
                balance: 5000.0,
                available_balance: 5000.0,
                currency: "USD".to_string(),
                status: AccountStatus::Active,
                daily_limit: None,
                daily_used: 0.0,
            },
        );
        accounts.insert(
            "ACC003".to_string(),
            PaymentAccount {
                id: "ACC003".to_string(),
                account_type: AccountType::Savings,
                balance: 20000.0,
                available_balance: 20000.0,
                currency: "USD".to_string(),
                status: AccountStatus::Frozen,
                daily_limit: None,
                daily_used: 0.0,
            },
        );
        accounts
    }

    fn create_test_payment(id: &str, payer: &str, payee: &str, amount: f64) -> Payment {
        Payment {
            id: id.to_string(),
            payer_account: payer.to_string(),
            payee_account: payee.to_string(),
            amount,
            currency: "USD".to_string(),
            payment_type: PaymentType::ACH,
            status: PaymentStatus::Initiated,
            initiated_at: 1000,
            completed_at: None,
            reference: format!("REF-{}", id),
            priority: PaymentPriority::Normal,
            attributes: HashMap::new(),
        }
    }

    #[test]
    fn test_process_valid_payment() {
        let accounts = create_test_accounts();
        let config = ProcessingConfig::default();

        let payments = vec![create_test_payment("P001", "ACC001", "ACC002", 100.0)];
        let result = PaymentProcessing::process_payments(&payments, &accounts, &config);

        assert_eq!(result.stats.total_count, 1);
        assert_eq!(result.stats.processed_count, 1);
        assert_eq!(result.stats.failed_count, 0);
        assert!(result.processed.contains(&"P001".to_string()));
    }

    #[test]
    fn test_insufficient_funds() {
        let accounts = create_test_accounts();
        let config = ProcessingConfig::default();

        let payments = vec![create_test_payment("P001", "ACC001", "ACC002", 15000.0)];
        let result = PaymentProcessing::process_payments(&payments, &accounts, &config);

        assert_eq!(result.stats.failed_count, 1);
        assert!(
            result
                .failed
                .iter()
                .any(|(id, reason)| { id == "P001" && reason.contains("Insufficient funds") })
        );
    }

    #[test]
    fn test_frozen_account() {
        let accounts = create_test_accounts();
        let config = ProcessingConfig::default();

        let payments = vec![create_test_payment("P001", "ACC003", "ACC002", 100.0)];
        let result = PaymentProcessing::process_payments(&payments, &accounts, &config);

        assert_eq!(result.stats.failed_count, 1);
        assert!(
            result
                .failed
                .iter()
                .any(|(id, reason)| { id == "P001" && reason.contains("not active") })
        );
    }

    #[test]
    fn test_daily_limit_exceeded() {
        let mut accounts = create_test_accounts();
        // Increase available balance to allow for the amount, so daily limit is the blocker
        accounts.get_mut("ACC001").unwrap().available_balance = 30000.0;
        accounts.get_mut("ACC001").unwrap().balance = 30000.0;
        let config = ProcessingConfig::default();

        // ACC001 has $25000 daily limit and $1000 already used, so $24001+ would exceed
        let payments = vec![create_test_payment("P001", "ACC001", "ACC002", 24500.0)];
        let result = PaymentProcessing::process_payments(&payments, &accounts, &config);

        assert_eq!(result.stats.failed_count, 1);
        assert!(
            result
                .failed
                .iter()
                .any(|(id, reason)| { id == "P001" && reason.contains("Daily limit") })
        );
    }

    #[test]
    fn test_account_not_found() {
        let accounts = create_test_accounts();
        let config = ProcessingConfig::default();

        let payments = vec![create_test_payment("P001", "ACC999", "ACC002", 100.0)];
        let result = PaymentProcessing::process_payments(&payments, &accounts, &config);

        assert_eq!(result.stats.failed_count, 1);
        assert!(
            result
                .failed
                .iter()
                .any(|(_, reason)| { reason.contains("not found") })
        );
    }

    #[test]
    fn test_validate_only() {
        let accounts = create_test_accounts();
        let config = ProcessingConfig::default();

        let payment = create_test_payment("P001", "ACC001", "ACC002", 100.0);
        let result = PaymentProcessing::validate_only(&payment, &accounts, &config);

        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validate_with_warnings() {
        let accounts = create_test_accounts();
        let config = ProcessingConfig::default();

        let payment = create_test_payment("P001", "ACC001", "ACC002", 6000.0);
        let result = PaymentProcessing::validate_only(&payment, &accounts, &config);

        assert!(result.is_valid);
        assert!(result.warnings.iter().any(|w| w.code == "LARGE_AMOUNT"));
    }

    #[test]
    fn test_payment_routing_realtime() {
        let mut payment = create_test_payment("P001", "ACC001", "ACC002", 100.0);
        payment.payment_type = PaymentType::RealTime;

        let routing = PaymentProcessing::get_routing(&payment);

        assert_eq!(routing.rail, "RTP");
        assert_eq!(routing.estimated_settlement_days, 0);
        assert_eq!(routing.fees, 0.50);
    }

    #[test]
    fn test_payment_routing_wire() {
        let mut payment = create_test_payment("P001", "ACC001", "ACC002", 10000.0);
        payment.payment_type = PaymentType::Wire;
        payment.priority = PaymentPriority::Normal;

        let routing = PaymentProcessing::get_routing(&payment);

        assert_eq!(routing.rail, "FEDWIRE");
        assert_eq!(routing.estimated_settlement_days, 1);
        assert_eq!(routing.fees, 15.0);
    }

    #[test]
    fn test_payment_routing_urgent_wire() {
        let mut payment = create_test_payment("P001", "ACC001", "ACC002", 10000.0);
        payment.payment_type = PaymentType::Wire;
        payment.priority = PaymentPriority::Urgent;

        let routing = PaymentProcessing::get_routing(&payment);

        assert_eq!(routing.estimated_settlement_days, 0);
        assert_eq!(routing.fees, 25.0);
        assert!(routing.requires_approval);
    }

    #[test]
    fn test_batch_payments() {
        let accounts = create_test_accounts();
        let config = ProcessingConfig::default();

        let payments = vec![
            create_test_payment("P001", "ACC001", "ACC002", 100.0),
            create_test_payment("P002", "ACC001", "ACC002", 200.0),
            create_test_payment("P003", "ACC001", "ACC002", 300.0),
        ];

        let result = PaymentProcessing::process_payments(&payments, &accounts, &config);

        assert_eq!(result.stats.total_count, 3);
        assert_eq!(result.stats.processed_count, 3);
        assert_eq!(result.stats.total_amount, 600.0);
    }

    #[test]
    fn test_process_by_priority() {
        let accounts = create_test_accounts();
        let config = ProcessingConfig::default();

        let mut p1 = create_test_payment("P001", "ACC001", "ACC002", 100.0);
        p1.priority = PaymentPriority::Low;

        let mut p2 = create_test_payment("P002", "ACC001", "ACC002", 200.0);
        p2.priority = PaymentPriority::Urgent;

        let mut p3 = create_test_payment("P003", "ACC001", "ACC002", 300.0);
        p3.priority = PaymentPriority::Normal;

        let payments = vec![p1, p2, p3];
        let results = PaymentProcessing::process_by_priority(&payments, &accounts, &config);

        // Should have results for 3 priority levels (Urgent, Normal, Low)
        assert_eq!(results.len(), 3);
        // First result should be Urgent
        assert_eq!(results[0].stats.total_count, 1);
        assert!(results[0].processed.contains(&"P002".to_string()));
    }

    #[test]
    fn test_fraud_check_large_payment() {
        let accounts = create_test_accounts();
        let mut config = ProcessingConfig::default();
        config.large_payment_threshold = Some(5000.0);
        config.max_amount = Some(100000.0);
        config.velocity_limit = Some(1000.0); // Increase to avoid velocity check triggering

        let mut payment = create_test_payment("P001", "ACC001", "ACC002", 8000.0);
        payment.payment_type = PaymentType::RealTime;

        let payments = vec![payment];
        let result = PaymentProcessing::process_payments(&payments, &accounts, &config);

        assert_eq!(result.stats.failed_count, 1);
        assert!(
            result
                .failed
                .iter()
                .any(|(_, reason)| { reason.contains("manual review") })
        );
    }

    #[test]
    fn test_payment_type_disabled() {
        let accounts = create_test_accounts();
        let mut config = ProcessingConfig::default();
        config.enabled_payment_types = vec![PaymentType::ACH]; // Only ACH enabled

        let mut payment = create_test_payment("P001", "ACC001", "ACC002", 100.0);
        payment.payment_type = PaymentType::Wire;

        let payments = vec![payment];
        let result = PaymentProcessing::process_payments(&payments, &accounts, &config);

        assert_eq!(result.stats.failed_count, 1);
        assert!(
            result
                .failed
                .iter()
                .any(|(_, reason)| { reason.contains("not enabled") })
        );
    }

    #[test]
    fn test_amount_limits() {
        let accounts = create_test_accounts();
        let mut config = ProcessingConfig::default();
        config.min_amount = Some(10.0);
        config.max_amount = Some(1000.0);

        // Test below minimum
        let p1 = create_test_payment("P001", "ACC001", "ACC002", 5.0);
        let result = PaymentProcessing::process_payments(&[p1], &accounts, &config);
        assert!(
            result
                .failed
                .iter()
                .any(|(_, r)| r.contains("below minimum"))
        );

        // Test above maximum
        let p2 = create_test_payment("P002", "ACC001", "ACC002", 2000.0);
        let result = PaymentProcessing::process_payments(&[p2], &accounts, &config);
        assert!(
            result
                .failed
                .iter()
                .any(|(_, r)| r.contains("exceeds maximum"))
        );
    }
}
