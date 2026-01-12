//! Journal transformation kernel.
//!
//! This module provides journal transformation for accounting:
//! - Transform journal entries between formats
//! - Validate entries
//! - Apply GL mappings

use crate::types::{
    ErrorSeverity, JournalEntry, JournalLine, JournalStatus, MappedAccount, MappingResult,
    TransformationResult, TransformationStats, ValidationError,
};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// Journal Transformation Kernel
// ============================================================================

/// Journal transformation kernel.
///
/// Transforms and validates journal entries.
#[derive(Debug, Clone)]
pub struct JournalTransformation {
    metadata: KernelMetadata,
}

impl Default for JournalTransformation {
    fn default() -> Self {
        Self::new()
    }
}

impl JournalTransformation {
    /// Create a new journal transformation kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("accounting/journal-transform", Domain::Accounting)
                .with_description("Journal entry transformation and GL mapping")
                .with_throughput(30_000)
                .with_latency_us(100.0),
        }
    }

    /// Transform journal entries using account mappings.
    pub fn transform(
        entries: &[JournalEntry],
        mapping: &MappingResult,
        config: &TransformConfig,
    ) -> TransformationResult {
        let mut transformed_entries = Vec::new();
        let mut errors = Vec::new();
        let mut total_debit = 0.0;
        let mut total_credit = 0.0;

        // Build mapping lookup
        let mapping_lookup: HashMap<String, Vec<&MappedAccount>> =
            mapping.mapped.iter().fold(HashMap::new(), |mut acc, m| {
                acc.entry(m.source_code.clone()).or_default().push(m);
                acc
            });

        for entry in entries {
            // Validate before transformation
            let entry_errors = Self::validate_entry(entry, config);
            if !entry_errors.is_empty() {
                if config.skip_invalid {
                    errors.extend(entry_errors);
                    continue;
                } else {
                    errors.extend(entry_errors);
                }
            }

            // Transform the entry
            match Self::transform_entry(entry, &mapping_lookup, config) {
                Ok(transformed) => {
                    for line in &transformed.lines {
                        total_debit += line.debit;
                        total_credit += line.credit;
                    }
                    transformed_entries.push(transformed);
                }
                Err(e) => {
                    errors.push(e);
                }
            }
        }

        let transformed_count = transformed_entries.len();
        let error_count = errors.len();

        TransformationResult {
            entries: transformed_entries,
            errors,
            stats: TransformationStats {
                total_entries: entries.len(),
                transformed_count,
                error_count,
                total_debit,
                total_credit,
            },
        }
    }

    /// Validate a journal entry.
    fn validate_entry(entry: &JournalEntry, config: &TransformConfig) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // Check for empty lines
        if entry.lines.is_empty() {
            errors.push(ValidationError {
                entry_id: entry.id,
                line_number: None,
                code: "EMPTY_ENTRY".to_string(),
                message: "Journal entry has no lines".to_string(),
                severity: ErrorSeverity::Error,
            });
        }

        // Check debit/credit balance
        let total_debit: f64 = entry.lines.iter().map(|l| l.debit).sum();
        let total_credit: f64 = entry.lines.iter().map(|l| l.credit).sum();

        if (total_debit - total_credit).abs() > config.balance_tolerance {
            errors.push(ValidationError {
                entry_id: entry.id,
                line_number: None,
                code: "UNBALANCED".to_string(),
                message: format!(
                    "Entry is unbalanced: debit={}, credit={}",
                    total_debit, total_credit
                ),
                severity: ErrorSeverity::Error,
            });
        }

        // Validate individual lines
        for line in &entry.lines {
            // Check for both debit and credit on same line
            if line.debit > 0.0 && line.credit > 0.0 {
                errors.push(ValidationError {
                    entry_id: entry.id,
                    line_number: Some(line.line_number),
                    code: "DUAL_SIDED".to_string(),
                    message: "Line has both debit and credit".to_string(),
                    severity: ErrorSeverity::Warning,
                });
            }

            // Check for zero amounts
            if line.debit == 0.0 && line.credit == 0.0 {
                errors.push(ValidationError {
                    entry_id: entry.id,
                    line_number: Some(line.line_number),
                    code: "ZERO_AMOUNT".to_string(),
                    message: "Line has zero amount".to_string(),
                    severity: ErrorSeverity::Warning,
                });
            }

            // Check for empty account code
            if line.account_code.is_empty() {
                errors.push(ValidationError {
                    entry_id: entry.id,
                    line_number: Some(line.line_number),
                    code: "EMPTY_ACCOUNT".to_string(),
                    message: "Line has empty account code".to_string(),
                    severity: ErrorSeverity::Error,
                });
            }
        }

        errors
    }

    /// Transform a single entry.
    fn transform_entry(
        entry: &JournalEntry,
        mapping_lookup: &HashMap<String, Vec<&MappedAccount>>,
        config: &TransformConfig,
    ) -> Result<JournalEntry, ValidationError> {
        let mut new_lines = Vec::new();
        let mut line_number = 1u32;

        for line in &entry.lines {
            let mappings = mapping_lookup.get(&line.account_code);

            match mappings {
                Some(mapped_accounts) => {
                    for mapped in mapped_accounts {
                        let new_line = JournalLine {
                            line_number,
                            account_code: mapped.target_code.clone(),
                            debit: line.debit * mapped.amount_ratio,
                            credit: line.credit * mapped.amount_ratio,
                            currency: line.currency.clone(),
                            entity_id: line.entity_id.clone(),
                            cost_center: line.cost_center.clone(),
                            description: line.description.clone(),
                        };
                        new_lines.push(new_line);
                        line_number += 1;
                    }
                }
                None => {
                    if config.preserve_unmapped {
                        // Keep original line
                        let mut preserved = line.clone();
                        preserved.line_number = line_number;
                        new_lines.push(preserved);
                        line_number += 1;
                    } else {
                        return Err(ValidationError {
                            entry_id: entry.id,
                            line_number: Some(line.line_number),
                            code: "UNMAPPED_ACCOUNT".to_string(),
                            message: format!("No mapping for account {}", line.account_code),
                            severity: ErrorSeverity::Error,
                        });
                    }
                }
            }
        }

        Ok(JournalEntry {
            id: entry.id,
            date: entry.date,
            posting_date: entry.posting_date,
            document_number: entry.document_number.clone(),
            lines: new_lines,
            status: entry.status,
            source_system: entry.source_system.clone(),
            description: entry.description.clone(),
        })
    }

    /// Aggregate entries by account.
    pub fn aggregate_by_account(entries: &[JournalEntry]) -> HashMap<String, AccountSummary> {
        let mut summaries: HashMap<String, AccountSummary> = HashMap::new();

        for entry in entries {
            for line in &entry.lines {
                let summary = summaries
                    .entry(line.account_code.clone())
                    .or_insert_with(|| AccountSummary {
                        account_code: line.account_code.clone(),
                        total_debit: 0.0,
                        total_credit: 0.0,
                        line_count: 0,
                        entry_count: 0,
                    });

                summary.total_debit += line.debit;
                summary.total_credit += line.credit;
                summary.line_count += 1;
            }
        }

        // Count distinct entries per account
        for entry in entries {
            let mut seen_accounts: std::collections::HashSet<&str> =
                std::collections::HashSet::new();
            for line in &entry.lines {
                if seen_accounts.insert(&line.account_code) {
                    if let Some(summary) = summaries.get_mut(&line.account_code) {
                        summary.entry_count += 1;
                    }
                }
            }
        }

        summaries
    }

    /// Group entries by period.
    pub fn group_by_period(
        entries: &[JournalEntry],
        period_type: PeriodType,
    ) -> HashMap<String, Vec<&JournalEntry>> {
        let mut groups: HashMap<String, Vec<&JournalEntry>> = HashMap::new();

        for entry in entries {
            let period_key = Self::get_period_key(entry.posting_date, period_type);
            groups.entry(period_key).or_default().push(entry);
        }

        groups
    }

    /// Get period key for a date.
    fn get_period_key(timestamp: u64, period_type: PeriodType) -> String {
        // Simplified period calculation
        let days = timestamp / 86400;
        match period_type {
            PeriodType::Daily => format!("D{}", days),
            PeriodType::Weekly => format!("W{}", days / 7),
            PeriodType::Monthly => format!("M{}", days / 30),
            PeriodType::Quarterly => format!("Q{}", days / 90),
            PeriodType::Yearly => format!("Y{}", days / 365),
        }
    }
}

impl GpuKernel for JournalTransformation {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Transformation configuration.
#[derive(Debug, Clone)]
pub struct TransformConfig {
    /// Skip invalid entries.
    pub skip_invalid: bool,
    /// Preserve unmapped accounts.
    pub preserve_unmapped: bool,
    /// Balance tolerance.
    pub balance_tolerance: f64,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            skip_invalid: false,
            preserve_unmapped: true,
            balance_tolerance: 0.01,
        }
    }
}

/// Account summary.
#[derive(Debug, Clone)]
pub struct AccountSummary {
    /// Account code.
    pub account_code: String,
    /// Total debit.
    pub total_debit: f64,
    /// Total credit.
    pub total_credit: f64,
    /// Line count.
    pub line_count: usize,
    /// Entry count.
    pub entry_count: usize,
}

/// Period type.
#[derive(Debug, Clone, Copy)]
pub enum PeriodType {
    /// Daily.
    Daily,
    /// Weekly.
    Weekly,
    /// Monthly.
    Monthly,
    /// Quarterly.
    Quarterly,
    /// Yearly.
    Yearly,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MappingStats;

    fn create_test_entry() -> JournalEntry {
        JournalEntry {
            id: 1,
            date: 1700000000,
            posting_date: 1700000000,
            document_number: "JE001".to_string(),
            lines: vec![
                JournalLine {
                    line_number: 1,
                    account_code: "1000".to_string(),
                    debit: 1000.0,
                    credit: 0.0,
                    currency: "USD".to_string(),
                    entity_id: "CORP".to_string(),
                    cost_center: None,
                    description: "Cash debit".to_string(),
                },
                JournalLine {
                    line_number: 2,
                    account_code: "4000".to_string(),
                    debit: 0.0,
                    credit: 1000.0,
                    currency: "USD".to_string(),
                    entity_id: "CORP".to_string(),
                    cost_center: None,
                    description: "Revenue credit".to_string(),
                },
            ],
            status: JournalStatus::Draft,
            source_system: "TEST".to_string(),
            description: "Test entry".to_string(),
        }
    }

    fn create_test_mapping() -> MappingResult {
        MappingResult {
            mapped: vec![
                MappedAccount {
                    source_code: "1000".to_string(),
                    target_code: "A1000".to_string(),
                    rule_id: "R1".to_string(),
                    amount_ratio: 1.0,
                },
                MappedAccount {
                    source_code: "4000".to_string(),
                    target_code: "R4000".to_string(),
                    rule_id: "R2".to_string(),
                    amount_ratio: 1.0,
                },
            ],
            unmapped: vec![],
            stats: MappingStats {
                total_accounts: 2,
                mapped_count: 2,
                unmapped_count: 0,
                rules_applied: 2,
                mapping_rate: 1.0,
            },
        }
    }

    #[test]
    fn test_journal_metadata() {
        let kernel = JournalTransformation::new();
        assert_eq!(kernel.metadata().id, "accounting/journal-transform");
        assert_eq!(kernel.metadata().domain, Domain::Accounting);
    }

    #[test]
    fn test_basic_transformation() {
        let entries = vec![create_test_entry()];
        let mapping = create_test_mapping();
        let config = TransformConfig::default();

        let result = JournalTransformation::transform(&entries, &mapping, &config);

        assert_eq!(result.stats.transformed_count, 1);
        assert!(result.errors.is_empty());
        assert_eq!(result.entries[0].lines[0].account_code, "A1000");
        assert_eq!(result.entries[0].lines[1].account_code, "R4000");
    }

    #[test]
    fn test_unbalanced_entry() {
        let mut entry = create_test_entry();
        entry.lines[0].debit = 1500.0; // Make unbalanced

        let errors = JournalTransformation::validate_entry(&entry, &TransformConfig::default());

        assert!(errors.iter().any(|e| e.code == "UNBALANCED"));
    }

    #[test]
    fn test_empty_entry() {
        let entry = JournalEntry {
            id: 1,
            date: 1700000000,
            posting_date: 1700000000,
            document_number: "JE001".to_string(),
            lines: vec![],
            status: JournalStatus::Draft,
            source_system: "TEST".to_string(),
            description: "Empty".to_string(),
        };

        let errors = JournalTransformation::validate_entry(&entry, &TransformConfig::default());

        assert!(errors.iter().any(|e| e.code == "EMPTY_ENTRY"));
    }

    #[test]
    fn test_preserve_unmapped() {
        let entries = vec![create_test_entry()];
        let mapping = MappingResult {
            mapped: vec![MappedAccount {
                source_code: "1000".to_string(),
                target_code: "A1000".to_string(),
                rule_id: "R1".to_string(),
                amount_ratio: 1.0,
            }],
            unmapped: vec!["4000".to_string()],
            stats: MappingStats {
                total_accounts: 2,
                mapped_count: 1,
                unmapped_count: 1,
                rules_applied: 1,
                mapping_rate: 0.5,
            },
        };

        let config = TransformConfig {
            preserve_unmapped: true,
            ..Default::default()
        };

        let result = JournalTransformation::transform(&entries, &mapping, &config);

        // Should preserve unmapped 4000 account
        assert_eq!(result.stats.transformed_count, 1);
        assert!(
            result.entries[0]
                .lines
                .iter()
                .any(|l| l.account_code == "4000")
        );
    }

    #[test]
    fn test_split_transformation() {
        let entries = vec![create_test_entry()];
        let mapping = MappingResult {
            mapped: vec![
                MappedAccount {
                    source_code: "1000".to_string(),
                    target_code: "A1001".to_string(),
                    rule_id: "R1".to_string(),
                    amount_ratio: 0.6,
                },
                MappedAccount {
                    source_code: "1000".to_string(),
                    target_code: "A1002".to_string(),
                    rule_id: "R1".to_string(),
                    amount_ratio: 0.4,
                },
                MappedAccount {
                    source_code: "4000".to_string(),
                    target_code: "R4000".to_string(),
                    rule_id: "R2".to_string(),
                    amount_ratio: 1.0,
                },
            ],
            unmapped: vec![],
            stats: MappingStats {
                total_accounts: 2,
                mapped_count: 2,
                unmapped_count: 0,
                rules_applied: 2,
                mapping_rate: 1.0,
            },
        };

        let result =
            JournalTransformation::transform(&entries, &mapping, &TransformConfig::default());

        // Should have 3 lines (1000 split to 2, 4000 to 1)
        assert_eq!(result.entries[0].lines.len(), 3);

        let a1001_line = result.entries[0]
            .lines
            .iter()
            .find(|l| l.account_code == "A1001")
            .unwrap();
        assert!((a1001_line.debit - 600.0).abs() < 0.01);
    }

    #[test]
    fn test_aggregate_by_account() {
        let entries = vec![create_test_entry(), create_test_entry()];

        let summaries = JournalTransformation::aggregate_by_account(&entries);

        let cash_summary = summaries.get("1000").unwrap();
        assert_eq!(cash_summary.total_debit, 2000.0);
        assert_eq!(cash_summary.line_count, 2);
        assert_eq!(cash_summary.entry_count, 2);
    }

    #[test]
    fn test_group_by_period() {
        let mut entry1 = create_test_entry();
        entry1.posting_date = 1700000000;

        let mut entry2 = create_test_entry();
        entry2.id = 2;
        entry2.posting_date = 1700000000 + 86400 * 35; // 35 days later

        let entries = vec![entry1, entry2];
        let groups = JournalTransformation::group_by_period(&entries, PeriodType::Monthly);

        // Should be in different monthly periods
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_skip_invalid() {
        let mut entry = create_test_entry();
        entry.lines[0].debit = 2000.0; // Unbalanced

        let entries = vec![entry];
        let mapping = create_test_mapping();

        let config = TransformConfig {
            skip_invalid: true,
            ..Default::default()
        };

        let result = JournalTransformation::transform(&entries, &mapping, &config);

        assert_eq!(result.stats.transformed_count, 0);
        assert!(!result.errors.is_empty());
    }
}
