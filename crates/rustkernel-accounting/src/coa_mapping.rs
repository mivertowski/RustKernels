//! Chart of accounts mapping kernel.
//!
//! This module provides chart of accounts mapping for accounting:
//! - Map accounts between different chart of accounts
//! - Apply transformation rules
//! - Handle entity-specific mappings

use crate::types::{
    Account, MappedAccount, MappingResult, MappingRule, MappingStats, MappingTransformation,
};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// Chart of Accounts Mapping Kernel
// ============================================================================

/// Chart of accounts mapping kernel.
///
/// Maps accounts from source to target chart of accounts.
#[derive(Debug, Clone)]
pub struct ChartOfAccountsMapping {
    metadata: KernelMetadata,
}

impl Default for ChartOfAccountsMapping {
    fn default() -> Self {
        Self::new()
    }
}

impl ChartOfAccountsMapping {
    /// Create a new chart of accounts mapping kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("accounting/coa-mapping", Domain::Accounting)
                .with_description("Entity-specific chart of accounts mapping")
                .with_throughput(50_000)
                .with_latency_us(50.0),
        }
    }

    /// Map accounts using rules.
    pub fn map_accounts(
        accounts: &[Account],
        rules: &[MappingRule],
        config: &MappingConfig,
    ) -> MappingResult {
        let mut mapped = Vec::new();
        let mut unmapped = Vec::new();
        let mut rules_applied = 0;
        let mut processed_count = 0;

        // Sort rules by priority
        let mut sorted_rules: Vec<_> = rules.iter().collect();
        sorted_rules.sort_by_key(|r| std::cmp::Reverse(r.priority));

        for account in accounts {
            if !account.is_active && !config.include_inactive {
                continue;
            }

            processed_count += 1;

            let mapping = Self::find_mapping(account, &sorted_rules, config);

            match mapping {
                Some(mappings) => {
                    rules_applied += 1;
                    mapped.extend(mappings);
                }
                None => {
                    if config.default_target.is_some() {
                        mapped.push(MappedAccount {
                            source_code: account.code.clone(),
                            target_code: config.default_target.clone().unwrap(),
                            rule_id: "default".to_string(),
                            amount_ratio: 1.0,
                        });
                    } else {
                        unmapped.push(account.code.clone());
                    }
                }
            }
        }

        let mapped_count = processed_count - unmapped.len();

        MappingResult {
            mapped,
            unmapped: unmapped.clone(),
            stats: MappingStats {
                total_accounts: processed_count,
                mapped_count,
                unmapped_count: unmapped.len(),
                rules_applied,
                mapping_rate: if processed_count > 0 {
                    mapped_count as f64 / processed_count as f64
                } else {
                    0.0
                },
            },
        }
    }

    /// Find mapping for an account.
    fn find_mapping(
        account: &Account,
        rules: &[&MappingRule],
        _config: &MappingConfig,
    ) -> Option<Vec<MappedAccount>> {
        for rule in rules {
            // Check entity filter
            if let Some(ref filter) = rule.entity_filter {
                if filter != &account.entity_id {
                    continue;
                }
            }

            // Check pattern match
            if Self::matches_pattern(&account.code, &rule.source_pattern) {
                let mappings = Self::apply_transformation(account, rule);
                return Some(mappings);
            }
        }
        None
    }

    /// Check if account code matches pattern.
    fn matches_pattern(code: &str, pattern: &str) -> bool {
        if pattern.contains('*') {
            // Simple wildcard matching
            let parts: Vec<&str> = pattern.split('*').collect();
            if parts.len() == 1 {
                return code == pattern;
            }

            let mut pos = 0;
            for (i, part) in parts.iter().enumerate() {
                if part.is_empty() {
                    continue;
                }
                if i == 0 {
                    // Must start with first part
                    if !code.starts_with(part) {
                        return false;
                    }
                    pos = part.len();
                } else if i == parts.len() - 1 {
                    // Must end with last part
                    if !code.ends_with(part) {
                        return false;
                    }
                } else {
                    // Must contain middle parts
                    if let Some(idx) = code[pos..].find(part) {
                        pos += idx + part.len();
                    } else {
                        return false;
                    }
                }
            }
            true
        } else {
            code == pattern
        }
    }

    /// Apply transformation to create mappings.
    fn apply_transformation(account: &Account, rule: &MappingRule) -> Vec<MappedAccount> {
        match &rule.transformation {
            MappingTransformation::Direct => {
                vec![MappedAccount {
                    source_code: account.code.clone(),
                    target_code: rule.target_code.clone(),
                    rule_id: rule.id.clone(),
                    amount_ratio: 1.0,
                }]
            }
            MappingTransformation::Split(splits) => splits
                .iter()
                .map(|(target, ratio)| MappedAccount {
                    source_code: account.code.clone(),
                    target_code: target.clone(),
                    rule_id: rule.id.clone(),
                    amount_ratio: *ratio,
                })
                .collect(),
            MappingTransformation::Aggregate => {
                vec![MappedAccount {
                    source_code: account.code.clone(),
                    target_code: rule.target_code.clone(),
                    rule_id: rule.id.clone(),
                    amount_ratio: 1.0,
                }]
            }
            MappingTransformation::Conditional {
                condition,
                if_true,
                if_false,
            } => {
                let target = if Self::evaluate_condition(account, condition) {
                    if_true.clone()
                } else {
                    if_false.clone()
                };
                vec![MappedAccount {
                    source_code: account.code.clone(),
                    target_code: target,
                    rule_id: rule.id.clone(),
                    amount_ratio: 1.0,
                }]
            }
        }
    }

    /// Evaluate a simple condition.
    fn evaluate_condition(account: &Account, condition: &str) -> bool {
        // Simple attribute-based conditions
        if let Some(stripped) = condition.strip_prefix("attr:") {
            let parts: Vec<&str> = stripped.splitn(2, '=').collect();
            if parts.len() == 2 {
                return account.attributes.get(parts[0]) == Some(&parts[1].to_string());
            }
        }

        // Account type conditions
        if let Some(type_str) = condition.strip_prefix("type:") {
            return match type_str {
                "asset" => account.account_type == crate::types::AccountType::Asset,
                "liability" => account.account_type == crate::types::AccountType::Liability,
                "equity" => account.account_type == crate::types::AccountType::Equity,
                "revenue" => account.account_type == crate::types::AccountType::Revenue,
                "expense" => account.account_type == crate::types::AccountType::Expense,
                _ => false,
            };
        }

        false
    }

    /// Validate mapping rules.
    pub fn validate_rules(rules: &[MappingRule]) -> Vec<RuleValidationError> {
        let mut errors = Vec::new();

        for rule in rules {
            // Check for empty patterns
            if rule.source_pattern.is_empty() {
                errors.push(RuleValidationError {
                    rule_id: rule.id.clone(),
                    message: "Source pattern is empty".to_string(),
                });
            }

            // Check for empty targets
            if rule.target_code.is_empty()
                && !matches!(rule.transformation, MappingTransformation::Split(_))
            {
                errors.push(RuleValidationError {
                    rule_id: rule.id.clone(),
                    message: "Target code is empty".to_string(),
                });
            }

            // Validate split ratios
            if let MappingTransformation::Split(splits) = &rule.transformation {
                let total: f64 = splits.iter().map(|(_, r)| r).sum();
                if (total - 1.0).abs() > 0.001 {
                    errors.push(RuleValidationError {
                        rule_id: rule.id.clone(),
                        message: format!("Split ratios sum to {}, expected 1.0", total),
                    });
                }
            }
        }

        errors
    }

    /// Build hierarchy from accounts.
    pub fn build_hierarchy(accounts: &[Account]) -> HashMap<String, Vec<String>> {
        let mut hierarchy: HashMap<String, Vec<String>> = HashMap::new();

        for account in accounts {
            if let Some(ref parent) = account.parent_code {
                hierarchy
                    .entry(parent.clone())
                    .or_default()
                    .push(account.code.clone());
            }
        }

        hierarchy
    }
}

impl GpuKernel for ChartOfAccountsMapping {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Mapping configuration.
#[derive(Debug, Clone, Default)]
pub struct MappingConfig {
    /// Include inactive accounts.
    pub include_inactive: bool,
    /// Default target for unmapped accounts.
    pub default_target: Option<String>,
    /// Strict mode (fail on unmapped).
    pub strict_mode: bool,
}

/// Rule validation error.
#[derive(Debug, Clone)]
pub struct RuleValidationError {
    /// Rule ID.
    pub rule_id: String,
    /// Error message.
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AccountType;

    fn create_test_accounts() -> Vec<Account> {
        vec![
            Account {
                code: "1000".to_string(),
                name: "Cash".to_string(),
                account_type: AccountType::Asset,
                parent_code: None,
                is_active: true,
                currency: "USD".to_string(),
                entity_id: "CORP".to_string(),
                attributes: HashMap::new(),
            },
            Account {
                code: "1100".to_string(),
                name: "Receivables".to_string(),
                account_type: AccountType::Asset,
                parent_code: Some("1000".to_string()),
                is_active: true,
                currency: "USD".to_string(),
                entity_id: "CORP".to_string(),
                attributes: HashMap::new(),
            },
            Account {
                code: "2000".to_string(),
                name: "Payables".to_string(),
                account_type: AccountType::Liability,
                parent_code: None,
                is_active: true,
                currency: "USD".to_string(),
                entity_id: "CORP".to_string(),
                attributes: HashMap::new(),
            },
        ]
    }

    fn create_test_rules() -> Vec<MappingRule> {
        vec![
            MappingRule {
                id: "R1".to_string(),
                source_pattern: "1*".to_string(),
                target_code: "A1000".to_string(),
                entity_filter: None,
                priority: 10,
                transformation: MappingTransformation::Direct,
            },
            MappingRule {
                id: "R2".to_string(),
                source_pattern: "2000".to_string(),
                target_code: "L2000".to_string(),
                entity_filter: None,
                priority: 5,
                transformation: MappingTransformation::Direct,
            },
        ]
    }

    #[test]
    fn test_coa_metadata() {
        let kernel = ChartOfAccountsMapping::new();
        assert_eq!(kernel.metadata().id, "accounting/coa-mapping");
        assert_eq!(kernel.metadata().domain, Domain::Accounting);
    }

    #[test]
    fn test_basic_mapping() {
        let accounts = create_test_accounts();
        let rules = create_test_rules();
        let config = MappingConfig::default();

        let result = ChartOfAccountsMapping::map_accounts(&accounts, &rules, &config);

        assert_eq!(result.stats.total_accounts, 3);
        assert_eq!(result.stats.mapped_count, 3);
        assert!(result.unmapped.is_empty());
    }

    #[test]
    fn test_wildcard_matching() {
        assert!(ChartOfAccountsMapping::matches_pattern("1000", "1*"));
        assert!(ChartOfAccountsMapping::matches_pattern("1100", "1*"));
        assert!(!ChartOfAccountsMapping::matches_pattern("2000", "1*"));
        assert!(ChartOfAccountsMapping::matches_pattern("ABC123", "*123"));
        assert!(ChartOfAccountsMapping::matches_pattern("TEST", "*"));
    }

    #[test]
    fn test_split_transformation() {
        let accounts = vec![Account {
            code: "5000".to_string(),
            name: "Mixed Expense".to_string(),
            account_type: AccountType::Expense,
            parent_code: None,
            is_active: true,
            currency: "USD".to_string(),
            entity_id: "CORP".to_string(),
            attributes: HashMap::new(),
        }];

        let rules = vec![MappingRule {
            id: "R1".to_string(),
            source_pattern: "5000".to_string(),
            target_code: String::new(),
            entity_filter: None,
            priority: 10,
            transformation: MappingTransformation::Split(vec![
                ("E5001".to_string(), 0.6),
                ("E5002".to_string(), 0.4),
            ]),
        }];

        let result =
            ChartOfAccountsMapping::map_accounts(&accounts, &rules, &MappingConfig::default());

        assert_eq!(result.mapped.len(), 2);
        assert!((result.mapped[0].amount_ratio - 0.6).abs() < 0.001);
        assert!((result.mapped[1].amount_ratio - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_entity_filter() {
        let accounts = vec![
            Account {
                code: "1000".to_string(),
                name: "Cash".to_string(),
                account_type: AccountType::Asset,
                parent_code: None,
                is_active: true,
                currency: "USD".to_string(),
                entity_id: "CORP_A".to_string(),
                attributes: HashMap::new(),
            },
            Account {
                code: "1000".to_string(),
                name: "Cash".to_string(),
                account_type: AccountType::Asset,
                parent_code: None,
                is_active: true,
                currency: "USD".to_string(),
                entity_id: "CORP_B".to_string(),
                attributes: HashMap::new(),
            },
        ];

        let rules = vec![MappingRule {
            id: "R1".to_string(),
            source_pattern: "1000".to_string(),
            target_code: "A1000".to_string(),
            entity_filter: Some("CORP_A".to_string()),
            priority: 10,
            transformation: MappingTransformation::Direct,
        }];

        let result =
            ChartOfAccountsMapping::map_accounts(&accounts, &rules, &MappingConfig::default());

        assert_eq!(result.stats.mapped_count, 1);
        assert_eq!(result.unmapped.len(), 1);
    }

    #[test]
    fn test_default_target() {
        let accounts = create_test_accounts();
        let rules: Vec<MappingRule> = vec![]; // No rules

        let config = MappingConfig {
            default_target: Some("UNMAPPED".to_string()),
            ..Default::default()
        };

        let result = ChartOfAccountsMapping::map_accounts(&accounts, &rules, &config);

        assert!(result.unmapped.is_empty());
        assert!(result.mapped.iter().all(|m| m.target_code == "UNMAPPED"));
    }

    #[test]
    fn test_validate_rules() {
        let rules = vec![
            MappingRule {
                id: "EMPTY".to_string(),
                source_pattern: "".to_string(),
                target_code: "T1".to_string(),
                entity_filter: None,
                priority: 1,
                transformation: MappingTransformation::Direct,
            },
            MappingRule {
                id: "BAD_SPLIT".to_string(),
                source_pattern: "1*".to_string(),
                target_code: String::new(),
                entity_filter: None,
                priority: 1,
                transformation: MappingTransformation::Split(vec![
                    ("T1".to_string(), 0.3),
                    ("T2".to_string(), 0.3),
                ]),
            },
        ];

        let errors = ChartOfAccountsMapping::validate_rules(&rules);

        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_build_hierarchy() {
        let accounts = create_test_accounts();
        let hierarchy = ChartOfAccountsMapping::build_hierarchy(&accounts);

        assert!(hierarchy.contains_key("1000"));
        assert_eq!(hierarchy.get("1000").unwrap().len(), 1);
        assert!(hierarchy.get("1000").unwrap().contains(&"1100".to_string()));
    }

    #[test]
    fn test_inactive_accounts() {
        let accounts = vec![Account {
            code: "1000".to_string(),
            name: "Inactive".to_string(),
            account_type: AccountType::Asset,
            parent_code: None,
            is_active: false,
            currency: "USD".to_string(),
            entity_id: "CORP".to_string(),
            attributes: HashMap::new(),
        }];

        let rules = create_test_rules();

        // Default: exclude inactive
        let result1 =
            ChartOfAccountsMapping::map_accounts(&accounts, &rules, &MappingConfig::default());
        assert_eq!(result1.stats.total_accounts, 0);

        // Include inactive
        let config = MappingConfig {
            include_inactive: true,
            ..Default::default()
        };
        let result2 = ChartOfAccountsMapping::map_accounts(&accounts, &rules, &config);
        assert_eq!(result2.stats.total_accounts, 1);
    }
}
