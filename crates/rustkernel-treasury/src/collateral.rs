//! Collateral optimization kernel.
//!
//! This module provides collateral optimization for treasury:
//! - Optimal allocation of collateral to requirements
//! - Cheapest-to-deliver optimization
//! - Eligibility and haircut handling

use crate::types::{
    AssetType, CollateralAllocation, CollateralAsset, CollateralOptimizationResult,
    CollateralRequirement,
};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// Collateral Optimization Kernel
// ============================================================================

/// Collateral optimization kernel.
///
/// Optimizes allocation of collateral assets to requirements.
#[derive(Debug, Clone)]
pub struct CollateralOptimization {
    metadata: KernelMetadata,
}

impl Default for CollateralOptimization {
    fn default() -> Self {
        Self::new()
    }
}

impl CollateralOptimization {
    /// Create a new collateral optimization kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("treasury/collateral-opt", Domain::TreasuryManagement)
                .with_description("Collateral allocation optimization")
                .with_throughput(5_000)
                .with_latency_us(1000.0),
        }
    }

    /// Optimize collateral allocation.
    pub fn optimize(
        assets: &[CollateralAsset],
        requirements: &[CollateralRequirement],
        config: &CollateralConfig,
    ) -> CollateralOptimizationResult {
        // Sort requirements by priority (highest first)
        let mut sorted_reqs: Vec<_> = requirements.iter().collect();
        sorted_reqs.sort_by_key(|r| std::cmp::Reverse(r.priority));

        // Track available collateral
        let mut available: HashMap<String, f64> = assets
            .iter()
            .filter(|a| !a.is_pledged || config.allow_rehypothecation)
            .map(|a| (a.id.clone(), a.eligible_value))
            .collect();

        let mut allocations = Vec::new();
        let mut total_allocated = 0.0;
        let mut shortfall = 0.0;

        for req in sorted_reqs {
            let mut remaining = req.required_amount;

            // Find eligible assets for this requirement
            let mut eligible_assets: Vec<_> = assets
                .iter()
                .filter(|a| {
                    req.eligible_types.contains(&a.asset_type)
                        && a.currency == req.currency
                        && available.get(&a.id).copied().unwrap_or(0.0) > 0.0
                })
                .collect();

            // Sort by optimization strategy
            match config.strategy {
                OptimizationStrategy::CheapestToDeliver => {
                    // Prefer assets with lower quality (highest haircut)
                    eligible_assets.sort_by(|a, b| {
                        b.haircut.partial_cmp(&a.haircut).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                OptimizationStrategy::HighestQuality => {
                    // Prefer assets with highest quality (lowest haircut)
                    eligible_assets.sort_by(|a, b| {
                        a.haircut.partial_cmp(&b.haircut).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                OptimizationStrategy::LargestFirst => {
                    // Prefer largest eligible values
                    eligible_assets.sort_by(|a, b| {
                        b.eligible_value.partial_cmp(&a.eligible_value).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }

            // Allocate collateral
            for asset in eligible_assets {
                if remaining <= 0.0 {
                    break;
                }

                let available_amount = available.get(&asset.id).copied().unwrap_or(0.0);
                if available_amount <= 0.0 {
                    continue;
                }

                let allocate_value = available_amount.min(remaining);
                let allocate_quantity = allocate_value / (1.0 - asset.haircut) / (asset.market_value / asset.quantity);

                allocations.push(CollateralAllocation {
                    asset_id: asset.id.clone(),
                    counterparty_id: req.counterparty_id.clone(),
                    quantity: allocate_quantity,
                    value: allocate_value,
                });

                *available.get_mut(&asset.id).unwrap() -= allocate_value;
                remaining -= allocate_value;
                total_allocated += allocate_value;
            }

            if remaining > 0.0 {
                shortfall += remaining;
            }
        }

        // Calculate excess
        let total_required: f64 = requirements.iter().map(|r| r.required_amount).sum();
        let excess = if total_allocated > total_required {
            total_allocated - total_required
        } else {
            0.0
        };

        // Calculate optimization score
        let score = Self::calculate_score(&allocations, assets, requirements, config);

        CollateralOptimizationResult {
            allocations,
            total_allocated,
            excess,
            shortfall,
            score,
        }
    }

    /// Calculate optimization score.
    fn calculate_score(
        allocations: &[CollateralAllocation],
        assets: &[CollateralAsset],
        requirements: &[CollateralRequirement],
        config: &CollateralConfig,
    ) -> f64 {
        if requirements.is_empty() {
            return 1.0;
        }

        let total_required: f64 = requirements.iter().map(|r| r.required_amount).sum();
        if total_required == 0.0 {
            return 1.0;
        }

        // Base score: coverage ratio
        let total_allocated: f64 = allocations.iter().map(|a| a.value).sum();
        let coverage = (total_allocated / total_required).min(1.0);

        // Quality factor: prefer using lower-quality assets if CTD strategy
        let quality_factor = if config.strategy == OptimizationStrategy::CheapestToDeliver {
            let avg_haircut: f64 = allocations
                .iter()
                .filter_map(|alloc| {
                    assets.iter().find(|a| a.id == alloc.asset_id).map(|a| a.haircut)
                })
                .sum::<f64>()
                / allocations.len().max(1) as f64;
            avg_haircut // Higher haircut = better for CTD
        } else {
            1.0 - allocations
                .iter()
                .filter_map(|alloc| {
                    assets.iter().find(|a| a.id == alloc.asset_id).map(|a| a.haircut)
                })
                .sum::<f64>()
                / allocations.len().max(1) as f64
        };

        // Concentration factor: penalize over-concentration
        let mut by_counterparty: HashMap<String, f64> = HashMap::new();
        for alloc in allocations {
            *by_counterparty.entry(alloc.counterparty_id.clone()).or_default() += alloc.value;
        }
        let concentration = if by_counterparty.len() > 1 {
            1.0 - Self::herfindahl_index(&by_counterparty.values().copied().collect::<Vec<_>>())
        } else {
            0.5
        };

        // Weighted score
        coverage * 0.6 + quality_factor * 0.25 + concentration * 0.15
    }

    /// Calculate Herfindahl index (concentration measure).
    fn herfindahl_index(values: &[f64]) -> f64 {
        let total: f64 = values.iter().sum();
        if total == 0.0 {
            return 0.0;
        }
        values.iter().map(|v| (v / total).powi(2)).sum()
    }

    /// Calculate total eligible collateral by asset type.
    pub fn eligible_by_type(assets: &[CollateralAsset]) -> HashMap<AssetType, f64> {
        let mut by_type: HashMap<AssetType, f64> = HashMap::new();
        for asset in assets {
            if !asset.is_pledged {
                *by_type.entry(asset.asset_type).or_default() += asset.eligible_value;
            }
        }
        by_type
    }

    /// Calculate utilization metrics.
    pub fn utilization_metrics(
        assets: &[CollateralAsset],
        allocations: &[CollateralAllocation],
    ) -> UtilizationMetrics {
        let total_available: f64 = assets.iter().map(|a| a.eligible_value).sum();
        let total_allocated: f64 = allocations.iter().map(|a| a.value).sum();

        let pledged_count = assets.iter().filter(|a| a.is_pledged).count();
        let pledged_value: f64 = assets
            .iter()
            .filter(|a| a.is_pledged)
            .map(|a| a.market_value)
            .sum();

        UtilizationMetrics {
            total_available,
            total_allocated,
            utilization_rate: if total_available > 0.0 {
                total_allocated / total_available
            } else {
                0.0
            },
            pledged_count: pledged_count as u64,
            pledged_value,
            free_collateral: total_available - total_allocated,
        }
    }
}

impl GpuKernel for CollateralOptimization {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Collateral optimization configuration.
#[derive(Debug, Clone)]
pub struct CollateralConfig {
    /// Optimization strategy.
    pub strategy: OptimizationStrategy,
    /// Allow rehypothecation of pledged assets.
    pub allow_rehypothecation: bool,
    /// Minimum allocation amount.
    pub min_allocation: f64,
    /// Maximum concentration per counterparty (0-1).
    pub max_concentration: f64,
}

impl Default for CollateralConfig {
    fn default() -> Self {
        Self {
            strategy: OptimizationStrategy::CheapestToDeliver,
            allow_rehypothecation: false,
            min_allocation: 0.0,
            max_concentration: 1.0,
        }
    }
}

/// Optimization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationStrategy {
    /// Use cheapest (highest haircut) assets first.
    CheapestToDeliver,
    /// Use highest quality (lowest haircut) assets first.
    HighestQuality,
    /// Use largest assets first.
    LargestFirst,
}

/// Collateral utilization metrics.
#[derive(Debug, Clone)]
pub struct UtilizationMetrics {
    /// Total available eligible collateral.
    pub total_available: f64,
    /// Total allocated collateral.
    pub total_allocated: f64,
    /// Utilization rate (0-1).
    pub utilization_rate: f64,
    /// Number of pledged assets.
    pub pledged_count: u64,
    /// Total pledged value.
    pub pledged_value: f64,
    /// Free collateral.
    pub free_collateral: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_assets() -> Vec<CollateralAsset> {
        vec![
            CollateralAsset {
                id: "CASH_001".to_string(),
                asset_type: AssetType::Cash,
                quantity: 1_000_000.0,
                market_value: 1_000_000.0,
                haircut: 0.0,
                eligible_value: 1_000_000.0,
                currency: "USD".to_string(),
                is_pledged: false,
                pledged_to: None,
            },
            CollateralAsset {
                id: "GOV_001".to_string(),
                asset_type: AssetType::GovBond,
                quantity: 100.0,
                market_value: 500_000.0,
                haircut: 0.02,
                eligible_value: 490_000.0,
                currency: "USD".to_string(),
                is_pledged: false,
                pledged_to: None,
            },
            CollateralAsset {
                id: "CORP_001".to_string(),
                asset_type: AssetType::CorpBond,
                quantity: 200.0,
                market_value: 400_000.0,
                haircut: 0.10,
                eligible_value: 360_000.0,
                currency: "USD".to_string(),
                is_pledged: false,
                pledged_to: None,
            },
        ]
    }

    fn create_test_requirements() -> Vec<CollateralRequirement> {
        vec![
            CollateralRequirement {
                counterparty_id: "CP_A".to_string(),
                required_amount: 300_000.0,
                currency: "USD".to_string(),
                eligible_types: vec![AssetType::Cash, AssetType::GovBond],
                priority: 1,
            },
            CollateralRequirement {
                counterparty_id: "CP_B".to_string(),
                required_amount: 200_000.0,
                currency: "USD".to_string(),
                eligible_types: vec![AssetType::Cash, AssetType::GovBond, AssetType::CorpBond],
                priority: 2,
            },
        ]
    }

    #[test]
    fn test_collateral_metadata() {
        let kernel = CollateralOptimization::new();
        assert_eq!(kernel.metadata().id, "treasury/collateral-opt");
        assert_eq!(kernel.metadata().domain, Domain::TreasuryManagement);
    }

    #[test]
    fn test_basic_optimization() {
        let assets = create_test_assets();
        let requirements = create_test_requirements();
        let config = CollateralConfig::default();

        let result = CollateralOptimization::optimize(&assets, &requirements, &config);

        assert!(result.shortfall < 0.01);
        assert!(result.total_allocated >= 500_000.0);
        assert!(!result.allocations.is_empty());
    }

    #[test]
    fn test_cheapest_to_deliver() {
        let assets = create_test_assets();
        let requirements = vec![CollateralRequirement {
            counterparty_id: "CP_A".to_string(),
            required_amount: 300_000.0,
            currency: "USD".to_string(),
            eligible_types: vec![AssetType::Cash, AssetType::GovBond, AssetType::CorpBond],
            priority: 1,
        }];

        let config = CollateralConfig {
            strategy: OptimizationStrategy::CheapestToDeliver,
            ..Default::default()
        };

        let result = CollateralOptimization::optimize(&assets, &requirements, &config);

        // Should prefer corporate bonds (highest haircut) first
        let first_alloc = &result.allocations[0];
        assert_eq!(first_alloc.asset_id, "CORP_001");
    }

    #[test]
    fn test_highest_quality() {
        let assets = create_test_assets();
        let requirements = vec![CollateralRequirement {
            counterparty_id: "CP_A".to_string(),
            required_amount: 300_000.0,
            currency: "USD".to_string(),
            eligible_types: vec![AssetType::Cash, AssetType::GovBond, AssetType::CorpBond],
            priority: 1,
        }];

        let config = CollateralConfig {
            strategy: OptimizationStrategy::HighestQuality,
            ..Default::default()
        };

        let result = CollateralOptimization::optimize(&assets, &requirements, &config);

        // Should prefer cash (lowest haircut) first
        let first_alloc = &result.allocations[0];
        assert_eq!(first_alloc.asset_id, "CASH_001");
    }

    #[test]
    fn test_shortfall() {
        let assets = create_test_assets();
        let requirements = vec![CollateralRequirement {
            counterparty_id: "CP_A".to_string(),
            required_amount: 5_000_000.0, // More than available
            currency: "USD".to_string(),
            eligible_types: vec![AssetType::Cash, AssetType::GovBond, AssetType::CorpBond],
            priority: 1,
        }];

        let config = CollateralConfig::default();
        let result = CollateralOptimization::optimize(&assets, &requirements, &config);

        assert!(result.shortfall > 0.0);
    }

    #[test]
    fn test_currency_filtering() {
        let assets = create_test_assets();
        let requirements = vec![CollateralRequirement {
            counterparty_id: "CP_A".to_string(),
            required_amount: 300_000.0,
            currency: "EUR".to_string(), // Different currency
            eligible_types: vec![AssetType::Cash, AssetType::GovBond],
            priority: 1,
        }];

        let config = CollateralConfig::default();
        let result = CollateralOptimization::optimize(&assets, &requirements, &config);

        // No assets match EUR, so shortfall should equal requirement
        assert!((result.shortfall - 300_000.0).abs() < 0.01);
    }

    #[test]
    fn test_pledged_assets() {
        let mut assets = create_test_assets();
        assets[0].is_pledged = true; // Cash is pledged

        let requirements = vec![CollateralRequirement {
            counterparty_id: "CP_A".to_string(),
            required_amount: 1_500_000.0,
            currency: "USD".to_string(),
            eligible_types: vec![AssetType::Cash, AssetType::GovBond, AssetType::CorpBond],
            priority: 1,
        }];

        let config = CollateralConfig {
            allow_rehypothecation: false,
            ..Default::default()
        };

        let result = CollateralOptimization::optimize(&assets, &requirements, &config);

        // Should not use pledged cash
        assert!(!result.allocations.iter().any(|a| a.asset_id == "CASH_001"));
    }

    #[test]
    fn test_eligible_by_type() {
        let assets = create_test_assets();
        let by_type = CollateralOptimization::eligible_by_type(&assets);

        assert_eq!(by_type.get(&AssetType::Cash), Some(&1_000_000.0));
        assert_eq!(by_type.get(&AssetType::GovBond), Some(&490_000.0));
        assert_eq!(by_type.get(&AssetType::CorpBond), Some(&360_000.0));
    }

    #[test]
    fn test_utilization_metrics() {
        let assets = create_test_assets();
        let allocations = vec![
            CollateralAllocation {
                asset_id: "CASH_001".to_string(),
                counterparty_id: "CP_A".to_string(),
                quantity: 500_000.0,
                value: 500_000.0,
            },
        ];

        let metrics = CollateralOptimization::utilization_metrics(&assets, &allocations);

        assert_eq!(metrics.total_available, 1_850_000.0);
        assert_eq!(metrics.total_allocated, 500_000.0);
        assert!((metrics.utilization_rate - 500_000.0 / 1_850_000.0).abs() < 0.001);
    }

    #[test]
    fn test_priority_ordering() {
        let assets = vec![CollateralAsset {
            id: "CASH_001".to_string(),
            asset_type: AssetType::Cash,
            quantity: 500_000.0,
            market_value: 500_000.0,
            haircut: 0.0,
            eligible_value: 500_000.0,
            currency: "USD".to_string(),
            is_pledged: false,
            pledged_to: None,
        }];

        let requirements = vec![
            CollateralRequirement {
                counterparty_id: "LOW".to_string(),
                required_amount: 300_000.0,
                currency: "USD".to_string(),
                eligible_types: vec![AssetType::Cash],
                priority: 1, // Lower priority
            },
            CollateralRequirement {
                counterparty_id: "HIGH".to_string(),
                required_amount: 300_000.0,
                currency: "USD".to_string(),
                eligible_types: vec![AssetType::Cash],
                priority: 10, // Higher priority
            },
        ];

        let config = CollateralConfig::default();
        let result = CollateralOptimization::optimize(&assets, &requirements, &config);

        // Higher priority should be fully allocated
        let high_alloc: f64 = result
            .allocations
            .iter()
            .filter(|a| a.counterparty_id == "HIGH")
            .map(|a| a.value)
            .sum();
        assert!((high_alloc - 300_000.0).abs() < 0.01);
    }
}
