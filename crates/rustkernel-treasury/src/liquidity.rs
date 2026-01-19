//! Liquidity optimization kernel.
//!
//! This module provides liquidity optimization for treasury:
//! - LCR (Liquidity Coverage Ratio) calculation
//! - NSFR (Net Stable Funding Ratio) calculation
//! - Liquidity optimization recommendations

use crate::types::{
    LCRResult, LiquidityAction, LiquidityActionType, LiquidityAssetType,
    LiquidityOptimizationResult, LiquidityOutflow, LiquidityPosition, NSFRResult, OutflowCategory,
};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// Liquidity Optimization Kernel
// ============================================================================

/// Liquidity optimization kernel.
///
/// Calculates LCR/NSFR ratios and recommends optimization actions.
#[derive(Debug, Clone)]
pub struct LiquidityOptimization {
    metadata: KernelMetadata,
}

impl Default for LiquidityOptimization {
    fn default() -> Self {
        Self::new()
    }
}

impl LiquidityOptimization {
    /// Create a new liquidity optimization kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("treasury/liquidity-opt", Domain::TreasuryManagement)
                .with_description("Liquidity ratio optimization (LCR/NSFR)")
                .with_throughput(5_000)
                .with_latency_us(1000.0),
        }
    }

    /// Calculate LCR (Liquidity Coverage Ratio).
    pub fn calculate_lcr(
        assets: &[LiquidityPosition],
        outflows: &[LiquidityOutflow],
        inflows: &[LiquidityInflow],
        config: &LCRConfig,
    ) -> LCRResult {
        // Calculate HQLA by level
        let mut hqla_breakdown: HashMap<String, f64> = HashMap::new();
        let mut level1 = 0.0;
        let mut level2a = 0.0;
        let mut level2b = 0.0;

        for asset in assets {
            let haircut_value = asset.amount * (1.0 - asset.lcr_haircut);

            match asset.asset_type {
                LiquidityAssetType::CashReserves | LiquidityAssetType::Level1HQLA => {
                    level1 += haircut_value;
                }
                LiquidityAssetType::Level2AHQLA => {
                    level2a += haircut_value;
                }
                LiquidityAssetType::Level2BHQLA => {
                    level2b += haircut_value;
                }
                _ => {}
            }
        }

        // Apply caps (Basel III compliant)
        // Level 2A cap: L2A <= cap% of total HQLA
        // This means L2A <= cap% * (L1 + L2A), solving: L2A <= L1 * cap / (1 - cap)
        let max_l2a_from_cap = if config.level2a_cap < 1.0 {
            level1 * config.level2a_cap / (1.0 - config.level2a_cap)
        } else {
            f64::INFINITY
        };
        let capped_level2a = level2a.min(max_l2a_from_cap);

        // Level 2B cap: L2B <= cap% of total HQLA (considering L1 + L2A + L2B)
        let max_l2b_from_cap = if config.level2b_cap < 1.0 {
            (level1 + capped_level2a) * config.level2b_cap / (1.0 - config.level2b_cap)
        } else {
            f64::INFINITY
        };
        let capped_level2b = level2b.min(max_l2b_from_cap);

        // Total Level 2 cap
        let max_total_l2_from_cap = if config.level2_total_cap < 1.0 {
            level1 * config.level2_total_cap / (1.0 - config.level2_total_cap)
        } else {
            f64::INFINITY
        };
        let total_level2 = (capped_level2a + capped_level2b).min(max_total_l2_from_cap);

        let hqla = level1 + total_level2;

        hqla_breakdown.insert("Level1".to_string(), level1);
        hqla_breakdown.insert("Level2A".to_string(), capped_level2a);
        hqla_breakdown.insert("Level2B".to_string(), capped_level2b);
        hqla_breakdown.insert("Total".to_string(), hqla);

        // Calculate net outflows
        let gross_outflows: f64 = outflows
            .iter()
            .filter(|o| o.days_to_maturity <= 30)
            .map(|o| o.amount * o.runoff_rate)
            .sum();

        let gross_inflows: f64 = inflows
            .iter()
            .filter(|i| i.days_to_maturity <= 30)
            .map(|i| i.amount * i.inflow_rate)
            .sum();

        // Inflow cap: max 75% of outflows
        let capped_inflows = gross_inflows.min(gross_outflows * config.inflow_cap);
        let net_outflows = (gross_outflows - capped_inflows).max(0.0);

        // Calculate LCR
        let lcr_ratio = if net_outflows > 0.0 {
            hqla / net_outflows
        } else {
            f64::INFINITY
        };

        let is_compliant = lcr_ratio >= config.min_lcr;
        // Buffer = excess HQLA above minimum requirement (negative if non-compliant)
        let buffer = hqla - (net_outflows * config.min_lcr);

        LCRResult {
            hqla,
            net_outflows,
            lcr_ratio,
            is_compliant,
            buffer,
            hqla_breakdown,
        }
    }

    /// Calculate NSFR (Net Stable Funding Ratio).
    pub fn calculate_nsfr(
        assets: &[LiquidityPosition],
        funding: &[FundingSource],
        config: &NSFRConfig,
    ) -> NSFRResult {
        // Calculate Available Stable Funding (ASF)
        let asf: f64 = funding
            .iter()
            .map(|f| f.amount * Self::get_asf_factor(f, config))
            .sum();

        // Calculate Required Stable Funding (RSF)
        let rsf: f64 = assets
            .iter()
            .map(|a| a.amount * Self::get_rsf_factor(a, config))
            .sum();

        // Calculate NSFR
        let nsfr_ratio = if rsf > 0.0 { asf / rsf } else { f64::INFINITY };

        let is_compliant = nsfr_ratio >= config.min_nsfr;
        let buffer = asf - (rsf * config.min_nsfr);

        NSFRResult {
            asf,
            rsf,
            nsfr_ratio,
            is_compliant,
            buffer,
        }
    }

    /// Get ASF (Available Stable Funding) factor for funding source.
    fn get_asf_factor(funding: &FundingSource, config: &NSFRConfig) -> f64 {
        match funding.funding_type {
            FundingType::Equity => 1.0,
            FundingType::LongTermDebt => {
                if funding.remaining_maturity_days > 365 {
                    1.0
                } else if funding.remaining_maturity_days > 180 {
                    config.asf_6m_1y
                } else {
                    config.asf_under_6m
                }
            }
            FundingType::RetailDeposit => {
                if funding.is_stable {
                    config.asf_stable_retail
                } else {
                    config.asf_less_stable_retail
                }
            }
            FundingType::WholesaleDeposit => {
                if funding.remaining_maturity_days > 365 {
                    1.0
                } else {
                    config.asf_wholesale
                }
            }
            FundingType::Other => config.asf_other,
        }
    }

    /// Get RSF (Required Stable Funding) factor for asset.
    fn get_rsf_factor(asset: &LiquidityPosition, config: &NSFRConfig) -> f64 {
        match asset.asset_type {
            LiquidityAssetType::CashReserves => 0.0,
            LiquidityAssetType::Level1HQLA => config.rsf_level1,
            LiquidityAssetType::Level2AHQLA => config.rsf_level2a,
            LiquidityAssetType::Level2BHQLA => config.rsf_level2b,
            LiquidityAssetType::OtherLiquid => {
                if asset.days_to_liquidate <= 30 {
                    config.rsf_other_liquid
                } else {
                    config.rsf_illiquid
                }
            }
            LiquidityAssetType::Illiquid => config.rsf_illiquid,
        }
    }

    /// Optimize liquidity ratios.
    pub fn optimize(
        assets: &[LiquidityPosition],
        outflows: &[LiquidityOutflow],
        inflows: &[LiquidityInflow],
        funding: &[FundingSource],
        config: &OptimizationConfig,
    ) -> LiquidityOptimizationResult {
        let lcr_before = Self::calculate_lcr(assets, outflows, inflows, &config.lcr_config);
        let nsfr_before = Self::calculate_nsfr(assets, funding, &config.nsfr_config);

        let mut actions = Vec::new();
        let mut total_cost = 0.0;

        // Generate optimization actions if below targets
        if !lcr_before.is_compliant || lcr_before.lcr_ratio < config.target_lcr {
            let lcr_actions = Self::generate_lcr_actions(assets, outflows, &lcr_before, config);
            for action in lcr_actions {
                total_cost += action.cost;
                actions.push(action);
            }
        }

        if !nsfr_before.is_compliant || nsfr_before.nsfr_ratio < config.target_nsfr {
            let nsfr_actions = Self::generate_nsfr_actions(assets, funding, &nsfr_before, config);
            for action in nsfr_actions {
                total_cost += action.cost;
                actions.push(action);
            }
        }

        // Calculate actual improvement by simulating actions applied
        let lcr_improvement = Self::calculate_lcr_improvement(
            assets,
            outflows,
            inflows,
            &actions,
            &lcr_before,
            &config.lcr_config,
        );

        LiquidityOptimizationResult {
            lcr: lcr_before,
            nsfr: nsfr_before,
            actions,
            total_cost,
            lcr_improvement,
        }
    }

    /// Generate LCR improvement actions.
    fn generate_lcr_actions(
        assets: &[LiquidityPosition],
        outflows: &[LiquidityOutflow],
        lcr: &LCRResult,
        config: &OptimizationConfig,
    ) -> Vec<LiquidityAction> {
        let mut actions = Vec::new();
        let shortfall = if lcr.buffer < 0.0 { -lcr.buffer } else { 0.0 };

        if shortfall == 0.0 {
            return actions;
        }

        // Action 1: Convert non-HQLA to HQLA
        for (i, asset) in assets.iter().enumerate() {
            if matches!(
                asset.asset_type,
                LiquidityAssetType::OtherLiquid | LiquidityAssetType::Illiquid
            ) {
                let convert_amount = asset.amount.min(shortfall);
                let cost = convert_amount * config.conversion_cost_rate;
                let lcr_impact = convert_amount * (1.0 - asset.lcr_haircut);

                actions.push(LiquidityAction {
                    action_type: LiquidityActionType::ConvertToHQLA,
                    target_id: format!("asset_{}", i),
                    amount: convert_amount,
                    lcr_impact,
                    cost,
                });

                if actions.iter().map(|a| a.lcr_impact).sum::<f64>() >= shortfall {
                    break;
                }
            }
        }

        // Action 2: Reduce outflow commitments
        for (i, outflow) in outflows.iter().enumerate() {
            if matches!(outflow.category, OutflowCategory::CommittedFacilities) {
                let reduce_amount = outflow.amount * 0.1; // Max 10% reduction
                let cost = reduce_amount * config.commitment_reduction_cost;
                let lcr_impact = reduce_amount * outflow.runoff_rate;

                actions.push(LiquidityAction {
                    action_type: LiquidityActionType::ReduceCommitment,
                    target_id: format!("outflow_{}", i),
                    amount: reduce_amount,
                    lcr_impact,
                    cost,
                });
            }
        }

        actions
    }

    /// Generate NSFR improvement actions.
    fn generate_nsfr_actions(
        assets: &[LiquidityPosition],
        funding: &[FundingSource],
        nsfr: &NSFRResult,
        config: &OptimizationConfig,
    ) -> Vec<LiquidityAction> {
        let mut actions = Vec::new();
        let shortfall = if nsfr.buffer < 0.0 { -nsfr.buffer } else { 0.0 };

        if shortfall == 0.0 {
            return actions;
        }

        // Action 1: Issue term funding
        for (i, fund) in funding.iter().enumerate() {
            if fund.remaining_maturity_days < 365
                && matches!(fund.funding_type, FundingType::WholesaleDeposit)
            {
                let extend_amount = fund.amount.min(shortfall);
                let cost = extend_amount
                    * config.term_funding_spread
                    * (365.0 - fund.remaining_maturity_days as f64)
                    / 365.0;
                let asf_improvement = extend_amount * (1.0 - config.nsfr_config.asf_wholesale);

                actions.push(LiquidityAction {
                    action_type: LiquidityActionType::IssueTerm,
                    target_id: format!("funding_{}", i),
                    amount: extend_amount,
                    lcr_impact: asf_improvement, // Reusing field for NSFR impact
                    cost,
                });

                if actions.iter().map(|a| a.lcr_impact).sum::<f64>() >= shortfall {
                    break;
                }
            }
        }

        // Action 2: Sell illiquid assets
        for (i, asset) in assets.iter().enumerate() {
            if matches!(asset.asset_type, LiquidityAssetType::Illiquid) {
                let sell_amount = asset.amount.min(shortfall);
                let cost = sell_amount * config.illiquid_sale_haircut;
                let rsf_reduction = sell_amount * config.nsfr_config.rsf_illiquid;

                actions.push(LiquidityAction {
                    action_type: LiquidityActionType::SellIlliquid,
                    target_id: format!("asset_{}", i),
                    amount: sell_amount,
                    lcr_impact: rsf_reduction,
                    cost,
                });
            }
        }

        actions
    }

    /// Calculate liquidity stress metrics.
    pub fn stress_test(
        assets: &[LiquidityPosition],
        outflows: &[LiquidityOutflow],
        inflows: &[LiquidityInflow],
        scenario: &StressScenario,
    ) -> StressTestResult {
        // Apply stress factors to outflows
        let stressed_outflows: Vec<LiquidityOutflow> = outflows
            .iter()
            .map(|o| {
                let stress_factor = scenario
                    .outflow_multipliers
                    .get(&o.category)
                    .copied()
                    .unwrap_or(scenario.default_outflow_multiplier);
                LiquidityOutflow {
                    category: o.category,
                    amount: o.amount,
                    currency: o.currency.clone(),
                    runoff_rate: (o.runoff_rate * stress_factor).min(1.0),
                    days_to_maturity: o.days_to_maturity,
                }
            })
            .collect();

        // Apply haircut to assets
        let stressed_assets: Vec<LiquidityPosition> = assets
            .iter()
            .map(|a| LiquidityPosition {
                id: a.id.clone(),
                asset_type: a.asset_type,
                amount: a.amount,
                currency: a.currency.clone(),
                hqla_level: a.hqla_level,
                lcr_haircut: (a.lcr_haircut + scenario.additional_haircut).min(1.0),
                days_to_liquidate: (a.days_to_liquidate as f64 * scenario.liquidation_delay_factor)
                    as u32,
            })
            .collect();

        // Reduce inflows
        let stressed_inflows: Vec<LiquidityInflow> = inflows
            .iter()
            .map(|i| LiquidityInflow {
                category: i.category.clone(),
                amount: i.amount,
                currency: i.currency.clone(),
                inflow_rate: i.inflow_rate * scenario.inflow_reduction,
                days_to_maturity: i.days_to_maturity,
            })
            .collect();

        let lcr_config = LCRConfig::default();
        let base_lcr = Self::calculate_lcr(assets, outflows, inflows, &lcr_config);
        let stressed_lcr = Self::calculate_lcr(
            &stressed_assets,
            &stressed_outflows,
            &stressed_inflows,
            &lcr_config,
        );

        StressTestResult {
            scenario_name: scenario.name.clone(),
            base_lcr: base_lcr.lcr_ratio,
            stressed_lcr: stressed_lcr.lcr_ratio,
            lcr_impact: stressed_lcr.lcr_ratio - base_lcr.lcr_ratio,
            survives_stress: stressed_lcr.is_compliant,
            days_until_breach: Self::estimate_days_until_breach(&stressed_lcr),
        }
    }

    /// Calculate actual LCR improvement by simulating actions applied.
    fn calculate_lcr_improvement(
        assets: &[LiquidityPosition],
        outflows: &[LiquidityOutflow],
        inflows: &[LiquidityInflow],
        actions: &[LiquidityAction],
        lcr_before: &LCRResult,
        config: &LCRConfig,
    ) -> f64 {
        if actions.is_empty() {
            return 0.0;
        }

        // Create modified copies of assets and outflows
        let mut modified_assets: Vec<LiquidityPosition> = assets.to_vec();
        let mut modified_outflows: Vec<LiquidityOutflow> = outflows.to_vec();

        // Apply each action
        for action in actions {
            match action.action_type {
                LiquidityActionType::ConvertToHQLA => {
                    // Parse asset index from target_id (e.g., "asset_0")
                    if let Some(idx_str) = action.target_id.strip_prefix("asset_") {
                        if let Ok(idx) = idx_str.parse::<usize>() {
                            if idx < modified_assets.len() {
                                // Extract data we need before modifying
                                let asset_id = modified_assets[idx].id.clone();
                                let asset_currency = modified_assets[idx].currency.clone();

                                // Reduce original asset amount
                                modified_assets[idx].amount -= action.amount;

                                // Add new Level 1 HQLA position
                                modified_assets.push(LiquidityPosition {
                                    id: format!("{}_converted", asset_id),
                                    asset_type: LiquidityAssetType::Level1HQLA,
                                    amount: action.amount,
                                    currency: asset_currency,
                                    hqla_level: Some(1),
                                    lcr_haircut: 0.0,
                                    days_to_liquidate: 1,
                                });
                            }
                        }
                    }
                }
                LiquidityActionType::ReduceCommitment => {
                    // Parse outflow index from target_id (e.g., "outflow_0")
                    if let Some(idx_str) = action.target_id.strip_prefix("outflow_") {
                        if let Ok(idx) = idx_str.parse::<usize>() {
                            if idx < modified_outflows.len() {
                                // Reduce outflow amount
                                modified_outflows[idx].amount -= action.amount;
                            }
                        }
                    }
                }
                _ => {
                    // Other action types (NSFR-related) don't affect LCR
                }
            }
        }

        // Recalculate LCR with modified positions
        let lcr_after = Self::calculate_lcr(&modified_assets, &modified_outflows, inflows, config);

        // Return the improvement in HQLA buffer (positive = improvement)
        lcr_after.buffer - lcr_before.buffer
    }

    /// Estimate days until LCR breach under stress.
    ///
    /// Uses a liquidity runoff model considering:
    /// - Current LCR ratio and buffer
    /// - Daily net outflow rate under stress
    /// - HQLA depletion trajectory
    fn estimate_days_until_breach(lcr: &LCRResult) -> Option<u32> {
        if lcr.is_compliant {
            return None;
        }

        // Calculate daily net outflow rate (30-day outflows spread daily)
        let daily_outflow = lcr.net_outflows / 30.0;

        if daily_outflow <= 0.0 {
            // No outflows, won't breach
            return None;
        }

        // Current HQLA level
        let current_hqla = lcr.hqla;

        // Calculate minimum HQLA needed for compliance (LCR = 100%)
        // LCR = HQLA / NetOutflows >= 1.0
        // We're already below compliance, so estimate how long until
        // HQLA depletes to a critical level (e.g., 50% of net outflows)
        let critical_hqla = lcr.net_outflows * 0.5;

        // Days until HQLA drops below critical threshold
        // Assuming linear HQLA depletion at daily outflow rate
        if current_hqla <= critical_hqla {
            // Already at critical level
            return Some(0);
        }

        let hqla_buffer = current_hqla - critical_hqla;
        let days = (hqla_buffer / daily_outflow).ceil() as u32;

        // Cap at reasonable maximum (regulatory typically look at 30-day horizon)
        Some(days.min(90))
    }
}

impl GpuKernel for LiquidityOptimization {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// LCR configuration.
#[derive(Debug, Clone)]
pub struct LCRConfig {
    /// Minimum LCR (usually 1.0 = 100%).
    pub min_lcr: f64,
    /// Level 2A cap.
    pub level2a_cap: f64,
    /// Level 2B cap.
    pub level2b_cap: f64,
    /// Total Level 2 cap.
    pub level2_total_cap: f64,
    /// Inflow cap.
    pub inflow_cap: f64,
}

impl Default for LCRConfig {
    fn default() -> Self {
        Self {
            min_lcr: 1.0,
            level2a_cap: 0.40,
            level2b_cap: 0.15,
            level2_total_cap: 0.40,
            inflow_cap: 0.75,
        }
    }
}

/// NSFR configuration.
#[derive(Debug, Clone)]
pub struct NSFRConfig {
    /// Minimum NSFR.
    pub min_nsfr: f64,
    /// ASF factor for stable retail deposits.
    pub asf_stable_retail: f64,
    /// ASF factor for less stable retail deposits.
    pub asf_less_stable_retail: f64,
    /// ASF factor for wholesale funding.
    pub asf_wholesale: f64,
    /// ASF factor for 6-month to 1-year funding.
    pub asf_6m_1y: f64,
    /// ASF factor for under 6-month funding.
    pub asf_under_6m: f64,
    /// ASF factor for other stable funding.
    pub asf_other: f64,
    /// RSF factor for Level 1 HQLA.
    pub rsf_level1: f64,
    /// RSF factor for Level 2A HQLA.
    pub rsf_level2a: f64,
    /// RSF factor for Level 2B HQLA.
    pub rsf_level2b: f64,
    /// RSF factor for other liquid assets.
    pub rsf_other_liquid: f64,
    /// RSF factor for illiquid assets.
    pub rsf_illiquid: f64,
}

impl Default for NSFRConfig {
    fn default() -> Self {
        Self {
            min_nsfr: 1.0,
            asf_stable_retail: 0.95,
            asf_less_stable_retail: 0.90,
            asf_wholesale: 0.50,
            asf_6m_1y: 0.50,
            asf_under_6m: 0.0,
            asf_other: 0.0,
            rsf_level1: 0.0,
            rsf_level2a: 0.15,
            rsf_level2b: 0.50,
            rsf_other_liquid: 0.50,
            rsf_illiquid: 1.0,
        }
    }
}

/// Optimization configuration.
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Target LCR.
    pub target_lcr: f64,
    /// Target NSFR.
    pub target_nsfr: f64,
    /// LCR config.
    pub lcr_config: LCRConfig,
    /// NSFR config.
    pub nsfr_config: NSFRConfig,
    /// Conversion cost rate.
    pub conversion_cost_rate: f64,
    /// Commitment reduction cost.
    pub commitment_reduction_cost: f64,
    /// Term funding spread.
    pub term_funding_spread: f64,
    /// Illiquid asset sale haircut.
    pub illiquid_sale_haircut: f64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            target_lcr: 1.1,
            target_nsfr: 1.1,
            lcr_config: LCRConfig::default(),
            nsfr_config: NSFRConfig::default(),
            conversion_cost_rate: 0.01,
            commitment_reduction_cost: 0.005,
            term_funding_spread: 0.02,
            illiquid_sale_haircut: 0.10,
        }
    }
}

/// Liquidity inflow.
#[derive(Debug, Clone)]
pub struct LiquidityInflow {
    /// Category.
    pub category: String,
    /// Amount.
    pub amount: f64,
    /// Currency.
    pub currency: String,
    /// Inflow rate.
    pub inflow_rate: f64,
    /// Days to maturity.
    pub days_to_maturity: u32,
}

/// Funding source for NSFR.
#[derive(Debug, Clone)]
pub struct FundingSource {
    /// Funding ID.
    pub id: String,
    /// Funding type.
    pub funding_type: FundingType,
    /// Amount.
    pub amount: f64,
    /// Currency.
    pub currency: String,
    /// Remaining maturity in days.
    pub remaining_maturity_days: u32,
    /// Is stable (for retail).
    pub is_stable: bool,
}

/// Funding type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FundingType {
    /// Equity/Tier 1 capital.
    Equity,
    /// Long-term debt.
    LongTermDebt,
    /// Retail deposits.
    RetailDeposit,
    /// Wholesale deposits.
    WholesaleDeposit,
    /// Other funding.
    Other,
}

/// Stress scenario.
#[derive(Debug, Clone)]
pub struct StressScenario {
    /// Scenario name.
    pub name: String,
    /// Outflow multipliers by category.
    pub outflow_multipliers: HashMap<OutflowCategory, f64>,
    /// Default outflow multiplier.
    pub default_outflow_multiplier: f64,
    /// Additional haircut on assets.
    pub additional_haircut: f64,
    /// Inflow reduction factor.
    pub inflow_reduction: f64,
    /// Liquidation delay factor.
    pub liquidation_delay_factor: f64,
}

impl Default for StressScenario {
    fn default() -> Self {
        let mut multipliers = HashMap::new();
        multipliers.insert(OutflowCategory::WholesaleFunding, 1.5);
        multipliers.insert(OutflowCategory::CommittedFacilities, 1.3);

        Self {
            name: "Standard Stress".to_string(),
            outflow_multipliers: multipliers,
            default_outflow_multiplier: 1.2,
            additional_haircut: 0.05,
            inflow_reduction: 0.8,
            liquidation_delay_factor: 1.5,
        }
    }
}

/// Stress test result.
#[derive(Debug, Clone)]
pub struct StressTestResult {
    /// Scenario name.
    pub scenario_name: String,
    /// Base LCR.
    pub base_lcr: f64,
    /// Stressed LCR.
    pub stressed_lcr: f64,
    /// LCR impact.
    pub lcr_impact: f64,
    /// Survives stress test.
    pub survives_stress: bool,
    /// Days until breach (if stressed below minimum).
    pub days_until_breach: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_assets() -> Vec<LiquidityPosition> {
        vec![
            LiquidityPosition {
                id: "CASH".to_string(),
                asset_type: LiquidityAssetType::CashReserves,
                amount: 100_000.0,
                currency: "USD".to_string(),
                hqla_level: Some(1),
                lcr_haircut: 0.0,
                days_to_liquidate: 0,
            },
            LiquidityPosition {
                id: "GOV_BOND".to_string(),
                asset_type: LiquidityAssetType::Level1HQLA,
                amount: 200_000.0,
                currency: "USD".to_string(),
                hqla_level: Some(1),
                lcr_haircut: 0.0,
                days_to_liquidate: 1,
            },
            LiquidityPosition {
                id: "CORP_BOND".to_string(),
                asset_type: LiquidityAssetType::Level2AHQLA,
                amount: 100_000.0,
                currency: "USD".to_string(),
                hqla_level: Some(2),
                lcr_haircut: 0.15,
                days_to_liquidate: 3,
            },
        ]
    }

    fn create_test_outflows() -> Vec<LiquidityOutflow> {
        vec![
            LiquidityOutflow {
                category: OutflowCategory::RetailDeposits,
                amount: 500_000.0,
                currency: "USD".to_string(),
                runoff_rate: 0.05,
                days_to_maturity: 30,
            },
            LiquidityOutflow {
                category: OutflowCategory::WholesaleFunding,
                amount: 200_000.0,
                currency: "USD".to_string(),
                runoff_rate: 0.25,
                days_to_maturity: 30,
            },
        ]
    }

    fn create_test_inflows() -> Vec<LiquidityInflow> {
        vec![LiquidityInflow {
            category: "Loans".to_string(),
            amount: 100_000.0,
            currency: "USD".to_string(),
            inflow_rate: 0.50,
            days_to_maturity: 30,
        }]
    }

    fn create_test_funding() -> Vec<FundingSource> {
        vec![
            FundingSource {
                id: "EQUITY".to_string(),
                funding_type: FundingType::Equity,
                amount: 100_000.0,
                currency: "USD".to_string(),
                remaining_maturity_days: u32::MAX,
                is_stable: true,
            },
            FundingSource {
                id: "RETAIL".to_string(),
                funding_type: FundingType::RetailDeposit,
                amount: 300_000.0,
                currency: "USD".to_string(),
                remaining_maturity_days: 365,
                is_stable: true,
            },
            FundingSource {
                id: "WHOLESALE".to_string(),
                funding_type: FundingType::WholesaleDeposit,
                amount: 200_000.0,
                currency: "USD".to_string(),
                remaining_maturity_days: 90,
                is_stable: false,
            },
        ]
    }

    #[test]
    fn test_liquidity_metadata() {
        let kernel = LiquidityOptimization::new();
        assert_eq!(kernel.metadata().id, "treasury/liquidity-opt");
        assert_eq!(kernel.metadata().domain, Domain::TreasuryManagement);
    }

    #[test]
    fn test_calculate_lcr() {
        let assets = create_test_assets();
        let outflows = create_test_outflows();
        let inflows = create_test_inflows();
        let config = LCRConfig::default();

        let lcr = LiquidityOptimization::calculate_lcr(&assets, &outflows, &inflows, &config);

        assert!(lcr.hqla > 0.0);
        assert!(lcr.net_outflows > 0.0);
        assert!(lcr.lcr_ratio > 0.0);
        assert!(lcr.hqla_breakdown.contains_key("Level1"));
    }

    #[test]
    fn test_hqla_breakdown() {
        let assets = create_test_assets();
        let outflows = create_test_outflows();
        let inflows = create_test_inflows();
        let config = LCRConfig::default();

        let lcr = LiquidityOptimization::calculate_lcr(&assets, &outflows, &inflows, &config);

        // Level 1 = Cash (100k) + Gov Bond (200k) = 300k
        assert!((lcr.hqla_breakdown.get("Level1").unwrap() - 300_000.0).abs() < 0.01);
    }

    #[test]
    fn test_level2_caps() {
        let assets = vec![
            LiquidityPosition {
                id: "CASH".to_string(),
                asset_type: LiquidityAssetType::CashReserves,
                amount: 100_000.0,
                currency: "USD".to_string(),
                hqla_level: Some(1),
                lcr_haircut: 0.0,
                days_to_liquidate: 0,
            },
            LiquidityPosition {
                id: "L2A".to_string(),
                asset_type: LiquidityAssetType::Level2AHQLA,
                amount: 1_000_000.0, // Large amount that should be capped
                currency: "USD".to_string(),
                hqla_level: Some(2),
                lcr_haircut: 0.15,
                days_to_liquidate: 3,
            },
        ];

        let outflows = create_test_outflows();
        let inflows = create_test_inflows();
        let config = LCRConfig::default();

        let lcr = LiquidityOptimization::calculate_lcr(&assets, &outflows, &inflows, &config);

        // Level 2A should be capped at 40% of total
        let l2a = *lcr.hqla_breakdown.get("Level2A").unwrap();
        let l1 = *lcr.hqla_breakdown.get("Level1").unwrap();
        assert!(l2a <= (l1 + l2a) * 0.40 + 0.01);
    }

    #[test]
    fn test_calculate_nsfr() {
        let assets = create_test_assets();
        let funding = create_test_funding();
        let config = NSFRConfig::default();

        let nsfr = LiquidityOptimization::calculate_nsfr(&assets, &funding, &config);

        assert!(nsfr.asf > 0.0);
        assert!(nsfr.rsf >= 0.0);
        assert!(nsfr.nsfr_ratio > 0.0);
    }

    #[test]
    fn test_asf_factors() {
        let funding = vec![FundingSource {
            id: "EQUITY".to_string(),
            funding_type: FundingType::Equity,
            amount: 100_000.0,
            currency: "USD".to_string(),
            remaining_maturity_days: u32::MAX,
            is_stable: true,
        }];
        let config = NSFRConfig::default();

        let nsfr = LiquidityOptimization::calculate_nsfr(&[], &funding, &config);

        // Equity has 100% ASF factor
        assert!((nsfr.asf - 100_000.0).abs() < 0.01);
    }

    #[test]
    fn test_optimization() {
        let assets = create_test_assets();
        let outflows = create_test_outflows();
        let inflows = create_test_inflows();
        let funding = create_test_funding();
        let config = OptimizationConfig::default();

        let result =
            LiquidityOptimization::optimize(&assets, &outflows, &inflows, &funding, &config);

        // Should have LCR and NSFR results
        assert!(result.lcr.hqla > 0.0);
        assert!(result.nsfr.asf > 0.0);
    }

    #[test]
    fn test_stress_test() {
        let assets = create_test_assets();
        let outflows = create_test_outflows();
        let inflows = create_test_inflows();
        let scenario = StressScenario::default();

        let result = LiquidityOptimization::stress_test(&assets, &outflows, &inflows, &scenario);

        // Stressed LCR should be lower than base
        assert!(result.stressed_lcr <= result.base_lcr);
        assert!(result.lcr_impact <= 0.0);
    }

    #[test]
    fn test_lcr_compliant() {
        let assets = vec![LiquidityPosition {
            id: "CASH".to_string(),
            asset_type: LiquidityAssetType::CashReserves,
            amount: 500_000.0, // Large cash balance
            currency: "USD".to_string(),
            hqla_level: Some(1),
            lcr_haircut: 0.0,
            days_to_liquidate: 0,
        }];

        let outflows = vec![LiquidityOutflow {
            category: OutflowCategory::RetailDeposits,
            amount: 100_000.0,
            currency: "USD".to_string(),
            runoff_rate: 0.05,
            days_to_maturity: 30,
        }];

        let inflows: Vec<LiquidityInflow> = vec![];
        let config = LCRConfig::default();

        let lcr = LiquidityOptimization::calculate_lcr(&assets, &outflows, &inflows, &config);

        assert!(lcr.is_compliant);
        assert!(lcr.buffer > 0.0);
    }

    #[test]
    fn test_empty_inputs() {
        let assets: Vec<LiquidityPosition> = vec![];
        let outflows: Vec<LiquidityOutflow> = vec![];
        let inflows: Vec<LiquidityInflow> = vec![];
        let config = LCRConfig::default();

        let lcr = LiquidityOptimization::calculate_lcr(&assets, &outflows, &inflows, &config);

        assert_eq!(lcr.hqla, 0.0);
        assert_eq!(lcr.net_outflows, 0.0);
    }

    #[test]
    fn test_inflow_cap() {
        let assets = create_test_assets();
        let outflows = vec![LiquidityOutflow {
            category: OutflowCategory::RetailDeposits,
            amount: 100_000.0,
            currency: "USD".to_string(),
            runoff_rate: 1.0, // 100% runoff
            days_to_maturity: 30,
        }];

        let inflows = vec![LiquidityInflow {
            category: "Loans".to_string(),
            amount: 200_000.0, // More than outflows
            currency: "USD".to_string(),
            inflow_rate: 1.0,
            days_to_maturity: 30,
        }];

        let config = LCRConfig::default();
        let lcr = LiquidityOptimization::calculate_lcr(&assets, &outflows, &inflows, &config);

        // Inflows should be capped at 75% of outflows
        // Gross outflows = 100k * 1.0 = 100k
        // Gross inflows = 200k * 1.0 = 200k, capped at 75k
        // Net outflows = 100k - 75k = 25k
        assert!((lcr.net_outflows - 25_000.0).abs() < 0.01);
    }
}
