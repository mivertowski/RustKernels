//! Netting calculation kernel.
//!
//! This module provides multilateral netting for clearing:
//! - Calculate net positions per party
//! - Reduce gross obligations to net obligations
//! - Calculate netting efficiency

use crate::types::{NetPosition, NettingConfig, NettingResult, PartySummary, Trade, TradeStatus};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// Netting Calculation Kernel
// ============================================================================

/// Netting calculation kernel.
///
/// Calculates net positions from gross trades to reduce settlement obligations.
#[derive(Debug, Clone)]
pub struct NettingCalculation {
    metadata: KernelMetadata,
}

impl Default for NettingCalculation {
    fn default() -> Self {
        Self::new()
    }
}

impl NettingCalculation {
    /// Create a new netting calculation kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("clearing/netting", Domain::Clearing)
                .with_description("Multilateral netting calculation")
                .with_throughput(10_000)
                .with_latency_us(500.0),
        }
    }

    /// Calculate net positions from trades.
    pub fn calculate(trades: &[Trade], config: &NettingConfig) -> NettingResult {
        // Filter trades
        let eligible_trades: Vec<_> = trades
            .iter()
            .filter(|t| {
                if !config.include_failed {
                    matches!(
                        t.status,
                        TradeStatus::Pending | TradeStatus::Validated | TradeStatus::Matched
                    )
                } else {
                    true
                }
            })
            .collect();

        let gross_trade_count = eligible_trades.len() as u64;

        // Build netting key
        let get_key = |trade: &Trade, party: &str| -> String {
            let mut key = party.to_string();
            if config.net_by_security {
                key.push_str(&format!(":{}", trade.security_id));
            }
            if config.net_by_settlement_date {
                key.push_str(&format!(":{}", trade.settlement_date));
            }
            if config.net_by_currency {
                // Extract currency from trade attributes, default to USD
                let currency = trade
                    .attributes
                    .get("currency")
                    .map(|s| s.as_str())
                    .unwrap_or("USD");
                key.push_str(&format!(":{}", currency));
            }
            key
        };

        // Helper to extract currency from trade
        let get_currency = |trade: &Trade| -> String {
            trade
                .attributes
                .get("currency")
                .cloned()
                .unwrap_or_else(|| "USD".to_string())
        };

        // Calculate net positions
        let mut positions_map: HashMap<String, NetPositionBuilder> = HashMap::new();

        for trade in &eligible_trades {
            let currency = get_currency(trade);

            // Buyer receives securities, pays money
            let buyer_key = get_key(trade, &trade.buyer_id);
            let buyer_pos = positions_map.entry(buyer_key).or_insert_with(|| {
                NetPositionBuilder::new(
                    trade.buyer_id.clone(),
                    trade.security_id.clone(),
                    currency.clone(),
                )
            });
            buyer_pos.add_receive(trade.quantity, trade.value(), trade.id);

            // Seller delivers securities, receives money
            let seller_key = get_key(trade, &trade.seller_id);
            let seller_pos = positions_map.entry(seller_key).or_insert_with(|| {
                NetPositionBuilder::new(
                    trade.seller_id.clone(),
                    trade.security_id.clone(),
                    currency,
                )
            });
            seller_pos.add_deliver(trade.quantity, trade.value(), trade.id);
        }

        // Convert to net positions
        let positions: Vec<_> = positions_map
            .into_values()
            .map(|builder| builder.build())
            .collect();

        let net_instruction_count = positions.len() as u64;

        // Calculate efficiency
        let efficiency = if gross_trade_count > 0 {
            1.0 - (net_instruction_count as f64 / (gross_trade_count * 2) as f64)
        } else {
            0.0
        };

        // Build party summaries
        let mut party_summary: HashMap<String, PartySummary> = HashMap::new();

        for pos in &positions {
            let summary = party_summary.entry(pos.party_id.clone()).or_default();

            if pos.net_quantity > 0 {
                summary.gross_receipts += pos.net_quantity;
            } else {
                summary.gross_deliveries += pos.net_quantity.unsigned_abs() as i64;
            }
            summary.net_position += pos.net_quantity;

            if pos.net_payment > 0 {
                summary.gross_payments -= pos.net_payment; // Positive net_payment means receiving
            } else {
                summary.gross_payments += pos.net_payment.unsigned_abs() as i64;
            }
            summary.net_payment += pos.net_payment;
            summary.trade_count += pos.trade_ids.len() as u64;
        }

        NettingResult {
            positions,
            gross_trade_count,
            net_instruction_count,
            efficiency,
            party_summary,
        }
    }

    /// Calculate bilateral net positions between two parties.
    pub fn calculate_bilateral(
        trades: &[Trade],
        party_a: &str,
        party_b: &str,
    ) -> BilateralNetResult {
        let relevant_trades: Vec<_> = trades
            .iter()
            .filter(|t| {
                (t.buyer_id == party_a && t.seller_id == party_b)
                    || (t.buyer_id == party_b && t.seller_id == party_a)
            })
            .collect();

        let mut a_receives = 0i64;
        let mut a_delivers = 0i64;
        let mut a_pays = 0i64;
        let mut a_collects = 0i64;

        for trade in &relevant_trades {
            if trade.buyer_id == party_a {
                // A buys from B: A receives securities, pays money
                a_receives += trade.quantity;
                a_pays += trade.value();
            } else {
                // A sells to B: A delivers securities, collects money
                a_delivers += trade.quantity;
                a_collects += trade.value();
            }
        }

        BilateralNetResult {
            party_a: party_a.to_string(),
            party_b: party_b.to_string(),
            trade_count: relevant_trades.len() as u64,
            net_securities_a: a_receives - a_delivers,
            net_payment_a: a_collects - a_pays,
        }
    }

    /// Get netting statistics by security.
    pub fn stats_by_security(result: &NettingResult) -> HashMap<String, SecurityNettingStats> {
        let mut stats: HashMap<String, SecurityNettingStats> = HashMap::new();

        for pos in &result.positions {
            let stat =
                stats
                    .entry(pos.security_id.clone())
                    .or_insert_with(|| SecurityNettingStats {
                        security_id: pos.security_id.clone(),
                        total_net_positions: 0,
                        total_trades: 0,
                        net_quantity: 0,
                        gross_volume: 0,
                    });

            stat.total_net_positions += 1;
            stat.total_trades += pos.trade_ids.len() as u64;
            stat.net_quantity += pos.net_quantity.unsigned_abs() as i64;
            stat.gross_volume += pos.trade_ids.len() as i64; // Approximation
        }

        stats
    }
}

impl GpuKernel for NettingCalculation {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Builder for net positions.
struct NetPositionBuilder {
    party_id: String,
    security_id: String,
    currency: String,
    receives: i64,
    delivers: i64,
    payments_in: i64,
    payments_out: i64,
    trade_ids: Vec<u64>,
}

impl NetPositionBuilder {
    fn new(party_id: String, security_id: String, currency: String) -> Self {
        Self {
            party_id,
            security_id,
            currency,
            receives: 0,
            delivers: 0,
            payments_in: 0,
            payments_out: 0,
            trade_ids: Vec::new(),
        }
    }

    fn add_receive(&mut self, quantity: i64, payment: i64, trade_id: u64) {
        self.receives += quantity;
        self.payments_out += payment;
        if !self.trade_ids.contains(&trade_id) {
            self.trade_ids.push(trade_id);
        }
    }

    fn add_deliver(&mut self, quantity: i64, payment: i64, trade_id: u64) {
        self.delivers += quantity;
        self.payments_in += payment;
        if !self.trade_ids.contains(&trade_id) {
            self.trade_ids.push(trade_id);
        }
    }

    fn build(self) -> NetPosition {
        NetPosition {
            party_id: self.party_id,
            security_id: self.security_id,
            net_quantity: self.receives - self.delivers,
            net_payment: self.payments_in - self.payments_out,
            currency: self.currency,
            trade_ids: self.trade_ids,
        }
    }
}

/// Bilateral netting result.
#[derive(Debug, Clone)]
pub struct BilateralNetResult {
    /// Party A.
    pub party_a: String,
    /// Party B.
    pub party_b: String,
    /// Trade count.
    pub trade_count: u64,
    /// Net securities for A (positive = A receives, negative = A delivers).
    pub net_securities_a: i64,
    /// Net payment for A (positive = A receives, negative = A pays).
    pub net_payment_a: i64,
}

/// Security netting statistics.
#[derive(Debug, Clone)]
pub struct SecurityNettingStats {
    /// Security ID.
    pub security_id: String,
    /// Total net positions.
    pub total_net_positions: u64,
    /// Total trades.
    pub total_trades: u64,
    /// Net quantity across all positions.
    pub net_quantity: i64,
    /// Gross volume.
    pub gross_volume: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_trades() -> Vec<Trade> {
        vec![
            Trade::new(
                1,
                "AAPL".to_string(),
                "A".to_string(),
                "B".to_string(),
                100,
                150,
                1700000000,
                1700172800,
            ),
            Trade::new(
                2,
                "AAPL".to_string(),
                "B".to_string(),
                "A".to_string(),
                50,
                151,
                1700000100,
                1700172800,
            ),
            Trade::new(
                3,
                "AAPL".to_string(),
                "A".to_string(),
                "C".to_string(),
                30,
                149,
                1700000200,
                1700172800,
            ),
            Trade::new(
                4,
                "MSFT".to_string(),
                "A".to_string(),
                "B".to_string(),
                200,
                300,
                1700000300,
                1700172800,
            ),
        ]
    }

    #[test]
    fn test_netting_metadata() {
        let kernel = NettingCalculation::new();
        assert_eq!(kernel.metadata().id, "clearing/netting");
        assert_eq!(kernel.metadata().domain, Domain::Clearing);
    }

    #[test]
    fn test_basic_netting() {
        let trades = create_test_trades();
        let config = NettingConfig::default();

        let result = NettingCalculation::calculate(&trades, &config);

        assert_eq!(result.gross_trade_count, 4);
        assert!(result.net_instruction_count < result.gross_trade_count * 2);
        assert!(result.efficiency > 0.0);
    }

    #[test]
    fn test_net_positions() {
        let trades = vec![
            Trade::new(
                1,
                "AAPL".to_string(),
                "A".to_string(),
                "B".to_string(),
                100,
                150,
                1700000000,
                1700172800,
            ),
            Trade::new(
                2,
                "AAPL".to_string(),
                "B".to_string(),
                "A".to_string(),
                100,
                150,
                1700000100,
                1700172800,
            ),
        ];
        let config = NettingConfig::default();

        let result = NettingCalculation::calculate(&trades, &config);

        // A buys 100, sells 100 = net 0
        let a_pos = result
            .positions
            .iter()
            .find(|p| p.party_id == "A" && p.security_id == "AAPL");
        if let Some(pos) = a_pos {
            assert_eq!(pos.net_quantity, 0);
        }
    }

    #[test]
    fn test_bilateral_netting() {
        let trades = create_test_trades();

        let result = NettingCalculation::calculate_bilateral(&trades, "A", "B");

        assert_eq!(result.trade_count, 3); // Trades 1, 2, 4
        // Trade 1: A buys 100 AAPL from B
        // Trade 2: B buys 50 AAPL from A (A sells)
        // Trade 4: A buys 200 MSFT from B
        // Net: A receives 100 - 50 = 50 AAPL, 200 MSFT
        assert!(result.net_securities_a > 0); // A is net buyer
    }

    #[test]
    fn test_netting_efficiency() {
        // Create trades that will net down significantly
        let trades = vec![
            Trade::new(
                1,
                "AAPL".to_string(),
                "A".to_string(),
                "B".to_string(),
                100,
                150,
                1700000000,
                1700172800,
            ),
            Trade::new(
                2,
                "AAPL".to_string(),
                "B".to_string(),
                "A".to_string(),
                100,
                150,
                1700000100,
                1700172800,
            ),
            Trade::new(
                3,
                "AAPL".to_string(),
                "A".to_string(),
                "B".to_string(),
                100,
                150,
                1700000200,
                1700172800,
            ),
            Trade::new(
                4,
                "AAPL".to_string(),
                "B".to_string(),
                "A".to_string(),
                100,
                150,
                1700000300,
                1700172800,
            ),
        ];
        let config = NettingConfig::default();

        let result = NettingCalculation::calculate(&trades, &config);

        // 4 gross trades -> should reduce significantly
        assert!(result.efficiency > 0.5);
    }

    #[test]
    fn test_party_summary() {
        let trades = create_test_trades();
        let config = NettingConfig::default();

        let result = NettingCalculation::calculate(&trades, &config);

        assert!(result.party_summary.contains_key("A"));
        assert!(result.party_summary.contains_key("B"));
        assert!(result.party_summary.contains_key("C"));
    }

    #[test]
    fn test_net_by_security() {
        let trades = create_test_trades();
        let config = NettingConfig::default();

        let result = NettingCalculation::calculate(&trades, &config);

        // Should have separate positions for AAPL and MSFT
        let aapl_positions: Vec<_> = result
            .positions
            .iter()
            .filter(|p| p.security_id == "AAPL")
            .collect();
        let msft_positions: Vec<_> = result
            .positions
            .iter()
            .filter(|p| p.security_id == "MSFT")
            .collect();

        assert!(!aapl_positions.is_empty());
        assert!(!msft_positions.is_empty());
    }

    #[test]
    fn test_exclude_failed_trades() {
        let mut trades = create_test_trades();
        trades[0].status = TradeStatus::Failed;

        let config = NettingConfig::default();

        let result = NettingCalculation::calculate(&trades, &config);

        assert_eq!(result.gross_trade_count, 3); // One excluded
    }

    #[test]
    fn test_include_failed_trades() {
        let mut trades = create_test_trades();
        trades[0].status = TradeStatus::Failed;

        let config = NettingConfig {
            include_failed: true,
            ..NettingConfig::default()
        };

        let result = NettingCalculation::calculate(&trades, &config);

        assert_eq!(result.gross_trade_count, 4); // All included
    }

    #[test]
    fn test_stats_by_security() {
        let trades = create_test_trades();
        let config = NettingConfig::default();

        let result = NettingCalculation::calculate(&trades, &config);
        let stats = NettingCalculation::stats_by_security(&result);

        assert!(stats.contains_key("AAPL"));
        assert!(stats.contains_key("MSFT"));
    }

    #[test]
    fn test_empty_trades() {
        let trades: Vec<Trade> = vec![];
        let config = NettingConfig::default();

        let result = NettingCalculation::calculate(&trades, &config);

        assert_eq!(result.gross_trade_count, 0);
        assert_eq!(result.net_instruction_count, 0);
        assert!((result.efficiency - 0.0).abs() < 0.001);
    }
}
