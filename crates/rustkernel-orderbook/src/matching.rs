//! Order matching engine kernel.
//!
//! This module provides a high-performance order matching engine:
//! - Price-time priority matching
//! - Support for limit and market orders
//! - Self-trade prevention
//! - Order book management

use std::collections::HashMap;
use std::time::Instant;

use async_trait::async_trait;

use crate::messages::{
    BatchOrderInput, BatchOrderOutput, CancelOrderInput, CancelOrderOutput, GetSnapshotInput,
    GetSnapshotOutput, ModifyOrderInput, ModifyOrderOutput, SubmitOrderInput, SubmitOrderOutput,
};
use crate::types::{
    EngineConfig, L2Snapshot, MatchResult, Order, OrderBook, OrderStatus, OrderType, Price,
    PriceLevel, Quantity, Side, TimeInForce, Trade,
};
use rustkernel_core::{
    domain::Domain,
    error::Result as KernelResult,
    kernel::KernelMetadata,
    traits::{BatchKernel, GpuKernel},
};

// ============================================================================
// Order Matching Engine Kernel
// ============================================================================

/// Order matching engine kernel.
///
/// High-performance price-time priority matching engine with sub-10μs latency.
#[derive(Debug)]
pub struct OrderMatchingEngine {
    metadata: KernelMetadata,
    /// Order books by symbol.
    books: HashMap<u32, OrderBook>,
    /// All orders by ID.
    orders: HashMap<u64, Order>,
    /// Next trade ID.
    next_trade_id: u64,
    /// Engine configuration.
    config: EngineConfig,
}

impl Default for OrderMatchingEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl OrderMatchingEngine {
    /// Create a new order matching engine.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(EngineConfig::default())
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: EngineConfig) -> Self {
        Self {
            metadata: KernelMetadata::ring("orderbook/matching", Domain::OrderMatching)
                .with_description("Price-time priority order matching")
                .with_throughput(100_000)
                .with_latency_us(10.0)
                .with_gpu_native(true),
            books: HashMap::new(),
            orders: HashMap::new(),
            next_trade_id: 1,
            config,
        }
    }

    /// Submit a new order.
    pub fn submit_order(&mut self, mut order: Order) -> MatchResult {
        // Validate order
        if let Err(status) = self.validate_order(&order) {
            order.status = status;
            let mut result = MatchResult::no_match(order.id, order.remaining);
            result.status = status;
            self.orders.insert(order.id, order);
            return result;
        }

        // Ensure order book exists
        self.books
            .entry(order.symbol_id)
            .or_insert_with(|| OrderBook::new(order.symbol_id));

        // Match the order
        let mut result = self.match_order_internal(&mut order);

        // Handle remaining quantity based on order type and TIF
        if !order.is_filled() {
            match (order.order_type, order.tif) {
                (OrderType::Market, _) | (_, TimeInForce::IOC) => {
                    // Cancel remaining
                    order.status = if result.trades.is_empty() {
                        OrderStatus::Canceled
                    } else {
                        OrderStatus::PartiallyFilled
                    };
                }
                (_, TimeInForce::FOK) => {
                    // Should have been fully filled or nothing
                    if !result.trades.is_empty() {
                        // This shouldn't happen with proper FOK handling
                        order.status = OrderStatus::PartiallyFilled;
                    } else {
                        order.status = OrderStatus::Canceled;
                    }
                }
                _ => {
                    // Add to book
                    self.add_to_book(&order);
                    order.status = if result.trades.is_empty() {
                        OrderStatus::New
                    } else {
                        OrderStatus::PartiallyFilled
                    };
                }
            }
        } else {
            order.status = OrderStatus::Filled;
        }

        // Update result status to match order status
        result.status = order.status;

        // Store order
        self.orders.insert(order.id, order);

        result
    }

    /// Cancel an order.
    pub fn cancel_order(&mut self, order_id: u64) -> Option<Order> {
        // First check if order exists and is active
        {
            let order = self.orders.get(&order_id)?;
            if !order.is_active() {
                return None;
            }
        }

        // Remove from book
        self.remove_from_book(order_id);

        // Update status
        let order = self.orders.get_mut(&order_id)?;
        order.status = OrderStatus::Canceled;
        Some(order.clone())
    }

    /// Modify an order (cancel and replace).
    pub fn modify_order(
        &mut self,
        order_id: u64,
        new_price: Option<Price>,
        new_quantity: Option<Quantity>,
    ) -> Option<MatchResult> {
        let old_order = self.orders.get(&order_id)?.clone();

        if !old_order.is_active() {
            return None;
        }

        // Cancel old order
        self.cancel_order(order_id)?;

        // Create new order with modifications
        let new_order = Order {
            id: order_id, // Reuse ID for simplicity
            price: new_price.unwrap_or(old_order.price),
            quantity: new_quantity.unwrap_or(old_order.remaining),
            remaining: new_quantity.unwrap_or(old_order.remaining),
            status: OrderStatus::New,
            ..old_order
        };

        Some(self.submit_order(new_order))
    }

    /// Get order by ID.
    pub fn get_order(&self, order_id: u64) -> Option<&Order> {
        self.orders.get(&order_id)
    }

    /// Get order book snapshot.
    pub fn get_snapshot(&self, symbol_id: u32, depth: usize) -> Option<L2Snapshot> {
        let book = self.books.get(&symbol_id)?;
        Some(L2Snapshot::from_book(
            book,
            depth,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        ))
    }

    /// Get order book.
    pub fn get_book(&self, symbol_id: u32) -> Option<&OrderBook> {
        self.books.get(&symbol_id)
    }

    /// Validate an order.
    fn validate_order(&self, order: &Order) -> Result<(), OrderStatus> {
        // Check quantity bounds
        if order.quantity.0 < self.config.min_order_size.0 {
            return Err(OrderStatus::Rejected);
        }
        if order.quantity.0 > self.config.max_order_size.0 {
            return Err(OrderStatus::Rejected);
        }

        // Check price tick size for limit orders
        if order.order_type == OrderType::Limit && self.config.tick_size.0 > 0 {
            if order.price.0 % self.config.tick_size.0 != 0 {
                return Err(OrderStatus::Rejected);
            }
        }

        Ok(())
    }

    /// Match an order against the book.
    fn match_order_internal(&mut self, order: &mut Order) -> MatchResult {
        let book = self.books.get_mut(&order.symbol_id).unwrap();
        let mut trades = Vec::new();

        let opposite_side = match order.side {
            Side::Buy => &mut book.asks,
            Side::Sell => &mut book.bids,
        };

        // Collect prices to match against
        let prices_to_match: Vec<Price> = match order.side {
            Side::Buy => opposite_side.keys().copied().collect(),
            Side::Sell => opposite_side.keys().rev().copied().collect(),
        };

        let mut levels_to_remove = Vec::new();

        for price in prices_to_match {
            // Check if order can match at this price
            let can_match = match order.order_type {
                OrderType::Market => true,
                OrderType::Limit => match order.side {
                    Side::Buy => price <= order.price,
                    Side::Sell => price >= order.price,
                },
                _ => false,
            };

            if !can_match {
                break;
            }

            // Get the price level
            let level = match opposite_side.get_mut(&price) {
                Some(l) => l,
                None => continue,
            };

            // Match against orders at this level
            let mut orders_to_remove = Vec::new();

            for &resting_order_id in &level.orders {
                if order.remaining.0 == 0 {
                    break;
                }

                let resting_order = match self.orders.get_mut(&resting_order_id) {
                    Some(o) => o,
                    None => continue,
                };

                // Self-trade prevention
                if self.config.self_trade_prevention
                    && order.trader_id == resting_order.trader_id
                {
                    continue;
                }

                // Calculate fill quantity
                let fill_qty = Quantity(order.remaining.0.min(resting_order.remaining.0));

                // Create trade
                let trade = Trade {
                    id: self.next_trade_id,
                    symbol_id: order.symbol_id,
                    buy_order_id: if order.side == Side::Buy {
                        order.id
                    } else {
                        resting_order_id
                    },
                    sell_order_id: if order.side == Side::Sell {
                        order.id
                    } else {
                        resting_order_id
                    },
                    price,
                    quantity: fill_qty,
                    timestamp: order.timestamp,
                    aggressor: order.side,
                };

                self.next_trade_id += 1;
                trades.push(trade);

                // Update quantities
                order.remaining.0 -= fill_qty.0;
                resting_order.remaining.0 -= fill_qty.0;

                // Update resting order status
                if resting_order.remaining.0 == 0 {
                    resting_order.status = OrderStatus::Filled;
                    orders_to_remove.push(resting_order_id);
                } else {
                    resting_order.status = OrderStatus::PartiallyFilled;
                }

                // Update book volume
                book.volume_24h.0 += fill_qty.0;
                book.last_price = Some(price);
            }

            // Remove filled orders from level
            for order_id in &orders_to_remove {
                if let Some(resting) = self.orders.get(order_id) {
                    level.remove_order(*order_id, resting.quantity);
                }
            }

            // Mark empty levels for removal
            if level.is_empty() {
                levels_to_remove.push(price);
            }

            if order.remaining.0 == 0 {
                break;
            }
        }

        // Remove empty levels
        for price in levels_to_remove {
            opposite_side.remove(&price);
        }

        // Build result
        if trades.is_empty() {
            MatchResult::no_match(order.id, order.remaining)
        } else if order.remaining.0 == 0 {
            MatchResult::full_fill(order.id, trades)
        } else {
            MatchResult::partial_fill(order.id, trades, order.remaining)
        }
    }

    /// Add order to book.
    fn add_to_book(&mut self, order: &Order) {
        let book = self
            .books
            .entry(order.symbol_id)
            .or_insert_with(|| OrderBook::new(order.symbol_id));

        let levels = match order.side {
            Side::Buy => &mut book.bids,
            Side::Sell => &mut book.asks,
        };

        let level = levels
            .entry(order.price)
            .or_insert_with(|| PriceLevel::new(order.price));

        level.add_order(order.id, order.remaining);
    }

    /// Remove order from book.
    fn remove_from_book(&mut self, order_id: u64) -> bool {
        let order = match self.orders.get(&order_id) {
            Some(o) => o.clone(),
            None => return false,
        };

        let book = match self.books.get_mut(&order.symbol_id) {
            Some(b) => b,
            None => return false,
        };

        let levels = match order.side {
            Side::Buy => &mut book.bids,
            Side::Sell => &mut book.asks,
        };

        if let Some(level) = levels.get_mut(&order.price) {
            level.remove_order(order_id, order.remaining);
            if level.is_empty() {
                levels.remove(&order.price);
            }
            true
        } else {
            false
        }
    }

    /// Process a batch of orders.
    pub fn process_batch(&mut self, orders: Vec<Order>) -> Vec<MatchResult> {
        orders
            .into_iter()
            .map(|o| self.submit_order(o))
            .collect()
    }

    /// Clear the engine (for testing).
    pub fn clear(&mut self) {
        self.books.clear();
        self.orders.clear();
        self.next_trade_id = 1;
    }
}

impl Clone for OrderMatchingEngine {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            books: self.books.clone(),
            orders: self.orders.clone(),
            next_trade_id: self.next_trade_id,
            config: self.config.clone(),
        }
    }
}

impl GpuKernel for OrderMatchingEngine {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[async_trait]
impl BatchKernel<SubmitOrderInput, SubmitOrderOutput> for OrderMatchingEngine {
    async fn execute(&self, input: SubmitOrderInput) -> KernelResult<SubmitOrderOutput> {
        let start = Instant::now();
        // Note: This requires &mut self, but BatchKernel takes &self
        // In practice, this would use interior mutability (Mutex/RwLock)
        // For now, we clone and process
        let mut engine = self.clone();
        let result = engine.submit_order(input.order);
        Ok(SubmitOrderOutput {
            result,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

#[async_trait]
impl BatchKernel<BatchOrderInput, BatchOrderOutput> for OrderMatchingEngine {
    async fn execute(&self, input: BatchOrderInput) -> KernelResult<BatchOrderOutput> {
        let start = Instant::now();
        let mut engine = self.clone();
        let results = engine.process_batch(input.orders);
        let total_trades: usize = results.iter().map(|r| r.trades.len()).sum();
        Ok(BatchOrderOutput {
            results,
            total_trades,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

#[async_trait]
impl BatchKernel<CancelOrderInput, CancelOrderOutput> for OrderMatchingEngine {
    async fn execute(&self, input: CancelOrderInput) -> KernelResult<CancelOrderOutput> {
        let start = Instant::now();
        let mut engine = self.clone();
        let canceled_order = engine.cancel_order(input.order_id);
        Ok(CancelOrderOutput {
            success: canceled_order.is_some(),
            canceled_order,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

#[async_trait]
impl BatchKernel<ModifyOrderInput, ModifyOrderOutput> for OrderMatchingEngine {
    async fn execute(&self, input: ModifyOrderInput) -> KernelResult<ModifyOrderOutput> {
        let start = Instant::now();
        let mut engine = self.clone();
        let result = engine.modify_order(input.order_id, input.new_price, input.new_quantity);
        Ok(ModifyOrderOutput {
            success: result.is_some(),
            result,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

#[async_trait]
impl BatchKernel<GetSnapshotInput, GetSnapshotOutput> for OrderMatchingEngine {
    async fn execute(&self, input: GetSnapshotInput) -> KernelResult<GetSnapshotOutput> {
        let start = Instant::now();
        let snapshot = self.get_snapshot(input.symbol_id, input.depth);
        Ok(GetSnapshotOutput {
            snapshot,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

// ============================================================================
// Ring Kernel Handler Implementation
// ============================================================================

use crate::ring_messages::{
    CancelOrderResponse, CancelOrderRing, QueryBookResponse, QueryBookRing, RingOrderStatus,
    RingString, SubmitOrderResponse, SubmitOrderRing,
};
use ringkernel_core::message::CorrelationId;
use ringkernel_core::RingContext;
use rustkernel_core::traits::RingKernelHandler;

/// Ring kernel handler for ultra-low latency order submission.
#[async_trait]
impl RingKernelHandler<SubmitOrderRing, SubmitOrderResponse> for OrderMatchingEngine {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: SubmitOrderRing,
    ) -> KernelResult<SubmitOrderResponse> {
        // Convert Ring message to domain Order
        let order = msg.to_order();

        // Clone self to get mutable access (in production, would use interior mutability)
        let mut engine = self.clone();
        let result = engine.submit_order(order);

        // Convert result to Ring response
        let status = match result.status {
            OrderStatus::New => RingOrderStatus::New,
            OrderStatus::PartiallyFilled => RingOrderStatus::PartiallyFilled,
            OrderStatus::Filled => RingOrderStatus::Filled,
            OrderStatus::Canceled => RingOrderStatus::Canceled,
            OrderStatus::Rejected | OrderStatus::Expired => RingOrderStatus::Rejected,
        };

        Ok(SubmitOrderResponse {
            correlation_id: msg.correlation_id,
            order_id: msg.order_id,
            status,
            filled_quantity: result.filled_quantity.0,
            avg_price: result.avg_price.0,
            remaining: result.remaining.0,
            trade_count: result.trades.len() as u32,
            error: RingString::empty(),
        })
    }
}

/// Ring kernel handler for order cancellation.
#[async_trait]
impl RingKernelHandler<CancelOrderRing, CancelOrderResponse> for OrderMatchingEngine {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: CancelOrderRing,
    ) -> KernelResult<CancelOrderResponse> {
        let mut engine = self.clone();
        let canceled = engine.cancel_order(msg.order_id);

        Ok(CancelOrderResponse {
            correlation_id: msg.correlation_id,
            order_id: msg.order_id,
            success: canceled.is_some(),
            remaining: canceled.map_or(0, |o| o.remaining.0),
        })
    }
}

/// Ring kernel handler for book queries.
#[async_trait]
impl RingKernelHandler<QueryBookRing, QueryBookResponse> for OrderMatchingEngine {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: QueryBookRing,
    ) -> KernelResult<QueryBookResponse> {
        let book = self.get_book(msg.symbol_id);

        let (best_bid, best_ask, bid_depth, ask_depth, spread, mid_price) = match book {
            Some(b) => {
                let (bid_d, ask_d) = b.depth_at_levels(msg.depth as usize);
                (
                    b.best_bid().map_or(0, |p| p.0),
                    b.best_ask().map_or(0, |p| p.0),
                    bid_d.0,
                    ask_d.0,
                    b.spread().map_or(0, |p| p.0),
                    b.mid_price().map_or(0, |p| p.0),
                )
            }
            None => (0, 0, 0, 0, 0, 0),
        };

        Ok(QueryBookResponse {
            correlation_id: msg.correlation_id,
            symbol_id: msg.symbol_id,
            best_bid,
            best_ask,
            bid_depth,
            ask_depth,
            spread,
            mid_price,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_engine() -> OrderMatchingEngine {
        OrderMatchingEngine::new()
    }

    #[test]
    fn test_engine_metadata() {
        let engine = create_engine();
        assert_eq!(engine.metadata().id, "orderbook/matching");
        assert_eq!(engine.metadata().domain, Domain::OrderMatching);
    }

    #[test]
    fn test_limit_order_submission() {
        let mut engine = create_engine();

        let order = Order::limit(
            1,
            100,
            Side::Buy,
            Price::from_f64(100.0),
            Quantity::from_f64(10.0),
            1000,
            0,
        );

        let result = engine.submit_order(order);

        assert_eq!(result.order_id, 1);
        assert_eq!(result.status, OrderStatus::New);
        assert!(result.trades.is_empty());
    }

    #[test]
    fn test_market_order_no_liquidity() {
        let mut engine = create_engine();

        let order = Order::market(1, 100, Side::Buy, Quantity::from_f64(10.0), 1000, 0);

        let result = engine.submit_order(order);

        // Market order with no liquidity should be canceled
        assert_eq!(result.status, OrderStatus::Canceled);
        assert!(result.trades.is_empty());
    }

    #[test]
    fn test_simple_match() {
        let mut engine = create_engine();

        // Add sell order
        let sell = Order::limit(
            1,
            100,
            Side::Sell,
            Price::from_f64(100.0),
            Quantity::from_f64(10.0),
            1000,
            0,
        );
        engine.submit_order(sell);

        // Add matching buy order
        let buy = Order::limit(
            2,
            100,
            Side::Buy,
            Price::from_f64(100.0),
            Quantity::from_f64(10.0),
            2000,
            1,
        );
        let result = engine.submit_order(buy);

        assert_eq!(result.status, OrderStatus::Filled);
        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.filled_quantity.0, Quantity::from_f64(10.0).0);
    }

    #[test]
    fn test_partial_fill() {
        let mut engine = create_engine();

        // Add sell order for 5 units
        let sell = Order::limit(
            1,
            100,
            Side::Sell,
            Price::from_f64(100.0),
            Quantity::from_f64(5.0),
            1000,
            0,
        );
        engine.submit_order(sell);

        // Buy order for 10 units (partial fill)
        let buy = Order::limit(
            2,
            100,
            Side::Buy,
            Price::from_f64(100.0),
            Quantity::from_f64(10.0),
            2000,
            1,
        );
        let result = engine.submit_order(buy);

        assert_eq!(result.status, OrderStatus::PartiallyFilled);
        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.filled_quantity.0, Quantity::from_f64(5.0).0);
        assert_eq!(result.remaining.0, Quantity::from_f64(5.0).0);
    }

    #[test]
    fn test_price_priority() {
        let mut engine = create_engine();

        // Add two sell orders at different prices
        let sell1 = Order::limit(
            1,
            100,
            Side::Sell,
            Price::from_f64(101.0),
            Quantity::from_f64(10.0),
            1000,
            0,
        );
        engine.submit_order(sell1);

        let sell2 = Order::limit(
            2,
            100,
            Side::Sell,
            Price::from_f64(100.0),
            Quantity::from_f64(10.0),
            1000,
            1,
        );
        engine.submit_order(sell2);

        // Buy should match against better price (100) first
        let buy = Order::limit(
            3,
            100,
            Side::Buy,
            Price::from_f64(101.0),
            Quantity::from_f64(10.0),
            2000,
            2,
        );
        let result = engine.submit_order(buy);

        assert_eq!(result.status, OrderStatus::Filled);
        assert_eq!(result.trades[0].price.to_f64(), 100.0);
        assert_eq!(result.trades[0].sell_order_id, 2); // Matched sell2
    }

    #[test]
    fn test_time_priority() {
        let mut engine = create_engine();

        // Add two sell orders at same price
        let sell1 = Order::limit(
            1,
            100,
            Side::Sell,
            Price::from_f64(100.0),
            Quantity::from_f64(5.0),
            1000,
            0, // Earlier timestamp
        );
        engine.submit_order(sell1);

        let sell2 = Order::limit(
            2,
            100,
            Side::Sell,
            Price::from_f64(100.0),
            Quantity::from_f64(5.0),
            1000,
            1, // Later timestamp
        );
        engine.submit_order(sell2);

        // Buy should match against earlier order first
        let buy = Order::limit(
            3,
            100,
            Side::Buy,
            Price::from_f64(100.0),
            Quantity::from_f64(5.0),
            2000,
            2,
        );
        let result = engine.submit_order(buy);

        assert_eq!(result.trades[0].sell_order_id, 1); // Matched sell1 (earlier)
    }

    #[test]
    fn test_cancel_order() {
        let mut engine = create_engine();

        let order = Order::limit(
            1,
            100,
            Side::Buy,
            Price::from_f64(100.0),
            Quantity::from_f64(10.0),
            1000,
            0,
        );
        engine.submit_order(order);

        let canceled = engine.cancel_order(1);

        assert!(canceled.is_some());
        assert_eq!(canceled.unwrap().status, OrderStatus::Canceled);
    }

    #[test]
    fn test_order_book_snapshot() {
        let mut engine = create_engine();

        // Add buy orders
        for i in 0..5 {
            let order = Order::limit(
                i,
                100,
                Side::Buy,
                Price::from_f64(100.0 - i as f64),
                Quantity::from_f64(10.0),
                1000,
                i,
            );
            engine.submit_order(order);
        }

        // Add sell orders
        for i in 5..10 {
            let order = Order::limit(
                i,
                100,
                Side::Sell,
                Price::from_f64(101.0 + (i - 5) as f64),
                Quantity::from_f64(10.0),
                1000,
                i,
            );
            engine.submit_order(order);
        }

        let snapshot = engine.get_snapshot(100, 3).unwrap();

        assert_eq!(snapshot.bids.len(), 3);
        assert_eq!(snapshot.asks.len(), 3);

        // Best bid should be 100.0
        assert_eq!(snapshot.bids[0].0.to_f64(), 100.0);
        // Best ask should be 101.0
        assert_eq!(snapshot.asks[0].0.to_f64(), 101.0);
    }

    #[test]
    fn test_self_trade_prevention() {
        let mut engine = create_engine();

        // Add sell order
        let sell = Order::limit(
            1,
            100,
            Side::Sell,
            Price::from_f64(100.0),
            Quantity::from_f64(10.0),
            1000, // Trader 1000
            0,
        );
        engine.submit_order(sell);

        // Same trader tries to buy
        let buy = Order::limit(
            2,
            100,
            Side::Buy,
            Price::from_f64(100.0),
            Quantity::from_f64(10.0),
            1000, // Same trader
            1,
        );
        let result = engine.submit_order(buy);

        // Should not match due to self-trade prevention
        assert!(result.trades.is_empty());
    }

    #[test]
    fn test_market_order_fill() {
        let mut engine = create_engine();

        // Add liquidity
        let sell = Order::limit(
            1,
            100,
            Side::Sell,
            Price::from_f64(100.0),
            Quantity::from_f64(10.0),
            1000,
            0,
        );
        engine.submit_order(sell);

        // Market buy
        let buy = Order::market(2, 100, Side::Buy, Quantity::from_f64(5.0), 2000, 1);
        let result = engine.submit_order(buy);

        assert_eq!(result.status, OrderStatus::Filled);
        assert_eq!(result.trades.len(), 1);
    }

    #[test]
    fn test_book_depth() {
        let mut engine = create_engine();

        // Add buy orders
        let buy = Order::limit(
            1,
            100,
            Side::Buy,
            Price::from_f64(100.0),
            Quantity::from_f64(10.0),
            1000,
            0,
        );
        engine.submit_order(buy);

        // Add sell orders
        let sell = Order::limit(
            2,
            100,
            Side::Sell,
            Price::from_f64(101.0),
            Quantity::from_f64(20.0),
            1000,
            1,
        );
        engine.submit_order(sell);

        let book = engine.get_book(100).unwrap();

        assert_eq!(book.best_bid().unwrap().to_f64(), 100.0);
        assert_eq!(book.best_ask().unwrap().to_f64(), 101.0);
        assert_eq!(book.bid_depth().to_f64(), 10.0);
        assert_eq!(book.ask_depth().to_f64(), 20.0);
    }

    #[test]
    fn test_modify_order() {
        let mut engine = create_engine();

        // Add order
        let order = Order::limit(
            1,
            100,
            Side::Buy,
            Price::from_f64(100.0),
            Quantity::from_f64(10.0),
            1000,
            0,
        );
        engine.submit_order(order);

        // Modify price
        let result = engine.modify_order(1, Some(Price::from_f64(99.0)), None);

        assert!(result.is_some());
        let order = engine.get_order(1).unwrap();
        assert_eq!(order.price.to_f64(), 99.0);
    }

    #[test]
    fn test_batch_processing() {
        let mut engine = create_engine();

        let orders = vec![
            Order::limit(
                1,
                100,
                Side::Sell,
                Price::from_f64(100.0),
                Quantity::from_f64(10.0),
                1000,
                0,
            ),
            Order::limit(
                2,
                100,
                Side::Buy,
                Price::from_f64(100.0),
                Quantity::from_f64(5.0),
                2000,
                1,
            ),
            Order::limit(
                3,
                100,
                Side::Buy,
                Price::from_f64(100.0),
                Quantity::from_f64(5.0),
                2000,
                2,
            ),
        ];

        let results = engine.process_batch(orders);

        assert_eq!(results.len(), 3);
        // First order rests
        assert_eq!(results[0].status, OrderStatus::New);
        // Second order fills
        assert_eq!(results[1].status, OrderStatus::Filled);
        // Third order fills remaining
        assert_eq!(results[2].status, OrderStatus::Filled);
    }

    #[test]
    fn test_multiple_symbols() {
        let mut engine = create_engine();

        // Symbol 100
        let order1 = Order::limit(
            1,
            100,
            Side::Buy,
            Price::from_f64(100.0),
            Quantity::from_f64(10.0),
            1000,
            0,
        );
        engine.submit_order(order1);

        // Symbol 200
        let order2 = Order::limit(
            2,
            200,
            Side::Sell,
            Price::from_f64(50.0),
            Quantity::from_f64(20.0),
            1000,
            1,
        );
        engine.submit_order(order2);

        assert!(engine.get_book(100).is_some());
        assert!(engine.get_book(200).is_some());
        assert!(engine.get_book(300).is_none());
    }

    #[test]
    fn test_order_validation() {
        let mut engine = create_engine();

        // Order too small
        let small_order = Order::limit(
            1,
            100,
            Side::Buy,
            Price::from_f64(100.0),
            Quantity(1), // Below minimum
            1000,
            0,
        );
        let result = engine.submit_order(small_order);
        assert_eq!(result.status, OrderStatus::Rejected);

        // Order too large
        let large_order = Order::limit(
            2,
            100,
            Side::Buy,
            Price::from_f64(100.0),
            Quantity::from_f64(10_000_000.0), // Above maximum
            1000,
            1,
        );
        let result = engine.submit_order(large_order);
        assert_eq!(result.status, OrderStatus::Rejected);
    }
}
