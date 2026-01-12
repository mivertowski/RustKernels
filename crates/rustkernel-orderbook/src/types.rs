//! Order book types and data structures.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

// ============================================================================
// Order Types
// ============================================================================

/// A trading order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Unique order ID.
    pub id: u64,
    /// Symbol/instrument ID.
    pub symbol_id: u32,
    /// Order side (buy/sell).
    pub side: Side,
    /// Order type.
    pub order_type: OrderType,
    /// Order price (for limit orders).
    pub price: Price,
    /// Order quantity.
    pub quantity: Quantity,
    /// Remaining quantity.
    pub remaining: Quantity,
    /// Time-in-force.
    pub tif: TimeInForce,
    /// Timestamp (nanoseconds since epoch).
    pub timestamp: u64,
    /// Trader/account ID.
    pub trader_id: u64,
    /// Order status.
    pub status: OrderStatus,
}

impl Order {
    /// Create a new limit order.
    pub fn limit(
        id: u64,
        symbol_id: u32,
        side: Side,
        price: Price,
        quantity: Quantity,
        trader_id: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            symbol_id,
            side,
            order_type: OrderType::Limit,
            price,
            quantity,
            remaining: quantity,
            tif: TimeInForce::GTC,
            timestamp,
            trader_id,
            status: OrderStatus::New,
        }
    }

    /// Create a new market order.
    pub fn market(
        id: u64,
        symbol_id: u32,
        side: Side,
        quantity: Quantity,
        trader_id: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            symbol_id,
            side,
            order_type: OrderType::Market,
            price: Price(0), // Market orders don't have a price
            quantity,
            remaining: quantity,
            tif: TimeInForce::IOC,
            timestamp,
            trader_id,
            status: OrderStatus::New,
        }
    }

    /// Check if order is filled.
    pub fn is_filled(&self) -> bool {
        self.remaining.0 == 0
    }

    /// Check if order is active.
    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::New | OrderStatus::PartiallyFilled
        )
    }
}

/// Order side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    /// Buy order.
    Buy,
    /// Sell order.
    Sell,
}

/// Order type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    /// Limit order (price specified).
    Limit,
    /// Market order (best available price).
    Market,
    /// Stop order.
    Stop,
    /// Stop-limit order.
    StopLimit,
}

/// Time-in-force specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeInForce {
    /// Good-till-canceled.
    GTC,
    /// Immediate-or-cancel.
    IOC,
    /// Fill-or-kill.
    FOK,
    /// Day order.
    Day,
}

/// Order status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    /// New order.
    New,
    /// Partially filled.
    PartiallyFilled,
    /// Fully filled.
    Filled,
    /// Canceled.
    Canceled,
    /// Rejected.
    Rejected,
    /// Expired.
    Expired,
}

// ============================================================================
// Price and Quantity Types
// ============================================================================

/// Price in fixed-point representation (price * 10^8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Price(pub i64);

impl Price {
    /// Create price from float.
    pub fn from_f64(price: f64) -> Self {
        Self((price * 100_000_000.0) as i64)
    }

    /// Convert to float.
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / 100_000_000.0
    }
}

impl Ord for Price {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for Price {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Quantity in fixed-point representation (quantity * 10^8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Quantity(pub u64);

impl Quantity {
    /// Create quantity from float.
    pub fn from_f64(qty: f64) -> Self {
        Self((qty * 100_000_000.0) as u64)
    }

    /// Convert to float.
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / 100_000_000.0
    }
}

// ============================================================================
// Order Book
// ============================================================================

/// Price level in the order book.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    /// Price at this level.
    pub price: Price,
    /// Orders at this level (in time priority).
    pub orders: Vec<u64>, // Order IDs
    /// Total quantity at this level.
    pub total_quantity: Quantity,
    /// Number of orders.
    pub order_count: u32,
}

impl PriceLevel {
    /// Create a new price level.
    pub fn new(price: Price) -> Self {
        Self {
            price,
            orders: Vec::new(),
            total_quantity: Quantity(0),
            order_count: 0,
        }
    }

    /// Add an order to this level.
    pub fn add_order(&mut self, order_id: u64, quantity: Quantity) {
        self.orders.push(order_id);
        self.total_quantity.0 += quantity.0;
        self.order_count += 1;
    }

    /// Remove an order from this level.
    pub fn remove_order(&mut self, order_id: u64, quantity: Quantity) -> bool {
        if let Some(pos) = self.orders.iter().position(|&id| id == order_id) {
            self.orders.remove(pos);
            self.total_quantity.0 = self.total_quantity.0.saturating_sub(quantity.0);
            self.order_count = self.order_count.saturating_sub(1);
            true
        } else {
            false
        }
    }

    /// Check if level is empty.
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }
}

/// Order book for a single symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Symbol ID.
    pub symbol_id: u32,
    /// Bid levels (price -> level), sorted descending by price.
    pub bids: BTreeMap<Price, PriceLevel>,
    /// Ask levels (price -> level), sorted ascending by price.
    pub asks: BTreeMap<Price, PriceLevel>,
    /// Last trade price.
    pub last_price: Option<Price>,
    /// 24h volume.
    pub volume_24h: Quantity,
}

impl OrderBook {
    /// Create a new order book.
    pub fn new(symbol_id: u32) -> Self {
        Self {
            symbol_id,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_price: None,
            volume_24h: Quantity(0),
        }
    }

    /// Get best bid price.
    pub fn best_bid(&self) -> Option<Price> {
        self.bids.keys().next_back().copied()
    }

    /// Get best ask price.
    pub fn best_ask(&self) -> Option<Price> {
        self.asks.keys().next().copied()
    }

    /// Get spread.
    pub fn spread(&self) -> Option<Price> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some(Price(ask.0 - bid.0)),
            _ => None,
        }
    }

    /// Get mid price.
    pub fn mid_price(&self) -> Option<Price> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some(Price((ask.0 + bid.0) / 2)),
            _ => None,
        }
    }

    /// Get total bid depth.
    pub fn bid_depth(&self) -> Quantity {
        Quantity(self.bids.values().map(|l| l.total_quantity.0).sum())
    }

    /// Get total ask depth.
    pub fn ask_depth(&self) -> Quantity {
        Quantity(self.asks.values().map(|l| l.total_quantity.0).sum())
    }

    /// Get depth at N levels.
    pub fn depth_at_levels(&self, n: usize) -> (Quantity, Quantity) {
        let bid_depth: u64 = self.bids.values().rev().take(n).map(|l| l.total_quantity.0).sum();
        let ask_depth: u64 = self.asks.values().take(n).map(|l| l.total_quantity.0).sum();
        (Quantity(bid_depth), Quantity(ask_depth))
    }
}

// ============================================================================
// Trade/Execution Types
// ============================================================================

/// A trade execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade ID.
    pub id: u64,
    /// Symbol ID.
    pub symbol_id: u32,
    /// Buy order ID.
    pub buy_order_id: u64,
    /// Sell order ID.
    pub sell_order_id: u64,
    /// Execution price.
    pub price: Price,
    /// Execution quantity.
    pub quantity: Quantity,
    /// Timestamp.
    pub timestamp: u64,
    /// Aggressor side (which order initiated the trade).
    pub aggressor: Side,
}

/// Matching result for an order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    /// Original order ID.
    pub order_id: u64,
    /// Order status after matching.
    pub status: OrderStatus,
    /// Filled quantity.
    pub filled_quantity: Quantity,
    /// Average fill price.
    pub avg_price: Price,
    /// Generated trades.
    pub trades: Vec<Trade>,
    /// Remaining quantity (if not fully filled).
    pub remaining: Quantity,
}

impl MatchResult {
    /// Create result for no matches.
    pub fn no_match(order_id: u64, quantity: Quantity) -> Self {
        Self {
            order_id,
            status: OrderStatus::New,
            filled_quantity: Quantity(0),
            avg_price: Price(0),
            trades: Vec::new(),
            remaining: quantity,
        }
    }

    /// Create result for full fill.
    pub fn full_fill(order_id: u64, trades: Vec<Trade>) -> Self {
        let filled: u64 = trades.iter().map(|t| t.quantity.0).sum();
        // Calculate weighted average price using f64 to avoid overflow
        let avg_price = if filled > 0 {
            let total_value: f64 = trades
                .iter()
                .map(|t| t.price.to_f64() * t.quantity.to_f64())
                .sum();
            let total_qty: f64 = trades.iter().map(|t| t.quantity.to_f64()).sum();
            Price::from_f64(total_value / total_qty)
        } else {
            Price(0)
        };

        Self {
            order_id,
            status: OrderStatus::Filled,
            filled_quantity: Quantity(filled),
            avg_price,
            trades,
            remaining: Quantity(0),
        }
    }

    /// Create result for partial fill.
    pub fn partial_fill(order_id: u64, trades: Vec<Trade>, remaining: Quantity) -> Self {
        let filled: u64 = trades.iter().map(|t| t.quantity.0).sum();
        // Calculate weighted average price using f64 to avoid overflow
        let avg_price = if filled > 0 {
            let total_value: f64 = trades
                .iter()
                .map(|t| t.price.to_f64() * t.quantity.to_f64())
                .sum();
            let total_qty: f64 = trades.iter().map(|t| t.quantity.to_f64()).sum();
            Price::from_f64(total_value / total_qty)
        } else {
            Price(0)
        };

        Self {
            order_id,
            status: OrderStatus::PartiallyFilled,
            filled_quantity: Quantity(filled),
            avg_price,
            trades,
            remaining,
        }
    }
}

// ============================================================================
// Order Book Snapshot
// ============================================================================

/// Level 2 market data snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2Snapshot {
    /// Symbol ID.
    pub symbol_id: u32,
    /// Bid levels (price, quantity).
    pub bids: Vec<(Price, Quantity)>,
    /// Ask levels (price, quantity).
    pub asks: Vec<(Price, Quantity)>,
    /// Timestamp.
    pub timestamp: u64,
}

impl L2Snapshot {
    /// Create from order book.
    pub fn from_book(book: &OrderBook, depth: usize, timestamp: u64) -> Self {
        let bids: Vec<_> = book
            .bids
            .iter()
            .rev()
            .take(depth)
            .map(|(&price, level)| (price, level.total_quantity))
            .collect();

        let asks: Vec<_> = book
            .asks
            .iter()
            .take(depth)
            .map(|(&price, level)| (price, level.total_quantity))
            .collect();

        Self {
            symbol_id: book.symbol_id,
            bids,
            asks,
            timestamp,
        }
    }
}

// ============================================================================
// Engine Configuration
// ============================================================================

/// Order matching engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Maximum order size.
    pub max_order_size: Quantity,
    /// Minimum order size.
    pub min_order_size: Quantity,
    /// Price tick size.
    pub tick_size: Price,
    /// Maximum price levels in book.
    pub max_price_levels: usize,
    /// Enable self-trade prevention.
    pub self_trade_prevention: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_order_size: Quantity::from_f64(1_000_000.0),
            min_order_size: Quantity::from_f64(0.001),
            tick_size: Price::from_f64(0.01),
            max_price_levels: 1000,
            self_trade_prevention: true,
        }
    }
}
