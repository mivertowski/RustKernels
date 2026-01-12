//! Ring message types for Order Matching Engine.
//!
//! These messages implement the `RingMessage` trait for GPU-native persistent
//! actor communication with ultra-low latency (<10μs P99).
//!
//! The OrderMatchingEngine is a Tier 1 critical-path kernel that benefits from
//! Ring mode for:
//! - Sub-microsecond message serialization via rkyv
//! - GPU-resident state for order book maintenance
//! - K2K messaging for cross-symbol coordination

use ringkernel_core::message::{CorrelationId, MessageId};
use ringkernel_derive::RingMessage;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

use crate::types::{Order, Price, Quantity, Side};

// ============================================================================
// Ring Message Type IDs (OrderMatching domain: 500-599)
// ============================================================================
//
// Type ID assignments within OrderMatching domain:
// - 0-9: Order submission messages
// - 10-19: Batch order messages
// - 20-29: Cancel/modify messages
// - 30-39: Query messages
// - 40-49: K2K coordination messages
// - 50+: Reserved

// ============================================================================
// Submit Order Ring Messages
// ============================================================================

/// Ring message for submitting a single order.
///
/// This is the primary message type for ultra-low latency order submission.
/// Type ID: 500 (OrderMatching.base + 0)
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, RingMessage)]
#[message(type_id = 500)]
#[archive(check_bytes)]
pub struct SubmitOrderRing {
    /// Message ID for tracking.
    #[message(id)]
    pub id: MessageId,
    /// Correlation ID for request-response pairing.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Order ID.
    pub order_id: u64,
    /// Symbol ID.
    pub symbol_id: u32,
    /// Order side.
    pub side: RingSide,
    /// Order type.
    pub order_type: RingOrderType,
    /// Price (fixed-point, 0 for market orders).
    pub price: i64,
    /// Quantity (fixed-point).
    pub quantity: u64,
    /// Trader ID.
    pub trader_id: u64,
    /// Timestamp (nanoseconds).
    pub timestamp: u64,
}

impl SubmitOrderRing {
    /// Create a new limit order submission message.
    pub fn limit(
        order_id: u64,
        symbol_id: u32,
        side: Side,
        price: Price,
        quantity: Quantity,
        trader_id: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            id: MessageId::generate(),
            correlation_id: CorrelationId::generate(),
            order_id,
            symbol_id,
            side: side.into(),
            order_type: RingOrderType::Limit,
            price: price.0,
            quantity: quantity.0,
            trader_id,
            timestamp,
        }
    }

    /// Create a new market order submission message.
    pub fn market(
        order_id: u64,
        symbol_id: u32,
        side: Side,
        quantity: Quantity,
        trader_id: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            id: MessageId::generate(),
            correlation_id: CorrelationId::generate(),
            order_id,
            symbol_id,
            side: side.into(),
            order_type: RingOrderType::Market,
            price: 0,
            quantity: quantity.0,
            trader_id,
            timestamp,
        }
    }

    /// Convert to domain Order type.
    pub fn to_order(&self) -> Order {
        match self.order_type {
            RingOrderType::Limit => Order::limit(
                self.order_id,
                self.symbol_id,
                self.side.into(),
                Price(self.price),
                Quantity(self.quantity),
                self.trader_id,
                self.timestamp,
            ),
            RingOrderType::Market => Order::market(
                self.order_id,
                self.symbol_id,
                self.side.into(),
                Quantity(self.quantity),
                self.trader_id,
                self.timestamp,
            ),
        }
    }
}

/// Ring message response for order submission.
///
/// Type ID: 501 (OrderMatching.base + 1)
#[derive(Debug, Clone, RingMessage, Archive, RkyvSerialize, RkyvDeserialize)]
#[message(type_id = 501)]
#[archive(check_bytes)]
pub struct SubmitOrderResponse {
    /// Correlation ID matching the request.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Order ID.
    pub order_id: u64,
    /// Result status.
    pub status: RingOrderStatus,
    /// Filled quantity.
    pub filled_quantity: u64,
    /// Average fill price (fixed-point).
    pub avg_price: i64,
    /// Remaining quantity.
    pub remaining: u64,
    /// Number of trades generated.
    pub trade_count: u32,
    /// Error message (empty if success).
    pub error: RingString,
}

// ============================================================================
// Cancel Order Ring Messages
// ============================================================================

/// Ring message for canceling an order.
///
/// Type ID: 520 (OrderMatching.base + 20)
#[derive(Debug, Clone, RingMessage, Archive, RkyvSerialize, RkyvDeserialize)]
#[message(type_id = 520)]
#[archive(check_bytes)]
pub struct CancelOrderRing {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Order ID to cancel.
    pub order_id: u64,
}

impl CancelOrderRing {
    /// Create a new cancel order message.
    pub fn new(order_id: u64) -> Self {
        Self {
            id: MessageId::generate(),
            correlation_id: CorrelationId::generate(),
            order_id,
        }
    }
}

/// Ring message response for order cancellation.
///
/// Type ID: 521 (OrderMatching.base + 21)
#[derive(Debug, Clone, RingMessage, Archive, RkyvSerialize, RkyvDeserialize)]
#[message(type_id = 521)]
#[archive(check_bytes)]
pub struct CancelOrderResponse {
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Order ID.
    pub order_id: u64,
    /// Whether cancellation succeeded.
    pub success: bool,
    /// Remaining quantity at cancellation.
    pub remaining: u64,
}

// ============================================================================
// Query Ring Messages
// ============================================================================

/// Ring message for querying order book state.
///
/// Type ID: 530 (OrderMatching.base + 30)
#[derive(Debug, Clone, RingMessage, Archive, RkyvSerialize, RkyvDeserialize)]
#[message(type_id = 530)]
#[archive(check_bytes)]
pub struct QueryBookRing {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Symbol ID.
    pub symbol_id: u32,
    /// Depth (number of price levels).
    pub depth: u32,
}

impl QueryBookRing {
    /// Create a new book query message.
    pub fn new(symbol_id: u32, depth: u32) -> Self {
        Self {
            id: MessageId::generate(),
            correlation_id: CorrelationId::generate(),
            symbol_id,
            depth,
        }
    }
}

/// Ring message response for book query.
///
/// Type ID: 531 (OrderMatching.base + 31)
#[derive(Debug, Clone, RingMessage, Archive, RkyvSerialize, RkyvDeserialize)]
#[message(type_id = 531)]
#[archive(check_bytes)]
pub struct QueryBookResponse {
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Symbol ID.
    pub symbol_id: u32,
    /// Best bid price.
    pub best_bid: i64,
    /// Best ask price.
    pub best_ask: i64,
    /// Bid depth at requested levels.
    pub bid_depth: u64,
    /// Ask depth at requested levels.
    pub ask_depth: u64,
    /// Spread.
    pub spread: i64,
    /// Mid price.
    pub mid_price: i64,
}

// ============================================================================
// K2K Coordination Messages (for cross-symbol matching)
// ============================================================================

/// K2K message for cross-symbol trade coordination.
///
/// Type ID: 540 (OrderMatching.base + 40)
#[derive(Debug, Clone, RingMessage, Archive, RkyvSerialize, RkyvDeserialize)]
#[message(type_id = 540)]
#[archive(check_bytes)]
pub struct CrossSymbolTrade {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Correlation ID.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Source symbol ID.
    pub source_symbol: u32,
    /// Target symbol ID.
    pub target_symbol: u32,
    /// Trade quantity.
    pub quantity: u64,
    /// Execution price.
    pub price: i64,
    /// Timestamp.
    pub timestamp: u64,
}

// ============================================================================
// Ring-Compatible Types (rkyv-serializable)
// ============================================================================

/// Ring-compatible order side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
#[repr(u8)]
pub enum RingSide {
    /// Buy order.
    Buy = 0,
    /// Sell order.
    Sell = 1,
}

impl From<Side> for RingSide {
    fn from(side: Side) -> Self {
        match side {
            Side::Buy => RingSide::Buy,
            Side::Sell => RingSide::Sell,
        }
    }
}

impl From<RingSide> for Side {
    fn from(side: RingSide) -> Self {
        match side {
            RingSide::Buy => Side::Buy,
            RingSide::Sell => Side::Sell,
        }
    }
}

/// Ring-compatible order type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
#[repr(u8)]
pub enum RingOrderType {
    /// Limit order.
    Limit = 0,
    /// Market order.
    Market = 1,
}

/// Ring-compatible order status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
#[repr(u8)]
pub enum RingOrderStatus {
    /// New order accepted.
    New = 0,
    /// Partially filled.
    PartiallyFilled = 1,
    /// Fully filled.
    Filled = 2,
    /// Canceled.
    Canceled = 3,
    /// Rejected.
    Rejected = 4,
}

/// Fixed-size string for Ring messages (64 bytes).
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
pub struct RingString {
    /// String data (null-terminated).
    pub data: [u8; 64],
    /// Actual length.
    pub len: u8,
}

impl RingString {
    /// Create an empty string.
    pub fn empty() -> Self {
        Self {
            data: [0; 64],
            len: 0,
        }
    }

    /// Create from a string slice.
    pub fn from_str(s: &str) -> Self {
        let bytes = s.as_bytes();
        let len = bytes.len().min(63) as u8;
        let mut data = [0u8; 64];
        data[..len as usize].copy_from_slice(&bytes[..len as usize]);
        Self { data, len }
    }

    /// Convert to string.
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.data[..self.len as usize]).unwrap_or("")
    }
}

impl Default for RingString {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_submit_order_ring() {
        let msg = SubmitOrderRing::limit(
            1,
            100,
            Side::Buy,
            Price::from_f64(100.0),
            Quantity::from_f64(10.0),
            1000,
            0,
        );
        assert_eq!(msg.order_id, 1);
        assert_eq!(msg.symbol_id, 100);
        assert_eq!(msg.side, RingSide::Buy);
    }

    #[test]
    fn test_ring_string() {
        let s = RingString::from_str("Hello, World!");
        assert_eq!(s.as_str(), "Hello, World!");
        assert_eq!(s.len, 13);
    }

    #[test]
    fn test_ring_string_truncation() {
        let long = "a".repeat(100);
        let s = RingString::from_str(&long);
        assert_eq!(s.len, 63);
    }
}
