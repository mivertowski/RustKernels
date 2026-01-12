//! Message types for order matching engine kernel.
//!
//! Input/output message types for the `BatchKernel` trait implementations
//! and Ring kernel messages for K2K communication.
//!
//! Note: Full RingMessage support requires FR-1 (KernelMessage ↔ RingMessage bridge)
//! from the RustCompute feature request. See docs/RUSTCOMPUTE_FEATURE_REQUEST.md

use rustkernel_derive::KernelMessage;
use serde::{Deserialize, Serialize};

use crate::types::{L2Snapshot, MatchResult, Order, Price, Quantity};

// ============================================================================
// Submit Order Messages
// ============================================================================

/// Input for submitting a single order.
///
/// Ring message type_id: 1000 (OrderMatching domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 1000, domain = "OrderMatching")]
pub struct SubmitOrderInput {
    /// Order to submit.
    pub order: Order,
}

impl SubmitOrderInput {
    /// Create a new submit order input.
    pub fn new(order: Order) -> Self {
        Self { order }
    }
}

/// Output from submitting a single order.
///
/// Ring message type_id: 1001 (OrderMatching domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 1001, domain = "OrderMatching")]
pub struct SubmitOrderOutput {
    /// Match result from the order submission.
    pub result: MatchResult,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Batch Order Messages
// ============================================================================

/// Input for processing a batch of orders.
///
/// Ring message type_id: 1010 (OrderMatching domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 1010, domain = "OrderMatching")]
pub struct BatchOrderInput {
    /// Orders to process.
    pub orders: Vec<Order>,
}

impl BatchOrderInput {
    /// Create a new batch order input.
    pub fn new(orders: Vec<Order>) -> Self {
        Self { orders }
    }

    /// Create from a single order.
    pub fn single(order: Order) -> Self {
        Self {
            orders: vec![order],
        }
    }
}

/// Output from processing a batch of orders.
///
/// Ring message type_id: 1011 (OrderMatching domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 1011, domain = "OrderMatching")]
pub struct BatchOrderOutput {
    /// Match results for each order.
    pub results: Vec<MatchResult>,
    /// Total trades generated.
    pub total_trades: usize,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Cancel Order Messages
// ============================================================================

/// Input for canceling an order.
///
/// Ring message type_id: 1020 (OrderMatching domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 1020, domain = "OrderMatching")]
pub struct CancelOrderInput {
    /// Order ID to cancel.
    pub order_id: u64,
}

impl CancelOrderInput {
    /// Create a new cancel order input.
    pub fn new(order_id: u64) -> Self {
        Self { order_id }
    }
}

/// Output from canceling an order.
///
/// Ring message type_id: 1021 (OrderMatching domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 1021, domain = "OrderMatching")]
pub struct CancelOrderOutput {
    /// Canceled order (if found and active).
    pub canceled_order: Option<Order>,
    /// Whether the cancellation succeeded.
    pub success: bool,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Modify Order Messages
// ============================================================================

/// Input for modifying an order.
///
/// Ring message type_id: 1030 (OrderMatching domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 1030, domain = "OrderMatching")]
pub struct ModifyOrderInput {
    /// Order ID to modify.
    pub order_id: u64,
    /// New price (optional).
    pub new_price: Option<Price>,
    /// New quantity (optional).
    pub new_quantity: Option<Quantity>,
}

impl ModifyOrderInput {
    /// Create a new modify order input.
    pub fn new(order_id: u64, new_price: Option<Price>, new_quantity: Option<Quantity>) -> Self {
        Self {
            order_id,
            new_price,
            new_quantity,
        }
    }

    /// Create to modify price only.
    pub fn modify_price(order_id: u64, new_price: Price) -> Self {
        Self {
            order_id,
            new_price: Some(new_price),
            new_quantity: None,
        }
    }

    /// Create to modify quantity only.
    pub fn modify_quantity(order_id: u64, new_quantity: Quantity) -> Self {
        Self {
            order_id,
            new_price: None,
            new_quantity: Some(new_quantity),
        }
    }
}

/// Output from modifying an order.
///
/// Ring message type_id: 1031 (OrderMatching domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 1031, domain = "OrderMatching")]
pub struct ModifyOrderOutput {
    /// Match result if modification succeeded (may match against book).
    pub result: Option<MatchResult>,
    /// Whether the modification succeeded.
    pub success: bool,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

// ============================================================================
// Order Book Snapshot Messages
// ============================================================================

/// Input for getting an order book snapshot.
///
/// Ring message type_id: 1040 (OrderMatching domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 1040, domain = "OrderMatching")]
pub struct GetSnapshotInput {
    /// Symbol ID.
    pub symbol_id: u32,
    /// Depth (number of price levels).
    pub depth: usize,
}

impl GetSnapshotInput {
    /// Create a new snapshot input.
    pub fn new(symbol_id: u32, depth: usize) -> Self {
        Self { symbol_id, depth }
    }

    /// Create with default depth (10 levels).
    pub fn with_default_depth(symbol_id: u32) -> Self {
        Self {
            symbol_id,
            depth: 10,
        }
    }
}

/// Output from getting an order book snapshot.
///
/// Ring message type_id: 1041 (OrderMatching domain)
#[derive(Debug, Clone, Serialize, Deserialize, KernelMessage)]
#[message(type_id = 1041, domain = "OrderMatching")]
pub struct GetSnapshotOutput {
    /// L2 order book snapshot (if symbol exists).
    pub snapshot: Option<L2Snapshot>,
    /// Computation time in microseconds.
    pub compute_time_us: u64,
}
