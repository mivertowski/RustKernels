//! # RustKernel Order Matching
//!
//! GPU-accelerated order book matching for HFT.
//!
//! ## Kernels
//! - `OrderMatchingEngine` - Price-time priority matching (<10μs P99)
//!
//! ## Features
//! - Price-time priority matching
//! - Support for limit and market orders
//! - Self-trade prevention
//! - Order book management with L2 snapshots
//! - Batch order processing

#![warn(missing_docs)]

pub mod matching;
pub mod messages;
pub mod ring_messages;
pub mod types;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::matching::*;
    pub use crate::messages::*;
    pub use crate::types::*;
}

// Re-export main kernel
pub use matching::OrderMatchingEngine;

// Re-export key types
pub use types::{
    EngineConfig, L2Snapshot, MatchResult, Order, OrderBook, OrderStatus, OrderType, Price,
    PriceLevel, Quantity, Side, TimeInForce, Trade,
};

/// Register all order matching kernels with a registry.
pub fn register_all(
    _registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering order matching kernels");
    Ok(())
}
