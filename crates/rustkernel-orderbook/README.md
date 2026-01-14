# rustkernel-orderbook

[![Crates.io](https://img.shields.io/crates/v/rustkernel-orderbook.svg)](https://crates.io/crates/rustkernel-orderbook)
[![Documentation](https://docs.rs/rustkernel-orderbook/badge.svg)](https://docs.rs/rustkernel-orderbook)
[![License](https://img.shields.io/crates/l/rustkernel-orderbook.svg)](LICENSE)

GPU-accelerated order book matching for high-frequency trading.

## Kernels (1)

- **OrderMatchingEngine** - Price-time priority matching (<10us P99)

## Features

- Price-time priority matching
- Support for limit and market orders
- Self-trade prevention
- Order book management with L2 snapshots
- Batch order processing
- Sub-10 microsecond P99 latency

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernel-orderbook = "0.1.0"
```

## Usage

```rust
use rustkernel_orderbook::prelude::*;

// Create an order matching engine
let engine = OrderMatchingEngine::new();

// Process orders
let matches = engine.process(&orders);
```

## License

Apache-2.0
