# Order Matching

**Crate**: `rustkernel-orderbook`
**Kernels**: 1
**Feature**: `orderbook`

High-performance order book matching engine for trading systems.

## Kernel Overview

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| OrderMatchingEngine | `orderbook/matching-engine` | Batch, Ring | GPU-accelerated order matching |

---

## Kernel Details

### OrderMatchingEngine

Ultra-low latency order matching with price-time priority.

**ID**: `orderbook/matching-engine`
**Modes**: Batch, Ring
**Latency**: <1μs per order (Ring mode)

#### Input

```rust
pub struct OrderInput {
    pub orders: Vec<Order>,
    pub symbol: String,
}

pub struct Order {
    pub id: String,
    pub side: Side,
    pub price: f64,
    pub quantity: u64,
    pub order_type: OrderType,
    pub time_in_force: TimeInForce,
    pub timestamp: u64,
}

pub enum Side {
    Buy,
    Sell,
}

pub enum OrderType {
    Limit,
    Market,
    StopLimit { trigger_price: f64 },
    IcebergLimit { display_qty: u64 },
}

pub enum TimeInForce {
    GTC,  // Good Till Cancelled
    IOC,  // Immediate Or Cancel
    FOK,  // Fill Or Kill
    GTD { expiry: u64 },  // Good Till Date
}
```

#### Output

```rust
pub struct OrderOutput {
    pub executions: Vec<Execution>,
    pub book_state: BookState,
    pub statistics: MatchingStatistics,
}

pub struct Execution {
    pub execution_id: String,
    pub buy_order_id: String,
    pub sell_order_id: String,
    pub price: f64,
    pub quantity: u64,
    pub timestamp: u64,
}

pub struct BookState {
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
    pub last_trade_price: f64,
    pub last_trade_quantity: u64,
}
```

#### Example

```rust
use rustkernel::orderbook::{OrderMatchingEngine, OrderInput, Order, Side, OrderType};

let kernel = OrderMatchingEngine::new();

let result = kernel.execute(OrderInput {
    orders: vec![
        Order {
            id: "O1".into(),
            side: Side::Buy,
            price: 100.50,
            quantity: 1000,
            order_type: OrderType::Limit,
            time_in_force: TimeInForce::GTC,
            timestamp: 1699000000,
        },
        Order {
            id: "O2".into(),
            side: Side::Sell,
            price: 100.50,
            quantity: 500,
            order_type: OrderType::Limit,
            time_in_force: TimeInForce::GTC,
            timestamp: 1699000001,
        },
    ],
    symbol: "AAPL".into(),
}).await?;

for exec in result.executions {
    println!("Execution: {} shares @ ${:.2}",
        exec.quantity,
        exec.price
    );
}
```

---

## Ring Mode for Live Trading

Ring mode maintains order book state on GPU for sub-microsecond matching:

```rust
use rustkernel::orderbook::OrderMatchingRing;

let ring = OrderMatchingRing::new("AAPL");

// Process incoming orders
async fn process_order(order: Order) -> Vec<Execution> {
    match order.order_type {
        OrderType::Limit => ring.add_limit_order(order).await?,
        OrderType::Market => ring.add_market_order(order).await?,
        _ => unimplemented!(),
    }
}

// Cancel order
ring.cancel_order("O123").await?;

// Query book state
let book = ring.get_book_snapshot().await?;
println!("Best bid: ${:.2}", book.bids[0].price);
println!("Best ask: ${:.2}", book.asks[0].price);
```

---

## Matching Rules

### Price-Time Priority

Orders are matched following price-time priority:

1. Best price first (highest bid, lowest ask)
2. Earlier orders at same price matched first

### Order Types

| Type | Behavior |
|------|----------|
| Limit | Rests on book until filled or cancelled |
| Market | Executes immediately at best available |
| Stop-Limit | Converts to limit when trigger price hit |
| Iceberg | Only displays partial quantity |

### Time in Force

| TIF | Behavior |
|-----|----------|
| GTC | Remains until filled or cancelled |
| IOC | Fill available, cancel remainder |
| FOK | Fill entire quantity or reject |
| GTD | Expires at specified time |

---

## Performance Characteristics

- **Throughput**: >1M orders/sec (batch)
- **Latency**: <1μs per order (ring)
- **Book depth**: Unlimited price levels
- **Symbols**: One ring per symbol

## Integration Notes

For production trading systems:

1. **Sequencer**: Orders must be sequenced before matching
2. **Persistence**: Log all orders and executions
3. **Risk checks**: Pre-trade risk should precede matching
4. **Market data**: Publish book updates after each match
