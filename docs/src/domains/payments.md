# Payments

**Crate**: `rustkernel-payments`
**Kernels**: 2
**Feature**: `payments`

Payment processing and flow analysis kernels for payment systems.

## Kernel Overview

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| PaymentProcessing | `payments/processing` | Batch, Ring | Process payment instructions |
| FlowAnalysis | `payments/flow-analysis` | Batch, Ring | Analyze payment flows |

---

## Kernel Details

### PaymentProcessing

Processes and validates payment instructions.

**ID**: `payments/processing`
**Modes**: Batch, Ring

#### Input

```rust
pub struct PaymentProcessingInput {
    pub payments: Vec<Payment>,
    pub validation_rules: Vec<ValidationRule>,
    pub processing_window: ProcessingWindow,
}

pub struct Payment {
    pub id: String,
    pub sender: PartyInfo,
    pub receiver: PartyInfo,
    pub amount: f64,
    pub currency: String,
    pub payment_type: PaymentType,
    pub value_date: u64,
    pub reference: String,
    pub remittance_info: Option<String>,
}

pub enum PaymentType {
    Wire,
    ACH,
    SEPA,
    SWIFT,
    RTP,  // Real-Time Payments
    FedNow,
}
```

#### Output

```rust
pub struct PaymentProcessingOutput {
    pub processed: Vec<ProcessedPayment>,
    pub rejected: Vec<RejectedPayment>,
    pub pending: Vec<String>,
    pub statistics: ProcessingStatistics,
}

pub struct ProcessedPayment {
    pub id: String,
    pub status: PaymentStatus,
    pub processing_time_us: u64,
    pub fees: f64,
}

pub struct RejectedPayment {
    pub id: String,
    pub reason: RejectionReason,
    pub failed_rules: Vec<String>,
}
```

#### Example

```rust
use rustkernel::payments::processing::{PaymentProcessing, PaymentProcessingInput};

let kernel = PaymentProcessing::new();

let result = kernel.execute(PaymentProcessingInput {
    payments: incoming_payments,
    validation_rules: default_rules(),
    processing_window: ProcessingWindow::Same_Day,
}).await?;

println!("Processed: {}", result.processed.len());
println!("Rejected: {}", result.rejected.len());

for rejection in result.rejected {
    println!("Rejected {}: {:?}", rejection.id, rejection.reason);
}
```

---

### FlowAnalysis

Analyzes payment flows for patterns, anomalies, and liquidity insights.

**ID**: `payments/flow-analysis`
**Modes**: Batch, Ring

#### Input

```rust
pub struct FlowAnalysisInput {
    pub payments: Vec<Payment>,
    pub analysis_type: FlowAnalysisType,
    pub time_window: TimeWindow,
}

pub enum FlowAnalysisType {
    VolumeAnalysis,
    NetworkAnalysis,
    AnomalyDetection,
    LiquidityForecasting,
}
```

#### Output

```rust
pub struct FlowAnalysisOutput {
    /// Flow statistics by counterparty
    pub counterparty_flows: Vec<CounterpartyFlow>,
    /// Detected anomalies
    pub anomalies: Vec<FlowAnomaly>,
    /// Network metrics
    pub network_metrics: NetworkMetrics,
    /// Liquidity forecast
    pub liquidity_forecast: Option<LiquidityForecast>,
}

pub struct CounterpartyFlow {
    pub counterparty: String,
    pub inflow_volume: f64,
    pub outflow_volume: f64,
    pub net_flow: f64,
    pub transaction_count: u32,
}
```

#### Example

```rust
use rustkernel::payments::flow::{FlowAnalysis, FlowAnalysisInput};

let kernel = FlowAnalysis::new();

let result = kernel.execute(FlowAnalysisInput {
    payments: daily_payments,
    analysis_type: FlowAnalysisType::NetworkAnalysis,
    time_window: TimeWindow::Days(30),
}).await?;

// Top counterparties by volume
for flow in result.counterparty_flows.iter().take(10) {
    println!("{}: in=${:.0}, out=${:.0}, net=${:.0}",
        flow.counterparty,
        flow.inflow_volume,
        flow.outflow_volume,
        flow.net_flow
    );
}
```

---

## Ring Mode for Real-Time Processing

```rust
use rustkernel::payments::processing::PaymentProcessingRing;

let ring = PaymentProcessingRing::new();

// Process payments as they arrive
for payment in payment_stream {
    let result = ring.process(payment).await?;

    match result.status {
        PaymentStatus::Processed => send_confirmation(result),
        PaymentStatus::Rejected(reason) => notify_sender(reason),
        PaymentStatus::Pending => queue_for_review(result),
    }
}
```

---

## Use Cases

- **Payment hubs**: Central payment processing
- **Real-time payments**: Instant payment validation
- **Correspondent banking**: SWIFT message processing
- **Treasury**: Cash position forecasting
