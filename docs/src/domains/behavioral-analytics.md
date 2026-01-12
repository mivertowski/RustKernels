# Behavioral Analytics

**Crate**: `rustkernel-behavioral`
**Kernels**: 6
**Feature**: `behavioral`

Behavioral profiling, forensic analysis, and event correlation kernels.

## Kernel Overview

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| BehavioralProfiling | `behavioral/profiling` | Batch, Ring | Build user behavior profiles |
| AnomalyProfiling | `behavioral/anomaly-profiling` | Batch, Ring | Detect profile deviations |
| FraudSignatureDetection | `behavioral/fraud-signature` | Batch, Ring | Match known fraud patterns |
| CausalGraphConstruction | `behavioral/causal-graph` | Batch | Build causal relationship graphs |
| ForensicQueryExecution | `behavioral/forensic-query` | Batch, Ring | Complex forensic queries |
| EventCorrelationKernel | `behavioral/event-correlation` | Batch, Ring | Correlate events across sources |

---

## Kernel Details

### BehavioralProfiling

Constructs behavioral profiles from historical activity data.

**ID**: `behavioral/profiling`
**Modes**: Batch, Ring

#### Input

```rust
pub struct ProfilingInput {
    pub entity_id: String,
    pub events: Vec<BehaviorEvent>,
    pub profile_type: ProfileType,
    pub time_window_days: u32,
}

pub struct BehaviorEvent {
    pub timestamp: u64,
    pub event_type: String,
    pub attributes: HashMap<String, String>,
    pub numeric_values: HashMap<String, f64>,
}

pub enum ProfileType {
    User,
    Account,
    Device,
    Session,
}
```

#### Output

```rust
pub struct ProfilingOutput {
    pub profile: BehaviorProfile,
    pub confidence: f64,
    pub data_quality: DataQuality,
}

pub struct BehaviorProfile {
    pub entity_id: String,
    pub typical_patterns: Vec<Pattern>,
    pub statistics: ProfileStatistics,
    pub risk_indicators: Vec<RiskIndicator>,
}
```

---

### EventCorrelationKernel

Correlates events across multiple data sources to identify related activities.

**ID**: `behavioral/event-correlation`
**Modes**: Batch, Ring

#### Example

```rust
use rustkernel::behavioral::correlation::{EventCorrelationKernel, CorrelationInput};

let kernel = EventCorrelationKernel::new();

let result = kernel.execute(CorrelationInput {
    events: vec![
        Event { source: "auth", type_: "login_failure", entity: "user123", ts: 1000 },
        Event { source: "auth", type_: "login_success", entity: "user123", ts: 1005 },
        Event { source: "api", type_: "data_export", entity: "user123", ts: 1010 },
    ],
    correlation_window_seconds: 60,
    correlation_rules: default_rules(),
}).await?;

for chain in result.correlated_chains {
    println!("Attack chain detected:");
    for event in chain.events {
        println!("  {} -> {}", event.source, event.type_);
    }
}
```

---

### CausalGraphConstruction

Builds causal relationship graphs from event sequences.

**ID**: `behavioral/causal-graph`
**Modes**: Batch

#### Output

```rust
pub struct CausalGraphOutput {
    pub nodes: Vec<CausalNode>,
    pub edges: Vec<CausalEdge>,
    pub root_causes: Vec<String>,
    pub impact_paths: Vec<ImpactPath>,
}

pub struct CausalEdge {
    pub from: String,
    pub to: String,
    pub strength: f64,
    pub lag_seconds: u64,
}
```

---

## Use Cases

### Security Operations

- Detect account takeover attempts
- Identify insider threats
- Correlate security events across systems

### Fraud Investigation

- Build fraud case timelines
- Identify related accounts
- Trace fund flows across entities

### User Analytics

- Understand user journeys
- Predict churn risk
- Personalize experiences
