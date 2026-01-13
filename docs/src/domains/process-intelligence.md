# Process Intelligence

**Crate**: `rustkernel-procint`
**Kernels**: 7
**Feature**: `procint`

Process mining and analysis kernels for business process optimization.

## Kernel Overview

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| DFGConstruction | `procint/dfg-construction` | Batch, Ring | Build Directly-Follows Graphs |
| PartialOrderAnalysis | `procint/partial-order-analysis` | Batch | Analyze process concurrency |
| ConformanceChecking | `procint/conformance-checking` | Batch, Ring | Check process compliance |
| OCPMPatternMatching | `procint/ocpm-pattern-matching` | Batch | Object-Centric Process Mining |
| NextActivityPrediction | `procint/next-activity-prediction` | Batch | Predict next activity in process |
| EventLogImputation | `procint/event-log-imputation` | Batch | Handle missing events in logs |
| DigitalTwin | `procint/digital-twin` | Batch | Process simulation for what-if analysis |

---

## Kernel Details

### DFGConstruction

Constructs Directly-Follows Graphs from event logs.

**ID**: `procint/dfg-construction`
**Modes**: Batch, Ring

#### Input

```rust
pub struct DFGInput {
    /// Event log entries
    pub events: Vec<ProcessEvent>,
    /// Minimum edge frequency threshold
    pub min_frequency: u32,
}

pub struct ProcessEvent {
    pub case_id: String,
    pub activity: String,
    pub timestamp: u64,
    pub resource: Option<String>,
    pub attributes: HashMap<String, String>,
}
```

#### Output

```rust
pub struct DFGOutput {
    /// Activities (nodes)
    pub activities: Vec<Activity>,
    /// Edges with frequencies
    pub edges: Vec<DFGEdge>,
    /// Start activities
    pub start_activities: Vec<String>,
    /// End activities
    pub end_activities: Vec<String>,
}

pub struct DFGEdge {
    pub from: String,
    pub to: String,
    pub frequency: u32,
    pub avg_duration_seconds: f64,
}
```

#### Example

```rust
use rustkernel::procint::dfg::{DFGConstruction, DFGInput};

let kernel = DFGConstruction::new();

let result = kernel.execute(DFGInput {
    events: vec![
        ProcessEvent { case_id: "C1".into(), activity: "Submit".into(), timestamp: 1000, .. },
        ProcessEvent { case_id: "C1".into(), activity: "Review".into(), timestamp: 2000, .. },
        ProcessEvent { case_id: "C1".into(), activity: "Approve".into(), timestamp: 3000, .. },
    ],
    min_frequency: 1,
}).await?;

for edge in result.edges {
    println!("{} -> {} (freq: {}, avg: {:.1}s)",
        edge.from,
        edge.to,
        edge.frequency,
        edge.avg_duration_seconds
    );
}
```

---

### ConformanceChecking

Checks if process executions conform to a reference model.

**ID**: `procint/conformance-checking`
**Modes**: Batch, Ring

#### Input

```rust
pub struct ConformanceInput {
    pub events: Vec<ProcessEvent>,
    pub reference_model: ProcessModel,
    pub tolerance: ConformanceTolerance,
}

pub struct ProcessModel {
    pub activities: Vec<String>,
    pub transitions: Vec<(String, String)>,
    pub start: String,
    pub end: String,
}
```

#### Output

```rust
pub struct ConformanceOutput {
    /// Fitness score (0.0 - 1.0)
    pub fitness: f64,
    /// Precision score
    pub precision: f64,
    /// Deviations found
    pub deviations: Vec<Deviation>,
    /// Per-case conformance
    pub case_conformance: HashMap<String, f64>,
}

pub struct Deviation {
    pub case_id: String,
    pub deviation_type: DeviationType,
    pub activity: String,
    pub description: String,
}
```

---

### OCPMPatternMatching

Object-Centric Process Mining for complex, multi-object processes.

**ID**: `procint/ocpm-pattern-matching`
**Modes**: Batch

#### Example

```rust
use rustkernel::procint::ocpm::{OCPMPatternMatching, OCPMInput};

let kernel = OCPMPatternMatching::new();

let result = kernel.execute(OCPMInput {
    events: order_events,
    object_types: vec!["Order".into(), "Item".into(), "Delivery".into()],
    patterns: vec![
        Pattern::BottleneckDetection,
        Pattern::ObjectLifecycle,
        Pattern::InteractionAnalysis,
    ],
}).await?;

for bottleneck in result.bottlenecks {
    println!("Bottleneck: {} (avg wait: {:.1}h)",
        bottleneck.activity,
        bottleneck.avg_wait_hours
    );
}
```

---

### NextActivityPrediction

Predicts the next activity in a process using sequence models.

**ID**: `procint/next-activity-prediction`
**Modes**: Batch

#### Example

```rust
use rustkernel::procint::prediction::{NextActivityPrediction, PredictionConfig};

let kernel = NextActivityPrediction::new();

let config = PredictionConfig {
    sequence_length: 10,
    top_k: 3,
};

let predictions = kernel.predict(&event_sequence, &config)?;
for (activity, prob) in predictions {
    println!("{}: {:.1}%", activity, prob * 100.0);
}
```

---

### EventLogImputation

Handles missing events, incorrect timestamps, and duplicates in event logs.

**ID**: `procint/event-log-imputation`
**Modes**: Batch

#### Example

```rust
use rustkernel::procint::imputation::{EventLogImputation, ImputationConfig};

let kernel = EventLogImputation::new();

let config = ImputationConfig {
    detect_missing: true,
    fix_timestamps: true,
    remove_duplicates: true,
};

let cleaned_log = kernel.impute(&raw_events, &config)?;
println!("Fixed {} issues", cleaned_log.issues_fixed);
```

---

### DigitalTwin

Process simulation for what-if analysis and optimization using Monte Carlo methods.

**ID**: `procint/digital-twin`
**Modes**: Batch

#### Example

```rust
use rustkernel::procint::simulation::{DigitalTwin, ProcessModel, SimulationConfig};

let kernel = DigitalTwin::new();

let config = SimulationConfig {
    num_simulations: 1000,
    time_horizon_hours: 24.0,
    seed: Some(42),
};

let result = kernel.simulate(&process_model, &config)?;
println!("Avg completion time: {:.2}h", result.avg_completion_time_hours);
println!("Bottleneck: {}", result.bottlenecks[0].activity);
```

---

## Use Cases

### Process Discovery

- Automatically discover process models from logs
- Identify common paths and variants
- Measure process performance

### Compliance Monitoring

- Ensure processes follow defined procedures
- Detect deviations in real-time
- Generate audit trails

### Process Optimization

- Identify bottlenecks using DigitalTwin simulation
- Analyze resource utilization
- Run what-if scenarios for capacity planning

### Predictive Analytics

- Predict next activities for proactive intervention
- Clean and impute event logs for better analysis
- Estimate remaining process time
