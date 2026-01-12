# Process Intelligence

**Crate**: `rustkernel-procint`
**Kernels**: 4
**Feature**: `procint`

Process mining and analysis kernels for business process optimization.

## Kernel Overview

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| DFGConstruction | `procint/dfg-construction` | Batch, Ring | Build Directly-Follows Graphs |
| PartialOrderAnalysis | `procint/partial-order-analysis` | Batch | Analyze process concurrency |
| ConformanceChecking | `procint/conformance-checking` | Batch, Ring | Check process compliance |
| OCPMPatternMatching | `procint/ocpm-pattern-matching` | Batch | Object-Centric Process Mining |

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

- Identify bottlenecks
- Analyze resource utilization
- Recommend improvements
