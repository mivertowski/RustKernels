# rustkernel-procint

[![Crates.io](https://img.shields.io/crates/v/rustkernel-procint.svg)](https://crates.io/crates/rustkernel-procint)
[![Documentation](https://docs.rs/rustkernel-procint/badge.svg)](https://docs.rs/rustkernel-procint)
[![License](https://img.shields.io/crates/l/rustkernel-procint.svg)](LICENSE)

GPU-accelerated process intelligence kernels for process mining and conformance checking.

## Kernels (7)

- **DFGConstruction** - Directly-follows graph construction from event logs
- **PartialOrderAnalysis** - Concurrency detection
- **ConformanceChecking** - Multi-model conformance (DFG/Petri/BPMN)
- **OCPMPatternMatching** - Object-centric process mining
- **NextActivityPrediction** - Markov/N-gram next activity prediction
- **EventLogImputation** - Event log quality detection and repair
- **DigitalTwin** - Monte Carlo process simulation for what-if analysis

## Features

- Directly-follows graph construction from event logs
- Process conformance checking against multiple model types
- Object-centric process mining for complex processes
- Predictive process monitoring
- Event log quality improvement
- Digital twin simulation

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernel-procint = "0.1.0"
```

## Usage

```rust
use rustkernel_procint::prelude::*;

// Build a directly-follows graph from event logs
let dfg = DFGConstruction::new();
let graph = dfg.build(&event_log);
```

## License

Apache-2.0
