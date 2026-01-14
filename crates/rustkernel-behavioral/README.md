# rustkernel-behavioral

[![Crates.io](https://img.shields.io/crates/v/rustkernel-behavioral.svg)](https://crates.io/crates/rustkernel-behavioral)
[![Documentation](https://docs.rs/rustkernel-behavioral/badge.svg)](https://docs.rs/rustkernel-behavioral)
[![License](https://img.shields.io/crates/l/rustkernel-behavioral.svg)](LICENSE)

GPU-accelerated behavioral analytics kernels for profiling and forensics.

## Kernels (6)

- **BehavioralProfiling** - Feature extraction for user behavior
- **AnomalyProfiling** - Deviation scoring from behavioral baseline
- **FraudSignatureDetection** - Known fraud pattern matching
- **CausalGraphConstruction** - DAG inference from event streams
- **ForensicQueryExecution** - Historical pattern search and analysis
- **EventCorrelationKernel** - Temporal event correlation and clustering

## Features

- User behavior profiling and baseline establishment
- Real-time anomaly detection against behavioral baselines
- Pattern matching for known fraud signatures
- Causal relationship discovery from events
- Forensic analysis and historical pattern search
- Event correlation across multiple streams

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernel-behavioral = "0.1.0"
```

## Usage

```rust
use rustkernel_behavioral::prelude::*;

// Profile user behavior
let profiler = BehavioralProfiling::new();
let profile = profiler.extract_features(&user_events);
```

## License

Apache-2.0
