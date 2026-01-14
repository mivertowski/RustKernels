# rustkernel-banking

[![Crates.io](https://img.shields.io/crates/v/rustkernel-banking.svg)](https://crates.io/crates/rustkernel-banking)
[![Documentation](https://docs.rs/rustkernel-banking/badge.svg)](https://docs.rs/rustkernel-banking)
[![License](https://img.shields.io/crates/l/rustkernel-banking.svg)](LICENSE)

GPU-accelerated banking kernels for fraud detection.

## Kernels (1)

- **FraudPatternMatch** - Multi-pattern fraud detection combining:
  - Aho-Corasick pattern matching
  - Rapid split (structuring) detection
  - Circular flow detection
  - Velocity and amount anomalies
  - Geographic anomaly (impossible travel)
  - Mule account detection

## Features

- Real-time fraud pattern matching
- Multi-pattern detection in a single pass
- Configurable detection thresholds
- Alert generation with severity scoring

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernel-banking = "0.1.0"
```

## Usage

```rust
use rustkernel_banking::prelude::*;

// Detect fraud patterns in transactions
let detector = FraudPatternMatch::new();
let alerts = detector.detect(&transactions);
```

## License

Apache-2.0
