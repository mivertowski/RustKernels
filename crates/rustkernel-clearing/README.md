# rustkernel-clearing

[![Crates.io](https://img.shields.io/crates/v/rustkernel-clearing.svg)](https://crates.io/crates/rustkernel-clearing)
[![Documentation](https://docs.rs/rustkernel-clearing/badge.svg)](https://docs.rs/rustkernel-clearing)
[![License](https://img.shields.io/crates/l/rustkernel-clearing.svg)](LICENSE)

GPU-accelerated clearing and settlement kernels.

## Kernels (5)

- **ClearingValidation** - Trade validation for clearing eligibility
- **DVPMatching** - Delivery vs payment matching
- **NettingCalculation** - Multilateral netting calculation
- **SettlementExecution** - Settlement instruction execution
- **ZeroBalanceFrequency** - Settlement efficiency metrics

## Features

- Trade validation with counterparty/security eligibility checks
- DVP instruction matching with tolerance-based scoring
- Multilateral netting to reduce gross obligations
- Settlement execution with priority and partial settlement support
- Zero balance frequency and efficiency metrics

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernel-clearing = "0.1.0"
```

## Usage

```rust
use rustkernel_clearing::prelude::*;

// Calculate netting positions
let netting = NettingCalculation::new();
let positions = netting.calculate(&trades);
```

## License

Apache-2.0
