# rustkernel-compliance

[![Crates.io](https://img.shields.io/crates/v/rustkernel-compliance.svg)](https://crates.io/crates/rustkernel-compliance)
[![Documentation](https://docs.rs/rustkernel-compliance/badge.svg)](https://docs.rs/rustkernel-compliance)
[![License](https://img.shields.io/crates/l/rustkernel-compliance.svg)](LICENSE)

GPU-accelerated compliance kernels for AML, KYC, sanctions screening, and transaction monitoring.

## Kernels (11)

### AML (6 kernels)
- **CircularFlowRatio** - SCC detection for circular transactions
- **ReciprocityFlowRatio** - Mutual transaction detection
- **RapidMovement** - Velocity analysis for structuring
- **AMLPatternDetection** - Multi-pattern FSM detection
- **FlowReversalPattern** - Transaction reversal detection (wash trading, round-tripping)
- **FlowSplitRatio** - Transaction splitting/structuring detection

### KYC (2 kernels)
- **RiskScoring** - Dynamic risk scoring
- **DocumentVerification** - Document validation

### Sanctions (2 kernels)
- **SanctionsScreening** - OFAC/UN/EU screening
- **PEPScreening** - Politically exposed person screening

### Transaction Monitoring (1 kernel)
- **TransactionMonitoring** - Real-time transaction analysis

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernel-compliance = "0.1.0"
```

## Usage

```rust
use rustkernel_compliance::prelude::*;

// Screen transactions for AML patterns
let aml = AMLPatternDetection::new();
let alerts = aml.detect(&transactions);
```

## License

Apache-2.0
