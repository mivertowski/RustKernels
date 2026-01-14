# rustkernel-accounting

[![Crates.io](https://img.shields.io/crates/v/rustkernel-accounting.svg)](https://crates.io/crates/rustkernel-accounting)
[![Documentation](https://docs.rs/rustkernel-accounting/badge.svg)](https://docs.rs/rustkernel-accounting)
[![License](https://img.shields.io/crates/l/rustkernel-accounting.svg)](LICENSE)

GPU-accelerated accounting kernels.

## Kernels (9)

- **ChartOfAccountsMapping** - Entity-specific CoA mapping
- **JournalTransformation** - GL mapping and transformation
- **GLReconciliation** - Account matching and reconciliation
- **NetworkAnalysis** - Intercompany relationship analysis
- **TemporalCorrelation** - Account correlation over time
- **NetworkGeneration** - Journal entry to accounting network transformation
- **NetworkGenerationRing** - Streaming network generation
- **SuspenseAccountDetection** - Centrality-based suspense account detection
- **GaapViolationDetection** - GAAP prohibited flow pattern detection

## Features

- Chart of accounts mapping and standardization
- Journal entry transformation and GL mapping
- Automated account reconciliation
- Intercompany transaction analysis
- Temporal correlation analysis
- Graph-based accounting network analysis
- GAAP compliance checking

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernel-accounting = "0.1.0"
```

## Usage

```rust
use rustkernel_accounting::prelude::*;

// Reconcile GL accounts
let reconciler = GLReconciliation::new();
let matches = reconciler.reconcile(&source, &target);
```

## License

Apache-2.0
