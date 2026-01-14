# rustkernel-audit

[![Crates.io](https://img.shields.io/crates/v/rustkernel-audit.svg)](https://crates.io/crates/rustkernel-audit)
[![Documentation](https://docs.rs/rustkernel-audit/badge.svg)](https://docs.rs/rustkernel-audit)
[![License](https://img.shields.io/crates/l/rustkernel-audit.svg)](LICENSE)

GPU-accelerated financial audit kernels.

## Kernels (2)

- **FeatureExtraction** - Audit feature vector extraction for ML analysis
- **HypergraphConstruction** - Multi-way relationship hypergraph construction

## Features

- Feature extraction for audit analytics
- Hypergraph construction for complex relationship modeling
- Support for multi-way relationships in audit data
- ML-ready feature vectors

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernel-audit = "0.1.0"
```

## Usage

```rust
use rustkernel_audit::prelude::*;

// Extract audit features
let extractor = FeatureExtraction::new();
let features = extractor.extract(&audit_data);
```

## License

Apache-2.0
