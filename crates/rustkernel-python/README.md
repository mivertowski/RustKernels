# rustkernels

**Python bindings for the RustKernels GPU-accelerated kernel library.**

[![PyPI](https://img.shields.io/pypi/v/rustkernels.svg)](https://pypi.org/project/rustkernels/)
[![Python](https://img.shields.io/pypi/pyversions/rustkernels.svg)](https://pypi.org/project/rustkernels/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)

---

## Overview

`rustkernels` provides native Python access to 106 GPU-accelerated kernels across 14 financial and analytical domains. Built with [PyO3](https://pyo3.rs), it wraps the Rust [RustKernels](https://github.com/mivertowski/RustKernels) library and its [RingKernel 0.4.2](https://crates.io/crates/ringkernel-core) runtime.

Use it directly from Python or Jupyter notebooks without needing a REST/gRPC server.

**Scope**: Batch kernel execution only. Ring kernels (sub-microsecond persistent actors) require the Rust runtime directly or the ecosystem service layer.

---

## Installation

```bash
pip install rustkernels
```

### From Source

Requires Rust 1.85+ and Python 3.10+.

```bash
pip install maturin
cd crates/rustkernel-python
maturin develop --features full
```

---

## Quick Start

```python
import rustkernels

# Check version
print(rustkernels.__version__)          # "0.4.0"
print(rustkernels.ringkernel_version)   # "0.4.2"

# Create a registry (auto-populates with all enabled kernels)
reg = rustkernels.KernelRegistry()
print(f"{len(reg)} kernels available")

# Execute a batch kernel
result = reg.execute("graph/betweenness_centrality", {
    "num_nodes": 4,
    "edges": [[0, 1], [1, 2], [2, 3], [0, 3]],
    "normalized": True,
})
print(result)

# Module-level convenience (uses a cached default registry)
result = rustkernels.execute("graph/betweenness_centrality", {
    "num_nodes": 4,
    "edges": [[0, 1], [1, 2], [2, 3], [0, 3]],
    "normalized": True,
})
```

---

## API Reference

### Registry

```python
reg = rustkernels.KernelRegistry()

# Discovery
reg.kernel_ids                     # list[str] — all registered kernel IDs
reg.batch_kernel_ids               # list[str] — batch-executable kernel IDs
reg.total_count                    # int
reg.stats                          # RegistryStats

# Lookup
reg.get("graph/betweenness")       # KernelMetadata | None
reg.by_domain("GraphAnalytics")    # list[KernelMetadata]
reg.by_mode("batch")               # list[KernelMetadata]
reg.search("centrality")           # list[KernelMetadata]
"graph/betweenness" in reg         # bool
len(reg)                           # int

# Execution
result = reg.execute("graph/betweenness_centrality", {...})  # dict
```

### Catalog

```python
rustkernels.list_domains()         # list[DomainInfo]
rustkernels.total_kernel_count()   # 106
rustkernels.enabled_domains()      # ["graph", "ml", ...]
```

### Data Classes

**KernelMetadata** — kernel identity and performance targets:
- `id`, `mode`, `domain`, `description`
- `expected_throughput`, `target_latency_us`
- `requires_gpu_native`, `version`

**RegistryStats** — aggregate counts:
- `total`, `batch_kernels`, `ring_kernels`
- `by_domain` (`dict[str, int]`)

**DomainInfo** — domain catalog entry:
- `name`, `description`, `kernel_count`, `feature`, `domain`

### Exceptions

All exceptions inherit from `rustkernels.KernelError`:

| Exception | Raised When |
|---|---|
| `KernelNotFoundError` | Kernel ID not in registry |
| `ValidationError` | Invalid input data |
| `SerializationError` | JSON serialization/deserialization failure |
| `ExecutionError` | Kernel launch, device, or internal error |
| `TimeoutError` | Execution timeout exceeded |
| `LicenseError` | License or domain restriction |
| `AuthorizationError` | Unauthorized access |
| `ResourceExhaustedError` | Rate limit or queue full |
| `ServiceUnavailableError` | Backend unavailable |

```python
try:
    result = reg.execute("graph/betweenness_centrality", bad_input)
except rustkernels.ValidationError as e:
    print(f"Bad input: {e}")
except rustkernels.KernelError as e:
    print(f"Kernel error: {e}")
```

---

## Domain Coverage

| Domain | Kernels | Examples |
|---|---|---|
| Graph Analytics | 28 | PageRank, Louvain, GNN inference, betweenness centrality |
| Statistical ML | 17 | K-Means, DBSCAN, isolation forest, SHAP values |
| Compliance | 11 | AML circular flow, sanctions screening, KYC scoring |
| Temporal Analysis | 7 | ARIMA, Prophet decomposition, change point detection |
| Risk Analytics | 5 | Monte Carlo VaR, credit scoring, stress testing |
| Process Intelligence | 7 | DFG construction, conformance checking, digital twin |
| Behavioral Analytics | 6 | Profiling, forensics, causal graph analysis |
| Treasury | 5 | Liquidity optimization, FX hedging, NSFR |
| Clearing | 5 | Multilateral netting, DVP matching, settlement |
| Accounting | 9 | Network generation, reconciliation, GAAP detection |
| Banking | 1 | Fraud pattern matching |
| Order Matching | 1 | Price-time priority order book |
| Payments | 2 | Payment processing, flow analysis |
| Audit | 2 | Feature extraction, hypergraph construction |

---

## Feature Flags

When building from source, domain features mirror the Rust crate:

```bash
# Default (graph, ml, compliance, temporal, risk)
maturin develop

# All 14 domains
maturin develop --features full

# Selective
maturin develop --features graph,compliance,procint
```

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.10+ |
| Rust | 1.85+ (build only) |
| RingKernel | 0.4.2 (bundled) |
| CUDA Toolkit | 12.0+ (optional; CPU fallback when unavailable) |

---

## License

Apache License, Version 2.0. See [LICENSE](../../LICENSE).
