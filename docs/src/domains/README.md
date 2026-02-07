# Kernel Catalogue

RustKernels provides **106 GPU-accelerated kernels** across **14 domain-specific crates**. This catalogue organizes kernels by business domain.

## Quick Reference

| Domain | Crate | Kernels | Primary Use Cases |
|--------|-------|---------|-------------------|
| [Graph Analytics](graph-analytics.md) | `rustkernel-graph` | 28 | Centrality, GNN inference, community detection |
| [Statistical ML](statistical-ml.md) | `rustkernel-ml` | 17 | Clustering, NLP, federated learning, healthcare |
| [Compliance](compliance.md) | `rustkernel-compliance` | 11 | AML, KYC, sanctions screening |
| [Temporal Analysis](temporal-analysis.md) | `rustkernel-temporal` | 7 | Forecasting, seasonality, anomalies |
| [Risk Analytics](risk-analytics.md) | `rustkernel-risk` | 5 | Credit, market, portfolio risk, correlation |
| [Banking](banking.md) | `rustkernel-banking` | 1 | Fraud pattern matching |
| [Behavioral Analytics](behavioral-analytics.md) | `rustkernel-behavioral` | 6 | Profiling, forensics, correlation |
| [Order Matching](order-matching.md) | `rustkernel-orderbook` | 1 | Order book matching engine |
| [Process Intelligence](process-intelligence.md) | `rustkernel-procint` | 7 | DFG, conformance, digital twin simulation |
| [Clearing](clearing.md) | `rustkernel-clearing` | 5 | Netting, settlement, DVP |
| [Treasury](treasury.md) | `rustkernel-treasury` | 5 | Cash flow, FX, liquidity |
| [Accounting](accounting.md) | `rustkernel-accounting` | 9 | Network generation, reconciliation |
| [Payments](payments.md) | `rustkernel-payments` | 2 | Payment processing, flow analysis |
| [Audit](audit.md) | `rustkernel-audit` | 2 | Feature extraction, hypergraph |

## Execution Support

Kernels fall into three registration categories based on their trait implementations:

### Fully Executable (via REST/gRPC)

Kernels implementing `BatchKernel<I, O>` are registered with `register_batch_typed()` and can be executed through the type-erased `BatchKernelDyn` interface used by REST and gRPC endpoints.

Examples: BetweennessCentrality, KMeans, DBSCAN, KYCScoring, ARIMAForecast, StressTesting

### Metadata-Only (Batch)

Kernels implementing `GpuKernel` only are registered with `register_batch_metadata_from()`. They are discoverable through metadata endpoints but require direct Rust API calls for execution.

Examples: GraphDensity, LouvainCommunity, IsolationForest, AMLPatternDetection

### Ring Kernels

Ring kernels are registered with `register_ring_metadata_from()`. They require the RingKernel persistent actor runtime for execution and communicate via lock-free ring buffers.

Examples: PageRankRing, DegreeCentralityRing, OrderMatchingRing, NetworkGenerationRing

## Using the Catalogue

Each domain page includes:

1. **Domain Overview** — Purpose and key use cases
2. **Kernel List** — All kernels with brief descriptions
3. **Kernel Details** — For each kernel:
   - Kernel ID and execution mode
   - Input/output types
   - Usage examples
   - Performance characteristics

## Feature Flags

Enable specific domains via Cargo features:

```toml
# Default domains (graph, ml, compliance, temporal, risk)
rustkernels = "0.4.0"

# Selective
rustkernels = { version = "0.4.0", features = ["accounting", "treasury"] }

# All domains
rustkernels = { version = "0.4.0", features = ["full"] }
```

## Kernel ID Convention

Kernel IDs follow the pattern `{domain}/{kernel-name}`:

```
graph/pagerank
ml/kmeans
compliance/aml-pattern-detection
risk/monte-carlo-var
accounting/network-generation
```

This enables hierarchical organization and clear domain ownership.
