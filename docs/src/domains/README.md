# Kernel Catalogue

RustKernels provides **82 GPU-accelerated kernels** across **14 domain-specific crates**. This catalogue organizes kernels by business domain.

## Quick Reference

| Domain | Crate | Kernels | Primary Use Cases |
|--------|-------|---------|-------------------|
| [Graph Analytics](graph-analytics.md) | `rustkernel-graph` | 21 | Centrality, community detection, similarity |
| [Statistical ML](statistical-ml.md) | `rustkernel-ml` | 8 | Clustering, anomaly detection, regression |
| [Compliance](compliance.md) | `rustkernel-compliance` | 9 | AML, KYC, sanctions screening |
| [Temporal Analysis](temporal-analysis.md) | `rustkernel-temporal` | 7 | Forecasting, seasonality, anomalies |
| [Risk Analytics](risk-analytics.md) | `rustkernel-risk` | 4 | Credit, market, portfolio risk |
| [Banking](banking.md) | `rustkernel-banking` | 1 | Fraud pattern matching |
| [Behavioral Analytics](behavioral-analytics.md) | `rustkernel-behavioral` | 6 | Profiling, forensics, correlation |
| [Order Matching](order-matching.md) | `rustkernel-orderbook` | 1 | Order book matching engine |
| [Process Intelligence](process-intelligence.md) | `rustkernel-procint` | 4 | DFG, conformance checking |
| [Clearing](clearing.md) | `rustkernel-clearing` | 5 | Netting, settlement, DVP |
| [Treasury](treasury.md) | `rustkernel-treasury` | 5 | Cash flow, FX, liquidity |
| [Accounting](accounting.md) | `rustkernel-accounting` | 7 | Network generation, reconciliation |
| [Payments](payments.md) | `rustkernel-payments` | 2 | Payment processing, flow analysis |
| [Audit](audit.md) | `rustkernel-audit` | 2 | Feature extraction, hypergraph |

## Kernels by Execution Mode

### Batch-Only Kernels (19)

Heavy computation kernels that only support batch mode:

- Graph: BetweennessCentrality, FullGraphMetrics
- ML: DBSCAN, HierarchicalClustering, IsolationForest
- Compliance: EntityResolution, TransactionMonitoring
- And more...

### Ring-Only Kernels (0)

Currently all Ring-capable kernels also support Batch mode.

### Dual-Mode Kernels (63)

Kernels supporting both Batch and Ring execution:

- All centrality measures (PageRank, Degree, Closeness, etc.)
- All clustering algorithms (KMeans, Louvain, etc.)
- All risk calculations (VaR, Credit Scoring, etc.)
- And more...

## Using the Catalogue

Each domain page includes:

1. **Domain Overview** - Purpose and key use cases
2. **Kernel List** - All kernels with brief descriptions
3. **Kernel Details** - For each kernel:
   - Kernel ID and modes
   - Input/output types
   - Usage examples
   - Performance characteristics

## Feature Flags

Enable specific domains via Cargo features:

```toml
# Default domains
rustkernel = "0.1.0"  # graph, ml, compliance, temporal, risk

# Selective
rustkernel = { version = "0.1.0", features = ["accounting", "treasury"] }

# All domains
rustkernel = { version = "0.1.0", features = ["full"] }
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
