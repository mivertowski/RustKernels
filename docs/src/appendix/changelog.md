# Changelog

All notable changes to RustKernels are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation site with mdBook
- Kernel catalogue with all 82 kernels documented
- Technical article: Accounting Network Generation

---

## [0.1.0] - 2026-01-12

### Added

#### Infrastructure
- `rustkernel` facade crate with domain re-exports
- `rustkernel-core` with core traits, registry, K2K messaging
- `rustkernel-derive` with `#[gpu_kernel]` and `#[derive(KernelMessage)]` macros
- `rustkernel-cli` command-line interface

#### Graph Analytics (21 kernels)
- PageRank, DegreeCentrality, BetweennessCentrality
- ClosenessCentrality, EigenvectorCentrality, KatzCentrality
- ModularityScore, LouvainCommunity, LabelPropagation
- JaccardSimilarity, CosineSimilarity, AdamicAdarIndex, CommonNeighbors
- GraphDensity, AveragePathLength, ClusteringCoefficient
- ConnectedComponents, FullGraphMetrics
- TriangleCounting, MotifDetection, KCliqueDetection

#### Statistical ML (8 kernels)
- KMeans, DBSCAN, HierarchicalClustering
- IsolationForest, LocalOutlierFactor, EnsembleVoting
- LinearRegression, RidgeRegression

#### Compliance (9 kernels)
- CircularFlowRatio, ReciprocityFlowRatio, RapidMovement
- AMLPatternDetection, KYCScoring, EntityResolution
- SanctionsScreening, PEPScreening, TransactionMonitoring

#### Temporal Analysis (7 kernels)
- ARIMAForecast, ProphetDecomposition, ChangePointDetection
- TimeSeriesAnomalyDetection, SeasonalDecomposition
- TrendExtraction, VolatilityAnalysis

#### Risk Analytics (4 kernels)
- CreditRiskScoring, MonteCarloVaR
- PortfolioRiskAggregation, StressTesting

#### Banking (1 kernel)
- FraudPatternMatch

#### Behavioral Analytics (6 kernels)
- BehavioralProfiling, AnomalyProfiling, FraudSignatureDetection
- CausalGraphConstruction, ForensicQueryExecution, EventCorrelationKernel

#### Order Matching (1 kernel)
- OrderMatchingEngine

#### Process Intelligence (4 kernels)
- DFGConstruction, PartialOrderAnalysis
- ConformanceChecking, OCPMPatternMatching

#### Clearing (5 kernels)
- ClearingValidation, DVPMatching, NettingCalculation
- SettlementExecution, ZeroBalanceFrequency

#### Treasury (5 kernels)
- CashFlowForecasting, CollateralOptimization
- FXHedging, InterestRateRisk, LiquidityOptimization

#### Accounting (7 kernels)
- ChartOfAccountsMapping, JournalTransformation
- GLReconciliation, NetworkAnalysis, TemporalCorrelation
- NetworkGeneration with enhanced features:
  - Account classification (11 classes)
  - VAT/tax detection (EU, GST/HST rates)
  - Transaction pattern recognition (14 patterns)
  - Confidence boosting
- NetworkGenerationRing (streaming mode)

#### Payments (2 kernels)
- PaymentProcessing, FlowAnalysis

#### Audit (2 kernels)
- FeatureExtraction, HypergraphConstruction

### Infrastructure Features
- Batch and Ring execution modes
- K2K (kernel-to-kernel) messaging patterns
- Fixed-point arithmetic for financial precision
- Enterprise licensing system
- Feature flags for selective compilation

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2026-01-12 | Initial release, 82 kernels |

---

## Migration Guides

### From DotCompute (C#)

RustKernels is a Rust port of DotCompute. Key differences:

1. **Async execution**: All kernel execution is async
2. **Ownership**: Rust ownership model affects API design
3. **Error handling**: Uses `Result<T, E>` instead of exceptions
4. **Ring messages**: Use rkyv serialization instead of protobuf

See migration guide (coming soon) for detailed instructions.
