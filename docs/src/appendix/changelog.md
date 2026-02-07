# Changelog

All notable changes to RustKernels are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [0.4.0] - 2026-02-07

### Added

#### Production-Ready Kernel Execution
- **TypeErasedBatchKernel**: Bridges typed `BatchKernel<I, O>` to `BatchKernelDyn` via JSON serialization, enabling REST/gRPC dispatch without compile-time type knowledge
- **TypeErasedRingKernel**: Equivalent wrapper for ring kernels
- **Factory Registration**: `register_batch_typed(factory)` with automatic type inference from `BatchKernel<I, O>` implementations
- **Metadata Registration**: `register_batch_metadata_from(factory)` and `register_ring_metadata_from(factory)` for discovery-only registration
- **Registry Execution**: `execute_batch(kernel_id, input_json)` convenience method for type-erased execution

#### Real Ecosystem Execution (Replacing All Stubs)
- **Axum**: `execute_kernel()` now performs real batch kernel dispatch with configurable timeout; ring kernels return HTTP 422 with guidance
- **Tower**: `KernelService::execute()` performs real batch kernel dispatch
- **Tonic gRPC**: `execute_kernel()` performs real execution with gRPC deadline support; exceeded deadlines return `DEADLINE_EXCEEDED`
- **Actix**: Actor handler performs real execution via `tokio::task::block_in_place` bridge
- **Health endpoint**: Aggregated component health with registry status and error rate
- **Metrics endpoint**: Per-domain kernel counts, batch/ring breakdown, error rate

#### Deep RingKernel 0.4.2 Integration
- Bidirectional domain conversion (`Domain::to_ring_domain()`, `Domain::from_ring_domain()`)
- New re-exports: `ControlBlock`, `Backend`, `KernelStatus`, `RuntimeMetrics`, `K2KConfig`, `Priority`
- Submodule re-exports: `checkpoint`, `dispatcher`, `health`, `pubsub`
- Ring message type ID ranges aligned with `ringkernel_core::domain::Domain` base offsets

### Changed
- Upgraded to RingKernel 0.4.2 from 0.3.1
- All 14 domain crates now use factory-based registration (`register_batch_typed`, `register_batch_metadata_from`, `register_ring_metadata_from`)
- Ecosystem integrations execute real kernels instead of returning mock responses
- Updated prelude with `BatchKernelDyn`, `RingKernelDyn`, `TypeErasedBatchKernel`, `TypeErasedRingKernel`

---

## [0.3.0] - 2026-01-28

### Added

#### New Kernels (24 kernels added, bringing total to 106)
- **Graph**: GNNInference, GraphAttention, TopologicalSort, CycleDetection, ShortestPath, BipartiteMatching, GraphColoring
- **ML**: EmbeddingGeneration, SemanticSimilarity, SecureAggregation, DrugInteractionPrediction, ClinicalPathwayConformance, StreamingIsolationForest, AdaptiveThreshold, SHAPValues, FeatureImportance
- **Compliance**: FlowReversalPattern, FlowSplitRatio
- **Process Intelligence**: DigitalTwin, NextActivityPrediction, EventLogImputation
- **Risk**: CorrelationStress
- **Accounting**: DuplicateDetection, CurrencyConversion

#### Enterprise Enhancements
- Ring message definitions for all 14 domains with `#[derive(RingMessage)]`
- K2K coordination patterns: `IterativeState`, `ScatterGatherState`, `FanOutTracker`, `PipelineTracker`
- Domain-specific Ring message type ID ranges (100–799)

### Changed
- Upgraded to RingKernel 0.3.1
- Graph analytics expanded from 21 to 28 kernels
- ML expanded from 8 to 17 kernels
- Risk analytics expanded from 4 to 5 kernels
- Process intelligence expanded from 4 to 7 kernels
- Accounting expanded from 7 to 9 kernels
- Compliance expanded from 9 to 11 kernels

---

## [0.2.0] - 2026-01-19

### Added

#### Enterprise Security (`rustkernel-core/src/security/`)
- **Authentication**: JWT and API key validation via `AuthConfig`
- **RBAC**: Role-based access control with `KernelPermission` (Execute, Configure, Monitor, Admin)
- **Multi-tenancy**: Tenant isolation with `TenantId` and resource quotas
- **Secrets Management**: `SecretStore` abstraction for credential management
- **Security Context**: Unified context for auth/tenant/permission propagation

#### Observability (`rustkernel-core/src/observability/`)
- **Metrics**: Prometheus-compatible metrics via `KernelMetrics`
- **Distributed Tracing**: OTLP export support via `KernelTracing`
- **Structured Logging**: Kernel context propagation with configurable levels
- **Alerting**: SLO-based alerts with `AlertRule` and multiple notification channels

#### Resilience Patterns (`rustkernel-core/src/resilience/`)
- **Circuit Breaker**: Failure isolation with configurable thresholds
- **Retry**: Exponential backoff with jitter via `RetryConfig`
- **Timeouts**: Deadline propagation in K2K chains via `DeadlineContext`
- **Health Checks**: Liveness, readiness, and startup probes via `HealthProbe`
- **Recovery Policies**: Configurable recovery strategies for kernel failures

#### Runtime Lifecycle (`rustkernel-core/src/runtime/`)
- **Lifecycle State Machine**: Starting -> Running -> Draining -> Stopped
- **Runtime Presets**: Development, production, and high-performance configurations
- **Graceful Shutdown**: Drain period with active connection tracking
- **Configuration Validation**: Runtime parameter validation with hot reload support

#### Memory Management (`rustkernel-core/src/memory/`)
- **Size-Stratified Pooling**: `KernelMemoryManager` with bucket-based allocation
- **Pressure Handling**: Configurable thresholds with `PressureLevel` enum
- **Multi-Phase Reductions**: `InterPhaseReduction<T>` for iterative algorithms
- **Analytics Contexts**: Workload-specific buffer management via `AnalyticsContextManager`
- **Sync Modes**: Cooperative, SoftwareBarrier, and MultiLaunch synchronization

#### Production Configuration (`rustkernel-core/src/config/`)
- **Unified Config**: `ProductionConfig` combining all enterprise settings
- **Builder Pattern**: `ProductionConfigBuilder` with fluent API
- **Environment Loading**: `from_env()` with `RUSTKERNEL_*` variable overrides
- **File Loading**: TOML configuration file support via `from_file()`

#### Ecosystem Integrations (`rustkernel-ecosystem/`)
- **New Crate**: `rustkernel-ecosystem` for service deployments
- **Axum REST API**: `KernelRouter` with endpoints for kernels, execute, health, metrics
- **Tower Middleware**: `TimeoutLayer`, `RateLimiterLayer`, `KernelService`
- **gRPC Server**: `KernelGrpcServer` via Tonic
- **Actix Actors**: `KernelActor` with message handlers

#### Enhanced Core Traits
- `GpuKernel`: Added `health_check()`, `shutdown()`, `refresh_config()` methods
- `BatchKernel`: Added `execute_with_context()` for auth/tenant propagation
- `RingKernelHandler`: Added `handle_secure()` for security context
- **New Trait**: `CheckpointableKernel` for recovery/restart support
- **New Trait**: `DegradableKernel` for graceful degradation
- **New Trait**: `IterativeKernel` for multi-pass algorithms

#### CLI Enhancements
- `rustkernel runtime status|show|init` — Runtime lifecycle management
- `rustkernel health [--format json]` — Component health checks
- `rustkernel config show|validate|generate|env` — Configuration management

### Changed
- Upgraded to RingKernel 0.3.1 from 0.2.0
- Workspace now includes 19 crates (added `rustkernel-ecosystem`)
- Updated Tokio to 1.48
- Enhanced prelude with all enterprise module exports

---

## [0.1.1] - 2026-01-15

### Changed
- Renamed crate from `rustkernel` to `rustkernels` (crate name taken on crates.io)
- Added consistent README files for all 18 crates
- Resolved all compiler warnings for clean build
- Fixed User-Agent header in crates.io API requests

---

## [0.1.0] - 2026-01-12

### Added

#### Infrastructure
- `rustkernel` facade crate with domain re-exports
- `rustkernel-core` with core traits, registry, K2K messaging
- `rustkernel-derive` with `#[gpu_kernel]` and `#[derive(KernelMessage)]` macros
- `rustkernel-cli` command-line interface

#### 82 Kernels across 14 Domains
- **Graph Analytics** (21): PageRank, DegreeCentrality, BetweennessCentrality, ClosenessCentrality, EigenvectorCentrality, KatzCentrality, ModularityScore, LouvainCommunity, LabelPropagation, JaccardSimilarity, CosineSimilarity, AdamicAdarIndex, CommonNeighbors, GraphDensity, AveragePathLength, ClusteringCoefficient, ConnectedComponents, FullGraphMetrics, TriangleCounting, MotifDetection, KCliqueDetection
- **Statistical ML** (8): KMeans, DBSCAN, HierarchicalClustering, IsolationForest, LocalOutlierFactor, EnsembleVoting, LinearRegression, RidgeRegression
- **Compliance** (9): CircularFlowRatio, ReciprocityFlowRatio, RapidMovement, AMLPatternDetection, KYCScoring, EntityResolution, SanctionsScreening, PEPScreening, TransactionMonitoring
- **Temporal Analysis** (7): ARIMAForecast, ProphetDecomposition, ChangePointDetection, TimeSeriesAnomalyDetection, SeasonalDecomposition, TrendExtraction, VolatilityAnalysis
- **Risk Analytics** (4): CreditRiskScoring, MonteCarloVaR, PortfolioRiskAggregation, StressTesting
- **Banking** (1): FraudPatternMatch
- **Behavioral Analytics** (6): BehavioralProfiling, AnomalyProfiling, FraudSignatureDetection, CausalGraphConstruction, ForensicQueryExecution, EventCorrelationKernel
- **Order Matching** (1): OrderMatchingEngine
- **Process Intelligence** (4): DFGConstruction, PartialOrderAnalysis, ConformanceChecking, OCPMPatternMatching
- **Clearing** (5): ClearingValidation, DVPMatching, NettingCalculation, SettlementExecution, ZeroBalanceFrequency
- **Treasury** (5): CashFlowForecasting, CollateralOptimization, FXHedging, InterestRateRisk, LiquidityOptimization
- **Accounting** (7): ChartOfAccountsMapping, JournalTransformation, GLReconciliation, NetworkAnalysis, TemporalCorrelation, NetworkGeneration, NetworkGenerationRing
- **Payments** (2): PaymentProcessing, FlowAnalysis
- **Audit** (2): FeatureExtraction, HypergraphConstruction

#### Infrastructure Features
- Batch and Ring execution modes
- K2K (kernel-to-kernel) messaging patterns
- Fixed-point arithmetic for financial precision
- Enterprise licensing system
- Feature flags for selective compilation

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.4.0 | 2026-02-07 | Production execution, type erasure, RingKernel 0.4.2, real ecosystem dispatch |
| 0.3.0 | 2026-01-28 | 24 new kernels (106 total), Ring messages, K2K coordination |
| 0.2.0 | 2026-01-19 | Enterprise features: security, observability, resilience, ecosystem crate |
| 0.1.1 | 2026-01-15 | Crate rename to rustkernels, documentation |
| 0.1.0 | 2026-01-12 | Initial release, 82 kernels across 14 domains |

---

## Migration Guides

### From 0.3.x to 0.4.0

1. **Registration**: Replace `register_metadata(kernel.metadata().clone())` with factory-based methods:
   - `register_batch_typed(MyKernel::new)` for kernels implementing `BatchKernel<I, O>`
   - `register_batch_metadata_from(MyKernel::new)` for metadata-only batch kernels
   - `register_ring_metadata_from(MyKernel::new)` for ring kernels
2. **Execution**: Use `registry.execute_batch(id, json_bytes)` for type-erased execution
3. **Ecosystem**: All service endpoints now execute real kernels — remove any mock/fallback code

### From DotCompute (C#)

RustKernels is a Rust port of DotCompute. Key differences:

1. **Async execution**: All kernel execution is async
2. **Ownership**: Rust ownership model affects API design
3. **Error handling**: Uses `Result<T, E>` instead of exceptions
4. **Ring messages**: Use rkyv serialization instead of protobuf
