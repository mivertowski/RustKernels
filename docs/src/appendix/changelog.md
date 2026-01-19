# Changelog

All notable changes to RustKernels are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- **Lifecycle State Machine**: Starting → Running → Draining → Stopped
- **Runtime Presets**: Development, production, and high-performance configurations
- **Graceful Shutdown**: Drain period with active connection tracking
- **Configuration Validation**: Runtime parameter validation with hot reload support

#### Memory Management (`rustkernel-core/src/memory/`)
- **Size-Stratified Pooling**: `KernelMemoryManager` with bucket-based allocation
- **Pressure Handling**: Configurable thresholds with `PressureLevel` enum
- **Multi-Phase Reductions**: `InterPhaseReduction<T>` for iterative algorithms (PageRank, K-Means)
- **Analytics Contexts**: Workload-specific buffer management via `AnalyticsContextManager`
- **Sync Modes**: Cooperative, SoftwareBarrier, and MultiLaunch synchronization

#### Production Configuration (`rustkernel-core/src/config/`)
- **Unified Config**: `ProductionConfig` combining all enterprise settings
- **Builder Pattern**: `ProductionConfigBuilder` with fluent API
- **Environment Loading**: `from_env()` with RUSTKERNEL_* variable overrides
- **File Loading**: TOML configuration file support via `from_file()`

#### Ecosystem Integrations (`rustkernel-ecosystem/`)
- **New Crate**: `rustkernel-ecosystem` for service deployments
- **Axum REST API**: `KernelRouter` with endpoints for kernels, execute, health, metrics
- **Tower Middleware**: `TimeoutLayer`, `RateLimiterLayer`, `KernelService`
- **gRPC Server**: `KernelGrpcServer` via Tonic
- **Actix Actors**: `KernelActor` with message handlers for GPU-persistent actors

#### Enhanced Core Traits
- `GpuKernel`: Added `health_check()`, `shutdown()`, `refresh_config()` methods
- `BatchKernel`: Added `execute_with_context()` for auth/tenant propagation
- `RingKernelHandler`: Added `handle_secure()` for security context
- **New Trait**: `CheckpointableKernel` for recovery/restart support
- **New Trait**: `DegradableKernel` for graceful degradation
- **New Trait**: `IterativeKernel` for multi-pass algorithms

#### CLI Enhancements
- `rustkernel runtime status|show|init` - Runtime lifecycle management
- `rustkernel health [--format json]` - Component health checks
- `rustkernel config show|validate|generate|env` - Configuration management

### Changed
- Upgraded to RingKernel 0.3.1 from 0.2.0
- Workspace now includes 19 crates (added `rustkernel-ecosystem`)
- Updated Tokio to 1.48
- Enhanced prelude with all enterprise module exports

### Documentation
- Updated CLAUDE.md with enterprise features
- Added code examples for all new modules

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
| 0.2.0 | 2026-01-19 | Enterprise features: security, observability, resilience, ecosystem |
| 0.1.1 | 2026-01-15 | Crate rename to rustkernels, documentation |
| 0.1.0 | 2026-01-12 | Initial release, 106 kernels across 14 domains |

---

## Migration Guides

### From DotCompute (C#)

RustKernels is a Rust port of DotCompute. Key differences:

1. **Async execution**: All kernel execution is async
2. **Ownership**: Rust ownership model affects API design
3. **Error handling**: Uses `Result<T, E>` instead of exceptions
4. **Ring messages**: Use rkyv serialization instead of protobuf

See migration guide (coming soon) for detailed instructions.
