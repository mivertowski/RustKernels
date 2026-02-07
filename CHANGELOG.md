# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2025-05-15

### Added

- **Python bindings** (`rustkernel-python`) — native Python access to all 106 batch kernels via PyO3
  - `pip install rustkernels` for Python 3.10+
  - `KernelRegistry` class with discovery (`kernel_ids`, `by_domain`, `search`) and `execute()` for batch kernels
  - Module-level `execute()` convenience function with cached default registry
  - Catalog API: `list_domains()`, `total_kernel_count()`, `enabled_domains()`
  - Full exception hierarchy mapping all `KernelError` variants
  - GIL-releasing async bridge (`py.allow_threads()` + internal tokio runtime)
  - Feature pass-through: `--features full` enables all 14 domains
  - `scripts/publish-python.sh` for PyPI publishing via maturin
- **Deep RingKernel 0.4.2 integration** across all crates
  - Bidirectional `Domain` conversion between RustKernels and RingKernel
  - Re-exports: `ControlBlock`, `Backend`, `KernelStatus`, `RuntimeMetrics`, `K2KConfig`, `Priority`
  - Submodule re-exports: `checkpoint`, `dispatcher`, `health`, `pubsub`
- **Production-ready kernel execution** — all 106 kernels fully implemented with real dispatch (no stubs)
- **Enterprise modules** in `rustkernel-core`:
  - Security: JWT/API key auth, RBAC, multi-tenancy, secrets management
  - Observability: Prometheus metrics, OTLP tracing, structured logging, SLO alerting
  - Resilience: circuit breakers, exponential retry, timeout propagation, health probes
  - Runtime: lifecycle state machine, graceful shutdown, production config presets
  - Memory: size-stratified pools, pressure handling, inter-phase reductions
- **Ecosystem service layer** (`rustkernel-ecosystem`):
  - Axum REST API (`KernelRouter`)
  - Tower middleware (timeout, rate limiting)
  - Tonic gRPC server (`KernelGrpcServer`)
  - Actix actor integration (`KernelActor`)
- **New kernels**:
  - Graph: `GNNInference`, `GraphAttention`
  - ML: `EmbeddingGeneration`, `SemanticSimilarity`, `SecureAggregation`, `DrugInteractionPrediction`, `ClinicalPathwayConformance`, `StreamingIsolationForest`, `AdaptiveThreshold`, `SHAPValues`, `FeatureImportance`
  - Process Intelligence: `DigitalTwin`, `NextActivityPrediction`, `EventLogImputation`
- **K2K coordination** patterns: `IterativeState`, `ScatterGatherState`, `FanOutTracker`, `PipelineTracker`
- **Type-erased batch execution** via `KernelRegistry::execute_batch(kernel_id, json_bytes)`
- **CLI tool** (`rustkernel-cli`) for kernel management

### Changed

- Workspace expanded to 20 crates (added `rustkernel-python`)
- `rustkernel-python` excluded from `default-members` (requires Python dev headers)
- `pyo3 = "0.23"` added to workspace dependencies

## [0.3.0] - 2025-03-01

### Added

- Initial 14-domain workspace with 106 kernel stubs
- Core traits: `GpuKernel`, `BatchKernel`, `RingKernelHandler`, `IterativeKernel`
- Kernel registry with typed factory registration
- Domain-based licensing system
- Proc macro derive: `#[gpu_kernel]`, `#[derive(KernelMessage)]`

[0.4.0]: https://github.com/mivertowski/RustKernels/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/mivertowski/RustKernels/releases/tag/v0.3.0
