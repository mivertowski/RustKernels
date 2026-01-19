//! # RustKernel Core
//!
//! Core abstractions, traits, and registry for the RustKernels GPU kernel library.
//!
//! This crate provides:
//! - Domain and kernel type definitions
//! - Kernel metadata and configuration
//! - Trait definitions for batch and ring kernels
//! - Kernel registry with auto-discovery
//! - Licensing and feature gating system
//! - Actix actor integration for GPU-backed actors
//! - Runtime lifecycle management (0.3.1)
//! - Enterprise security, observability, and resilience patterns (0.3.1)

#![warn(missing_docs)]
#![warn(clippy::all)]

// Core modules
pub mod domain;
pub mod error;
pub mod k2k;
pub mod kernel;
pub mod license;
pub mod messages;
pub mod registry;
pub mod slo;
pub mod test_kernels;
pub mod traits;

// Enterprise modules (0.3.1)
pub mod config;
pub mod memory;
pub mod observability;
pub mod resilience;
pub mod runtime;
pub mod security;

// Re-exports from ringkernel-core for convenience
pub use ringkernel_core::{
    HlcTimestamp, MessageHeader, MessageId, MessageQueue, RingContext, RingKernelError, RingMessage,
};

// Re-export types from specific modules
pub use ringkernel_core::hlc::HlcClock;
pub use ringkernel_core::k2k::{K2KBroker, K2KEndpoint, K2KMessage};
pub use ringkernel_core::message::MessageEnvelope;
pub use ringkernel_core::runtime::{
    KernelHandle, KernelId, KernelState, LaunchOptions, RingKernelRuntime,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::domain::Domain;
    pub use crate::error::{KernelError, Result};
    pub use crate::k2k::{
        FanOutTracker, IterativeConvergenceSummary, IterativeState, K2KControlMessage, K2KPriority,
        K2KWorkerResult, PipelineTracker, ScatterGatherState, kernel_id_to_u64,
    };
    pub use crate::kernel::{KernelMetadata, KernelMode};
    pub use crate::license::{DevelopmentLicense, License, LicenseError, LicenseValidator};
    pub use crate::messages::{
        BatchMessage, CorrelationId, KernelRequest, KernelResponse, KernelResult,
    };
    pub use crate::registry::{KernelRegistry, RegistryStats};
    pub use crate::slo::{SLOResult, SLOValidator};
    pub use crate::test_kernels::{EchoKernel, MatMul, ReduceSum, VectorAdd};
    pub use crate::traits::{
        BatchKernel, CheckpointableKernel, DegradableKernel, ExecutionContext, GpuKernel,
        HealthStatus, IterativeKernel, KernelConfig, RingKernelHandler, SecureRingContext,
    };

    // Runtime lifecycle (0.3.1)
    pub use crate::runtime::{
        KernelRuntime, LifecycleState, RuntimeBuilder, RuntimeConfig, RuntimeHandle,
        RuntimePreset, RuntimeStats,
    };

    // Resilience patterns (0.3.1)
    pub use crate::resilience::{
        CircuitBreaker, CircuitBreakerConfig, CircuitState, DeadlineContext, HealthCheck,
        HealthCheckResult, HealthProbe, RecoveryPolicy, ResilienceConfig, RetryConfig,
        TimeoutConfig,
    };

    // Security (0.3.1)
    pub use crate::security::{
        AuthConfig, KernelPermission, Permission, PermissionSet, Role, SecurityConfig,
        SecurityContext, TenantId,
    };

    // Memory management (0.3.1)
    pub use crate::memory::{
        AnalyticsContext, AnalyticsContextManager, InterPhaseReduction, KernelMemoryManager,
        MemoryConfig, MemoryError, MemoryStats, PressureLevel, ReductionConfig, SyncMode,
    };

    // Production configuration (0.3.1)
    pub use crate::config::{ProductionConfig, ProductionConfigBuilder};

    // Re-exports from ringkernel-core
    pub use ringkernel_core::k2k::{K2KBroker, K2KEndpoint};
    pub use ringkernel_core::runtime::{KernelHandle, KernelId, KernelState, LaunchOptions};
    pub use ringkernel_core::{HlcTimestamp, MessageId, RingContext, RingMessage};
}
