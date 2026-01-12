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

#![warn(missing_docs)]
#![warn(clippy::all)]

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
    pub use crate::traits::{BatchKernel, GpuKernel, IterativeKernel, RingKernelHandler};

    // Re-exports from ringkernel-core
    pub use ringkernel_core::k2k::{K2KBroker, K2KEndpoint};
    pub use ringkernel_core::runtime::{KernelHandle, KernelId, KernelState, LaunchOptions};
    pub use ringkernel_core::{HlcTimestamp, MessageId, RingContext, RingMessage};
}
