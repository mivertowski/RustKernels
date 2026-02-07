//! Error types for RustKernels.

use crate::domain::Domain;
use thiserror::Error;

/// Result type alias using `KernelError`.
pub type Result<T> = std::result::Result<T, KernelError>;

/// Errors that can occur during kernel operations.
#[derive(Debug, Error)]
pub enum KernelError {
    /// Kernel not found in registry.
    #[error("Kernel not found: {0}")]
    KernelNotFound(String),

    /// Kernel already registered.
    #[error("Kernel already registered: {0}")]
    KernelAlreadyRegistered(String),

    /// Invalid kernel state transition.
    #[error("Invalid state transition from {from} to {to}")]
    InvalidStateTransition {
        /// Current state.
        from: String,
        /// Attempted target state.
        to: String,
    },

    /// Kernel is not in active state.
    #[error("Kernel is not active: {0}")]
    KernelNotActive(String),

    /// Input validation failed.
    #[error("Input validation failed: {0}")]
    ValidationError(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Deserialization error.
    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    /// Message queue is full.
    #[error("Message queue full (capacity: {capacity})")]
    QueueFull {
        /// Queue capacity.
        capacity: usize,
    },

    /// Message queue is empty.
    #[error("Message queue empty")]
    QueueEmpty,

    /// Message too large for queue.
    #[error("Message too large: {size} bytes (max: {max} bytes)")]
    MessageTooLarge {
        /// Actual message size.
        size: usize,
        /// Maximum allowed size.
        max: usize,
    },

    /// Timeout waiting for response.
    #[error("Timeout waiting for response after {0:?}")]
    Timeout(std::time::Duration),

    /// GPU kernel launch failed.
    #[error("Kernel launch failed: {0}")]
    LaunchFailed(String),

    /// GPU compilation error.
    #[error("GPU compilation error: {0}")]
    CompilationError(String),

    /// GPU device error.
    #[error("GPU device error: {0}")]
    DeviceError(String),

    /// Backend not available.
    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),

    /// License error.
    #[error("License error: {0}")]
    LicenseError(#[from] crate::license::LicenseError),

    /// SLO violation.
    #[error("SLO violation: {0}")]
    SLOViolation(String),

    /// Domain not supported.
    #[error("Domain not supported: {0}")]
    DomainNotSupported(Domain),

    /// Internal error.
    #[error("Internal error: {0}")]
    InternalError(String),

    /// IO error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Actor error.
    #[error("Actor error: {0}")]
    ActorError(String),

    /// RingKernel error (from underlying runtime).
    #[error("RingKernel error: {0}")]
    RingKernelError(String),

    /// K2K (Kernel-to-Kernel) communication error.
    #[error("K2K error: {0}")]
    K2KError(String),

    // Enterprise errors (0.3.1)
    /// Unauthorized access.
    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    /// Resource exhausted (quota exceeded, rate limited, etc.).
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Service unavailable (circuit open, degraded, etc.).
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),
}

impl KernelError {
    /// Create a validation error.
    #[must_use]
    pub fn validation(msg: impl Into<String>) -> Self {
        KernelError::ValidationError(msg.into())
    }

    /// Create an internal error.
    #[must_use]
    pub fn internal(msg: impl Into<String>) -> Self {
        KernelError::InternalError(msg.into())
    }

    /// Create a kernel not found error.
    #[must_use]
    pub fn not_found(id: impl Into<String>) -> Self {
        KernelError::KernelNotFound(id.into())
    }

    /// Create a launch failed error.
    #[must_use]
    pub fn launch_failed(msg: impl Into<String>) -> Self {
        KernelError::LaunchFailed(msg.into())
    }

    /// Create a device error.
    #[must_use]
    pub fn device(msg: impl Into<String>) -> Self {
        KernelError::DeviceError(msg.into())
    }

    /// Create a K2K error.
    #[must_use]
    pub fn k2k(msg: impl Into<String>) -> Self {
        KernelError::K2KError(msg.into())
    }

    /// Returns true if this is a recoverable error (safe to retry).
    #[must_use]
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            KernelError::QueueFull { .. }
                | KernelError::QueueEmpty
                | KernelError::Timeout(_)
                | KernelError::ServiceUnavailable(_)
                | KernelError::ResourceExhausted(_)
        )
    }

    /// Returns true if this is a client error (invalid input, not found, etc.).
    #[must_use]
    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            KernelError::KernelNotFound(_)
                | KernelError::ValidationError(_)
                | KernelError::DeserializationError(_)
                | KernelError::Unauthorized(_)
                | KernelError::DomainNotSupported(_)
        )
    }

    /// Returns true if this is a license-related error.
    #[must_use]
    pub fn is_license_error(&self) -> bool {
        matches!(self, KernelError::LicenseError(_))
    }

    /// Returns the suggested HTTP status code for this error.
    ///
    /// Centralizes HTTP status mapping so that all ecosystem integrations
    /// (Axum, Tower, gRPC, Actix) use consistent status codes.
    #[must_use]
    pub fn http_status_code(&self) -> u16 {
        match self {
            KernelError::KernelNotFound(_) => 404,
            KernelError::KernelAlreadyRegistered(_) => 409,
            KernelError::ValidationError(_) => 400,
            KernelError::DeserializationError(_) => 400,
            KernelError::SerializationError(_) => 500,
            KernelError::Unauthorized(_) => 401,
            KernelError::ResourceExhausted(_) => 429,
            KernelError::ServiceUnavailable(_) => 503,
            KernelError::Timeout(_) => 504,
            KernelError::LicenseError(_) => 403,
            KernelError::DomainNotSupported(_) => 403,
            KernelError::QueueFull { .. } => 503,
            KernelError::MessageTooLarge { .. } => 413,
            _ => 500,
        }
    }

    /// Returns a machine-readable error code string.
    #[must_use]
    pub fn error_code(&self) -> &'static str {
        match self {
            KernelError::KernelNotFound(_) => "KERNEL_NOT_FOUND",
            KernelError::KernelAlreadyRegistered(_) => "KERNEL_ALREADY_REGISTERED",
            KernelError::InvalidStateTransition { .. } => "INVALID_STATE_TRANSITION",
            KernelError::KernelNotActive(_) => "KERNEL_NOT_ACTIVE",
            KernelError::ValidationError(_) => "VALIDATION_ERROR",
            KernelError::SerializationError(_) => "SERIALIZATION_ERROR",
            KernelError::DeserializationError(_) => "DESERIALIZATION_ERROR",
            KernelError::QueueFull { .. } => "QUEUE_FULL",
            KernelError::QueueEmpty => "QUEUE_EMPTY",
            KernelError::MessageTooLarge { .. } => "MESSAGE_TOO_LARGE",
            KernelError::Timeout(_) => "TIMEOUT",
            KernelError::LaunchFailed(_) => "LAUNCH_FAILED",
            KernelError::CompilationError(_) => "COMPILATION_ERROR",
            KernelError::DeviceError(_) => "DEVICE_ERROR",
            KernelError::BackendNotAvailable(_) => "BACKEND_NOT_AVAILABLE",
            KernelError::LicenseError(_) => "LICENSE_ERROR",
            KernelError::SLOViolation(_) => "SLO_VIOLATION",
            KernelError::DomainNotSupported(_) => "DOMAIN_NOT_SUPPORTED",
            KernelError::InternalError(_) => "INTERNAL_ERROR",
            KernelError::IoError(_) => "IO_ERROR",
            KernelError::ConfigError(_) => "CONFIG_ERROR",
            KernelError::ActorError(_) => "ACTOR_ERROR",
            KernelError::RingKernelError(_) => "RINGKERNEL_ERROR",
            KernelError::K2KError(_) => "K2K_ERROR",
            KernelError::Unauthorized(_) => "UNAUTHORIZED",
            KernelError::ResourceExhausted(_) => "RESOURCE_EXHAUSTED",
            KernelError::ServiceUnavailable(_) => "SERVICE_UNAVAILABLE",
        }
    }
}

/// Convert from ringkernel-core errors.
impl From<ringkernel_core::RingKernelError> for KernelError {
    fn from(err: ringkernel_core::RingKernelError) -> Self {
        KernelError::RingKernelError(err.to_string())
    }
}
