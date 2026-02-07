//! Python exception hierarchy mapping `KernelError` variants.

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use rustkernel_core::error::KernelError;

// Base exception — all RustKernels errors inherit from this.
pyo3::create_exception!(rustkernels, KernelErrorBase, PyException);

// Specific exception types.
pyo3::create_exception!(rustkernels, KernelNotFoundError, KernelErrorBase);
pyo3::create_exception!(rustkernels, ValidationError, KernelErrorBase);
pyo3::create_exception!(rustkernels, SerializationError, KernelErrorBase);
pyo3::create_exception!(rustkernels, ExecutionError, KernelErrorBase);
pyo3::create_exception!(rustkernels, TimeoutError, KernelErrorBase);
pyo3::create_exception!(rustkernels, LicenseError, KernelErrorBase);
pyo3::create_exception!(rustkernels, AuthorizationError, KernelErrorBase);
pyo3::create_exception!(rustkernels, ResourceExhaustedError, KernelErrorBase);
pyo3::create_exception!(rustkernels, ServiceUnavailableError, KernelErrorBase);

/// Convert a `KernelError` into the appropriate Python exception.
pub fn kernel_error_to_py(err: KernelError) -> PyErr {
    let msg = err.to_string();
    match &err {
        KernelError::KernelNotFound(_) => KernelNotFoundError::new_err(msg),
        KernelError::ValidationError(_) => ValidationError::new_err(msg),
        KernelError::SerializationError(_) | KernelError::DeserializationError(_) => {
            SerializationError::new_err(msg)
        }
        KernelError::Timeout(_) => TimeoutError::new_err(msg),
        KernelError::LicenseError(_) | KernelError::DomainNotSupported(_) => {
            LicenseError::new_err(msg)
        }
        KernelError::Unauthorized(_) => AuthorizationError::new_err(msg),
        KernelError::ResourceExhausted(_) | KernelError::QueueFull { .. } => {
            ResourceExhaustedError::new_err(msg)
        }
        KernelError::ServiceUnavailable(_) => ServiceUnavailableError::new_err(msg),
        // Everything else is an execution-level error.
        _ => ExecutionError::new_err(msg),
    }
}

/// Register all exception types on the Python module.
pub fn register_exceptions(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add("KernelError", parent.py().get_type::<KernelErrorBase>())?;
    parent.add(
        "KernelNotFoundError",
        parent.py().get_type::<KernelNotFoundError>(),
    )?;
    parent.add("ValidationError", parent.py().get_type::<ValidationError>())?;
    parent.add(
        "SerializationError",
        parent.py().get_type::<SerializationError>(),
    )?;
    parent.add("ExecutionError", parent.py().get_type::<ExecutionError>())?;
    parent.add("TimeoutError", parent.py().get_type::<TimeoutError>())?;
    parent.add("LicenseError", parent.py().get_type::<LicenseError>())?;
    parent.add(
        "AuthorizationError",
        parent.py().get_type::<AuthorizationError>(),
    )?;
    parent.add(
        "ResourceExhaustedError",
        parent.py().get_type::<ResourceExhaustedError>(),
    )?;
    parent.add(
        "ServiceUnavailableError",
        parent.py().get_type::<ServiceUnavailableError>(),
    )?;
    Ok(())
}
