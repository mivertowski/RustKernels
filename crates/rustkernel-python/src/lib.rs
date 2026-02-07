//! Python bindings for RustKernels.
//!
//! Provides `rustkernels` Python package with batch kernel discovery and execution.

mod errors;
mod execution;
mod metadata;
mod registry;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::OnceLock;

use metadata::{PyDomainInfo, PyKernelMetadata, PyRegistryStats};
use registry::PyKernelRegistry;

/// Cached default registry for module-level convenience functions.
static DEFAULT_REGISTRY: OnceLock<rustkernel_core::registry::KernelRegistry> = OnceLock::new();

fn default_registry() -> &'static rustkernel_core::registry::KernelRegistry {
    DEFAULT_REGISTRY.get_or_init(|| {
        let reg = rustkernel_core::registry::KernelRegistry::new();
        rustkernels::register_all(&reg).expect("failed to populate default registry");
        reg
    })
}

// ── Module-level functions ──────────────────────────────────────────────────

/// Execute a batch kernel using the default (cached) registry.
#[pyfunction]
#[pyo3(signature = (kernel_id, input))]
fn execute<'py>(
    py: Python<'py>,
    kernel_id: &str,
    input: &Bound<'py, PyDict>,
) -> PyResult<PyObject> {
    let registry = default_registry();
    let json_module = py.import("json")?;
    let json_str: String = json_module.call_method1("dumps", (input,))?.extract()?;
    let output_bytes = execution::execute_batch(py, registry, kernel_id, json_str.as_bytes())?;
    let output_str = std::str::from_utf8(&output_bytes).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("invalid UTF-8 in output: {e}"))
    })?;
    let result = json_module.call_method1("loads", (output_str,))?;
    Ok(result.into())
}

/// List all domains with their metadata.
#[pyfunction]
fn list_domains() -> Vec<PyDomainInfo> {
    rustkernels::catalog::domains()
        .into_iter()
        .map(Into::into)
        .collect()
}

/// Total kernel count across all domains.
#[pyfunction]
fn total_kernel_count() -> usize {
    rustkernels::catalog::total_kernel_count()
}

/// Domains enabled at compile time.
#[pyfunction]
fn enabled_domains() -> Vec<String> {
    rustkernels::catalog::enabled_domains()
        .into_iter()
        .map(String::from)
        .collect()
}

// ── Module definition ───────────────────────────────────────────────────────

#[pymodule]
fn _rustkernels(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Version constants
    m.add("__version__", rustkernels::version::VERSION)?;
    m.add(
        "ringkernel_version",
        rustkernels::version::MIN_RINGKERNEL_VERSION,
    )?;

    // Classes
    m.add_class::<PyKernelRegistry>()?;
    m.add_class::<PyKernelMetadata>()?;
    m.add_class::<PyRegistryStats>()?;
    m.add_class::<PyDomainInfo>()?;

    // Exceptions
    errors::register_exceptions(m)?;

    // Module-level functions
    m.add_function(wrap_pyfunction!(execute, m)?)?;
    m.add_function(wrap_pyfunction!(list_domains, m)?)?;
    m.add_function(wrap_pyfunction!(total_kernel_count, m)?)?;
    m.add_function(wrap_pyfunction!(enabled_domains, m)?)?;

    Ok(())
}
