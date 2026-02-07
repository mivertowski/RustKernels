//! `PyKernelRegistry` — Python wrapper around `KernelRegistry`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustkernel_core::kernel::KernelMode;
use rustkernel_core::registry::KernelRegistry;
use std::sync::Arc;

use crate::errors::kernel_error_to_py;
use crate::execution;
use crate::metadata::{PyKernelMetadata, PyRegistryStats, parse_domain};

/// GPU kernel registry. Create one to discover and execute batch kernels.
///
/// On construction, all kernels enabled at compile time are registered automatically.
#[pyclass(name = "KernelRegistry")]
pub struct PyKernelRegistry {
    inner: Arc<KernelRegistry>,
}

#[pymethods]
impl PyKernelRegistry {
    #[new]
    fn new() -> PyResult<Self> {
        let registry = KernelRegistry::new();
        rustkernels::register_all(&registry).map_err(kernel_error_to_py)?;
        Ok(Self {
            inner: Arc::new(registry),
        })
    }

    /// List of all registered kernel IDs.
    #[getter]
    fn kernel_ids(&self) -> Vec<String> {
        let mut ids = self.inner.all_kernel_ids();
        ids.sort();
        ids
    }

    /// List of batch-executable kernel IDs.
    #[getter]
    fn batch_kernel_ids(&self) -> Vec<String> {
        let mut ids = self.inner.batch_kernel_ids();
        ids.sort();
        ids
    }

    /// Total number of registered kernels.
    #[getter]
    fn total_count(&self) -> usize {
        self.inner.total_count()
    }

    /// Registry statistics.
    #[getter]
    fn stats(&self) -> PyRegistryStats {
        self.inner.stats().into()
    }

    /// Get metadata for a kernel by ID, or `None` if not found.
    fn get(&self, kernel_id: &str) -> Option<PyKernelMetadata> {
        self.inner.get(kernel_id).map(Into::into)
    }

    /// Get all kernels belonging to a domain (e.g. `"GraphAnalytics"`).
    fn by_domain(&self, domain: &str) -> PyResult<Vec<PyKernelMetadata>> {
        let d = parse_domain(domain)
            .ok_or_else(|| PyValueError::new_err(format!("unknown domain: {domain}")))?;
        Ok(self
            .inner
            .by_domain(d)
            .into_iter()
            .map(Into::into)
            .collect())
    }

    /// Get all kernels matching a mode (`"batch"` or `"ring"`).
    fn by_mode(&self, mode: &str) -> PyResult<Vec<PyKernelMetadata>> {
        let m = match mode {
            "batch" => KernelMode::Batch,
            "ring" => KernelMode::Ring,
            _ => return Err(PyValueError::new_err(format!("unknown mode: {mode}"))),
        };
        Ok(self.inner.by_mode(m).into_iter().map(Into::into).collect())
    }

    /// Search kernels by substring (case-insensitive) on ID and description.
    fn search(&self, pattern: &str) -> Vec<PyKernelMetadata> {
        self.inner
            .search(pattern)
            .into_iter()
            .map(Into::into)
            .collect()
    }

    /// Execute a batch kernel and return the result as a Python `dict`.
    ///
    /// `input` must be a `dict` that is JSON-serializable.
    #[pyo3(signature = (kernel_id, input))]
    fn execute<'py>(
        &self,
        py: Python<'py>,
        kernel_id: &str,
        input: &Bound<'py, PyDict>,
    ) -> PyResult<PyObject> {
        let json_module = py.import("json")?;
        let json_str: String = json_module.call_method1("dumps", (input,))?.extract()?;
        let input_bytes = json_str.as_bytes();

        let output_bytes = execution::execute_batch(py, &self.inner, kernel_id, input_bytes)?;

        let output_str = std::str::from_utf8(&output_bytes).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("invalid UTF-8 in output: {e}"))
        })?;
        let result = json_module.call_method1("loads", (output_str,))?;
        Ok(result.into())
    }

    fn __contains__(&self, kernel_id: &str) -> bool {
        self.inner.contains(kernel_id)
    }

    fn __len__(&self) -> usize {
        self.inner.total_count()
    }

    fn __repr__(&self) -> String {
        format!("KernelRegistry(kernels={})", self.inner.total_count())
    }
}
