//! Async bridge: global tokio runtime with GIL release for batch execution.

use pyo3::prelude::*;
use rustkernel_core::registry::KernelRegistry;
use std::sync::OnceLock;
use tokio::runtime::Runtime;

use crate::errors::kernel_error_to_py;

/// Global tokio runtime, created on first use.
static RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        Runtime::new().expect("failed to create tokio runtime for rustkernel-python")
    })
}

/// Execute a batch kernel by ID, releasing the GIL while the kernel runs.
///
/// `input_json` is the serialized JSON input; returns serialized JSON output.
pub fn execute_batch(
    py: Python<'_>,
    registry: &KernelRegistry,
    kernel_id: &str,
    input_json: &[u8],
) -> PyResult<Vec<u8>> {
    py.allow_threads(|| {
        runtime()
            .block_on(registry.execute_batch(kernel_id, input_json))
            .map_err(kernel_error_to_py)
    })
}
