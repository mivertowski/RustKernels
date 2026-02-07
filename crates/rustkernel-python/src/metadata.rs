//! Python-facing metadata types.

use pyo3::prelude::*;
use rustkernel_core::domain::Domain;
use rustkernel_core::kernel::KernelMetadata;
use rustkernel_core::registry::RegistryStats;
use rustkernels::catalog::DomainInfo;
use std::collections::HashMap;

/// Kernel metadata exposed to Python.
#[pyclass(frozen, name = "KernelMetadata")]
#[derive(Clone)]
pub struct PyKernelMetadata {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub mode: String,
    #[pyo3(get)]
    pub domain: String,
    #[pyo3(get)]
    pub description: String,
    #[pyo3(get)]
    pub expected_throughput: u64,
    #[pyo3(get)]
    pub target_latency_us: f64,
    #[pyo3(get)]
    pub requires_gpu_native: bool,
    #[pyo3(get)]
    pub version: u32,
}

#[pymethods]
impl PyKernelMetadata {
    fn __repr__(&self) -> String {
        format!(
            "KernelMetadata(id='{}', mode='{}', domain='{}')",
            self.id, self.mode, self.domain
        )
    }

    fn __str__(&self) -> String {
        self.id.clone()
    }
}

impl From<KernelMetadata> for PyKernelMetadata {
    fn from(m: KernelMetadata) -> Self {
        Self {
            id: m.id,
            mode: m.mode.as_str().to_owned(),
            domain: m.domain.as_str().to_owned(),
            description: m.description,
            expected_throughput: m.expected_throughput,
            target_latency_us: m.target_latency_us,
            requires_gpu_native: m.requires_gpu_native,
            version: m.version,
        }
    }
}

/// Registry statistics exposed to Python.
#[pyclass(frozen, name = "RegistryStats")]
#[derive(Clone)]
pub struct PyRegistryStats {
    #[pyo3(get)]
    pub total: usize,
    #[pyo3(get)]
    pub batch_kernels: usize,
    #[pyo3(get)]
    pub ring_kernels: usize,
    #[pyo3(get)]
    pub by_domain: HashMap<String, usize>,
}

#[pymethods]
impl PyRegistryStats {
    fn __repr__(&self) -> String {
        format!(
            "RegistryStats(total={}, batch={}, ring={})",
            self.total, self.batch_kernels, self.ring_kernels
        )
    }
}

impl From<RegistryStats> for PyRegistryStats {
    fn from(s: RegistryStats) -> Self {
        let by_domain = s
            .by_domain
            .into_iter()
            .map(|(d, count)| (d.as_str().to_owned(), count))
            .collect();
        Self {
            total: s.total,
            batch_kernels: s.batch_kernels,
            ring_kernels: s.ring_kernels,
            by_domain,
        }
    }
}

/// Domain information exposed to Python.
#[pyclass(frozen, name = "DomainInfo")]
#[derive(Clone)]
pub struct PyDomainInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub description: String,
    #[pyo3(get)]
    pub kernel_count: usize,
    #[pyo3(get)]
    pub feature: String,
    #[pyo3(get)]
    pub domain: String,
}

#[pymethods]
impl PyDomainInfo {
    fn __repr__(&self) -> String {
        format!(
            "DomainInfo(name='{}', kernels={})",
            self.name, self.kernel_count
        )
    }
}

impl From<DomainInfo> for PyDomainInfo {
    fn from(d: DomainInfo) -> Self {
        Self {
            name: d.name.to_owned(),
            description: d.description.to_owned(),
            kernel_count: d.kernel_count,
            feature: d.feature.to_owned(),
            domain: d.domain.as_str().to_owned(),
        }
    }
}

/// Parse a domain string into the Rust `Domain` enum.
pub fn parse_domain(s: &str) -> Option<Domain> {
    Domain::parse(s)
}
