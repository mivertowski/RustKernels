"""RustKernels — GPU-accelerated kernel library for financial services.

Provides Python bindings to the RustKernels Rust library for batch kernel
discovery and execution.

Quick start::

    import rustkernels

    reg = rustkernels.KernelRegistry()
    print(reg.kernel_ids)

    result = rustkernels.execute("graph/betweenness_centrality", {
        "num_nodes": 4,
        "edges": [[0, 1], [1, 2], [2, 3], [0, 3]],
        "normalized": True,
    })
"""

from rustkernels._rustkernels import (
    # Version
    __version__,
    ringkernel_version,
    # Classes
    KernelRegistry,
    KernelMetadata,
    RegistryStats,
    DomainInfo,
    # Exceptions
    KernelError,
    KernelNotFoundError,
    ValidationError,
    SerializationError,
    ExecutionError,
    TimeoutError,
    LicenseError,
    AuthorizationError,
    ResourceExhaustedError,
    ServiceUnavailableError,
    # Functions
    execute,
    list_domains,
    total_kernel_count,
    enabled_domains,
)

__all__ = [
    "__version__",
    "ringkernel_version",
    "KernelRegistry",
    "KernelMetadata",
    "RegistryStats",
    "DomainInfo",
    "KernelError",
    "KernelNotFoundError",
    "ValidationError",
    "SerializationError",
    "ExecutionError",
    "TimeoutError",
    "LicenseError",
    "AuthorizationError",
    "ResourceExhaustedError",
    "ServiceUnavailableError",
    "execute",
    "list_domains",
    "total_kernel_count",
    "enabled_domains",
]
