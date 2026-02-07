"""Tests for KernelRegistry discovery API."""

import rustkernels


def test_create_registry():
    reg = rustkernels.KernelRegistry()
    assert len(reg) > 0


def test_kernel_ids_sorted():
    reg = rustkernels.KernelRegistry()
    ids = reg.kernel_ids
    assert ids == sorted(ids)


def test_batch_kernel_ids():
    reg = rustkernels.KernelRegistry()
    batch_ids = reg.batch_kernel_ids
    assert isinstance(batch_ids, list)
    assert len(batch_ids) > 0


def test_total_count():
    reg = rustkernels.KernelRegistry()
    assert reg.total_count == len(reg)
    assert reg.total_count > 0


def test_stats():
    reg = rustkernels.KernelRegistry()
    stats = reg.stats
    assert stats.total == reg.total_count
    assert stats.batch_kernels >= 0
    assert stats.ring_kernels >= 0
    assert stats.batch_kernels + stats.ring_kernels == stats.total
    assert isinstance(stats.by_domain, dict)
    assert len(stats.by_domain) > 0


def test_get_existing():
    reg = rustkernels.KernelRegistry()
    ids = reg.kernel_ids
    assert len(ids) > 0
    meta = reg.get(ids[0])
    assert meta is not None
    assert meta.id == ids[0]


def test_get_nonexistent():
    reg = rustkernels.KernelRegistry()
    assert reg.get("nonexistent/kernel") is None


def test_contains():
    reg = rustkernels.KernelRegistry()
    ids = reg.kernel_ids
    assert ids[0] in reg
    assert "nonexistent/kernel" not in reg


def test_by_domain():
    reg = rustkernels.KernelRegistry()
    graph_kernels = reg.by_domain("GraphAnalytics")
    assert isinstance(graph_kernels, list)
    assert len(graph_kernels) > 0
    for k in graph_kernels:
        assert k.domain == "GraphAnalytics"


def test_by_domain_invalid():
    reg = rustkernels.KernelRegistry()
    try:
        reg.by_domain("NotADomain")
        assert False, "should have raised"
    except ValueError:
        pass


def test_by_mode():
    reg = rustkernels.KernelRegistry()
    batch = reg.by_mode("batch")
    assert isinstance(batch, list)
    for k in batch:
        assert k.mode == "batch"


def test_by_mode_invalid():
    reg = rustkernels.KernelRegistry()
    try:
        reg.by_mode("invalid")
        assert False, "should have raised"
    except ValueError:
        pass


def test_search():
    reg = rustkernels.KernelRegistry()
    results = reg.search("centrality")
    assert isinstance(results, list)
    # Should find at least betweenness, degree, etc. in graph domain
    assert len(results) > 0


def test_repr():
    reg = rustkernels.KernelRegistry()
    r = repr(reg)
    assert "KernelRegistry" in r
    assert str(reg.total_count) in r
