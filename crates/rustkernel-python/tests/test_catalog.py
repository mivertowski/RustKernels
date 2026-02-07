"""Tests for catalog and version APIs."""

import rustkernels


def test_version():
    assert isinstance(rustkernels.__version__, str)
    assert rustkernels.__version__ == "0.4.0"


def test_ringkernel_version():
    assert rustkernels.ringkernel_version == "0.4.2"


def test_list_domains():
    domains = rustkernels.list_domains()
    assert isinstance(domains, list)
    assert len(domains) == 14
    names = [d.name for d in domains]
    assert "Graph Analytics" in names
    assert "Statistical ML" in names


def test_domain_info_fields():
    domains = rustkernels.list_domains()
    d = domains[0]
    assert isinstance(d.name, str)
    assert isinstance(d.description, str)
    assert isinstance(d.kernel_count, int)
    assert isinstance(d.feature, str)
    assert isinstance(d.domain, str)
    assert d.kernel_count > 0


def test_total_kernel_count():
    assert rustkernels.total_kernel_count() == 106


def test_enabled_domains():
    enabled = rustkernels.enabled_domains()
    assert isinstance(enabled, list)
    assert len(enabled) > 0
    # Default features at minimum
    assert "graph" in enabled
    assert "ml" in enabled


def test_domain_info_repr():
    domains = rustkernels.list_domains()
    r = repr(domains[0])
    assert "DomainInfo" in r
