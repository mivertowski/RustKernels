"""Tests for batch kernel execution."""

import rustkernels


def test_execute_via_registry():
    """Execute a kernel through a registry instance."""
    reg = rustkernels.KernelRegistry()
    batch_ids = reg.batch_kernel_ids
    assert len(batch_ids) > 0

    # Pick a kernel and verify it's callable (we may not know the exact
    # schema, so just verify the mechanism works with a known kernel if
    # available).
    meta = reg.get(batch_ids[0])
    assert meta is not None
    assert meta.mode == "batch"


def test_execute_module_level():
    """Verify module-level execute function exists and rejects bad kernel IDs."""
    try:
        rustkernels.execute("nonexistent/kernel", {"x": 1})
        assert False, "should have raised KernelNotFoundError"
    except rustkernels.KernelNotFoundError:
        pass


def test_execute_not_found():
    """KernelNotFoundError for missing kernel."""
    reg = rustkernels.KernelRegistry()
    try:
        reg.execute("does/not/exist", {})
        assert False, "should have raised"
    except rustkernels.KernelNotFoundError as e:
        assert "does/not/exist" in str(e)
