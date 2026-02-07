"""Tests for exception hierarchy."""

import rustkernels


def test_exception_hierarchy():
    """All specific exceptions inherit from KernelError."""
    assert issubclass(rustkernels.KernelNotFoundError, rustkernels.KernelError)
    assert issubclass(rustkernels.ValidationError, rustkernels.KernelError)
    assert issubclass(rustkernels.SerializationError, rustkernels.KernelError)
    assert issubclass(rustkernels.ExecutionError, rustkernels.KernelError)
    assert issubclass(rustkernels.TimeoutError, rustkernels.KernelError)
    assert issubclass(rustkernels.LicenseError, rustkernels.KernelError)
    assert issubclass(rustkernels.AuthorizationError, rustkernels.KernelError)
    assert issubclass(rustkernels.ResourceExhaustedError, rustkernels.KernelError)
    assert issubclass(rustkernels.ServiceUnavailableError, rustkernels.KernelError)


def test_kernel_error_is_exception():
    """KernelError itself is a proper Exception subclass."""
    assert issubclass(rustkernels.KernelError, Exception)


def test_catch_base_exception():
    """Catching KernelError also catches specific subtypes."""
    try:
        rustkernels.execute("nonexistent/kernel", {})
        assert False, "should have raised"
    except rustkernels.KernelError:
        pass  # Caught via base class


def test_catch_specific_exception():
    """Catching a specific exception works."""
    try:
        rustkernels.execute("nonexistent/kernel", {})
        assert False, "should have raised"
    except rustkernels.KernelNotFoundError as e:
        assert "nonexistent/kernel" in str(e)


def test_exception_message():
    """Exceptions carry a useful message."""
    try:
        rustkernels.execute("nonexistent/kernel", {})
    except rustkernels.KernelNotFoundError as e:
        msg = str(e)
        assert len(msg) > 0
        assert "nonexistent/kernel" in msg
