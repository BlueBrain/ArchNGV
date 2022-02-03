"""Generic utilities."""
from collections.abc import Iterable


def is_iterable(v):
    """Check if `v` is any iterable (strings are considered scalar)."""
    return isinstance(v, Iterable) and not isinstance(v, str)


def ensure_list(v):
    """Convert iterable / wrap scalar into list (strings are considered scalar)."""
    if is_iterable(v):
        return list(v)
    else:
        return [v]
