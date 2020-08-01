"""Generic utilities."""
import collections
import six


def is_iterable(v):
    """Check if `v` is any iterable (strings are considered scalar)."""
    return isinstance(v, collections.Iterable) and not isinstance(v, six.string_types)


def ensure_list(v):
    """Convert iterable / wrap scalar into list (strings are considered scalar)."""
    if is_iterable(v):
        return list(v)
    else:
        return [v]
