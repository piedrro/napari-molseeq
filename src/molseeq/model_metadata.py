"""Editable copies of MolCore JSON metadata for public model updates."""

from collections.abc import Mapping


def mutable_metadata(value):
    """Copy nested mappings and sequences without retaining frozen containers."""
    if isinstance(value, Mapping):
        return {key: mutable_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [mutable_metadata(item) for item in value]
    return value
