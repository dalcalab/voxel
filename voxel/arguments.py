"""
Utilities for flexible, torch-style variadic method arguments.
"""

from __future__ import annotations

import voxel as vx


# sentinel marking a space argument that has no default and must be provided
required = object()


def merge_components(value, components: tuple):
    """
    Merge a leading argument with trailing variadic positional components.

    Enables torch-style component arguments, in which a scalar or length-3
    parameter can also be passed as separate positional values, e.g.
    `f(1, 1, 2)` in addition to `f((1, 1, 2))` and `f(1)`.

    Args:
        value: First (or only) argument value.
        components (tuple): Additional positional components. If empty,
            `value` is returned unchanged.

    Returns:
        The original value if no components are given, otherwise the
        tuple `(value, *components)`.
    """
    if not components:
        return value
    if value is None:
        raise TypeError('cannot combine None with additional positional components')
    if any(isinstance(c, (str, vx.Space)) for c in components):
        raise TypeError('unexpected string in positional argument components')
    return (value, *components)


def extract_space(components: tuple, space, default=required) -> tuple:
    """
    Pop a trailing coordinate-space designation from variadic positional components.

    Preserves positional space arguments for methods converted to variadic
    signatures, e.g. `shift((1, 0, 0), 'voxel')` and `shift(1, 0, 0, 'voxel')`.

    Args:
        components (tuple): Variadic components, potentially ending with a
            Space instance or space name string.
        space (Space | str | None): Space provided as a keyword argument, or None.
        default: Fallback when no space is provided. If left as `required`,
            a missing space raises a TypeError.

    Returns:
        tuple: The remaining components and the resolved space.
    """
    if components and isinstance(components[-1], (str, vx.Space)):
        if space is not None:
            raise TypeError('space was provided both positionally and as a keyword argument')
        space = components[-1]
        components = components[:-1]
    if space is None:
        if default is required:
            raise TypeError("missing required argument: 'space'")
        space = default
    return components, space
