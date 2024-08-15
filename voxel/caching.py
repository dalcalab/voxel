"""
Class utilities for caching parameters computed on-the-fly.
"""

from __future__ import annotations

import functools


def cached(func: callable) -> property:
    """
    Decorator that converts a method into a cached property stored in
    `self._property_cache`. The property is read-only.
    """
    @functools.wraps(func)
    def wrapper(self):
        if func.__name__ not in self._property_cache:
            self._property_cache[func.__name__] = func(self)
        return self._property_cache[func.__name__]
    
    # make the property read-only
    wrapper = property(wrapper)
    return wrapper


def cached_transferable(func: callable) -> property:
    """
    Decorator that converts a method into a cached property stored in
    `self._transferable_property_cache`. The property is read-only.
    A transferable property is a property that can be easily transferred
    to a new instance of the same class.
    """
    @functools.wraps(func)
    def wrapper(self):
        if func.__name__ not in self._property_cache:
            self._transferable_property_cache[func.__name__] = func(self)
        return self._transferable_property_cache[func.__name__]

    # make the property read-only
    wrapper = property(wrapper)
    return wrapper


def init_property_cache(obj: object) -> None:
    """
    Initializes the property cache for an object with functions using
    the `cached` decorator.
    """
    if not hasattr(obj, '_property_cache'):
        obj._property_cache = {}
    if not hasattr(obj, '_transferable_property_cache'):
        obj._transferable_property_cache = {}


def transfer_property_cache(source: object, target: object) -> None:
    """
    Copy the transferable property cache from one object to another.
    """
    if hasattr(source, '_transferable_property_cache'):
        target._transferable_property_cache = source._transferable_property_cache.copy()


def empty_property_cache(obj: object) -> None:
    """
    Clears the entire property cache for an object with
    function using the `cached` decorator.
    """
    if hasattr(obj, '_property_cache'):
        obj._property_cache.clear()
    if hasattr(obj, '_transferable_property_cache'):
        obj._transferable_property_cache.clear()
