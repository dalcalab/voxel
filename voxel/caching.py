"""
Class utilities for caching parameters computed on-the-fly.
"""

from __future__ import annotations

import functools


def cached(func):
    """
    Decorator that converts a method into a cached property stored in
    `self._property_cache`. The property is read-only.
    """
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, '_property_cache'):
            self._property_cache = {}
        if func.__name__ not in self._property_cache:
            self._property_cache[func.__name__] = func(self)
        return self._property_cache[func.__name__]
    
    # make the property read-only
    wrapper = property(wrapper)
    return wrapper


def empty_property_cache(obj: object) -> None:
    """
    Clears the entire property cache for an object with
    function using the `cached` decorator.

    Args:
        obj (object): The object of the cache to clear.
    """
    if hasattr(obj, '_property_cache'):
        obj._property_cache.clear()
