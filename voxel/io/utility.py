"""
Methods related to file reading and writing.
"""

from __future__ import annotations

import os
import pathlib
import numpy as np
import torch


def check_file_readability(filename: os.PathLike) -> None:
    """
    Raise an exception if a file cannot be read.

    Args:
        filename (PathLike): Path to file.
    """
    if not isinstance(filename, pathlib.Path):
        filename = pathlib.Path(filename)

    if filename.is_dir():
        raise ValueError(f'{filename} is a directory, not a file')

    if not filename.is_file():
        raise FileNotFoundError(f'{filename} is not a file')

    if not os.access(filename, os.R_OK):
        raise PermissionError(f'{filename} is not a readable file')


class IOProtocol:
    """
    Abstract (and private) protocol class to implement filetype-specific reading and writing.

    Subclasses must override the `load` and `save` methods, and set the `name` and `extensions`
    global class members.
    """
    name = ''
    extensions = []

    @classmethod
    def primary_extension(cls) -> str:
        """
        Return the primary (first) file extension of the protocol.

        Returns:
            str: The primary file extension.
        """
        if not cls.extensions:
            return ''
        elif isinstance(cls.extensions, str):
            return cls.extensions
        return cls.extensions[0]

    @classmethod
    def enforce_extension(cls, filename: os.PathLike) -> pathlib.Path:
        """
        Enforce a valid protocol extension on a filename. Returns the corrected filename.

        Args:
            filename (PathLike): The filename to enforce the extension on.

        Returns:
            Path: The filename with the enforced extension.
        """
        if str(filename).lower().endswith(cls.extensions):
            return filename
        return pathlib.Path(filename).with_suffix(cls.primary_extension())

    def load(self, filename: os.PathLike) -> object:
        """
        File load function to be implemented for each subclass.

        Args:
            filename (PathLike): The filename to load.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplemented(f'reading file format of {os.path.basename(filename)} is not implemented yet')

    def save(self, obj: object, filename: os.PathLike) -> None:
        """
        File save function to be implemented for each subclass.

        Args:
            obj (object): The object to save.
            filename (PathLike): The filename to save the object to.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplemented(f'writing file format of {os.path.basename(filename)} is not implemented yet')


def find_protocol_by_name(protocols: list, fmt: str) -> IOProtocol | None:
    """
    Find IO protocol by format name.

    Args:
        protocols (list of IOProtocol): List of IOProtocol classes to search.
        fmt (str): File format name.

    Returns:
        IOProtocol: Matched IO protocol class or None if not found.
    """
    fmt = fmt.lower()
    return next((p for p in protocols if fmt == p.name), None)


def find_protocol_by_extension(protocols: list, filename: os.PathLike) -> IOProtocol | None:
    """
    Find IO protocol by extension type.

    Args:
        protocols (list of IOProtocol): List of IOProtocol classes to search.
        filename (PathLike): Filename to grab extension of.

    Returns:
        IOProtocol: Matched IO protocol class or None if not found.
    """
    lowercase = str(filename).lower()
    return next((p for p in protocols if lowercase.endswith(p.extensions)), None)


def get_all_extensions(protocols: list) -> list:
    """
    Returns all extensions in a list of protocols.

    Args:
        protocols (list of str): List of IOProtocol classes to search.

    Returns:
        list of str: List of extensions.
    """
    extensions = []
    for protocol in protocols:
        if isinstance(protocol.extensions, str):
            extensions.append(protocol.extensions)
        else:
            extensions.extend(protocol.extensions)
    return extensions


def numpy_to_tensor(x, dtype: torch.dtype = None, copy: bool = False) -> torch.Tensor:
    """
    Safely convert a numpy array to a tensor.

    Args:
        x: Numpy ndarray or array-like to convert
        dtype (dtype, optional): Optional torch dtype
        copy (bool, optional): Force a copy even if sharing is possible

    Returns:
        Tensor: Converted tensor
    """
    if isinstance(x, torch.Tensor):
        return x

    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    # only numeric/boolean dtypes are meaningful for tensors
    if x.dtype.kind not in "biufc?":  # bool/int/uint/float/complex/bool
        raise TypeError(f"unsupported dtype kind '{x.dtype.kind}' for tensor conversion")

    # normalize endianness: make sure dtype is native (or non-endian, e.g. '|u1')
    # np.dtype.byteorder: '=' native, '<' little, '>' big, '|' not applicable
    bo = x.dtype.byteorder
    if bo not in ('=', '|'):
        # if platform is little-endian and x is big-endian (or vice versa), swap
        if (bo == '>' and np.little_endian) or (bo == '<' and not np.little_endian):
            # byteswap returns a view by default - newbyteorder('=') sets native
            x = x.byteswap().view(x.dtype.newbyteorder('='))

    # if requested, or if array is read-only, make a copy so torch can safely own/write
    if copy or (hasattr(x, "flags") and not x.flags.writeable):
        x = x.copy()

    # use as_tensor to share memory when possible but copy when required
    return torch.as_tensor(x, dtype=dtype)
