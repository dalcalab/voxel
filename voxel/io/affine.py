"""
Reading and writing affines to various file formats.
"""

import os
import torch
import voxel as vx

from .utility import IOProtocol


def load_affine(filename: os.PathLike, fmt: str = None) -> vx.AffineMatrix:
    """
    Load an affine matrix from a file.

    Args:
        filename (PathLike): The path to the file to load.
        fmt (str, optional): The format of the file. If None, the format is
            determined by the file extension.
    
    Returns:
        AffineMatrix: The loaded affine matrix.
    """
    vx.io.utility.check_file_readability(filename)

    if fmt is None:
        proto = vx.io.utility.find_protocol_by_extension(affine_io_protocols, filename)
        if proto is None:
            raise ValueError(f'cannot determine file format from extension for {filename}')
    else:
        proto = vx.io.protocol.find_protocol_by_name(affine_io_protocols, fmt)
        if proto is None:
            raise ValueError(f'unknown file format {fmt}')

    return proto().load(filename)


def save_affine(affine: vx.AffineMatrix, filename: os.PathLike, fmt: str = None, **kwargs) -> None:
    """
    Save an affine matrix to a file.

    Args:
        affine (AffineMatrix): The affine matrix to save.
        filename (PathLike): The path to the file to save.
        fmt (str, optional): The format of the file. If None, the format is
            determined by the file extension.
        kwargs: Additional arguments to pass to the file writing method.
    """
    if fmt is None:
        proto = vx.io.utility.find_protocol_by_extension(affine_io_protocols, filename)
        if proto is None:
            raise ValueError(f'cannot determine file format from extension for {filename}')
    else:
        proto = vx.io.protocol.find_protocol_by_name(affine_io_protocols, fmt)
        if proto is None:
            raise ValueError(f'unknown file format {fmt}')
        filename = proto.enforce_extension(filename)

    proto().save(affine, filename, **kwargs)


class PytorchMatrixIO(IOProtocol):
    """
    IO protocol for storing a simple affine matrix in a pytorch file.
    """
    name = 'torch'
    extensions = ('.pth', '.pt')

    def load(self, filename: os.PathLike) -> vx.AffineMatrix:
        return vx.AffineMatrix(torch.load(filename, weights_only=False))

    def save(self, affine: vx.AffineMatrix, filename: os.PathLike) -> None:
        torch.save(affine.tensor.detach().cpu(), filename)


affine_io_protocols = [
    PytorchMatrixIO,
]
