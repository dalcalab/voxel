"""
Reading and writing affines to various file formats.
"""

from __future__ import annotations

import os
import torch
import voxel as vx

from .utility import IOProtocol


def load_affine(filename: os.PathLike, fmt: str | None = None) -> vx.AffineMatrix:
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
        proto = vx.io.utility.find_protocol_by_name(affine_io_protocols, fmt)
        if proto is None:
            raise ValueError(f'unknown file format {fmt}')

    return proto().load(filename)


def save_affine(affine: vx.AffineMatrix, filename: os.PathLike, fmt: str | None = None, **kwargs) -> None:
    """
    Save an affine matrix to a file.

    Args:
        affine (AffineMatrix): The affine matrix to save.
        filename (PathLike): The path to the file to save.
        fmt (str, optional): The format of the file. If None, the format is
            determined by the file extension.
        **kwargs (Any): Additional arguments to pass to the file writing method.
    """
    if fmt is None:
        proto = vx.io.utility.find_protocol_by_extension(affine_io_protocols, filename)
        if proto is None:
            raise ValueError(f'cannot determine file format from extension for {filename}')
    else:
        proto = vx.io.utility.find_protocol_by_name(affine_io_protocols, fmt)
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


class TextMatrixIO(IOProtocol):
    """
    IO protocol for storing an affine matrix as plain whitespace-delimited text.

    The file holds four rows of four numbers (or three rows, in which case the
    homogeneous bottom row is assumed). Lines beginning with `#` are ignored.
    This matches the layout used by FSL `.mat` files and `numpy.savetxt`.
    """
    name = 'text'
    extensions = ('.txt', '.mat')

    def load(self, filename: os.PathLike) -> vx.AffineMatrix:
        rows = []
        with open(filename) as f:
            for line in f:
                line = line.split('#', 1)[0].strip()
                if line:
                    rows.append([float(x) for x in line.replace(',', ' ').split()])
        if len(rows) not in (3, 4) or any(len(r) != 4 for r in rows):
            raise ValueError(f'expected a 3x4 or 4x4 matrix in {filename}, '
                             f'got {len(rows)} rows of {[len(r) for r in rows]} values')
        matrix = torch.tensor(rows, dtype=torch.float64)
        return vx.AffineMatrix(matrix, dtype=torch.float64)

    def save(self, affine: vx.AffineMatrix, filename: os.PathLike) -> None:
        matrix = affine.tensor.detach().cpu().to(torch.float64).tolist()
        with open(filename, 'w') as f:
            for row in matrix:
                f.write(' '.join(f'{x:.17g}' for x in row) + '\n')


affine_io_protocols = [
    PytorchMatrixIO,
    TextMatrixIO,
]
