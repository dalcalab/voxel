"""
Reading and writing spatial transforms.
"""

from __future__ import annotations

import os
import json
import zipfile
import numpy as np
import torch
import voxel as vx

from .utility import IOProtocol


def load_transform(filename: os.PathLike, fmt: str | None = None) -> vx.AnyTransform:
    """
    Load a transform from a file.

    Args:
        filename (PathLike): The path to the file to load.
        fmt (str, optional): The format of the file. If None, the format is
            determined by the file extension.

    Returns:
        AnyTransform: The loaded affine, vector field, field pair, or warp, or a
            `TransformSeries` when the file holds several steps.
    """
    vx.io.utility.check_file_readability(filename)
    return _protocol(filename, fmt)().load(filename)


def save_transform(transform: vx.AnyTransform, filename: os.PathLike, fmt: str | None = None, **kwargs) -> None:
    """
    Save a transform to a file. The native `.vxx` format holds any transform or
    series, other formats hold affines only.

    Args:
        transform (AnyTransform): The transform to save.
        filename (PathLike): The path to the file to save.
        fmt (str, optional): The format of the file. If None, the format is
            determined by the file extension.
        **kwargs (Any): Additional arguments to pass to the file writing method.
    """
    proto = _protocol(filename, fmt)
    if fmt is not None:
        filename = proto.enforce_extension(filename)
    proto().save(transform, filename, **kwargs)


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
    affine = load_transform(filename, fmt)
    if not isinstance(affine, vx.AffineMatrix):
        raise TypeError(f'{filename} holds a {type(affine).__name__}, not an affine')
    return affine


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
    save_transform(affine, filename, fmt, **kwargs)


def _protocol(filename: os.PathLike, fmt: str | None) -> type[IOProtocol]:
    """
    The transform protocol for a format name, or for the extension of a filename.
    """
    if fmt is None:
        proto = vx.io.utility.find_protocol_by_extension(transform_io_protocols, filename)
        if proto is None:
            raise ValueError(f'cannot determine file format from extension for {filename}')
    else:
        proto = vx.io.utility.find_protocol_by_name(transform_io_protocols, fmt)
        if proto is None:
            raise ValueError(f'unknown file format {fmt}')
    return proto


class VoxelTransformIO(IOProtocol):
    """
    IO protocol for the native transform format, a zip archive of a JSON manifest
    and one npy array per tensor. The manifest lists the steps of a series in the
    order they apply, each an affine, a vector field, a field pair, or a warp. It
    is readable without torch, by any zip and npy reader.
    """
    name = 'voxel'
    extensions = ('.vxx',)

    def load(self, filename: os.PathLike) -> vx.AnyTransform:
        with zipfile.ZipFile(filename) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            if manifest.get('format') != 'voxel transform':
                raise ValueError(f'{filename} is not a voxel transform file')

            def array(name):
                with archive.open(name) as file:
                    return torch.as_tensor(np.lib.format.read_array(file))

            steps = [_decode(record, array) for record in manifest['steps']]
        return steps[0] if len(steps) == 1 else vx.TransformSeries(steps)

    def save(self, transform: vx.AnyTransform, filename: os.PathLike) -> None:
        steps = list(transform) if isinstance(transform, vx.TransformSeries) else [transform]
        arrays = {}
        records = [_encode(step, str(index), arrays) for index, step in enumerate(steps)]
        manifest = {'format': 'voxel transform', 'version': 1, 'steps': records}
        with zipfile.ZipFile(filename, 'w') as archive:
            archive.writestr('manifest.json', json.dumps(manifest, indent=1))
            for name, data in arrays.items():
                with archive.open(name, 'w') as file:
                    np.lib.format.write_array(file, data)


def _encode(transform, prefix: str, arrays: dict) -> dict:
    """
    The manifest record of a transform, with its tensors added to `arrays` by file name.
    """
    def array(name, tensor):
        arrays[f'{prefix}.{name}.npy'] = tensor.detach().cpu().contiguous().numpy()
        return f'{prefix}.{name}.npy'

    def geometry(geometry):
        return {'shape': list(geometry.baseshape), 'matrix': array('geometry', geometry.tensor)}

    if isinstance(transform, vx.AffineMatrix):
        return {'type': 'affine', 'matrix': array('matrix', transform.tensor)}
    if isinstance(transform, vx.VectorFieldPair):
        return {'type': 'pair',
                'forward': _encode(transform.forward, f'{prefix}.forward', arrays),
                'reverse': _encode(transform.reverse, f'{prefix}.reverse', arrays)}
    if isinstance(transform, vx.Warp):
        return {'type': 'warp', 'coordinates': array('coordinates', transform.coordinates),
                'geometry': geometry(transform.geometry)}
    if isinstance(transform, vx.VectorField):
        return {'type': 'field', 'space': vx.space.space_names[transform.space.code],
                'vectors': array('vectors', transform.tensor), 'geometry': geometry(transform.geometry)}
    raise TypeError(f'cannot save a {type(transform).__name__} as a transform')


def _decode(record: dict, array) -> vx.AnyTransform:
    """
    The transform of a manifest record, reading its tensors by file name with `array`.
    """
    def geometry(record):
        return vx.AcquisitionGeometry(record['shape'], vx.AffineMatrix(array(record['matrix'])))

    kind = record['type']
    if kind == 'affine':
        matrix = array(record['matrix'])
        return vx.AffineMatrix(matrix, dtype=matrix.dtype)
    if kind == 'pair':
        return vx.VectorFieldPair(_decode(record['forward'], array), _decode(record['reverse'], array))
    if kind == 'warp':
        return vx.Warp(array(record['coordinates']), geometry(record['geometry']))
    if kind == 'field':
        return vx.VectorField(array(record['vectors']), geometry(record['geometry']), space=record['space'])
    raise ValueError(f'unknown transform type {kind!r}')


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
        if not isinstance(affine, vx.AffineMatrix):
            raise TypeError(f'a text file holds an affine only, not a {type(affine).__name__}')
        matrix = affine.tensor.detach().cpu().to(torch.float64).tolist()
        with open(filename, 'w') as f:
            for row in matrix:
                f.write(' '.join(f'{x:.17g}' for x in row) + '\n')


transform_io_protocols = [
    VoxelTransformIO,
    TextMatrixIO,
]
