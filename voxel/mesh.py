"""
Triangular mesh representation and manipulation.
"""

from __future__ import annotations

import os
import torch
import voxel as vx


class Mesh:

    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor) -> None:
        """
        Triangular mesh topology in 3D world space.

        Args:
            vertices (Tensor): Vertex coordinates of shape (V, 3).
            faces (Tensor): Triangular face integer indices of shape (F, 3).
        """
        self.vertices = vertices
        self.faces = faces

    @property
    def vertices(self) -> torch.Tensor:
        """
        Point positions represented by a (V, 3) tensor.
        """
        return self._vertices

    @vertices.setter
    def vertices(self, vertices: torch.Tensor):
        if vertices.ndim != 2 or vertices.shape[-1] != 3:
            raise ValueError('expected shape (V, 3) for vertex tensor, '
                             f'but got {vertices.shape}')
        self._vertices = vertices

    @property
    def num_vertices(self) -> int:
        """
        Total number of vertices in the mesh.
        """
        return len(self.vertices)

    @property
    def faces(self) -> torch.Tensor:
        """
        Triangle faces represented by a (F, 3) tensor.
        """
        return self._faces

    @faces.setter
    def faces(self, faces: torch.Tensor):
        if faces.ndim != 2 or faces.shape[-1] != 3:
            raise ValueError(f'expected shape (F, 3) for faces array, but got {faces.shape}')
        self._faces = faces.int()

    @property
    def num_faces(self) -> int:
        """
        Total number of faces in the mesh.
        """
        return len(self.faces)

    def cpu(self) -> Mesh:
        """
        Move the mesh vertex and face tensors to the CPU.

        Returns:
            Mesh: A new mesh instance with with the data on the CPU.
        """
        return Mesh(self.vertices.cpu(), self.faces.cpu())

    def cuda(self) -> Mesh:
        """
        Move the mesh vertex and face tensors to the GPU.

        Returns:
            Mesh: A new mesh instance with with the data on the GPU.
        """
        return Mesh(self.vertices.cuda(), self.faces.cuda())

    def save(self, filename: os.PathLike, fmt: str = None, **kwargs) -> None:
        """
        Save the mesh to a file.

        Args:
            filename (PathLike): The path to the file to save.
            fmt (str, optional): The format of the file. If None, the format is
                determined by the file extension.
            kwargs: Additional arguments passed to the file writing method.
        """
        vx.save_mesh(self, filename, fmt=fmt, **kwargs)

    def transform(self, transform: vx.AffineMatrix) -> Mesh:
        """
        Transform mesh vertex coordinates with an affine matrix.

        Args:
            transform (vx.AffineMatrix): Affine transformation.

        Returns:
            Mesh: Transformed mesh.
        """
        return Mesh(transform.transform(self.vertices), self.faces)


def construct_box_mesh(min_point: torch.Tensor, max_point: torch.Tensor) -> Mesh:
    """
    Construct a rectangular box mesh from a lower and upper corner coordinate.

    Args:
        min_point (Tensor): Coordinate at the lower corner bound.
        max_point (Tensor): Coordinate at the upper corner bound.

    Returns:
        Mesh: Rectangular mesh.
    """
    mask = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]], device=min_point.device)

    faces = torch.tensor([
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [0, 3, 7],
        [0, 7, 4],
        [1, 2, 6],
        [1, 6, 5]], device=min_point.device)

    points = (mask == 0) * (min_point - 0.5) + mask * (max_point + 0.5)
    return Mesh(points, faces)
