"""
Triangular mesh representation and manipulation.
"""

from __future__ import annotations

import os
import torch
import numpy as np
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

    def new(self, vertices: torch.Tensor) -> Mesh:
        """
        Construct a new mesh instance with the provided vertices, while
        preserving any unchanged properties of the original.

        Args:
            vertices (Tensor): The new vertices tensor replacement.
        """
        return self.__class__(vertices, self.faces)

    def to(self, device: torch.Device) -> Mesh:
        """
        Move the mesh (vertex and faces tensors) to a device.

        Args:
            device: A torch device.

        Returns:
            Mesh: A new mesh instance.
        """
        if device is None:
            return self
        return Mesh(self.vertices.to(device), self.faces.to(device))

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

    @property
    def triangles(self) -> torch.Tensor:
        """
        Triangle coordinate arrary with shape (F, 3, 3).
        """
        return self.vertices[self.faces]

    @property
    def triangles_cross(self) -> torch.Tensor:
        """
        Vertex cross-product with shape (F, 3).
        """
        vecs = torch.diff(self.triangles, dim=1)
        return torch.cross(vecs[:, 0], vecs[:, 1])

    @property
    def edges(self) -> torch.Tensor:
        """
        All directional edges in the mesh, with shape (E, 2).
        """
        return self.faces[:, [0, 1, 1, 2, 2, 0]].view((-1, 2))

    @property
    def edge_face(self) -> torch.Tensor:
        """
        Face indices corresponding to each directional edge in the
        mesh, with shape (E,).
        """
        arange = torch.arange(self.num_faces, device=self.faces.device)
        return arange.repeat_interleave(3).reshape(-1)

    @property
    def unique_edge_indices(self) -> tuple:
        """
        Indices that extract all unique edges from the directional edge list.
        """
        device = self.faces.device
        aligned = self.edges.sort(dim=1)[0]

        # similar result to lexsort (TODO: this is very slow)
        idx = aligned[:, 1].argsort(dim=-1, stable=True)
        order = idx.gather(-1, aligned[:, 0].gather(-1, idx).argsort(dim=-1, stable=True))

        pef = aligned[order]
        shift = torch.cat([torch.tensor([True], device=device),
                           torch.any(pef[1:] != pef[:-1], dim=-1),
                           torch.tensor([True], device=device)])
        matched = shift.nonzero(as_tuple=False).squeeze(-1)
        repeated = torch.repeat_interleave(torch.arange(len(matched) - 1, device=device),
                                           (matched[1:] - matched[:-1]))
        reverse = repeated[order.argsort()]
        indices = order[matched[:-1]]
        return indices, reverse

    @property
    def unique_edges(self):
        """
        Unique bi-directional edges in the mesh, with shape (U, 2).
        """
        return self.edges[self.unique_edge_indices[0]]

    @property
    def adjacent_faces(self):
        """
        Adjacent faces indices corresponding to each edge in `unique_edges`.
        """
        indices = self.unique_edge_indices[0].tile((1, 2))
        indices[:, 1] += 1
        return self.edge_face[indices]

    @property
    def face_normals(self) -> torch.Tensor:
        """
        Face (unit) normals with shape (F, 3).
        """
        return normalize(self.triangles_cross)

    @property
    def face_areas(self) -> torch.Tensor:
        """
        Face areas with shape (F,)
        """
        return (self.triangles_cross ** 2).sum(-1).sqrt() / 2

    @property
    def face_angles(self) -> torch.Tensor:
        """
        Face angles (in radians) with shape (F, 3).
        """
        triangles = self.triangles
        u = normalize(triangles[:, 1] - triangles[:, 0])
        v = normalize(triangles[:, 2] - triangles[:, 0])
        w = normalize(triangles[:, 2] - triangles[:, 1])
        angles = torch.zeros((self.num_faces, 3), device=u.device)
        angles[:, 0] = ( u * v).sum(-1).clamp(-1, 1).arccos()
        angles[:, 1] = (-u * w).sum(-1).clamp(-1, 1).arccos()
        angles[:, 2] = torch.pi - angles[:, 0] - angles[:, 1]
        return angles

    @property
    def vertex_normals(self) -> torch.Tensor:
        """
        Vertex (unit) normals, computed from face normals weighted by their angle.
        """
        scalars = self.face_angles.view(-1, 1) * self.face_normals.repeat_interleave(3, dim=0)
        indices = self.faces.type(torch.int64).view(-1, 1).expand(-1, 3)
        normals = torch.zeros_like(self.vertices).scatter_add(-2, indices, scalars)
        return normalize(normals)

    def gather(self, features: torch.Tensor, reduce: str = 'mean'):
        """
        """
        edges = self.edges
        source = features[edges[:, 0]]
        indices = edges[:, 1].type(torch.int64).view(-1, 1).expand(-1, source.shape[-1])
        reduced = torch.zeros_like(features).scatter_reduce(-2, indices, source,
                                                            reduce=reduce, include_self=False)
        return reduced

    def transform(self, transform: vx.AffineMatrix) -> Mesh:
        """
        Transform mesh vertex coordinates with an affine matrix.

        Args:
            transform (AffineMatrix): Affine transformation.

        Returns:
            Mesh: Transformed mesh.
        """
        return Mesh(transform.transform(self.vertices), self.faces)

    def bounds(self, margin: float | torch.Tensor = None) -> Mesh:
        """
        Compute a box mesh enclosing the vertex bounds.

        Args:
            margin (float or Tensor, optional): Margin (in vertex units) to expand
                the cropping boundary. Can be a positive or negative delta.

        Returns:
            Mesh: Bounding box mesh.
        """
        min_point = self.vertices.amin(dim=0).float()
        max_point = self.vertices.amax(dim=0).float()

        # expand (or shrink) margin around border
        if margin is not None:
            threes = margin.shape == (3,)
            min_point -= margin if threes else margin[0]
            min_point += margin if threes else margin[1]

        return construct_box_mesh(min_point, max_point)

    def extract_submesh(self, vertex_mask: torch.Tensor) -> Mesh:
        """
        Extract a submesh containing only the vertices included in an input mask.

        Args:
            vertex_mask (Tensor): A boolean vertex mask indicating which vertices
                to include in the submesh.

        Returns:
            Mesh: Extracted mesh with a subset of vertices and remapped face indices.
        """
        vertex_mask = vertex_mask.bool()
        new_num_vertices = vertex_mask.count_nonzero()

        mapping = torch.full((self.num_vertices,), -1, dtype=torch.int32, device=vertex_mask.device)
        mapping[vertex_mask] = torch.arange(new_num_vertices, dtype=torch.int32, device=vertex_mask.device)

        faces = mapping[self.faces]
        faces = faces[(faces >= 0).all(dim=1)]

        return Mesh(self.vertices[vertex_mask], faces)

    def largest_connected_components(self, k: int = 1) -> torch.Tensor:
        """
        Compute a mask indicating the top k largest connected components in the mesh graph, where
        'largest' is defined by number of vertices. This method uses scipy under the hood and
        therefore runs on the CPU.

        Args:
            k (int, optional): The top k largest components to include. Defaults to 1.

        Returns:
            Tensor: A boolean vertex mask representing vertices in the top k components.
        """
        import scipy.sparse as sp

        # TODO: extract this out to a different function (note: using unique edges
        # is maybe faster, but would need to change connected_components arg to undirected
        edges = self.faces[:, [0, 1, 1, 2, 2, 0]].view((-1, 2))

        # convert to a sparse scipy matrix and compute the components
        N = self.num_vertices
        row, col = edges.detach().cpu().T.numpy()
        adj = sp.coo_matrix((np.ones(len(row)), (row, col)), (N, N))
        num_found, components = sp.csgraph.connected_components(adj)

        if num_found <= k:
            return torch.ones(self.num_vertices, device=self.vertices.device, dtype=torch.bool)

        # compile the top largest into a single vertex mask
        _, count = np.unique(components, return_counts=True)
        subset_np = np.in1d(components, count.argsort()[-k:])
        vertex_mask = torch.from_numpy(subset_np).to(device=self.vertices.device, dtype=torch.bool)
        return vertex_mask


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


def normalize(vector):
    # TODO move this
    return vector / (vector * vector).sum(-1).sqrt().unsqueeze(-1)
