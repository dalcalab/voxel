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
        vx.caching.init_property_cache(self)
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
        mesh = self.__class__(vertices, self.faces)
        vx.caching.transfer_property_cache(self, mesh)
        return mesh

    @property
    def device(self) -> torch.Device:
        """
        The device of the mesh vertex and face tensors.
        """
        return self.vertices.device

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

    def type(self, dtype: torch.dtype) -> Mesh:
        """
        Cast the mesh vertices to a new data type.

        Args:
            dtype (torch.dtype): The target vertex data type.

        Returns:
            Mesh: A new mesh instance with the casted vertices.
        """
        return self.new(self.vertices.type(dtype))

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

    @vx.caching.cached
    def triangles(self) -> torch.Tensor:
        """
        Triangle coordinate arrary with shape (F, 3, 3).
        """
        return self.vertices[self.faces]

    @vx.caching.cached
    def triangles_cross(self) -> torch.Tensor:
        """
        Vertex cross-product with shape (F, 3).
        """
        vecs = torch.diff(self.triangles, dim=1)
        return torch.cross(vecs[:, 0], vecs[:, 1], dim=1)

    @vx.caching.cached_transferable
    def edges(self) -> torch.Tensor:
        """
        All directional edges in the mesh, with shape (E, 2). Note these are not unique.
        """
        return self.faces.repeat_interleave(2, dim=-1).roll(-1, dims=-1).view((-1, 2))

    @vx.caching.cached_transferable
    def uniform_laplacian(self) -> torch.Tensor:
        """
        The sparse uniform Laplacian matrix for the mesh connectivity.
        Note that it is generally faster to use the `gather()` or `smooth_features()`
        methods for diffusing features on the mesh graph.
        """
        V = self.num_vertices

        # compute the uniform weight matrix
        dest = self.edges[:, 0]
        count = torch.bincount(dest, minlength=V)
        weight = count[dest]
        val = torch.where(weight > 0.0, 1.0 / weight, weight)
        L = torch.sparse_coo_tensor(self.edges.T.type(torch.int64), val, (V, V))

        # subtract identity
        idx = torch.arange(V, device=self.device)
        idx = torch.stack([idx, idx], dim=0)
        # NOTE: once sparse torch.eye is implemented globally, use that instead
        ones = torch.ones(V, dtype=torch.float32, device=self.device)
        L -= torch.sparse_coo_tensor(idx, ones, (V, V))
        return L

    @vx.caching.cached_transferable
    def edge_face(self) -> torch.Tensor:
        """
        Face indices corresponding to each directional edge in the
        mesh, with shape (E,).
        """
        arange = torch.arange(self.num_faces, device=self.faces.device)
        return arange.repeat_interleave(3).reshape(-1)

    @vx.caching.cached_transferable
    def unique_edge_indices(self) -> tuple:
        """
        Indices that extract all unique edges from the directional edge list.
        """
        aligned = self.edges.sort(dim=1)[0]

        # this is a way to do lexsort in pytorch, which is faster
        # than using torch.unique
        idx = aligned[:, 1].argsort(dim=-1, stable=True)
        order = idx.gather(-1, aligned[:, 0].gather(-1, idx).argsort(dim=-1, stable=True))

        pef = aligned[order]
        device = self.faces.device
        shift = torch.cat([torch.tensor([True], device=device),
                           torch.any(pef[1:] != pef[:-1], dim=-1),
                           torch.tensor([True], device=device)])
        matched = shift.nonzero(as_tuple=False).squeeze(-1)
        repeated = torch.repeat_interleave(torch.arange(len(matched) - 1, device=device),
                                           (matched[1:] - matched[:-1]))
        reverse = repeated[order.argsort()]
        indices = order[matched[:-1]]

        repeats = torch.all(pef[:-1] == pef[1:], dim=-1)
        matched = repeats.nonzero(as_tuple=False).squeeze(-1)
        bidir_indices = order[torch.stack([matched, matched + 1], dim=1)]

        return indices, reverse, bidir_indices

    @vx.caching.cached_transferable
    def unique_edges(self):
        """
        Unique bi-directional edges in the mesh, with shape (U, 2).
        """
        return self.edges[self.unique_edge_indices[0]]

    @vx.caching.cached_transferable
    def adjacent_faces(self):
        """
        Adjacent faces indices corresponding to each edge in `unique_edges`.
        """
        return self.edge_face[self.unique_edge_indices[2]]

    @vx.caching.cached
    def face_normals(self) -> torch.Tensor:
        """
        Face (unit) normals with shape (F, 3).
        """
        return torch.nn.functional.normalize(self.triangles_cross)

    @vx.caching.cached
    def face_areas(self) -> torch.Tensor:
        """
        Face areas with shape (F,)
        """
        return (self.triangles_cross ** 2).sum(-1).sqrt() / 2

    @vx.caching.cached
    def face_angles(self) -> torch.Tensor:
        """
        Face angles (in radians) with shape (F, 3).
        """
        triangles = self.triangles
        u = torch.nn.functional.normalize(triangles[:, 1] - triangles[:, 0])
        v = torch.nn.functional.normalize(triangles[:, 2] - triangles[:, 0])
        w = torch.nn.functional.normalize(triangles[:, 2] - triangles[:, 1])
        angles = torch.zeros((self.num_faces, 3), device=u.device)
        angles[:, 0] = ( u * v).sum(-1).clamp(-1, 1).arccos()
        angles[:, 1] = (-u * w).sum(-1).clamp(-1, 1).arccos()
        angles[:, 2] = torch.pi - angles[:, 0] - angles[:, 1]
        return angles

    @vx.caching.cached
    def vertex_normals(self) -> torch.Tensor:
        """
        Vertex (unit) normals, computed from face normals weighted by their angle.
        """
        scalars = self.face_angles.view(-1, 1) * self.face_normals.repeat_interleave(3, dim=0)
        indices = self.faces.type(torch.int64).view(-1, 1).expand(-1, 3)
        normals = torch.zeros_like(self.vertices).scatter_add(-2, indices, scalars)
        return torch.nn.functional.normalize(normals)

    @vx.caching.cached
    def vertex_areas(self) -> torch.Tensor:
        """
        The total face surface area contributed to each vertex.
        """
        area_contributions = self.face_areas.unsqueeze(-1) * self.face_angles / torch.pi
        source = area_contributions.view(-1)
        indices = self.faces.view(-1).long()
        zeros = torch.zeros(self.num_vertices, dtype=self.vertices.dtype, device=self.device)
        return zeros.scatter_reduce(0, indices, source, reduce='sum', include_self=False)

    def flip_faces(self) -> vx.Mesh:
        """
        Flip triangular face directions.
        """
        return vx.Mesh(self.vertices, self.faces.flip(-1))

    def gather(self, features: torch.Tensor, reduce: str = 'mean') -> torch.Tensor:
        """
        """
        edges = self.edges
        source = features[edges[:, 0]]
        indices = edges[:, 1].type(torch.int64).view(-1, 1).expand(-1, source.shape[-1])
        reduced = torch.zeros_like(features).scatter_reduce(-2, indices, source,
                                                            reduce=reduce, include_self=False)
        return reduced

    def smooth_mesh(self, alpha: float = 0.5, iterations: int = 1) -> Mesh:
        """
        Smooth the mesh vertex positions with the uniform Laplacian operator.

        Args:
            alpha (float, optional): Smoothing factor between 0 and 1. Defaults to 0.5.
            iterations (int, optional): Number of smoothing iterations. Defaults to 1.

        Returns:
            Mesh: Smoothed mesh.
        """
        return self.new(self.smooth_features(self.vertices, alpha=alpha, iterations=iterations))

    def smooth_features(self, features: torch.Tensor, alpha: float = 0.5, iterations: int = 1) -> torch.Tensor:
        """
        Smooth vertex features with the uniform Laplacian operator. Note that this does not actually use
        the sparse Laplacian matrix, but rather the vertex gather method, which is roughly 2x faster.

        Args:
            features (Tensor): Input features to smooth of the shape $(V,)$ or $(V, C)$.
            alpha (float, optional): Smoothing factor between 0 and 1. Defaults to 0.5.
            iterations (int, optional): Number of smoothing iterations. Defaults to 1.

        Returns:
            Tensor: Smoothed features matching the input shape.
        """
        is1d = features.ndim == 1
        if is1d:
            features = features.view(-1, 1)
        for _ in range(iterations):
            features = (1 - alpha) * features + alpha * self.gather(features)
        if is1d:
            features = features.squeeze(-1)
        return features

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
            margin = vx.slicing.conform_coordinates(margin, 2)
            min_point -= margin[:, 0]
            max_point += margin[:, 1]

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

        # convert to a sparse scipy matrix and compute the components
        N = self.num_vertices
        row, col = self.edges.detach().cpu().T.numpy()
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
