"""
Bounding box utilities.
"""

from __future__ import annotations

import pathlib
import torch
import voxel as vx


# corner sign layout shared by corner_points() and the box face list below
_CORNER_SIGNS = torch.tensor([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]],
    dtype=torch.float32) * 2 - 1

# consistently outward-wound triangular faces for the corner order above
_BOX_FACES = torch.tensor([
    [0, 2, 1],
    [0, 3, 2],
    [4, 5, 6],
    [4, 6, 7],
    [0, 1, 5],
    [0, 5, 4],
    [1, 2, 6],
    [1, 6, 5],
    [2, 3, 7],
    [2, 7, 6],
    [3, 0, 4],
    [3, 4, 7]],
    dtype=torch.int32)


class BoundingBox:
    """
    An oriented 3D bounding box defined by a center, rotation, and half-length
    extent. Instances are immutable: all modifying operations return new boxes,
    and the corner points are computed lazily and cached.
    """

    def __init__(self,
        center: torch.Tensor | None = None,
        rotation: torch.Tensor | None = None,
        extent: torch.Tensor | None = None):
        """
        Args:
            center (Tensor, optional): Box center of shape (3,). Defaults to the origin.
            rotation (Tensor, optional): Box rotation matrix of shape (3, 3).
                Defaults to the identity.
            extent (Tensor, optional): Non-negative box half-lengths of shape (3,).
                Defaults to ones.
        """
        provided = {}
        device = None
        dtype = torch.float32
        for name, value, shape in (('center', center, (3,)),
                                   ('rotation', rotation, (3, 3)),
                                   ('extent', extent, (3,))):
            if value is None:
                continue
            value = torch.as_tensor(value)
            if value.shape != shape:
                raise ValueError(f'box {name} must have shape {shape}, got {tuple(value.shape)}')
            if not value.is_floating_point():
                value = value.float()
            if device is None:
                device = value.device
            elif value.device != device:
                raise ValueError(f'box parameters must share a device, got '
                                 f'{device} and {value.device} - move them explicitly')
            dtype = torch.promote_types(dtype, value.dtype)
            provided[name] = value

        device = torch.device('cpu') if device is None else device
        make = lambda name, default: default(device=device, dtype=dtype) \
            if provided.get(name) is None else provided[name].to(dtype)
        self._center = make('center', lambda **kw: torch.zeros(3, **kw))
        self._rotation = make('rotation', lambda **kw: torch.eye(3, **kw))
        self._extent = make('extent', lambda **kw: torch.ones(3, **kw))

        if (self._extent < 0).any():
            raise ValueError('box extent must be non-negative')

        # lazily cached corner points - safe since box parameters never mutate
        self._corners = None

    @property
    def center(self) -> torch.Tensor:
        """
        Box center coordinate of shape (3,).
        """
        return self._center

    @property
    def rotation(self) -> torch.Tensor:
        """
        Box rotation matrix of shape (3, 3), with columns as box axes.
        """
        return self._rotation

    @property
    def extent(self) -> torch.Tensor:
        """
        Box half-lengths of shape (3,) along each box axis.
        """
        return self._extent

    @property
    def device(self) -> torch.device:
        """
        Device of the box parameters.
        """
        return self._center.device

    @classmethod
    def from_min_max(cls,
        min_point: torch.Tensor,
        max_point: torch.Tensor) -> BoundingBox:
        """
        Construct an axis-aligned bounding box from lower and upper corner coordinates.

        Args:
            min_point (Tensor): Lower corner coordinate of shape (3,).
            max_point (Tensor): Upper corner coordinate of shape (3,).

        Returns:
            BoundingBox: Axis-aligned bounding box.
        """
        min_point = torch.as_tensor(min_point).float()
        max_point = torch.as_tensor(max_point).float()
        return cls(center=(min_point + max_point) / 2, extent=(max_point - min_point) / 2)

    @classmethod
    def from_points(cls, points: torch.Tensor) -> BoundingBox:
        """
        Construct an axis-aligned bounding box enclosing a point cloud.

        Args:
            points (Tensor): Coordinate point cloud of shape (N, 3).

        Returns:
            BoundingBox: Axis-aligned bounding box.
        """
        points = torch.as_tensor(points)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(f'points must have shape (N, 3), got {tuple(points.shape)}')
        return cls.from_min_max(points.amin(dim=0), points.amax(dim=0))

    def to(self, device: torch.device) -> BoundingBox:
        """
        Move all bounding box parameters to a device.

        Args:
            device (device): The target device.

        Returns:
            BoundingBox: A new bounding box instance.
        """
        box = BoundingBox(self._center.to(device), self._rotation.to(device),
                          self._extent.to(device))
        if self._corners is not None:
            box._corners = self._corners.to(device)
        return box

    def cpu(self) -> BoundingBox:
        """
        Move all bounding box parameters to the CPU.

        Returns:
            BoundingBox: A new bounding box instance.
        """
        return self.to('cpu')

    def cuda(self) -> BoundingBox:
        """
        Move all bounding box parameters to the GPU.

        Returns:
            BoundingBox: A new bounding box instance.
        """
        return self.to('cuda')

    def detach(self) -> BoundingBox:
        """
        Detach all bounding box parameters from the current graph.

        Returns:
            BoundingBox: A new bounding box instance.
        """
        box = BoundingBox(self._center.detach(), self._rotation.detach(),
                          self._extent.detach())
        if self._corners is not None:
            box._corners = self._corners.detach()
        return box

    def save(self, filename: pathlib.Path) -> None:
        """
        Save the bounding box to file.

        Args:
            filename (Path): File destination.
        """
        params = dict(center=self.center, rotation=self.rotation, extent=self.extent)
        torch.save(params, filename)

    def corner_points(self) -> torch.Tensor:
        """
        Compute the corner points of the bounding box. The result is
        cached after the first call.

        Returns:
            Tensor: Corner point tensor of shape (8, 3).
        """
        if self._corners is None:
            signs = _CORNER_SIGNS.to(device=self.device, dtype=self._center.dtype)
            self._corners = self._center + (signs * self._extent) @ self._rotation.T
        return self._corners

    def min_max_coords(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the minimum and maximum coordinates of the bounding box.

        Returns:
            tuple: Minimum and maximum coordinates.
        """
        points = self.corner_points()
        min_coord = points.amin(dim=0)
        max_coord = points.amax(dim=0)
        return min_coord, max_coord

    def mesh(self) -> vx.Mesh:
        """
        Construct a rectangular box mesh with outward-facing windings from
        the bounding box.

        Returns:
            Mesh: Rectangular box mesh.
        """
        faces = _BOX_FACES.to(self.device)
        # a reflected frame would flip the outward windings
        if torch.linalg.det(self._rotation) < 0:
            faces = faces.flip(-1)
        return vx.Mesh(self.corner_points(), faces)

    def geometry(self, spacing: torch.Tensor | None = None) -> vx.AcquisitionGeometry:
        """
        Construct an acquisition geometry from the bounding box.

        Args:
            spacing (Tensor, optional): Desired voxel spacing of the geometry. Defaults to ones.

        Returns:
            AcquisitionGeometry: Acquisition geometry.
        """
        if spacing is None:
            spacing = torch.ones(3, device=self.device, dtype=torch.float32)
        elif not torch.is_tensor(spacing):
            spacing = torch.tensor(spacing, device=self.device, dtype=torch.float32)
        if spacing.ndim == 0:
            spacing = spacing.repeat(3)
        if spacing.ndim != 1 or spacing.shape[0] != 3:
            raise ValueError(f'expected 3D spacing, got {spacing.ndim}D')

        # compute volume size
        baseshape = torch.ceil(2 * self.extent / spacing).int()

        # compute half length for re-centering
        half = spacing * (baseshape - 1) / 2

        # compute a voxel to world matrix
        T = torch.eye(4, device=self.device, dtype=self._center.dtype)
        T[:3, :3] = self.rotation @ torch.diag(spacing)
        T[:3, 3] = self.center - self.rotation @ half

        return vx.AcquisitionGeometry(baseshape, T)

    def shift(self, delta: float | torch.Tensor) -> BoundingBox:
        """
        Translate the bounding box center.

        Args:
            delta (float or Tensor): Translation delta, as a scalar or (3,) tensor,
                applied in the ambient coordinate frame.

        Returns:
            BoundingBox: Shifted bounding box.
        """
        delta = self._conform_units(delta)
        return BoundingBox(self._center + delta, self._rotation, self._extent)

    def scale(self, factor: float | torch.Tensor) -> BoundingBox:
        """
        Scale the bounding box extent about its center.

        Args:
            factor (float or Tensor): Scaling factor, as a scalar or (3,) tensor
                of per-axis factors in the box frame.

        Returns:
            BoundingBox: Scaled bounding box.
        """
        factor = self._conform_units(factor)
        return BoundingBox(self._center, self._rotation, self._extent * factor)

    def pad(self, margin: float | torch.Tensor) -> BoundingBox:
        """
        Pad (or shrink) the bounding box sides by a margin.

        Args:
            margin (float or Tensor): Margin delta for each side of the box, in box-frame
                units. Can be a scalar, a (3,) tensor of per-axis margins, or a (3, 2)
                tensor of per-side (lower, upper) margins. Negative values shrink the box.

        Returns:
            BoundingBox: Padded bounding box.
        """
        margin = self._conform_units(margin, 2)
        lower, upper = margin[:, 0], margin[:, 1]
        extent = self._extent + (lower + upper) / 2
        center = self._center + self._rotation @ ((upper - lower) / 2)
        return BoundingBox(center, self._rotation, extent)

    def trim(self, margin: float | torch.Tensor) -> BoundingBox:
        """
        Trim the bounding box sides by a margin. This is the inverse of `pad`.

        Args:
            margin (float or Tensor): Margin delta for each side of the box, in box-frame
                units. Can be a scalar, a (3,) tensor of per-axis margins, or a (3, 2)
                tensor of per-side (lower, upper) margins.

        Returns:
            BoundingBox: Trimmed bounding box.
        """
        return self.pad(-self._conform_units(margin, 2))

    def rotate(self, rotation: torch.Tensor, degrees: bool = True) -> BoundingBox:
        """
        Applies a rotation to the bounding box.

        Args:
            rotation (Tensor): Rotation angles. If `degrees` is True, the
                angles are in degrees, otherwise they are in radians.
            degrees (bool, optional): Whether the angles are defined as degrees or,
                alternatively, as radians.

        Returns:
            BoundingBox: Rotated bounding box.
        """
        matrix = vx.affine.angles_to_rotation_matrix(rotation, degrees)[:3, :3]
        rotated = matrix @ self.rotation
        return BoundingBox(self.center, rotated, self.extent)

    def transform(self, matrix: vx.AffineMatrix) -> BoundingBox:
        """
        Transform the bounding box through an affine matrix. This is exact as
        long as the box axes remain orthogonal under the transform (any rotation
        combined with scaling along the box axes), and an approximation under shear.

        Args:
            matrix (AffineMatrix): Affine transform to apply.

        Returns:
            BoundingBox: Transformed bounding box.
        """
        if not isinstance(matrix, vx.AffineMatrix):
            matrix = vx.AffineMatrix(matrix)
        linear = matrix.tensor.to(self.device)[:3, :3].to(self._center.dtype)
        # map the unit box axes (rotation columns) and renormalize, folding
        # any change in axis length into the extent
        axes = linear @ self._rotation
        factor = axes.norm(dim=0)
        return BoundingBox(center=matrix.transform(self._center),
                           rotation=axes / factor,
                           extent=self._extent * factor)

    def fit_extent(self, points: torch.Tensor) -> BoundingBox:
        """
        Fit the extent of the bounding box to a set of points, keeping
        the current rotation.

        Args:
            points (Tensor, BoundingBox, Mesh, or AcquisitionGeometry): Coordinate
                point cloud of shape (N, 3), or an object defining one.

        Returns:
            BoundingBox: Bounding box with refit extent.
        """
        if isinstance(points, BoundingBox):
            points = points.corner_points()
        elif isinstance(points, vx.AcquisitionGeometry):
            points = points.bounds().corner_points()
        elif isinstance(points, vx.Mesh):
            points = points.vertices

        assert points.ndim == 2 and points.shape[1] == 3, "points should be of shape (N, 3)"

        # center points and project onto eigenvectors
        centroid = points.mean(dim=0)
        centered_points = points - centroid
        projected = centered_points @ self.rotation

        # find min and max along each principal axis
        min_proj = projected.amin(dim=0)
        max_proj = projected.amax(dim=0)

        # compute OBB parameters and recompute the center in global coordinates
        center_local = (min_proj + max_proj) / 2
        extents = (max_proj - min_proj) / 2
        obb_center = centroid + self.rotation @ center_local

        return BoundingBox(obb_center, self.rotation, extents)

    def fine_tune(self, points: torch.Tensor) -> BoundingBox:
        """
        Orient the bounding box to minimize the volume of the box enclosing a point cloud.

        Args:
            points (Tensor): Coordinate point cloud of shape (N, 3).

        Returns:
            BoundingBox: Fine-tuned bounding box.
        """
        return obbox_fine_tune(points, self.rotation)

    def _conform_units(self, units: float | torch.Tensor, num: int | None = None) -> torch.Tensor:
        """
        Conform scalar or per-axis units to the box device and dtype.

        Args:
            units (float or Tensor): Units of size (1,) or (3,) or (3, num).
            num (int, optional): Number of units per axis, see
                `vx.slicing.conform_coordinates`.

        Returns:
            Tensor: Units of shape (3,) or (3, num).
        """
        units = torch.as_tensor(units, device=self.device, dtype=self._center.dtype)
        return vx.slicing.conform_coordinates(units, num)


def load_bounding_box(filename: pathlib.Path) -> BoundingBox:
    """
    Load a bounding box from file.

    Args:
        filename (Path): Target file to load.

    Returns:
        BoundingBox: Loaded bounding box.
    """
    return BoundingBox(**torch.load(filename, weights_only=False))


def obbox(points: torch.Tensor, initialize: bool = True, fine_tune: bool = True) -> BoundingBox:
    """
    Compute an oriented bounding box (OBB).

    Args:
        points (Tensor): Coordinate point cloud of shape (N, 3).
        initialize (bool, optional): Whether to initialize the rotation with PCA.
        fine_tune (bool, optional): Whether to fine-tune the bounds to minimize the volume.

    Returns:
        BoundingBox: Oriented bounding box.
    """
    assert initialize or fine_tune, 'either `initialize` or `fine-tune` should be enabled'

    bounds = obbox_pca(points) if initialize else BoundingBox()

    if fine_tune:
        rotation = bounds.rotation if initialize else None
        bounds = obbox_fine_tune(points, rotation)

    return bounds


def obbox_pca(points: torch.Tensor) -> BoundingBox:
    """
    Compute an oriented bounding box (OBB) using PCA.

    Args:
        points (Tensor): Coordinate point cloud of shape (N, 3).

    Returns:
        BoundingBox: Oriented bounding box.
    """
    assert points.ndim == 2 and points.shape[1] == 3, 'points should be of shape (N, 3)'

    # center points
    centroid = points.mean(dim=0)
    centered_points = points - centroid

    # compute covariance matrix (3x3)
    cov = centered_points.t() @ centered_points / points.shape[0]

    # eigenvalues and eigenvectors
    _, eigenvectors = torch.linalg.eigh(cov)

    # project points onto eigenvectors
    projected = centered_points @ eigenvectors

    # find min and max along each principal axis
    min_proj = projected.amin(dim=0)
    max_proj = projected.amax(dim=0)

    # compute OBB parameters
    extent = (max_proj - min_proj) / 2

    # compute the center in global coordinates
    center_local = (min_proj + max_proj) / 2
    obb_center = centroid + eigenvectors @ center_local

    return BoundingBox(obb_center, eigenvectors, extent)


def obbox_fine_tune(points: torch.Tensor, initial_rotation: torch.Tensor | None = None) -> BoundingBox:
    """
    Fine-tune an oriented bounding box (OBB) to minimize the volume of the box
    enclosing a point cloud.

    Args:
        points (Tensor): Coordinate point cloud of shape (N, 3).
        initial_rotation (Tensor, optional): Initial rotation matrix of shape (3, 3).

    Returns:
        BoundingBox: Oriented bounding box.
    """
    assert points.ndim == 2 and points.shape[1] == 3, 'points should be of shape (N, 3)'

    stepsize = 1e-2
    maxsteps = 200

    if initial_rotation is None:
        initial_rotation = torch.eye(3, device=points.device)

    with torch.no_grad():

        centroid = points.mean(dim=0)
        centered_points = points - centroid

        # initialize angles and center deltas
        angles = torch.zeros(3, device=points.device)
        center = torch.zeros(3, device=points.device)

        # compute initial volume for reference and as a gradient scaling factor
        projected = centered_points @ initial_rotation
        min_proj = projected.amin(dim=0)
        max_proj = projected.amax(dim=0)
        inital_volume = (max_proj - min_proj).prod()

        # optimization loop
        history = []
        for step in range(maxsteps):

            # construct rotation matrix
            zero = torch.zeros((), device=points.device)
            one = torch.ones((), device=points.device)
            cos = angles.cos()
            sin = angles.sin()
            rx = torch.stack([one, zero, zero, zero, cos[0], sin[0], zero, -sin[0], cos[0]]).view(3, 3)
            ry = torch.stack([cos[1], zero, sin[1], zero, one, zero, -sin[1], zero, cos[1]]).view(3, 3)
            rz = torch.stack([cos[2], sin[2], zero, -sin[2], cos[2], zero, zero, zero, one]).view(3, 3)
            delta_rotation = rx @ ry @ rz

            rotation = initial_rotation @ delta_rotation
            translated = centered_points + center
            projected = translated @ rotation

            min_proj, min_idx = projected.min(dim=0)
            max_proj, max_idx = projected.max(dim=0)

            diff = max_proj - min_proj
            volume = diff.prod()

            # compute first set of projected gradients
            d_max = volume / diff
            d_projected = torch.zeros_like(projected)
            d_projected[min_idx, [0, 1, 2]] = -d_max
            d_projected[max_idx, [0, 1, 2]] =  d_max

            # center gradients
            d_translated = d_projected @ rotation.T
            d_center = d_translated.sum(dim=0)
            d_rotation = translated.T @ d_projected
            d_delta_rotation = initial_rotation.T @ d_rotation

            # matrix composition gradients
            d_rx = d_delta_rotation @ rz.T @ ry.T
            d_ry = rx.T @ d_delta_rotation @ rz.T
            d_rz = (rx @ ry).T @ d_delta_rotation

            # sine gradients
            d_cos = torch.stack([d_rx[1, 1] + d_rx[2, 2], d_ry[0, 0] + d_ry[2, 2], d_rz[0, 0] + d_rz[1, 1]])
            d_sin = torch.stack([d_rx[1, 2] - d_rx[2, 1], d_ry[0, 2] - d_ry[2, 0], d_rz[0, 1] - d_rz[1, 0]])
            d_angles = d_cos * -sin + d_sin * cos
            d_angles /= inital_volume

            # update angles and center
            angles = angles - stepsize * d_angles
            center = center - stepsize * d_center

            # normalized volume relative to the initial volume
            relative_cost = volume / inital_volume

            # early stopping if flat improvement - the window was chosen somewhat arbitrarily
            window = 20
            threshold = sum(history[-window:]) / window
            if step > window and relative_cost > threshold:
                break

            history.append(relative_cost.item())

    # use the original rotation if the cost is higher
    if relative_cost > 1:
        rotation = initial_rotation
        projected = centered_points @ rotation
        min_proj = projected.amin(dim=0)
        max_proj = projected.amax(dim=0)

    # compute OBB parameters
    center_local = (min_proj + max_proj) / 2
    obb_center = centroid + rotation @ center_local
    extent = (max_proj - min_proj) / 2

    return BoundingBox(obb_center, rotation, extent)
