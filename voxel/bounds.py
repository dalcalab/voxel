"""
Bounding box utilities.
"""

from __future__ import annotations

import pathlib
import torch
import voxel as vx


class BoundingBox:

    def __init__(self,
        center: torch.Tensor = None,
        rotation: torch.Tensor = None,
        extent: torch.Tensor = None):
        """
        Args:
            center (torch.Tensor): Center of the bounding box.
            rotation (torch.Tensor): Rotation of the bounding box.
            extent (torch.Tensor): Extents of the bounding box.
        """
        self.center = torch.zeros(3) if center is None else center
        self.rotation = torch.eye(3) if rotation is None else rotation
        self.extent = torch.ones(3) if extent is None else extent

    def to(self, device: torch.Device) -> BoundingBox:
        """
        Move all bounding box parameters to a device.

        Args:
            device: A torch device.

        Returns:
            BoundingBox: A new bounding box instance.
        """
        return BoundingBox(self.center.to(device), self.rotation.to(device), self.extent.to(device))

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
        Compute the corner points of the bounding box.

        Returns:
            Tensor: Corner point tensor of shape (8, 3).
        """
        signs = torch.tensor([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]],
            dtype=torch.float32,
            device=self.center.device) * 2 - 1

        T = torch.eye(4, device=self.center.device, dtype=self.center.dtype)
        T[:3, :3] = self.rotation @ torch.diag(self.extent)
        T[:3, 3] = self.center
        affine = vx.AffineMatrix(T)

        return affine.transform(signs)

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
        Construct a rectangular box mesh from the bounding box.
        
        Returns:
            Mesh: Rectangular box mesh.
        """
        vertices = self.corner_points()

        # triangular faces
        faces = torch.tensor([
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0]],
            dtype=torch.int32,
            device=self.center.device)
        
        return vx.Mesh(vertices, faces)
    
    def geometry(self, spacing: torch.Tensor = None) -> vx.AcquisitionGeometry:
        """
        Construct an acquisition geometry from the bounding box.

        Args:
            spacing (torch.Tensor): Desired voxel spacing of the geometry.
        
        Returns:
            AcquisitionGeometry: Acquisition geometry.
        """
        if spacing is None:
            spacing = torch.ones(3, device=self.center.device, dtype=torch.float32)
        elif not torch.is_tensor(spacing):
            spacing = torch.tensor(spacing, device=self.center.device, dtype=torch.float32)
        if spacing.ndim == 0:
            spacing = spacing.repeat(3)
        if spacing.ndim != 1 or spacing.shape[0] != 3:
            raise ValueError(f'expected 3D spacing, got {spacing.ndim}D')

        # compute volume size
        baseshape = torch.ceil(2 * self.extent / spacing).int()

        # compute half length for re-centering
        half = spacing * (baseshape - 1) / 2

        # compute a voxel to world matrix
        T = torch.eye(4, device=self.center.device)
        T[:3, :3] = self.rotation @ torch.diag(spacing)
        T[:3, 3] = self.center - self.rotation @ half

        return vx.AcquisitionGeometry(baseshape, T)

    def expand(self, margin: float = None, factor: float = None) -> BoundingBox:
        """
        Expand the bounding box by a margin or a factor.

        Args:
            margin (float, optional): Margin to expand the bounding box.
            factor (float, optional): Factor to scale the bounding box.
        
        Returns:
            BoundingBox: Expanded bounding box.
        """
        if margin is not None:
            extent = self.extent + margin
        elif factor is not None:
            extent = self.extent * factor
        else:
            raise ValueError('either `margin` or `factor` should be provided')
        return BoundingBox(self.center, self.rotation, extent)

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

    def fit_extent(self, points: torch.Tensor) -> BoundingBox:
        """
        Fit the extent of the bounding box to a set of points.

        Args:
            points (torch.Tensor): Coordinate point cloud.

        Returns:
            BoundingBox: Bounding box with refit extent.
        """
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


def load_bounding_box(filename: pathlib.Path) -> BoundingBox:
    """
    Load a bounding box from file.

    Args:
        filename (Path): Target file to load.
    
    returns:
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


def obbox_fine_tune(points: torch.Tensor, initial_rotation: torch.Tensor = None) -> BoundingBox:
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
