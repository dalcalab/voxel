"""
Mapping of image voxel coordinates to world space.
"""

from __future__ import annotations

import torch
import voxel as vx


class AcquisitionGeometry(vx.AffineMatrix):
    """
    Geometry representating the linear relationship between volumetric
    image coordinates and a three-dimensional world space.
    """

    def __init__(self,
        baseshape: torch.Size,
        matrix: vx.AffineMatrix | None = None,
        slice_direction: int | None = None,
        device: torch.device | None = None) -> None:
        """
        Args:
            baseshape (Size): The spatial shape (3D dimensions) of the acquisition.
            matrix (AffineMatrix, optional): Affine matrix representing a voxel-to-world
                coordinate transform. If None, it defaults to a shifted identity in
                which the image volume is centered at the world origin.
            slice_direction (int, optional): Voxel-space index representing the dimension
                of slice direction, also known as the through-plane acquisition direction.
                If None, the direction is inferred from the largest voxel spacing.
            device (device, optional): Device to store the matrix on.
        """
        if matrix is None:
            matrix = torch.eye(4, device=device)
            matrix[:3, -1] = -(torch.tensor(baseshape, device=device) - 1) / 2
        super().__init__(matrix, device)
        self._baseshape = torch.Size(baseshape)
        self._explicit_slice_direction = slice_direction
        self.metadata = {}

    def __setitem__(self, indexing, item):
        """
        Prevents modification of the matrix tensor.
        """
        raise AttributeError('AcquisitionGeometry matrix is read-only')

    @property
    def device(self) -> torch.device:
        """
        Device of the matrix tensor.
        """
        return self.tensor.device

    @property
    def baseshape(self) -> torch.Size:
        """
        Spatial shape (3D dimensions) of the acquisition.
        """
        return self._baseshape

    @vx.caching.cached
    def spacing(self) -> torch.Tensor:
        """
        Spacing between voxel centers.
        """
        return torch.linalg.qr(self[:3, :3])[1].diagonal().abs()
 
    @vx.caching.cached
    def slice_direction(self) -> int:
        """
        Index representing the dimension of the slice acquisition
        direction, or through-plane, direction of the acquisition.
        """
        if self._explicit_slice_direction is not None:
            return self._explicit_slice_direction
        else:
            return self.spacing.argmax()

    @vx.caching.cached
    def in_plane_directions(self) -> list:
        """
        List of two indices representing the dimensions of the in-plane
        acquisition directions, which are orthogonal to the slice direction.
        """
        return [d for d in range(3) if d != self.slice_direction]

    @vx.caching.cached
    def slice_spacing(self) -> torch.Tensor:
        """
        Spacing (separation) between slices.
        """
        return self.spacing[self.slice_direction]

    @vx.caching.cached
    def in_plane_spacing(self) -> torch.Tensor:
        """
        Spacings between voxel centers in the in-plane directions.
        """
        return self.spacing[self.in_plane_directions]

    def is_isotropic(self, rtol: float = 1e-2) -> bool:
        """
        Determine if voxel spacing is isotropic within a relative tolerance.

        Args:
            rtol (float): Relative tolerance of difference across spacings.

        Returns:
            bool: True if isotropic spacing.
        """
        mean = self.in_plane_spacing.mean()
        return self.spacing.allclose(mean, atol=0, rtol=rtol)

    def shift(self, delta: float | torch.Tensor, space: vx.Space) -> AcquisitionGeometry:
        """
        Shift, or translate, the acquisition geometry.

        Args:
            delta (float or torch.Tensor): The shift amount.
            space (Space): The space in which to apply the shift.

        Returns:
            AcquisitionGeometry: The shifted geometry.
        """
        trf = torch.eye(4, device=self.device)
        trf[:3, 3] = torch.as_tensor(delta, device=self.device)
        matrix = trf @ self.tensor if vx.Space(space) =='world' else self.tensor @ trf
        geometry = AcquisitionGeometry(self.baseshape, matrix,
                                       slice_direction=self._explicit_slice_direction)
        return geometry

    def scale(self, factor: float | torch.Tensor, space: vx.Space) -> AcquisitionGeometry:
        """
        Scale the acquisition geometry.

        Args:
            factor (float or torch.Tensor): The scaling factor.
            space (Space): The space in which to apply the scale.

        Returns:
            AcquisitionGeometry: The scaled geometry.
        """
        diag = torch.ones(4, device=self.device)
        diag[:3] = torch.as_tensor(factor, device=self.device)
        trf = torch.diag(diag)
        matrix = trf @ self.tensor if vx.Space(space) =='world' else self.tensor @ trf
        geometry = AcquisitionGeometry(self.baseshape, matrix,
                                       slice_direction=self._explicit_slice_direction)
        return geometry


def cast_acquisition_geometry(obj: vx.Volume | AcquisitionGeometry) -> AcquisitionGeometry:
    """
    Cast item to an AcquisitionGeometry

    Args:
        obj (Volume | AcquisitionGeometry): Object to cast.
    
    Returns:
        AcquisitionGeometry
    """
    if isinstance(obj, vx.Volume):
        return obj.geometry
    elif isinstance(obj, vx.AcquisitionGeometry):
        return obj
    else:
        raise ValueError(f'cannot cast {type(obj)} to AcquisitionGeometry')
