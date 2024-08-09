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

    @vx.caching.cached
    def orientation(self) -> Orientation:
        """
        """
        return Orientation(self)

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
            delta (float or Tensor): The shift amount.
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

    def rotate(self,
        rotation: torch.Tensor,
        space: vx.Space,
        degrees: bool = True) -> AcquisitionGeometry:
        """
        Rotate the acquisition geometry.

        Args:
            rotation (Tensor): Rotation angles (x, y, z). If `degrees` is True, the
                angles are in degrees, otherwise they are in radians.
            space (Space): The space in which to apply the rotation.
            degrees (bool, optional): Whether the angles are defined as degrees or,
                alternatively, as radians.

        Returns:
            AcquisitionGeometry: The rotated geometry.
        """
        rotation = torch.as_tensor(rotation, device=self.device)
        trf = vx.affine.angles_to_rotation_matrix(rotation, degrees=degrees)
        matrix = trf @ self if vx.Space(space) =='world' else self @ trf
        geometry = AcquisitionGeometry(self.baseshape, matrix,
                                       slice_direction=self._explicit_slice_direction)
        return geometry

    def reorient(self, target):
        """
        """
        source = self.orientation
        target = cast_orientation(target)

        perm = source.dims[target.dims]
        flip = source.flip * target.flip[perm.argsort()]

        baseshape = torch.tensor(self.baseshape)

        trf = vx.AffineMatrix(torch.diag(flip)[:, perm])
        trf[:3, -1] = (baseshape - 1) * (flip < 0)

        slice_direction = None
        if self._explicit_slice_direction is not None:
            slice_direction = perm.argsort()[self.slice_direction]

        return AcquisitionGeometry(baseshape[perm], self @ trf, slice_direction=slice_direction)

    def bounds(self, margin: float | torch.Tensor = None) -> vx.Mesh:
        """
        Compute a box mesh enclosing the bounds of the grid.

        Args:
            margin (float or Tensor, optional): Margin (in world units) to expand
                the cropping boundary. Can be a positive or negative delta.

        Returns:
            Mesh: Bounding box mesh in world-space coordinates.
        """
        minc = torch.zeros(3, device=self.device)
        maxc = torch.tensor(self.baseshape, device=self.device).float() - 1
        
        # expand (or shrink) margin around border
        if margin is not None:
            margin = (margin / self.spacing).round().int().view(-1, 3).expand(2, 3)
            minc -= margin[0]
            maxc += margin[1]

        # build the world-space bounding box mesh
        mesh = vx.mesh.construct_box_mesh(minc, maxc)
        return mesh.transform(self.geometry)

    def fit_to_bounds(self,
        bounds: vx.Mesh,
        margin: float | torch.Tensor = None) -> AcquisitionGeometry:
        """
        """
        points = self.inverse().transform(bounds.vertices.detach())
        minc = points.amin(0).floor().int()
        maxc = points.amax(0).ceil().int() + 1

        if margin is not None:
            margin = (margin / self.spacing).round().int().view(-1, 3).expand(2, 3)
            minc -= margin[0]
            maxc += margin[1]

        geometry = AcquisitionGeometry(maxc - minc, self.shift(minc, 'voxel'),
                                       slice_direction=self._explicit_slice_direction)
        return geometry

    def voxel_to_local(self) -> vx.AffineMatrix:
        """
        Transform that converts voxel coordiates to flipped local grid coordinates in
        the range [-1, 1]. These local coordinates are used for grid sampling in torch.

        Returns:
            AffineMatrix: Transformed local coordinates.
        """
        mat = torch.eye(4)
        mat[:3, :3] *= 2 / (torch.tensor(self.baseshape) - 1)
        mat[:3, -1] = -1
        mat[[0, 2]] = mat[[2, 0]]
        return vx.AffineMatrix(mat)


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


class Orientation:

    def __init__(self, item) -> None:

        self.axes = 'LRPAIS'

        if isinstance(item, str):
            try:
                indices = [self.axes.index(x) for x in item]
            except:
                raise ValueError(f'orientation string must contain one of {self.axes}')
            self.dims = torch.as_tensor([0, 0, 1, 1, 2, 2])[indices]
            self.flip = torch.as_tensor([-1, 1, -1, 1, -1, 1])[indices]

        elif isinstance(item, vx.AffineMatrix):
            tensor = item[:3, :3]
            self.dims = tensor.abs().argmax(1)
            self.flip = tensor[tensor.abs().argmax(0), [0, 1, 2]].sign().int()

        else:
            return ValueError(f'cannot initialize Orientation from {type(item)}')

    def name(self) -> str:
        """
        """
        indices = self.dims * 2 + (self.flip > 0)
        return ''.join([self.axes[i] for i in indices])


def cast_orientation(obj) -> Orientation:
    """
    Cast item to an Orientation

    Args:
        obj (any): Object to cast.
    
    Returns:
        Orientation
    """
    if isinstance(obj, Orientation):
        return obj
    return Orientation(obj)
