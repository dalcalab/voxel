"""
Affine transforms in three dimensions.
"""

from __future__ import annotations

import torch
import voxel as vx


class AffineMatrix:
    """
    Affine matrix (4x4) transform for a 3D coordinate system.
    """

    def __init__(self,
        data: torch.Tensor | None = None,
        device: torch.device | None = None) -> None:
        """
        Args:
            data (Tensor, optional): A 3x3, 3x4, or 4x4 tensor. Default: if
                None, the matrix is initialized with the identitiy.
            device (device, optional): Device of the constructed matrix.
        """
        vx.caching.init_property_cache(self)

        if data is None:
            data = torch.eye(4, device=device)
        elif isinstance(data, AffineMatrix):
            data = data.tensor

        data = torch.as_tensor(data, device=device).float()

        if data.shape == (3, 3):
            row = torch.zeros((3, 1), dtype=data.dtype, device=data.device)
            data = torch.cat((data, row), dim=1)

        if data.shape == (3, 4):
            row = torch.tensor([[0, 0, 0, 1]], dtype=data.dtype, device=data.device)
            data = torch.cat((data, row), dim=0)
        elif data.shape != (4, 4):
            raise ValueError('Input matrix must be 3x3, 3x4, or 4x4.')

        self._tensor = data

    @property
    def tensor(self) -> torch.Tensor:
        """
        Tensor data of shape (4, 4).
        """
        return self._tensor

    def __getitem__(self, indexing) -> torch.Tensor:
        return self.tensor[indexing]

    def __setitem__(self, indexing, item):
        self.tensor[indexing] = item

    def __repr__(self) -> str:
        name = self.__class__.__name__
        tensor_str = str(self.tensor.detach().cpu().numpy())
        tensor_str = tensor_str.replace('\n', f'\n{" " * (len(name) + 1)}')
        return f'{name}({tensor_str})'

    def __matmul__(self, other: AffineMatrix | torch.Tensor) -> AffineMatrix | torch.Tensor:
        isaffine = isinstance(other, AffineMatrix) or \
                    (torch.is_tensor(other) and other.shape == (4, 4))
        if isaffine:
            other = AffineMatrix(other).tensor
        result = self.tensor.to(other.device) @ other.type(self.tensor.dtype)
        return AffineMatrix(result) if isaffine else result

    def inverse(self) -> AffineMatrix:
        """
        Invert the matrix.

        Returns:
            AffineMatrix: Inverted affine matrix.
        """
        return AffineMatrix(self.tensor.inverse())

    def transform(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply the matrix transformation to a set of 3D coordinates.

        Args:
            coordinates (Tensor): A tensor of coordinates with shape (..., 3).

        Returns:
            Tensor: Transformed coordinates with the same shape as the input.
        """
        if coords.shape[-1] != 3:
            raise ValueError('Coordinates must have a last dimension of size 3.')

        # reshape to a 2D tensor for the transformation
        coords_reshaped = coords.view(-1, 3)

        # convert to homogeneous coordinates
        ones = torch.ones((coords_reshaped.shape[0], 1), dtype=coords.dtype, device=coords.device)
        coords_homogeneous = torch.cat((coords_reshaped, ones), dim=1)

        # apply the transformation, convert back to cartesian, and reshape
        transformed_coords = coords_homogeneous @ self.tensor.T.to(coords.device)
        return transformed_coords[:, :3].view(coords.shape)


class AffineVolumeTransform(AffineMatrix):
    """
    Affine transform matrix in world or voxel space that contains metadata
    about the source and target acquistion geometry.
    """

    def __init__(self,
        data: torch.Tensor | AffineMatrix,
        space: vx.Space,
        source: vx.AcquisitionGeometry | vx.Volume,
        target: vx.AcquisitionGeometry | vx.Volume) -> None:
        """
        Args:
            data (Tensor | AffineMatrix): A 3x3, 3x4, or 4x4 tensor matrix.
            space (Space): The coordinate space of the transform.
            source (AcquisitionGeometry | Volume): The source acquisition geometry.
            target (AcquisitionGeometry | Volume): The target acquisition geometry.
        """
        super().__init__(data)
        self._source = vx.acquisition.cast_acquisition_geometry(source)
        self._target = vx.acquisition.cast_acquisition_geometry(target)
        self._space = vx.Space(space)
    
    @property
    def space(self) -> vx.Space:
        """
        Coordinate space of the transform.
        """
        return self._space
    
    @property
    def source(self) -> vx.AcquisitionGeometry:
        """
        Source acquisition geometry.
        """
        return self._source
    
    @property
    def target(self) -> vx.AcquisitionGeometry:
        """
        Target acquisition geometry.
        """
        return self._target

    def inverse(self) -> AffineVolumeTransform:
        """
        Invert the transform.

        Returns:
            AffineVolumeTransform: Inverted affine transform.
        """
        inverse = self.tensor.inverse()
        return AffineVolumeTransform(inverse, self.space, self.target, self.source)

    def convert(self,
        space: vx.Space | None = None,
        source: vx.AcquisitionGeometry | vx.Volume | None = None,
        target: vx.AcquisitionGeometry | vx.Volume | None = None) -> AffineVolumeTransform:
        """
        Convert transform for a new set of coordinate space, source, or target.

        Parameters
        ----------
        space (Space, optional): Desired coordinate space.
        source (AcquisitionGeometry or Volume, optional): Desired source geometry.
        target (AcquisitionGeometry or Volume, optional): Desired target geometry.

        Returns:
            AffineVolumeTransform: Converted affine transform.
        """
        
        # check if the desired space is the same as the embedded space
        space = self.space if space is None else vx.Space(space)
        same_space = space == self.space

        # check if the desired source is the same as the embedded source
        if source is None:
            source = self.source
            same_source = True
        else:
            source = vx.acquisition.cast_acquisition_geometry(source)
            same_source = torch.allclose(source.tensor, self.source.tensor, atol=1e-4, rtol=0)

        # check if the desired target is the same as the embedded target
        if target is None:
            target = self.target
            same_target = True
        else:
            target = vx.acquisition.cast_acquisition_geometry(target)
            same_target = torch.allclose(target.tensor, self.target.tensor, atol=1e-4, rtol=0)

        # return self if no changes are needed
        if all((same_space, same_source, same_target)):
            return self

        if same_source and same_target:
            # just a simple conversion of transform coordinate space, without
            # changing source and target information
            a = self.target if space == 'world' else self.target.inverse()
            b = self.source if space == 'voxel' else self.source.inverse()
            affine = a @ self @ b
        else:
            # if source and target info is changing, we need to recompute the
            # transform by first converting it to world-space
            affine = self if self.space == 'world' else self.target @ self @ self.source.inverse()
            # then back into the desired coordinate space
            if space == 'voxel':
                affine = target.inverse() @ affine @ source

        return AffineVolumeTransform(affine, space, source, target)


def translation_matrix(translation: torch.Tensor) -> AffineMatrix:
    """
    Compute a 3D translation matrix from translation vector.

    Args:
        translation (Tensor): Translation vector.
    
    Returns:
        AffineMatrix: Translation affine matrix.
    """
    if translation.shape != (3,):
        raise ValueError('Translation vector must have a shape of (3,).')
    matrix = torch.eye(4, dtype=torch.float64, device=translation.device)
    matrix[:3, 3] = translation
    return AffineMatrix(matrix)


def angles_to_rotation_matrix(
    rotation: torch.Tensor,
    degrees: bool = True) -> AffineMatrix:
    """
    Compute a 3D rotation matrix from rotation angles.

    Args:
        rotation (Tensor): Rotation angles. If `degrees` is True, the
            angles are in degrees, otherwise they are in radians.
        degrees (bool, optional): Whether the angles are defined as degrees or,
            alternatively, as radians.

    Returns:
        AffineMatrix: Rotation affine matrix.
    """
    if degrees:
        rotation = torch.deg2rad(rotation)

    c, s = torch.cos(rotation[0]), torch.sin(rotation[0])
    rx = torch.tensor([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=torch.float64)
    c, s = torch.cos(rotation[1]), torch.sin(rotation[1])
    ry = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float64)
    c, s = torch.cos(rotation[2]), torch.sin(rotation[2])
    rz = torch.tensor([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=torch.float64)
    matrix = rx @ ry @ rz

    return AffineMatrix(matrix.to(rotation.device))
