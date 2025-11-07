"""
Affine transforms in three dimensions.
"""

from __future__ import annotations

from typing import TypeVar

import os
import torch
import voxel as vx


T = TypeVar('T', bound='AffineMatrix')


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

    def _from_tensor_with_new_properties(self: T, tensor: torch.Tensor) -> T:
        """
        Base class utility function that creates a new object instance, with a
        new matrix tensor, but the same metadata. This should be reimplemented
        for subclasses. This function should be called in scenarios only when the
        matrix has new properties (e.g. device or data type), not new values.
        """
        return self.__class__(tensor)

    def save(self, filename: os.PathLike, fmt: str = None) -> None:
        """
        Save the affine matrix to a file.

        Args:
            filename (PathLike): The path to the file to save.
            fmt (str, optional): The format of the file. If None, the format is
                determined by the file extension.
        """
        vx.save_affine(self, filename, fmt=fmt)

    def detach(self: T) -> T:
        """
        Detach the matrix tensor from the current computational graph.

        Returns:
            A new affine with the detached matrix tensor.
        """
        return self._from_tensor_with_new_properties(self.tensor.detach())

    def to(self: T, device: torch.Device) -> T:
        """
        Move the matrix tensor to a device.

        Args:
            device (Device): The target device.

        Returns:
            A new affine with the matrix tensor on the target device.
        """
        return self._from_tensor_with_new_properties(self.tensor.to(device))

    def cuda(self: T) -> T:
        """
        Move the matrix tensor to the GPU.

        Returns:
            A new affine with the matrix tensor on the GPU.
        """
        return self._from_tensor_with_new_properties(self.tensor.cuda())

    def cpu(self: T) -> T:
        """
        Move the matrix tensor to the CPU.

        Returns:
            A new affine with the matrix tensor on the CPU.
        """
        return self._from_tensor_with_new_properties(self.tensor.cpu())

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
        coords_reshaped = coords.reshape(-1, 3)

        # convert to homogeneous coordinates
        ones = torch.ones((coords_reshaped.shape[0], 1), dtype=coords.dtype, device=coords.device)
        coords_homogeneous = torch.cat((coords_reshaped, ones), dim=1)

        # apply the transformation, convert back to cartesian, and reshape
        transformed_coords = coords_homogeneous @ self.tensor.T.to(coords.device)
        return transformed_coords[:, :3].reshape(coords.shape)


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


def compose_affine(
    translation : torch.Tensor = None,
    rotation : torch.Tensor = None,
    scale : torch.Tensor = None,
    shear : torch.Tensor = None,
    degrees : bool = True,
    device : torch.device = None) -> AffineMatrix:
    """
    Composes an affine matrix from a set of translation, rotation, scale,
    and shear transform components.

    Args:
        translation (Tensor, optional): Translation vector.
        rotation (Tensor, optional): Rotation angles.
        scale (Tensor, optional): Scaling factors.
        shear (Tensor, optional): Shearing factors.
        degrees (bool, optional): Whether the rotation angles are in degrees.
        device (device, optional): Device of the generated matrix.

    Returns:
        AffineMatrix: Composed affine matrix.
    """
    # check translation
    translation = torch.zeros(3) if translation is None else torch.as_tensor(translation)
    if len(translation) != 3:
        raise ValueError(f'translation must be of shape (3,)')

    # check rotation angles
    rotation = torch.zeros(3) if rotation is None else torch.as_tensor(rotation)
    if rotation.ndim == 0 or rotation.ndim != 0 and rotation.shape[0] != 3:
        raise ValueError(f'rotation must be of shape (3,)')

    # check scaling factor
    scale = torch.ones(3) if scale is None else torch.as_tensor(scale)
    if scale.ndim == 0:
        scale = scale.repeat(3)
    if scale.shape[0] != 3:
        raise ValueError(f'scale must be of size (3,)')

    # check shearing
    shear = torch.zeros(3) if shear is None else torch.as_tensor(shear)
    if shear.ndim == 0:
        shear = shear.view(1)
    if shear.shape[0] != 3:
        raise ValueError(f'shear must be of shape (3,)')

    # start from translation
    T = torch.eye(4, dtype=torch.float64)
    T[:3, -1] = translation

    # rotation matrix
    R = torch.eye(4, dtype=torch.float64)
    R[:3, :3] = angles_to_rotation_matrix(rotation, degrees=degrees)[:3, :3]

    # scaling
    Z = torch.diag(torch.cat([scale, torch.ones(1, dtype=torch.float64)]))

    # shear matrix
    S = torch.eye(4, dtype=torch.float64)
    S[0][1] = shear[0]
    S[0][2] = shear[1]
    S[1][2] = shear[2]

    # compose component matrices
    matrix = T @ R @ Z @ S

    return AffineMatrix(torch.as_tensor(matrix, dtype=torch.float32, device=device))


def random_affine(
    max_translation: float = 0,
    max_rotation: float = 0,
    max_scaling: float = 0,
    device: torch.device = None) -> AffineMatrix:
    """
    Generate a random affine transformation matrix.

    Args:
        max_translation (float, optional): Maximum translation in each direction.
        max_rotation (float, optional): Maximum rotation in each direction.
        max_scaling (float, optional): Maximum fractional scaling in each direction.
        device (device, optional): Device of the generated affine matrix.

    Returns:
        AffineMatrix: Random affine matrix.
    """
    translation = None
    if max_translation < 0:
        raise ValueError('max_translation must be a positive value')
    if max_translation > 0:
        translation_range = sorted([-max_translation, max_translation])
        translation = torch.distributions.uniform.Uniform(*translation_range).sample((3,))

    rotation = None
    if max_rotation < 0:
        raise ValueError('max_rotation must be a positive value')
    if max_rotation > 0:
        rotation_range = sorted([-max_rotation, max_rotation])
        rotation = torch.distributions.uniform.Uniform(*rotation_range).sample((3,))

    scale = None
    if max_scaling < 0:
        raise ValueError('max_scaling must be a positive value')
    if max_scaling > 0:
        scale = (1 + torch.rand(3) * max_scaling) ** torch.randn(3).sign()

    aff = compose_affine(
        translation=translation,
        rotation=rotation,
        scale=scale,
        device=device)
    return aff


def least_squares_alignment(
    source: torch.Tensor | vx.Mesh,
    target: torch.Tensor | vx.Mesh,
    weights: torch.Tensor = None,
    regularization: float = 1e-6) -> AffineMatrix:
    """
    Compute an affine least squares alignment between two 3D point sets.

    Args:
        source (Tensor or Mesh): Source point set.
        target (Tensor or Mesh): Target point set.
        weights (Tensor, optional): Weights for each point in the source set.
        regularization (float, optional): Regularization scale. Default is 1e-6.
    
    Returns:
        AffineMatrix: Affine alignment matrix.
    """
    if isinstance(source, vx.Mesh):
        source = source.vertices
    if isinstance(target, vx.Mesh):
        target = target.vertices

    # check input shapes
    assert source.shape == target.shape, 'source and target points must have the same shape'
    assert source.shape[1] == 3, 'source and target must be 3D point sets'

    # configure the weight matrix
    if weights is not None:
        assert weights.shape[0] == source.shape[0], 'weights must match the number of points'
        # TODO: ensure that weights are positive
        W = torch.diag(weights)
    else:
        W = torch.eye(len(source), device=source.device, dtype=source.dtype)

    #  extend source to shape (N, 4)
    source = torch.cat([source, torch.ones(source.shape[0], 1, device=source.device)], dim=1)

    # init regularization matrix
    R = regularization * torch.eye(4, device=source.device, dtype=source.dtype)

    # compute weighted least squared estimator
    M = (source.T @ W @ source + R).inverse() @ (source.T @ W @ target)

    return vx.AffineMatrix(M.T)
