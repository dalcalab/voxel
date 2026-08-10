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
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32) -> None:
        """
        Args:
            data (Tensor, optional): A 3x3, 3x4, or 4x4 tensor. If None, the
                matrix is initialized with the identity.
            device (device, optional): Device of the constructed matrix.
            dtype (dtype, optional): Data type of the constructed matrix.
                Defaults to `float32`. Use `float64` to retain full-precision.
        """
        vx.caching.init_property_cache(self)

        if data is None:
            data = torch.eye(4, device=device)
        elif isinstance(data, AffineMatrix):
            data = data.tensor

        data = torch.as_tensor(data, device=device).to(dtype)

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

    def __matmul__(self, rhs: AffineMatrix | torch.Tensor) -> AffineMatrix | torch.Tensor:
        rhs = rhs.tensor if isinstance(rhs, AffineMatrix) else rhs
        result = self.tensor @ rhs.type(self.tensor.dtype)
        return AffineMatrix(result) if result.shape == (4, 4) else result

    def __rmatmul__(self, lhs: AffineMatrix | torch.Tensor) -> AffineMatrix | torch.Tensor:
        lhs = lhs.tensor if isinstance(lhs, AffineMatrix) else lhs
        result = lhs.type(self.tensor.dtype) @ self.tensor
        return AffineMatrix(result) if result.shape == (4, 4) else result

    def _from_tensor_with_new_properties(self: T, tensor: torch.Tensor) -> T:
        """
        Base class utility function that creates a new object instance, with a
        new matrix tensor, but the same metadata. This should be reimplemented
        for subclasses. This function should be called in scenarios only when the
        matrix has new properties (e.g. device or data type), not new values.
        """
        return self.__class__(tensor)

    def save(self, filename: os.PathLike, fmt: str | None = None) -> None:
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

    def to(self: T, device: torch.device) -> T:
        """
        Move the matrix tensor to a device.

        Args:
            device (device): The target device.

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
        return AffineMatrix(self.tensor.inverse(), dtype=self.tensor.dtype)

    def transform(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply the matrix transformation to a set of 3D coordinates.

        Args:
            coords (Tensor): A tensor of coordinates with shape (..., 3).

        Returns:
            Tensor: Transformed coordinates with the same shape as the input.
        """
        if coords.shape[-1] != 3:
            raise ValueError('Coordinates must have a last dimension of size 3.')

        # split into linear and translation parts and apply directly over the
        # trailing axis, broadcasting across any leading dimensions
        matrix = self.tensor.to(coords.device)
        linear = matrix[:3, :3].to(coords.dtype)
        translation = matrix[:3, 3].to(coords.dtype)
        return coords @ linear.T + translation


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
    degrees: bool = True,
    dtype: torch.dtype = torch.float32) -> AffineMatrix:
    """
    Compute a 3D rotation matrix from rotation angles.

    Angles follow the standard right-handed convention (a positive rotation
    about x carries +y toward +z) and are composed as `Rx @ Ry @ Rz`,
    i.e. intrinsic x-y-z order, consistent with
    `quaternion_to_rotation_matrix` and scipy's `Rotation.from_euler('XYZ')`.

    Args:
        rotation (Tensor): Rotation angles (x, y, z). If `degrees` is True, the
            angles are in degrees, otherwise they are in radians.
        degrees (bool, optional): Whether the angles are defined as degrees or,
            alternatively, as radians.
        dtype (dtype, optional): Data type of the generated matrix. Defaults to
            `float32`. Use `float64` to retain full-precision gradients.

    Returns:
        AffineMatrix: Rotation affine matrix.
    """
    rotation = torch.as_tensor(rotation)
    if not torch.is_floating_point(rotation):
        rotation = rotation.double()
    if degrees:
        rotation = torch.deg2rad(rotation)

    zero = torch.zeros((), dtype=rotation.dtype, device=rotation.device)
    one = torch.ones((), dtype=rotation.dtype, device=rotation.device)

    c, s = torch.cos(rotation[0]), torch.sin(rotation[0])
    rx = torch.stack([
        torch.stack([one, zero, zero]),
        torch.stack([zero, c, -s]),
        torch.stack([zero, s, c]),
    ])
    c, s = torch.cos(rotation[1]), torch.sin(rotation[1])
    ry = torch.stack([
        torch.stack([c, zero, s]),
        torch.stack([zero, one, zero]),
        torch.stack([-s, zero, c]),
    ])
    c, s = torch.cos(rotation[2]), torch.sin(rotation[2])
    rz = torch.stack([
        torch.stack([c, -s, zero]),
        torch.stack([s, c, zero]),
        torch.stack([zero, zero, one]),
    ])
    matrix = rx @ ry @ rz

    return AffineMatrix(matrix, dtype=dtype)


def quaternion_to_rotation_matrix(
    quaternion: torch.Tensor,
    dtype: torch.dtype = torch.float32) -> AffineMatrix:
    """
    Compute a 3D rotation matrix from a quaternion.

    The quaternion is expected in scalar-first `(w, x, y, z)` order and is
    normalized to unit length before the matrix is built, so any non-unit input
    still yields a valid rotation.

    Args:
        quaternion (Tensor): Quaternion of shape (4,) in `(w, x, y, z)` order.
            It need not be normalized.
        dtype (dtype, optional): Data type of the generated matrix. Defaults to
            `float32`. Use `float64` to retain full-precision gradients.

    Returns:
        AffineMatrix: Rotation affine matrix.
    """
    quaternion = torch.as_tensor(quaternion)
    if not torch.is_floating_point(quaternion):
        quaternion = quaternion.double()
    if quaternion.shape != (4,):
        raise ValueError('quaternion must be of shape (4,)')

    # normalize to a unit quaternion so the result is a valid rotation
    quaternion = quaternion / torch.linalg.norm(quaternion)
    w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]

    matrix = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])

    return AffineMatrix(matrix, dtype=dtype)


def compose_affine(
    translation: torch.Tensor | None = None,
    rotation: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    shear: torch.Tensor | None = None,
    degrees: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32) -> AffineMatrix:
    """
    Composes an affine matrix from a set of translation, rotation, scale,
    and shear transform components.

    Args:
        translation (Tensor, optional): Translation vector.
        rotation (Tensor, optional): Rotation, given either as euler angles of
            shape (3,) or as a scalar-first `(w, x, y, z)` quaternion.
        scale (Tensor, optional): Scaling factors.
        shear (Tensor, optional): Shearing factors.
        degrees (bool, optional): Whether the rotation angles are in degrees.
        device (device, optional): Device of the generated matrix.
        dtype (dtype, optional): Data type of the generated matrix. Defaults to
            `float32`. Use `float64` to retain full-precision gradients.

    Returns:
        AffineMatrix: Composed affine matrix.
    """
    # prefer an input tensor's device when one is not explicitly provided
    inputs = [t for t in (translation, rotation, scale, shear) if isinstance(t, torch.Tensor)]
    if device is None and inputs:
        device = inputs[0].device

    def as_param(value: torch.Tensor) -> torch.Tensor:
        # cast to float64 while keeping the value attached to the graph
        return torch.as_tensor(value, device=device).double()

    zero = torch.zeros((), dtype=torch.float64, device=device)
    one = torch.ones((), dtype=torch.float64, device=device)
    bottom = torch.tensor([[0, 0, 0, 1]], dtype=torch.float64, device=device)

    def homogeneous(linear: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        # assemble a 4x4 matrix from a 3x3 linear block and a translation vector
        top = torch.cat([linear, trans.unsqueeze(1)], dim=1)
        return torch.cat([top, bottom], dim=0)

    # build only the requested components, preserving the T @ R @ Z @ S order
    components = []

    # translation (with an identity linear block)
    if translation is not None:
        translation = as_param(translation)
        if len(translation) != 3:
            raise ValueError(f'translation must be of shape (3,)')
        eye3 = torch.eye(3, dtype=torch.float64, device=device)
        components.append(homogeneous(eye3, translation))

    # rotation matrix; a length-3 input is treated as euler angles, while a
    # length-4 input is treated as a scalar-first (w, x, y, z) quaternion
    if rotation is not None:
        rotation = as_param(rotation)
        if rotation.ndim != 1 or rotation.shape[0] not in (3, 4):
            raise ValueError(f'rotation must be of shape (3,) for angles or (4,) for a quaternion')
        if rotation.shape[0] == 4:
            R = quaternion_to_rotation_matrix(rotation, dtype=torch.float64)
        else:
            R = angles_to_rotation_matrix(rotation, degrees=degrees, dtype=torch.float64)
        components.append(R.tensor)

    # scaling
    if scale is not None:
        scale = as_param(scale)
        if scale.ndim == 0:
            scale = scale.repeat(3)
        if scale.shape[0] != 3:
            raise ValueError(f'scale must be of size (3,)')
        components.append(torch.diag(torch.cat([scale, one.view(1)])))

    # shear matrix
    if shear is not None:
        shear = as_param(shear)
        if shear.ndim == 0:
            shear = shear.view(1)
        if shear.shape[0] != 3:
            raise ValueError(f'shear must be of shape (3,)')
        shear_block = torch.stack([
            torch.stack([one, shear[0], shear[1]]),
            torch.stack([zero, one, shear[2]]),
            torch.stack([zero, zero, one]),
        ])
        components.append(homogeneous(shear_block, torch.zeros(3, dtype=torch.float64, device=device)))

    # compose the provided component matrices, defaulting to the identity
    matrix = torch.eye(4, dtype=torch.float64, device=device)
    for component in components:
        matrix = matrix @ component

    return AffineMatrix(matrix, dtype=dtype)


def random_affine(
    max_translation: float = 0,
    max_rotation: float = 0,
    max_scaling: float = 0,
    device: torch.device | None = None) -> AffineMatrix:
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
    weights: torch.Tensor | None = None,
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
    ones = torch.ones(source.shape[0], 1, device=source.device, dtype=source.dtype)
    source = torch.cat([source, ones], dim=1)

    # init regularization matrix
    R = regularization * torch.eye(4, device=source.device, dtype=source.dtype)

    # compute weighted least squared estimator
    M = (source.T @ W @ source + R).inverse() @ (source.T @ W @ target)

    return vx.AffineMatrix(M.T)
