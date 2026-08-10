"""
Dense deformation structures for warping volumes and integrating flow fields.
"""

from __future__ import annotations

import torch
import voxel as vx


class VectorField(vx.Volume):
    """
    A 3-channel volume of 3D vectors, e.g. displacements or flow velocities,
    whose values are expressed in either world or voxel units (`space`).
    """

    def __init__(self,
        tensor: torch.Tensor | vx.Volume,
        geometry: vx.AcquisitionGeometry | vx.AffineMatrix | None = None,
        space: vx.Space = None) -> None:
        """
        Args:
            tensor (Tensor or Volume): Vector data tensor of shape $(3, W, H, D)$,
                or a 3-channel volume, whose geometry is adopted when `geometry`
                is not explicitly provided.
            geometry (AcquisitionGeometry or AffineMatrix, optional): Affine geometry
                or matrix representing the voxel-to-world coordinate transform.
            space (Space): The coordinate space of the vector values, either
                'world' or 'voxel'. Required.
        """
        if isinstance(tensor, vx.Volume):
            geometry = tensor.geometry if geometry is None else geometry
            tensor = tensor.tensor
        if tensor.ndim != 4 or tensor.shape[0] != 3:
            raise ValueError(f'vector field must have shape (3, W, H, D), got {tuple(tensor.shape)}')
        if space is None:
            raise ValueError("vector field requires a vector value space, either 'world' or 'voxel'")
        super().__init__(tensor, geometry)
        self.space = vx.Space(space)

    def new(self,
        tensor: torch.Tensor,
        geometry: vx.AcquisitionGeometry | None = None,
        keep_labels: bool = True) -> VectorField | vx.Volume:
        """
        Construct a new instance with the provided tensor, preserving unchanged
        properties. The result remains a VectorField with the same vector space
        when the tensor still holds 3 channels and decays to a Volume otherwise.

        Args:
            tensor (Tensor): The new data tensor replacement.
            geometry (AcquisitionGeometry, optional): The new geometry. If None,
                the current geometry is propagated.
            keep_labels (bool, optional): Unused for vector fields. Present to
                match the Volume interface.

        Returns:
            VectorField or Volume: The new instance.
        """
        geometry = self.geometry if geometry is None else geometry
        if tensor.ndim == 4 and tensor.shape[0] == 3:
            return VectorField(tensor, geometry, space=self.space)
        return vx.Volume(tensor, geometry)

    def in_space(self, space: vx.Space) -> VectorField:
        """
        Convert the vector values to a target coordinate space, rotating and
        scaling them through the linear part of the geometry.

        Args:
            space (Space): Target space of the vector values, either 'world'
                or 'voxel'.

        Returns:
            VectorField: The converted field, or the unchanged instance when
                already in the target space.
        """
        space = vx.Space(space)
        if space == self.space:
            return self
        dtype = self.dtype if self.tensor.is_floating_point() else torch.float32
        linear = self.geometry.tensor[:3, :3].to(device=self.device, dtype=dtype)
        if space == 'voxel':
            linear = torch.inverse(linear)
        vectors = torch.einsum('ij,jwhd->iwhd', linear, self.tensor.to(dtype))
        return VectorField(vectors, self.geometry, space=space)

    def as_warp(self) -> Warp:
        """
        Convert to an absolute world-space coordinate warp, built as the
        identity grid displaced by the world-space vectors.

        Returns:
            Warp: The corresponding coordinate warp.
        """
        vectors = self.in_space('world').tensor.permute(1, 2, 3, 0)
        grid = self.geometry.transform(vx.volume.volume_grid(self.baseshape, device=self.device))
        return Warp(grid + vectors, self.geometry)

    def integrate(self, coordinates: torch.Tensor, dt: float, method: str = 'euler',
                  time: float = 1, space: vx.Space = None,
                  exact_gradient: bool = True) -> torch.Tensor:
        """
        Integrate points through the flow field with fixed-step numerical
        integration. Points beyond the field extent see zero velocity.

        Args:
            coordinates (Tensor): Points to integrate, of shape $(..., 3)$.
            dt (float): Integration step size.
            method (str, optional): Integration scheme, either forward 'euler'
                or midpoint 'rk2'.
            time (float, optional): Total integration time.
            space (Space, optional): Coordinate space of the input and returned
                points. If None, the space of the field vectors is assumed.
            exact_gradient (bool, optional): When disabled, backprop may use
                cheaper approximate gradients that do not exactly match the
                integrated result. For rk2, the midpoint velocity is excluded
                from the graph, roughly halving backward time and memory at the
                cost of a small dt-dependent gradient bias. The forward result
                is unaffected.

        Returns:
            Tensor: The integrated points, of shape $(..., 3)$.
        """
        if method not in ('euler', 'rk2'):
            raise ValueError(f"integration method must be 'euler' or 'rk2', got '{method}'")
        space = self.space if space is None else vx.Space(space)
        dtype = torch.promote_types(
            self.dtype if self.tensor.is_floating_point() else torch.float32,
            coordinates.dtype if coordinates.is_floating_point() else torch.float32)

        # convert the field vectors and input points to the local sampling
        # space once up front, so each step reduces to a single sample-and-add
        # without any per-step coordinate round trips
        vector_linear = self.geometry.local_coordinate_transform(self.space, dtype=dtype)[:3, :3]
        to_local = self.geometry.local_coordinate_transform(space, dtype=dtype)
        field = vx.Volume(torch.einsum('ij,jwhd->iwhd', vector_linear, self.tensor.to(dtype)), self.geometry)
        local = to_local.transform(coordinates.to(dtype))

        steps = max(1, round(time / dt))
        step = time / steps
        for _ in range(steps):
            if method == 'rk2':
                # the midpoint velocity is only an evaluation position, so it can
                # be excluded from the graph entirely for approximate gradients
                with torch.set_grad_enabled(exact_gradient and torch.is_grad_enabled()):
                    midpoint = field.sample(local, space='local')
                velocity = field.sample(local + 0.5 * step * midpoint, space='local')
            else:
                velocity = field.sample(local, space='local')
            local = local + step * velocity
        return to_local.inverse().transform(local)

    def exponentiate(self, steps: int) -> VectorField:
        """
        Exponentiate a stationary velocity field by scaling and squaring.

        Args:
            steps (int): Number of scaling and squaring iterations.

        Returns:
            VectorField: The displacement field of the flowed deformation,
                expressed in the vector space of this field.
        """
        dtype = self.dtype if self.tensor.is_floating_point() else torch.float32

        # run the scaling and squaring entirely in the local sampling space
        # (see integrate), converting the displacements back at the end
        linear = self.geometry.local_coordinate_transform(self.space, dtype=dtype)[:3, :3]
        grid = vx.volume.volume_grid(self.baseshape, localshape=self.baseshape,
                                     device=self.device).to(dtype)
        scaled = torch.einsum('ij,jwhd->iwhd', linear, self.tensor.to(dtype)) / (2 ** steps)
        vectors = scaled.permute(1, 2, 3, 0)
        for _ in range(steps):
            current = vx.Volume(vectors.permute(3, 0, 1, 2), self.geometry)
            vectors = vectors + current.sample(grid + vectors, space='local')
        vectors = vectors @ torch.inverse(linear).T
        return VectorField(vectors.permute(3, 0, 1, 2), self.geometry, space=self.space)


class Warp:
    """
    A dense warp of absolute world-space coordinates, held as a $(W, H, D, 3)$
    tensor over the grid defined by `geometry` on the fixed-side domain. Each
    grid point maps to the world coordinate where a moving volume should be
    sampled (a pull-back mapping).
    """

    def __init__(self, coordinates: torch.Tensor | vx.Volume, geometry: vx.AcquisitionGeometry = None):
        """
        Args:
            coordinates (Tensor or Volume): World coordinate tensor of shape
                $(W, H, D, 3)$, or a 3-channel volume, whose geometry is adopted.
            geometry (AcquisitionGeometry, optional): Geometry of the underlying
                fixed-side grid domain.
        """
        if isinstance(coordinates, vx.Volume):
            if coordinates.num_channels != 3:
                raise ValueError(f'warp field must have 3 channels, got {coordinates.num_channels}')
            geometry = coordinates.geometry
            coordinates = coordinates.tensor.permute(1, 2, 3, 0)
        elif coordinates.ndim != 4 or coordinates.shape[-1] != 3:
            raise ValueError(f'warp field must have shape (W, H, D, 3), got {tuple(coordinates.shape)}')
        self.coordinates = coordinates
        self.geometry = geometry

    @property
    def baseshape(self) -> torch.Size:
        """
        The spatial 3D $(W, H, D)$ shape of the warp grid.
        """
        return self.coordinates.shape[:-1]

    @property
    def device(self) -> torch.device:
        """
        Device of the warp coordinate tensor.
        """
        return self.coordinates.device

    def as_volume(self) -> vx.Volume:
        """
        Convert the warp to a volume of 3-channel coordinate features.

        Returns:
            Volume: The warp coordinates as a volume.
        """
        return vx.Volume(self.coordinates.permute(3, 0, 1, 2), self.geometry)

    def as_displacement_field(self) -> VectorField:
        """
        Convert to the field of world-space displacements between the warp
        coordinates and the identity grid.

        Returns:
            VectorField: The displacement field.
        """
        grid = self.geometry.transform(vx.volume.volume_grid(self.baseshape, device=self.device))
        vectors = (self.coordinates - grid).permute(3, 0, 1, 2)
        return VectorField(vectors, geometry=self.geometry, space='world')

    def map(self, volume: vx.Volume, mode: str = 'linear', padding_mode: str = 'zeros') -> vx.Volume:
        """
        Warp a volume by sampling it at the mapped coordinates, producing a
        pulled-back volume on the warp grid.

        Args:
            volume (Volume): The moving volume to warp.
            mode (str, optional): The sampling mode, either 'linear' or 'nearest'.
            padding_mode (str, optional): Padding mode for outside grid values.

        Returns:
            Volume: The warped volume.
        """
        sampled = volume.sample(self.coordinates, space='world', mode=mode, padding_mode=padding_mode)
        return vx.Volume(sampled.permute(3, 0, 1, 2), self.geometry)


def compose_transforms(*transforms: vx.AffineMatrix | Warp) -> vx.AffineMatrix | Warp:
    """
    Merge a sequence of transforms into a single equivalent transform, where
    the argument order is the order in which the transforms are applied to a
    volume. Affine matrices are assumed to be world-space transforms. Points
    that a warp maps beyond the extent of a preceding warp see zero
    displacement from it.

    Args:
        *transforms (AffineMatrix or Warp): Transforms to merge.

    Returns:
        AffineMatrix or Warp: The merged transform, an affine matrix when all
            inputs are affine and a warp otherwise.
    """
    if not transforms:
        raise ValueError('expected at least one transform to compose')
    affine = None
    warp = None
    for transform in transforms:
        if isinstance(transform, Warp):
            coordinates = transform.coordinates
            if warp is not None:
                # pull the running warp back through the next one by sampling
                # its displacement field at the new mapping coordinates
                displacement = warp.as_displacement_field()
                coordinates = coordinates + displacement.sample(coordinates, space='world')
            elif affine is not None:
                coordinates = affine.inverse().transform(coordinates)
            warp = Warp(coordinates, transform.geometry)
        elif isinstance(transform, vx.AffineMatrix):
            if warp is not None:
                # an affine after a warp only moves the output domain
                geometry = vx.AcquisitionGeometry(warp.baseshape, transform @ warp.geometry)
                warp = Warp(warp.coordinates, geometry)
            else:
                affine = transform if affine is None else transform @ affine
        else:
            raise TypeError(f'cannot compose transform of type {type(transform).__name__}')
    return affine if warp is None else warp
