"""
Methods related to a volumetric image grid with a world-space geometry.
"""

from __future__ import annotations

import os
import torch
import voxel as vx


class Volume:
    """
    A multi-channel volumetric (3D) image with a world-space representation.

    The volume grid has dimensions $(C, W, H, D)$ where $C$ is the number of
    feature channels and $W, H, D$ are the spatial width, height, and depth
    of the image (called the **baseshape**).
    """

    def __init__(self,
        tensor: torch.Tensor,
        geometry: vx.AcquisitionGeometry | vx.AffineMatrix | None = None) -> None:
        """
        Args:
            tensor (Tensor): Image data tensor of shape $(C, W, H, D)$ or $(W, H, D)$. 
            geometry (AcquisitionGeometry or AffineMatrix, optional): Affine geometry
                or matrix representing the voxel-to-world coordinate transform. If
                None, it defaults to a shifted identity in which the image volume
                is centered at the world origin.
        """
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 4:
            raise ValueError(f'expected 3D or 4D features, got a {tensor.ndim}D input')
        self._tensor = tensor
        self.geometry = geometry

    # -------------------------------------------------------------------------
    # property getters and setters and core methods
    # -------------------------------------------------------------------------

    @property
    def tensor(self) -> torch.Tensor:
        """
        The volume feature tensor, always of shape $(C, W, H, D)$.
        """
        return self._tensor

    @property
    def geometry(self) -> vx.AcquisitionGeometry:
        """
        The acquisition geometry representing the transformation from
        voxel-center coordinates to world-space (or scanner) coordinates.
        """
        return self._geometry

    @geometry.setter
    def geometry(self, geometry: vx.AcquisitionGeometry):
        if not isinstance(geometry, vx.AcquisitionGeometry):
            geometry = vx.AcquisitionGeometry(self.baseshape, geometry)
        elif geometry.baseshape != self.baseshape:
            raise ValueError(f'acquisition geometry shape {tuple(geometry.baseshape)} must '
                             f'match the image base shape {tuple(self.baseshape)}')
        self._geometry = geometry

    @property
    def shape(self) -> torch.Size:
        """
        The 4D $(C, W, H, D)$ shape of the volume, including channel dimension.
        """
        return self.tensor.shape

    @property
    def baseshape(self) -> torch.Size:
        """
        The spatial 3D $(W, H, D)$ shape of the volume, excluding channel dimension.
        """
        return self.tensor.shape[1:]

    @property
    def num_channels(self) -> int:
        """
        The number of feature channels (the first volume dimension size).
        """
        return self.tensor.shape[0]

    @property
    def device(self) -> torch.device:
        """
        Device of the volume tensor.
        """
        return self.tensor.device

    @property
    def dtype(self) -> torch.dtype:
        """
        Datatype of the volume tensor.
        """
        return self.tensor.dtype

    def new(self,
        tensor: torch.Tensor,
        geometry: vx.AcquisitionGeometry | None = None) -> Volume:
        """
        Construct a new volume instance with the provided features tensor, while
        preserving any unchanged properties of the original volume.

        Args:
            tensor (Tensor): The new image tensor replacement.
            geometry (AcquisitionGeometry, optional): The new geometry. If None,
                the current geometry will be propagated.
        """
        geometry = self.geometry if geometry is None else geometry
        return self.__class__(tensor, geometry)

    def save(self, filename: os.PathLike, fmt: str = None) -> None:
        """
        Save the volume to a file.

        Args:
            filename (PathLike): The path to the file to save.
            fmt (str, optional): The format of the file. If None, the format is
                determined by the file extension.
        """
        vx.save_volume(self, filename, fmt=fmt)

    # -------------------------------------------------------------------------
    #  numerical and tensor operations for volume data manipulation
    # -------------------------------------------------------------------------

    def apply(self, func: callable) -> Volume:
        """
        Apply a function to the volume tensor and return a new instance.

        Args:
            func (callable): The function to apply.

        Returns:
            Volume: A new volume instance.
        """
        return self.new(func(self.tensor))

    def detach(self) -> Volume:
        """
        Detach the volume tensor from the current computational graph.

        Returns:
            Volume: A new volume instance with the detached tensor.
        """
        return self.new(self.tensor.detach())

    def to(self, device: torch.Device) -> Volume:
        """
        Move the volume tensor to a device.

        Args:
            device (Device): The target device.

        Returns:
            Volume: A new volume instance with the tensor on the target device.
        """
        if device is None:
            return self
        return self.new(self.tensor.to(device))

    def cuda(self) -> Volume:
        """
        Move the volume tensor to the GPU.

        Returns:
            Volume: A new volume instance with the tensor on the GPU.
        """
        return self.new(self.tensor.cuda())

    def cpu(self) -> Volume:
        """
        Move the volume tensor to the CPU.

        Returns:
            Volume: A new volume instance with the tensor on the CPU.
        """
        return self.new(self.tensor.cpu())

    def type(self, dtype: torch.dtype) -> Volume:
        """
        Convert the volume tensor to a specified data type.

        Args:
            dtype (torch.dtype): The target data type.

        Returns:
            Volume: A new volume instance.
        """
        if self.tensor.dtype == dtype:
            return self
        return self.new(self.tensor.type(dtype))

    def float(self) -> Volume:
        """
        Convert the volume tensor to float data type.

        Returns:
            Volume: A new float volume instance.
        """
        return self.new(self.tensor.float())

    def half(self) -> Volume:
        """
        Convert the volume tensor to half-precision float data type.

        Returns:
            Volume: A new half-precision float volume instance.
        """
        return self.new(self.tensor.half())

    def int(self) -> Volume:
        """
        Convert the volume tensor to integer data type.

        Returns:
            Volume: A new integer volume instance.
        """
        return self.new(self.tensor.int())

    def bool(self) -> Volume:
        """
        Convert the volume tensor to boolean data type.

        Returns:
            Volume: A new boolean volume instance.
        """
        return self.new(self.tensor.bool())

    def max(self, dim: int | None = None) -> Volume | torch.Tensor:
        """
        Get the maximum value in the volume tensor.

                Args:
            dim (int, optional): The dimension or dimensions to
                reduce. If None, all dimensions are reduced. If
                the dimension is 0 (channel axis), a single-channel
                volume is returned.

        Returns:
            Tensor or Volume: The maximum value(s) or volume.
        """
        reduced = self.tensor.amax(dim=dim)
        return self.new(reduced) if dim == 0 else reduced

    def min(self, dim: int | None = None) -> Volume | torch.Tensor:
        """
        Get the minimum value in the volume features.

        Args:
            dim (int, optional): The dimension or dimensions to
                reduce. If None, all dimensions are reduced. If
                the dimension is 0 (channel axis), a single-channel
                volume is returned.

        Returns:
            Tensor or Volume: The mininum value(s) or volume.
        """
        reduced = self.tensor.amin(dim=dim)
        return self.new(reduced) if dim == 0 else reduced

    def sum(self, dim: int | None = None) -> Volume | torch.Tensor:
        """
        Compute the sum of all voxels.

        Args:
            dim (int, optional): The dimension or dimensions to
                reduce. If None, all dimensions are reduced. If
                the dimension is 0 (channel axis), a single-channel
                volume is returned.

        Returns:
            Tensor or Volume: The summed value(s) or volume.
        """
        reduced = self.tensor.sum(dim=dim)
        return self.new(reduced) if dim == 0 else reduced

    def mean(self, dim: int | None = None) -> Volume | torch.Tensor:
        """
        Compute the mean of all voxels.

        Args:
            dim (int, optional): The dimension or dimensions to
                reduce. If None, all dimensions are reduced. If
                the dimension is 0 (channel axis), a single-channel
                volume is returned.

        Returns:
            Tensor or Volume: The mean value(s) or volume.
        """
        reduced = self.tensor.mean(dim=dim)
        return self.new(reduced) if dim == 0 else reduced

    def floor(self) -> Volume:
        """
        Apply the floor operation to the volume features.

        Returns:
            Volume: A new floored volume instance.
        """
        return self.new(self.tensor.floor())

    def ceil(self) -> Volume:
        """
        Apply the ceil operation to the volume features.

        Returns:
            Volume: A new ceiled volume instance.
        """
        return self.new(self.tensor.ceil())

    def abs(self) -> Volume:
        """
        Compute absolute values of the volume features.

        Returns:
            Volume: A new volume instance.
        """
        return self.new(self.tensor.abs())

    def exp(self) -> Volume:
        """
        Compute exponential of the elements in the volume features.

        Returns:
            Volume: A new exponentiated volume instance.
        """
        return self.new(self.tensor.exp())

    def isnan(self) -> Volume:
        """
        Compute a mask of NaN values in the volume.

        Returns:
            Volume: A new volume mask instance.
        """
        return self.new(self.tensor.isnan())

    def clamp(self,
        min: float = None,
        max: float = None,
        inplace: bool = False) -> Volume:
        """
        Clamp the values in the volume tensor.

        Args:
            min (float, optional): Minimum value to clamp to.
            max (float, optional): Maximum value to clamp to.
            inplace (bool): Whether to perform the operation in-place.

        Returns:
            Volume: A new (if not in-place) clamped volume instance.
        """
        if inplace:
            return self.new(self.tensor.clamp_(min=min, max=max))
        else:
            return self.new(self.tensor.clamp(min=min, max=max))

    def maximum(self, other: Volume) -> Volume:
        """
        Computes the element-wise maximum between two volumes.

        Args:
            other (Volume): The input volume to compare against.

        Returns:
            Volume: A maximized volume instance.
        """
        return self.new(self.tensor.maximum(other.tensor))

    def minimum(self, other: Volume) -> Volume:
        """
        Computes the element-wise minimum between two volumes.

        Args:
            other (Volume): The input volume to compare against.

        Returns:
            Volume: A minimized volume instance.
        """
        return self.new(self.tensor.minimum(other.tensor))

    def zeros_like(self,
        channels: int | None = None,
        dtype: torch.dtype | None = None) -> Volume:
        """
        Create a volume of zeros with the same geometry and
        device as the current instance.

        Args:
            channels (int, optional): Number of channels in the new volume.
                If None, will default to the existing number.
            dtype (torch.dtype, optional): Target data type.

        Returns:
            Volume: A new volume instance filled with zeros.
        """
        if channels is None:
            channels = self.num_channels
        return self.geometry.zeros_like(channels, dtype=dtype)

    def ones_like(self,
        channels: int | None = None,
        dtype: torch.dtype | None = None) -> Volume:
        """
        Create a volume of ones with the same geometry and
        device as the current instance.

        Args:
            channels (int, optional): Number of channels in the new volume.
                If None, will default to the existing number.
            dtype (torch.dtype, optional): Target data type.

        Returns:
            Volume: A new volume instance filled with ones.
        """
        if channels is None:
            channels = self.num_channels
        return self.geometry.ones_like(channels, dtype=dtype)

    def full_like(self,
        fill: float,
        channels: int | None = None,
        dtype: torch.dtype | None = None) -> Volume:
        """
        Create a volume filled with a specific value and with the same
        geometry and device as the current instance.

        Args:
            fill (float): The fill value.
            channels (int, optional): Number of channels in the new volume.
                If None, will default to the existing number.
            dtype (torch.dtype, optional): Target data type.

        Returns:
            Volume: A new filled volume instance.
        """
        if channels is None:
            channels = self.num_channels
        return self.geometry.full_like(fill, channels, dtype=dtype)

    def rand_like(self,
        channels: int | None = None,
        dtype: torch.dtype | None = None) -> Volume:
        """
        Create a volume of random values with the same geometry and
        device as the current instance. Values are sampled from a uniform
        distribution on the interval [0, 1).

        Args:
            channels (int, optional): Number of channels in the new volume.
                If None, will default to the existing number.
            dtype (torch.dtype, optional): Target data type.

        Returns:
            Volume: A new random volume instance.
        """
        if channels is None:
            channels = self.num_channels
        return self.geometry.rand_like(channels, dtype=dtype)

    def randn_like(self,
        channels: int | None = None,
        dtype: torch.dtype | None = None) -> Volume:
        """
        Create a volume of random values with the same geometry and
        device as the current instance. Values are sampled from a normal
        distribution with mean 0 and variance 1

        Args:
            channels (int, optional): Number of channels in the new volume.
                If None, will default to the existing number.
            dtype (torch.dtype, optional): Target data type.

        Returns:
            Volume: A new random volume instance.
        """
        if channels is None:
            channels = self.num_channels
        return self.geometry.randn_like(channels, dtype=dtype)

    def isin(self, elements: torch.Tensor) -> Volume:
        """
        Tests if each element of `elements` is in the volume.

        Args:
            elements (Tensor or Scalar): Values against which to test each voxel.

        Returns:
            Volume: A boolean volume that is True when a voxel value is
                in `elements` and False otherwise.
        """
        if isinstance(elements, (list, tuple)):
            elements = torch.tensor(elements, device=self.device)
        return self.new(torch.isin(self.tensor, elements))

    def unique(self, **kwargs) -> torch.Tensor:
        """
        Compute the unique elements of volume.

        Args:
            **kwargs: Additional arguments passed to the underlying
                call to `torch.unique()`.

        Returns:
            Tensor: The output list of unique scalar elements.
        """
        return self.tensor.unique(**kwargs)

    def quantile(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute the q-th quantile of the voxel data. 

        Args:
            q (float): A scalar quantile in the range [0, 1].
        
        Returns:
            Tensor: The quantile scalar value.
        """
        if q < 0 or q > 1:
            raise ValueError(f'quantile must be between 0 and 1, got {q}')
        if q == 0:
            return self.tensor.min()
        if q == 1:
            return self.tensor.max()
        flattened = self.tensor.flatten()
        if q > 0.5:
            k = int(flattened.numel() * (1.0 - q)) + 1
            return flattened.topk(k, largest=True, sorted=False).values.min()
        else:
            k = int(flattened.numel() * q) + 1
            return flattened.topk(k, largest=False, sorted=False).values.max()

    # -------------------------------------------------------------------------
    # indexing / operator overloads for tensor-style voxel data manipulation
    # -------------------------------------------------------------------------

    # assignment

    def __getitem__(self, indexing) -> torch.Tensor | Volume:
        # a regular boolean tensor-based indexing should be treated the
        # same as it would for a normal tensor
        if isinstance(indexing, torch.Tensor):
            return self.tensor[indexing]
        # the same goes for boolean volume indexing (in which case we'll
        # just use the underlying tensor)
        elif isinstance(indexing, Volume):
            indexing = indexing.tensor
            # if we get a one-channel boolean mask for the indexing,
            # we should auto-broadcast it to match the target channels
            if indexing.shape[0] == 1 and self.num_channels > 1:
                indexing = indexing.expand(self.num_channels, -1, -1, -1)
            return self.tensor[indexing]
        # in all circumstances (ex: slicing tuple or bounding box), call
        # the crop function which actually returns a new volume
        return self.crop(indexing)

    def __setitem__(self, indexing, value) -> None:
        self.tensor[_cast_volume_as_tensor(indexing)] = _cast_volume_as_tensor(value)

    # comparison operators

    def __eq__(self, other) -> Volume:
        return self.new(self.tensor == _cast_volume_as_tensor(other))
    
    def __ne__(self, other) -> Volume:
        return self.new(self.tensor != _cast_volume_as_tensor(other))

    def __lt__(self, other) -> Volume:
        return self.new(self.tensor < _cast_volume_as_tensor(other))

    def __le__(self, other) -> Volume:
        return self.new(self.tensor <= _cast_volume_as_tensor(other))

    def __gt__(self, other) -> Volume:
        return self.new(self.tensor > _cast_volume_as_tensor(other))

    def __ge__(self, other) -> Volume:
        return self.new(self.tensor >= _cast_volume_as_tensor(other))

    # unary operators

    def __pos__(self) -> Volume:
        return self.new(+self.tensor)

    def __neg__(self) -> Volume:
        return self.new(-self.tensor)

    # binary operators

    def __and__(self, other) -> Volume:
        return self.new(self.tensor & _cast_volume_as_tensor(other))

    def __or__(self, other) -> Volume:
        return self.new(self.tensor | _cast_volume_as_tensor(other))

    def __xor__(self, other) -> Volume:
        return self.new(self.tensor ^ _cast_volume_as_tensor(other))

    def __add__(self, other) -> Volume:
        return self.new(self.tensor + _cast_volume_as_tensor(other))

    def __radd__(self, other) -> Volume:
        return self.new(_cast_volume_as_tensor(other) + self.tensor)

    def __sub__(self, other) -> Volume:
        return self.new(self.tensor - _cast_volume_as_tensor(other))

    def __rsub__(self, other) -> Volume:
        return self.new(_cast_volume_as_tensor(other) - self.tensor)

    def __mul__(self, other) -> Volume:
        return self.new(self.tensor * _cast_volume_as_tensor(other))

    def __rmul__(self, other) -> Volume:
        return self.new(_cast_volume_as_tensor(other) * self.tensor)

    def __truediv__(self, other) -> Volume:
        return self.new(self.tensor / _cast_volume_as_tensor(other))

    def __rtruediv__(self, other) -> Volume:
        return self.new(_cast_volume_as_tensor(other) / self.tensor)

    def __pow__(self, other) -> Volume:
        return self.new(self.tensor ** _cast_volume_as_tensor(other))

    # assignment operators

    def __iadd__(self, other) -> None:
        self._tensor += _cast_volume_as_tensor(other)
        return self

    def __isub__(self, other) -> None:
        self._tensor -= _cast_volume_as_tensor(other)
        return self

    def __imul__(self, other) -> None:
        self._tensor *= _cast_volume_as_tensor(other)
        return self

    def __itruediv__(self, other) -> None:
        self._tensor /= _cast_volume_as_tensor(other)
        return self

    # -------------------------------------------------------------------------
    # methods for manipulating spatial geometry and computing coordinates
    # -------------------------------------------------------------------------

    def sample(self,
        points: torch.Tensor | vx.Mesh,
        space: vx.Space,
        mode: str = 'linear') -> torch.Tensor:
        """
        Sample volume features at a set of points.

        Args:
            points (Tensor | Mesh): A set of points in world or voxel coordinates with
                shape $(N, 3)$. If the input is a mesh, the vertex positions are used.
            space (Space): The coordinate space of the input points or mesh.
            mode (str, optional): The sampling mode, either 'linear' or 'nearest'.

        Returns:
            Tensor: The sampled features, with shape $(N, C)$.
        """
        if isinstance(points, vx.Mesh):
            points = points.vertices
        if vx.Space(space) == 'world':
            points = self.geometry.inverse().transform(points)
        points = self.geometry.voxel_to_local().transform(points)
        grid = points.view(1, len(points), 1, 1, 3)
        sampled = torch.nn.functional.grid_sample(self.tensor.unsqueeze(0), grid,
            align_corners=True, mode=('bilinear' if mode == 'linear' else 'nearest'))
        return sampled.squeeze(dim=(0, 3, 4)).swapaxes(0, 1)

    def tesselate(self, threshold: float = 0.5, space: vx.Space = 'world') -> vx.Mesh:
        """
        Tesselate a mesh around connected voxel components.
        This is not differentiable.

        Args:
            threshold (float, optional): Scalar threshold that determines
                whether a voxel is inside or outside the mesh boundary.
            space (Space, optional): The coordinate space of mesh vertices. Default
                is the world coordinate space.

        Returns:
            Mesh: Tesselated mesh.
        """
        try:
            from pytorch3d.ops.marching_cubes import marching_cubes
        except ImportError as exc:
            raise ImportError('mesh tesselation requires that the '
                              'pytorch3d package is installed') from exc

        # 
        padded = self.detach().pad(self.geometry.spacing)
        vertices, faces = marching_cubes(padded.tensor.float(), threshold,
                                         return_local_coords=False)
        if len(vertices[0]) == 0:
            raise ValueError('empty volume - could not tesselate')
        mesh = vx.Mesh(vertices[0].flip(-1) - 1, faces[0])

        # 
        if vx.Space(space) == 'world':
            mesh = mesh.transform(self.geometry)
        return mesh

    def bounds(self,
        nonzero: bool = False,
        margin: float | torch.Tensor = None,
        space: vx.Space = 'world') -> vx.Mesh:
        """
        Compute a box mesh enclosing the bounds of the volume grid or the non-zero
        voxels in the image.

        Args:
            nonzero (bool): If True, compute the bounds around all non-zero voxels,
                otherwise use the extent of the image grid.
            margin (float or Tensor, optional): Margin to expand the cropping boundary.
                Can be a positive or negative delta.
            space (Space, optional): Space of the margin values, either 'voxel' or 'world'.

        Returns:
            Mesh: Bounding box mesh in world-space coordinates.
        """
        if nonzero:
            # compute the bounding box around all nonzero voxels
            tensor = self.tensor if self.num_channels > 1 else self.tensor.sum(0)
            nonzero = tensor.view(self.baseshape).nonzero()
            if nonzero.shape[0] == 0:
                raise ValueError('cannot compute nonzero bounds on an empty volume')
            min_point = nonzero.amin(dim=0).float()
            max_point = nonzero.amax(dim=0).float()
        else:
            # just use the bounds of the volume extent
            min_point = torch.zeros(3, device=self.device)
            max_point = torch.tensor(self.baseshape, device=self.device).float() - 1
        
        # expand (or shrink) margin around border
        if margin is not None:
            margin = self.geometry.conform_units(margin, space, 'world', 2)
            min_point -= margin[:, 0]
            max_point += margin[:, 1]

        # build the world-space bounding box mesh
        mesh = vx.mesh.construct_box_mesh(min_point, max_point)
        return mesh.transform(self.geometry)

    def centroids(self, space: vx.Space) -> torch.Tensor:
        """
        Compute the centroids (centers of mass) for each volume channel.
        All negative values are clamped to zero before computing the centroids.

        Args:
            space (Space): The coordinate space of computed centroids.

        Returns:
            Tensor: Per-channel coordinates of shape (C, 3).
        """
        clamped_tensor = self.tensor.clamp(min=0).float()

        # compute the centroids with a differentiable coordinate weighting
        coord = lambda a: (a * torch.arange(a.shape[-1], device=self.device)).sum(-1) / (a.sum(-1) + 1e-6)
        z_mean = clamped_tensor.mean(-1)
        x = coord(z_mean.mean(-1))
        y = coord(z_mean.mean(-2))
        z = coord(clamped_tensor.mean(-2).mean(-2))
        centroids = torch.stack([x, y, z], dim=-1)

        # transform to world-space if necessary
        if vx.Space(space) == 'world':
            centroids = self.geometry.transform(centroids)
        return centroids

    def crop(self, cropping: tuple | vx.Mesh, margin: float | torch.Tensor = None) -> Volume:
        """
        Crop the volume to some bounding, either defined by a voxel slicing
        tuple or a bounding box mesh.

        Args:
            cropping (tuple or Mesh): Cropping defined by either a tuple of slices
                or a bounding box mesh.
            margin (float or Tensor, optional): Margin (in world units) to expand
                the cropping boundary. Can be a positive or negative delta. The
                boundary will be clipped if it extends beyond the shape of the volume.

        Returns:
            Volume: The cropped volume instance.
        """

        # transform to voxel units
        if margin is not None:
            margin = (margin / self.geometry.spacing).round().int()

        if isinstance(cropping, vx.Mesh):
            # if we get a mesh as input, assume its a bounding box, but really
            # any set of mesh points could work here
            world2voxel = self.geometry.inverse()
            points = world2voxel.transform(cropping.vertices.detach())
            minc = points.amin(0).cpu().ceil().int()
            maxc = points.amax(0).cpu().floor().int()

            # extend the boundary
            if margin is not None:
                minc -= margin
                maxc += margin

            # make sure the coordinates are clamped within the volume extent
            minc = minc.clamp(min=0)
            maxc = maxc.clamp(max=torch.tensor(self.baseshape))
            stride = None

            # convert coordinate bounds to a 4D slicing tuple
            slicing = (slice(None), *vx.slicing.coordinates_to_slicing(minc, maxc))

        elif isinstance(cropping, (tuple, int, slice, type(...))):

            # conform single indexing items to a tuple format
            if not isinstance(cropping, tuple):
                cropping = (cropping,)

            # if we get a tuple assume its a tuple of slices
            slicing = vx.slicing.expand_slicing(cropping, 4)

            # do not allow cropping to remove a spatial dimension
            if any(isinstance(s, int) for s in slicing[1:]):
                raise ValueError('cannot remove a spatial dimension when cropping a volume')

            # extend the boundary
            minc, maxc, stride = vx.slicing.slicing_to_coordinates(cropping[1:], self.baseshape)
            if margin is not None:
                minc = (minc - margin).clamp(min=0)
                maxc = (maxc + margin).clamp(max=torch.tensor(self.baseshape))
                slicing = (slicing[0], *vx.slicing.coordinates_to_slicing(minc, maxc, stride))
        else:
            raise ValueError(f'unknown cropping item: {type(cropping)}')

        # update the geometry based on any inferred voxel shift or scale
        geometry = self.geometry
        if any(minc != 0):
            geometry = geometry.shift(minc, space='voxel')
        if stride is not None:
            geometry = geometry.scale(stride, space='voxel')

        # apply the cropping
        cropped_tensor = self.tensor[slicing]
        cropped_geometry = vx.AcquisitionGeometry(baseshape=cropped_tensor.shape[-3:],
                                                  matrix=geometry.tensor,
                                                  slice_direction=geometry._explicit_slice_direction)
        return self.new(cropped_tensor, cropped_geometry)

    def crop_to_nonzero(self, margin: float  | torch.Tensor = None) -> Volume:
        """
        Crop the volume to the bounding box around nonzero voxels.

        Args:
            margin (float or Tensor, optional): Margin (in world units) to expand
                the cropping boundary. Can be a positive or negative delta. The
                boundary will be clipped if it extends beyond the shape of the volume.

        Returns:
            Volume: The cropped volume instance.
        """
        nonzero = self.tensor.nonzero()
        # note: we're using nonzero() directly here instead of calling self.bounds() to avoid
        # the unnecessary transformation into world space then back again)
        minc = nonzero.amin(0)
        maxc = nonzero.amax(0)
        return self.crop(vx.slicing.coordinates_to_slicing(minc, maxc), margin=margin)

    def reorient(self, orientation: vx.Orientation) -> Volume:
        """
        Transform the volume to a new orientation. This is faster than
        achieving the same result through resampling.

        Args:
            orientation (Orientation): The target orientation.
        
        Returns:
            Volume: The reoriented volume instance.
        """
        source = self.geometry.orientation
        target = vx.cast_orientation(orientation)

        perm = source.dims.argsort()[target.dims]
        tensor = self.tensor
        if (perm != torch.tensor((0, 1, 2))).any():
            tensor = tensor.permute(0, *(perm + 1))

        flip = source.flip[perm] * target.flip
        indices = (flip < 0).argwhere()
        if len(indices) > 0:
            tensor = tensor.flip(*(indices + 1))

        return self.new(tensor, self.geometry.reorient(orientation))

    def resample_like(self,
        target: Volume | vx.AcquisitionGeometry,
        mode: str = 'linear',
        padding_mode: str = 'zeros',
        antialias: bool = False) -> Volume:
        """
        Resample the volume features to match the geometry of a target volume.

        Args:
            target (Volume | AcquisitionGeometry): Target acquisition geometry.
            mode (str, optional): Interpolation mode.
            padding_mode (str, optional): Padding mode for outside grid values.
            antialias (bool, optional): If True, will apply a Gaussian filter
                before resampling to avoid aliasing artifacts.

        Returns:
            Volume: Resampled volume instance.
        """
        if isinstance(target, Volume):
            target = target.geometry

        # check if the matrices are similar because we might be able to avoid any
        # actual resampling if that's the case. first, we check the rotation and scale
        if torch.allclose(self.geometry.tensor[:, :3], target.tensor[:, :3], atol=1e-4, rtol=0):

            # then check if the source and target have matching baseshapes and matrices, because
            # in that case we don't have to modify anything at all
            if target.baseshape == self.baseshape and \
               torch.allclose(self.geometry.tensor[:3, -1], target.tensor[:3, -1], atol=1e-4, rtol=0):
                return self.new(self.tensor, target)

            # otherwise, it's possible the difference between image spaces is only a voxel-shift,
            # in which case we can just crop and/or pad -- much faster than resampling.
            # we need to check if the voxel-space translations are all integers
            delta = (self.geometry.inverse() @ target).transform(torch.zeros(3))
            delta_rounded = delta.round()
            if torch.allclose(delta, delta_rounded, atol=1e-4, rtol=0):

                # these are the relative shifts in voxels at the lower (origin) and upper corners
                lower = delta_rounded.int().cpu()
                upper = lower + torch.tensor(target.baseshape) - torch.tensor(self.baseshape)

                # apply any necessary cropping to the tensor
                minc = lower.clamp(min=0)
                maxc = upper.clamp(max=0) + torch.tensor(self.baseshape)
                slicing = (slice(None), *[slice(a, b) for a, b in zip(minc, maxc)])
                resampled = self.tensor[slicing]

                # apply any necessary padding to the tensor
                a = lower.clamp(max=0).abs()
                b = upper.clamp(min=0)
                padding = torch.stack((b, a), dim=1).flatten()
                if (padding != 0).any():
                    mode = dict(zeros='constant', reflection='reflect', border='replicate').get(padding_mode)
                    if mode is None:
                        raise ValueError(f'no padding mode equivolent for {padding_mode}')
                    reverse = list(reversed([int(d) for d in padding]))
                    resampled = torch.nn.functional.pad(resampled, reverse, mode=mode)

                return self.new(resampled, target)
    
        # if we got here, it means have to resort to doing a grid interpolation, so first
        # build the coordinate grid for the target image
        transform = self.geometry.inverse() @ target

        if antialias:
            # if antialiasing is enabled, we'll need to do some extra work. the goal is to smooth
            # intensities along a downsampling direction. since there is no guarantee that the
            # target geometry dimensions are aligned with the source grid (e.g. consider a rotation
            # resampling), we can't just smooth the source image and sample it using the target grid.
            # instead, we need to compute an intermediate grid that is aligned with the target so that
            # we can apply smoothing kernels in each downsampling directions of the target geometry.
            # the intermediate grid is a multiple of the target grid (determined by the downsample factor)
            # and we can use strided convolutions to get the final resampled image efficiently.

            if mode != 'linear':
                raise ValueError('antialiasing only supported with linear interpolation')

            inter_space = vx.AcquisitionGeometry(target.baseshape, transform)
            down_factor = inter_space.spacing.clamp(1).floor().int()

            # blur with a sigma of 1/3 of the downsample factor
            sigma = (down_factor > 1) * (down_factor.float() / 3)

            # compute the padding required for the Gaussian kernel (hardcoded the truncate value)
            # along with the updated grid transform
            truncate = 2
            padding = (truncate * sigma + 0.5).int()
            intermediate_baseshape = [int(s * f) + p * 2 for s, f, p in zip(target.baseshape, down_factor, padding)]
            transform = inter_space.scale(1 / down_factor, space='voxel').shift(-padding, space='voxel')
        else:
            intermediate_baseshape = target.baseshape

        grid = volume_grid(intermediate_baseshape, transform=transform,
                           device=self.device, localshape=self.baseshape)

        resampled = torch.nn.functional.grid_sample(
                        input=self.tensor.float().unsqueeze(0),
                        grid=grid.unsqueeze(0),
                        mode=('bilinear' if mode == 'linear' else mode),
                        padding_mode=padding_mode,
                        align_corners=True).squeeze(0)

        if antialias:
            resampled = vx.filters.gaussian_blur(resampled, sigma, stride=tuple(down_factor),
                                                 padding='valid', truncate=truncate)

        # probably ideal to keep the data type consistent when using nearest neighbor sampling
        if mode == 'nearest':
            resampled = resampled.type(self.dtype)

        return self.new(resampled, target)

    def resample(self,
        spacing: float | torch.Tensor = None,
        in_plane_spacing: float | torch.Tensor = None,
        slice_spacing: float | torch.Tensor = None,
        mode: str = 'linear',
        padding_mode: str = 'zeros',
        antialias: bool = False) -> Volume:
        """
        Resample voxel features to a new voxel grid spacing.

        Args:
            spacing (float |Tensor): Target voxel spacing. An isotropic target
                is assumed if a scalar is provided.
            in_plane_spacing (float | Tensor): Target in-plane voxel spacing. Mutually
                exclusive with the `spacing` argument.
            slice_spacing (float | Tensor): Target slice spacing. Mutually exclusive
                except with the `spacing` argument.
            mode (str, optional): Interpolation mode.
            padding_mode (str, optional): Padding mode for outside grid values.
            antialias (bool, optional): If True, will apply a Gaussian filter
                before resampling to avoid aliasing artifacts.

        Returns:
            Volume: Volume resampled to the target voxel spacing.
        """
        if spacing is None and in_plane_spacing is None and slice_spacing is None:
            raise ValueError('must provide either spacing, in_plane_spacing, or slice_spacing')
        if spacing is not None:
            if in_plane_spacing is not None or slice_spacing is not None:
                raise ValueError('cannot provide spacing with in_plane_spacing or slice_spacing')
        else:
            spacing = self.geometry.spacing.clone()
            if in_plane_spacing is not None:
                spacing[self.geometry.in_plane_directions] = in_plane_spacing
            if slice_spacing is not None:
                spacing[self.geometry.slice_direction] = slice_spacing

        target = self.geometry.resample(spacing)
        return self.resample_like(target, mode=mode, padding_mode=padding_mode, antialias=antialias)

    def reshape(self, baseshape: torch.Size) -> Volume:
        """
        Modify the spatial extent of the volume, cropping or padding around the
        center image to fit a given **baseshape**.

        This method is symmetric in that performing a reverse reshape operation
        will always yeild the original geometry.

        args:
            baseshape (torch.Size): Target spatial (3D) shape.
        
        returns:
            Volume: Reshaped volume instance.
        """
        return self.resample_like(self.geometry.reshape(baseshape), mode='nearest')

    def pad(self, delta: float | torch.Tensor, space: vx.Space) -> Volume:
        """
        Pad the spatial extent of the volume by a given delta. Note that
        a negative delta value will result in trimming (cropping).

        args:
            delta (float or Tensor, optional): Delta of specified units to
                pad (or crop) the volume by in each direction. Can be of size
                $(1,)$, $(3,)$, or $(3, 2)$.
            space (Space): The coordinate space of the delta values, either
                'voxel' or 'world'.

        returns:
            Volume: Reshaped volume instance.
        """
        return self.resample_like(self.geometry.pad(delta, space), mode='nearest')

    def trim(self, delta: float | torch.Tensor, space: vx.Space) -> Volume:
        """
        Trim the spatial extent of the volume by a given delta. This is
        equivalent to padding with negative delta values.

        args:
            delta (float or Tensor, optional): Delta of specified units to
                trim the volume by in each direction. Can be of size $(1,)$,
                $(3,)$, or $(3, 2)$.
            space (Space): The coordinate space of the delta values, either
                'voxel' or 'world'.

        returns:
            Volume: Reshaped volume instance.
        """
        return self.pad(-delta, space)

    def transform(self,
        transform: vx.AffineVolumeTransform | vx.AffineMatrix,
        resample: bool = False,
        negate: bool = False,
        mode: str = 'linear') -> Volume:
        """
        Apply a spatial transform to the volume. By default, this method will not
        resample the image data and instead transform the world geometry.

        Args:
            transform (AffineVolumeTransform or AffineMatrix): Transform to apply. Assume
                a world-space transform if an AffineMatrix is provided.
            resample (bool, optional): If True, the volume will be transformed and
                resampled in voxel space, otherwise only the geometry will be updated.
            negate (bool, optional): If True, the inverse transform is applied to the
                geometry so that image features do not move in world space. This option
                can only be enabled when resampling is enabled.
            mode (str, optional): Interpolation mode if resampling.

        Returns:
            Volume: Transformed volume.
        """

        # if the transform is just a simple matrix, assume it's a world-space transform
        if not isinstance(transform, vx.AffineVolumeTransform):
            transform = vx.AffineVolumeTransform(transform, space='world', source=self, target=self)

        if not resample:
            # just apply the transform to the acquisition geometry
            if negate:
                raise ValueError('cannot negate transform when resampling is disabled')
            transform = transform.convert(space='world')
            return self.new(self.tensor, transform @ self.geometry)

        # if we're resampling, convert to a voxel-to-voxel transform
        target = transform.target
        inverted = transform.convert(space='voxel', source=self).inverse()

        # construct the transformed resampling grid
        grid = volume_grid(target.baseshape, transform=inverted,
                           device=self.device, localshape=self.baseshape)

        interpolated = torch.nn.functional.grid_sample(
                        self.tensor.unsqueeze(0).float(),
                        grid.unsqueeze(0),
                        mode=('bilinear' if mode == 'linear' else mode),
                        padding_mode='zeros',
                        align_corners=True).squeeze(0)

        if negate:
            # apply inverse transform to the geometry to cancel out world space changes
            target = inverted.convert(space='world') @ target

        return self.new(interpolated, target)

    def pool(self,
        scale: int = 2,
        mode: str = 'mean',
        spacing_ratio_thresh: float = None) -> Volume:
        """
        Pool the voxel data with a sliding window.

        By default, this will pool over all dimensions, but it can be conditionally
        disabled for the slice dimension based on the ratio of slice vs in-plane spacing, 
        i.e. the value of `geometry.spacing_ratio`. For example, if the slice spacing is
        `spacing_ratio_thresh` times greater than the in-plane spacing, the slice dimension
        will not be pooled. Mind that if the resulting pooled volume has a slice spacing
        less than the in-plane spacing, it will be resampled to an isotropic resolution.

        There is no analogous `unpool` method because there is complexity in determining
        the desired unpooling strategy. To return to the original geometry, instead use
        the `resample_like` method. If no reference geometry is available, just use
        the `reshape` method to upsample.

        Args:
            scale (int, optional): The size of the pooling window. Defaults to 2.
            mode (str, optional): Pooling mode - can be 'mean' or 'max'. Defaults to 'mean'.
            spacing_ratio_thresh (float, optional): Slice spacing ration that determines
                whether the slice dimension is pooled. This is disabled by default.

        Returns:
            Volume: Pooled volume.
        """

        # dowsample factors for each dimension
        factors = [1 if d == 1 else scale for d in self.baseshape]

        # check if we should pool the slice dimension based on the spacing ratio threshold
        spacing_ratio = self.geometry.spacing_ratio
        slice_dim_pooling = spacing_ratio_thresh is None or spacing_ratio < spacing_ratio_thresh

        if not slice_dim_pooling:
            factors[self.geometry.slice_direction] = 1

        # apply the appropriate pooling operation
        if mode == 'max':
            func = torch.nn.functional.max_pool3d
        elif mode == 'mean':
            func = torch.nn.functional.avg_pool3d
        else:
            raise ValueError(f'unknown pooling mode \'{mode}\'')

        pooled_tensor = func(self.tensor, factors, ceil_mode=True)

        # adjust the geometry based on the pooling factors
        shift = [0.5 * (f - 1) for f in factors]
        adjusted = self.geometry.shift(shift, space='voxel').scale(factors, space='voxel')
        pooled = self.new(pooled_tensor, vx.AcquisitionGeometry(pooled_tensor.shape[1:], adjusted))

        # if the slice dimension was not pooled and the resulting slice spacing
        # is less than the in-plane spacing, we need to resample the slice dimension
        if not slice_dim_pooling and spacing_ratio < scale:
            spacing = pooled.geometry.spacing.clone()
            spacing[self.geometry.slice_direction] = spacing[self.geometry.in_plane_directions].mean()
            pooled = pooled.resample(spacing, padding_mode='border')

        # sanity check
        if pooled.geometry.spacing_ratio < 0.99:
            raise ValueError('unexpected spacing ratio after pooling operation')

        return pooled

    # -------------------------------------------------------------------------
    # image filtering and statistical normalization
    # -------------------------------------------------------------------------

    def smooth(self, sigma: float | torch.Tensor, truncate: float = 2) -> Volume:
        """
        Apply Gaussian smoothing to the image features.

        Args:
            sigma (float | Tensor): Smoothing sigma in world space units.
            truncate (float, optional): The number of standard deviations to extend
            the kernel before truncating.

        Returns:
            Volume: Smoothed volume.
        """
        scaled = torch.as_tensor(sigma, device=self.device) / self.geometry.spacing.to(self.device)
        return self.new(vx.filters.gaussian_blur(self.tensor, scaled, truncate=truncate))

    def dilate(self, iterations: int = 1) -> Volume:
        """
        Dilate the volume.

        Args:
            iterations (int, optional): Number of dilation iterations.

        Returns:
            Volume: Dilated volume of the same data type.
        """
        return self.new(vx.filters.dilate(self.tensor, iterations))


def _cast_volume_as_tensor(other: object) -> object:
    """
    If provided a Volume, cast to a Tensor, otherwise return the input.
    """
    return other.tensor if isinstance(other, Volume) else other


def volume_grid(
    baseshape: torch.Size,
    transform: vx.AffineMatrix | None = None,
    localshape: torch.Size | None = None,
    device: torch.device | None = None) -> torch.Tensor:
    """
    Construct a grid of 3D voxel coordinates of the shape (W, H, D, 3).

    Args:
        baseshape (Size): Spatial (3D) shape of the volume grid.
        transform (AffineMatrix, optional): Grid coordinate transform.
        localshape (Size, optional): If True, the grid is normalized
            between [1, -1] using the provided spatial shape and the
            coordinate order is swapped (for torch sampling methods).
        device (torch.device | None, optional): Device on which to
            allocate the grid data.

    Returns:
        Tensor: Grid volume tensor.
    """
    ranges = [torch.arange(s, dtype=torch.float32, device=device) for s in baseshape]
    grid = torch.stack(torch.meshgrid(*ranges, indexing='ij'), dim=-1)
    if transform:
        grid = transform.transform(grid)
    if localshape is not None:
        div = torch.tensor(localshape).maximum(torch.tensor(2)).to(grid.device) - 1
        grid = (grid / div * 2 - 1).flip(-1)
    return grid


def stack(*vols):
    """
    Concatenates (stacks) multiple volumes channel-wise. Assumes
    volumes are in the same image space (with the same base shape).

    Args:
        *vols (sequence of Volumes): Volumes to merge.

    Returns:
        Volume: Single channel-stacked volume instance.
    """
    if len(vols) == 1 and not isinstance(vols[0], Volume):
        vols = vols[0]
    if len(vols) == 1:
        return vols[0]
    return vols[0].new(torch.cat([v.tensor for v in vols], dim=0))


def volumes_equal(
    a: Volume,
    b: Volume,
    vol_tol: float = 1e-6,
    geom_tol: float = 1e-6) -> bool:
    """
    Check if two volumes are equal within a given tolerance.

    Args:
        a, b (Volume): Volumes to compare.
        vol_tol (float, optional): Absolute tolerance for volume tensor comparison.
        geom_tol (float, optional): Absolute tolerance for geometry comparison.

    Returns:
        bool: True if the volumes are equal, False otherwise.
    """
    if not a.tensor.allclose(b.tensor, atol=vol_tol, rtol=0):
        return False
    if not a.geometry.tensor.allclose(b.geometry.tensor, atol=geom_tol, rtol=0):
        return False
    return True
