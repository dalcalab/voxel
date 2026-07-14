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
        geometry: vx.AcquisitionGeometry | vx.AffineMatrix | None = None,
        labels: vx.LabelLookup | None = None) -> None:
        """
        Args:
            tensor (Tensor): Image data tensor of shape $(C, W, H, D)$ or $(W, H, D)$.
            geometry (AcquisitionGeometry or AffineMatrix, optional): Affine geometry
                or matrix representing the voxel-to-world coordinate transform. If
                None, it defaults to a shifted identity in which the image volume
                is centered at the world origin.
            labels (LabelLookup, optional): A lookup table annotating the integer
                values of a label-mapped volume with names and colors.
        """
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 4:
            raise ValueError(f'expected 3D or 4D features, got a {tensor.ndim}D input')
        self._tensor = tensor
        self.geometry = geometry
        self.labels = labels

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
            geometry = vx.AcquisitionGeometry(self.baseshape, matrix=geometry, device=self.device)
        elif geometry.baseshape != self.baseshape:
            raise ValueError(f'acquisition geometry shape {tuple(geometry.baseshape)} must '
                             f'match the image base shape {tuple(self.baseshape)}')
        self._geometry = geometry

    @property
    def labels(self) -> vx.LabelLookup | None:
        """
        The label lookup table annotating the integer values of this volume,
        or None if the volume has no associated labels.
        """
        return self._labels

    @labels.setter
    def labels(self, labels: vx.LabelLookup | None):
        if labels is not None and not isinstance(labels, vx.LabelLookup):
            raise TypeError(f'volume labels must be a LabelLookup or None, got {type(labels).__name__}')
        self._labels = labels

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
        geometry: vx.AcquisitionGeometry | None = None,
        keep_labels: bool = True) -> Volume:
        """
        Construct a new volume instance with the provided features tensor, while
        preserving any unchanged properties of the original volume.

        Args:
            tensor (Tensor): The new image tensor replacement.
            geometry (AcquisitionGeometry, optional): The new geometry. If None,
                the current geometry will be propagated.
            keep_labels (bool, optional): Whether to propagate the current label
                lookup table to the new volume. Should be False for operations
                that no longer produce an integer label map.
        """
        geometry = self.geometry if geometry is None else geometry
        labels = self.labels if keep_labels else None
        return self.__class__(tensor, geometry, labels)

    def copy(self) -> Volume:
        """
        Copy the volume instance. Only the data tensor is copied,
        not the underlying geometry.
        """
        return self.new(self.tensor.clone())

    def save(self, filename: os.PathLike, fmt: str | None = None) -> None:
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
        return self.new(self.tensor.detach(), self.geometry.detach())

    def to(self, device: torch.device) -> Volume:
        """
        Move the volume tensor to a device.

        Args:
            device (device): The target device.

        Returns:
            Volume: A new volume instance with the tensor on the target device.
        """
        if device is None:
            return self
        return self.new(self.tensor.to(device), self.geometry.to(device))

    def cuda(self) -> Volume:
        """
        Move the volume tensor to the GPU.

        Returns:
            Volume: A new volume instance with the tensor on the GPU.
        """
        return self.new(self.tensor.cuda(), self.geometry.cuda())

    def cpu(self) -> Volume:
        """
        Move the volume tensor to the CPU.

        Returns:
            Volume: A new volume instance with the tensor on the CPU.
        """
        return self.new(self.tensor.cpu(), self.geometry.cpu())

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

    def log(self) -> Volume:
        """
        Compute the natural logarithm of the volume features.

        Returns:
            Volume: A new log-transformed volume instance.
        """
        return self.new(self.tensor.log())

    def sqrt(self) -> Volume:
        """
        Compute the square root of the volume features.

        Returns:
            Volume: A new square-rooted volume instance.
        """
        return self.new(self.tensor.sqrt())

    def square(self) -> Volume:
        """
        Compute the square of the volume features.

        Returns:
            Volume: A new squared volume instance.
        """
        return self.new(self.tensor.square())

    def pow(self, exponent: float) -> Volume:
        """
        Compute the power of the volume features.

        Args:
            exponent (float): The exponent value.

        Returns:
            Volume: A new powered volume instance.
        """
        return self.new(self.tensor.pow(exponent))

    def isnan(self) -> Volume:
        """
        Compute a mask of NaN values in the volume.

        Returns:
            Volume: A new volume mask instance.
        """
        return self.new(self.tensor.isnan())

    def clamp(self,
        min: float | None = None,
        max: float | None = None,
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

    def all(self, dim: int | None = None) -> Volume | torch.Tensor:
        """
        Check if all elements in the volume are True.

        Args:
            dim (int, optional): The dimension or dimensions to
                reduce. If None, all dimensions are reduced. If
                the dimension is 0 (channel axis), a single-channel
                volume is returned.

        Returns:
            Tensor or Volume: The all-True value(s) or volume.
        """
        kwargs = {} if dim is None else {'dim': dim}
        reduced = self.tensor.all(**kwargs)
        return self.new(reduced) if dim == 0 else reduced

    def any(self, dim: int | None = None) -> Volume | torch.Tensor:
        """
        Check if any elements in the volume are True.

        Args:
            dim (int, optional): The dimension or dimensions to
                reduce. If None, all dimensions are reduced. If
                the dimension is 0 (channel axis), a single-channel
                volume is returned.

        Returns:
            Tensor or Volume: The any-True value(s) or volume.
        """
        kwargs = {} if dim is None else {'dim': dim}
        reduced = self.tensor.any(**kwargs)
        return self.new(reduced) if dim == 0 else reduced

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
        channels = channels or self.num_channels
        dtype = dtype or self.dtype
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
        channels = channels or self.num_channels
        dtype = dtype or self.dtype
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
        channels = channels or self.num_channels
        dtype = dtype or self.dtype
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
        channels = channels or self.num_channels
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
        channels = channels or self.num_channels
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
            **kwargs (Any): Additional arguments passed to the underlying
                call to `torch.unique()`.

        Returns:
            Tensor: The output list of unique scalar elements.
        """
        return self.tensor.unique(**kwargs)

    def quantile(self, q: float) -> torch.Tensor:
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

    def softmax(self, dim: int = 0) -> Volume | torch.Tensor:
        """
        Apply a softmax along a dimension of the volume tensor.

        Args:
            dim (int, optional): The dimension along which to apply the softmax.
                Defaults to the channel axis (0), in which case a volume is returned.

        Returns:
            Tensor or Volume: Softmaxed probabilities.
        """
        reduced = self.tensor.softmax(dim=dim)
        return self.new(reduced, keep_labels=False) if dim == 0 else reduced

    def argmax(self, dim: int | None = None) -> Volume | torch.Tensor:
        """
        Get the maximum index in the volume tensor.

        Args:
            dim (int, optional): The dimension or dimensions to
                reduce. If None, all dimensions are reduced. If
                the dimension is 0 (channel axis), a single-channel
                volume is returned.

        Returns:
            Tensor or Volume: The maximum indices or volume.
        """
        reduced = self.tensor.argmax(dim=dim)
        return self.new(reduced) if dim == 0 else reduced

    def recode(self,
        mapping: vx.LabelLookup | torch.Tensor | list,
        reverse: bool = False,
        background: bool = False) -> vx.Volume:
        """
        Remap the integer label values of the volume via a lookup.

        In the forward direction, the volume is treated as an index map and each
        voxel value `i` is replaced with `mapping[i]`. In the reverse direction,
        each voxel value is replaced with its position in `mapping` (the inverse
        operation).

        A `LabelLookup` is treated as its ordered integer `indices`. When the
        forward direction is used with a `LabelLookup`, the recoded voxel values
        are the real label values it describes, so the lookup is attached to the
        returned volume as its `labels`.

        Args:
            mapping (LabelLookup or Tensor or list): The ordered values to map
                index positions to.
            reverse (bool, optional): Map values back to their index positions.
            background (bool, optional): Prepend the background label 0 to
                `mapping` (unless already present) so it occupies the first index.

        Returns:
            Volume: A new label-map volume with remapped values.
        """
        assert self.num_channels == 1, f'cannot recode volume with {self.num_channels} channels'
        assert not torch.is_floating_point(self.tensor), f'recode requires volume of type int, got {self.dtype}'
        recoded = vx.labels.recode(self.tensor, mapping, reverse=reverse, background=background)
        volume = self.new(recoded, keep_labels=False)
        if not reverse and isinstance(mapping, vx.LabelLookup):
            volume.labels = mapping
        return volume

    def onehot(self,
        labels: int | torch.Tensor | vx.LabelLookup = -1,
        background: bool = False) -> vx.Volume:
        """
        One hot encode a label volume, with one channel per class.

        Args:
            labels (int or Tensor or LabelLookup, optional): The classes to encode.
                If an integer, it is the total number of classes (with -1 inferring
                one greater than the largest voxel value). If a tensor or lookup of
                label values, the volume is first recoded so those values map to the
                one-hot channels, in order.
            background (bool, optional): When `labels` is a tensor or lookup, reserve
                the first channel for the background label 0 (unless already present).

        Returns:
            Volume: The one-hot encoded volume.
        """
        assert self.num_channels == 1, f'cannot one hot volume with {self.num_channels} channels'
        assert not torch.is_floating_point(self.tensor), f'one hot requires volume of type int, got {self.dtype}'
        tensor = vx.labels.onehot(self.tensor.squeeze(0), labels=labels, background=background)
        return self.new(tensor, keep_labels=False)

    def collapse(self,
        labels: torch.Tensor | vx.LabelLookup = None,
        background: bool = False) -> vx.Volume:
        """
        Collapse a multi-channel (one-hot or probabilistic) volume into a single
        channel label map. This is the inverse of `onehot`.

        The channel axis is reduced with an argmax, and the resulting per-voxel
        channel index is optionally recoded into label values.

        Args:
            labels (Tensor or LabelLookup, optional): The label values that the
                channels correspond to. If None, the channel indices are returned
                directly. If a tensor or lookup, the channel index is recoded into
                the corresponding label value.
            background (bool, optional): When `labels` is provided, treat the first
                channel as the background label 0 (unless already present).

        Returns:
            Volume: A single-channel label map volume.
        """
        reduced = vx.labels.collapse(self.tensor, labels=labels, background=background)
        volume = self.new(reduced, keep_labels=False)
        if isinstance(labels, vx.LabelLookup):
            volume.labels = labels
        return volume

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
            return self.tensor[self._conform_volume_mask(indexing)]
        elif isinstance(indexing, list):
            # a list of indices should be treated as a list of channel reshuffling indices
            if not all(isinstance(i, int) for i in indexing):
                raise ValueError('channel list indexing must be a list of integers')
            return self.new(self.tensor[indexing])
        # in all circumstances (ex: slicing tuple or bounding box), call
        # the crop function which actually returns a new volume
        return self.crop(indexing)

    def _conform_volume_mask(self, indexing: Volume) -> torch.Tensor:
        # if we get a one-channel boolean mask for the indexing,
        # we should auto-broadcast it to match the target channels
        indexing = indexing.tensor
        if indexing.shape[0] == 1 and self.num_channels > 1:
            indexing = indexing.expand(self.num_channels, -1, -1, -1)
        return indexing

    def __setitem__(self, indexing, value) -> None:
        if isinstance(indexing, Volume):
            indexing = self._conform_volume_mask(indexing)
        self.tensor[_cast_volume_as_tensor(indexing)] = _cast_volume_as_tensor(value)

    def __contains__(self, item) -> bool:
        return item in self.tensor

    # comparison operators

    def __eq__(self, other) -> Volume:
        return self.new(self.tensor == _cast_volume_as_tensor(other), keep_labels=False)

    def __ne__(self, other) -> Volume:
        return self.new(self.tensor != _cast_volume_as_tensor(other), keep_labels=False)

    def __lt__(self, other) -> Volume:
        return self.new(self.tensor < _cast_volume_as_tensor(other), keep_labels=False)

    def __le__(self, other) -> Volume:
        return self.new(self.tensor <= _cast_volume_as_tensor(other), keep_labels=False)

    def __gt__(self, other) -> Volume:
        return self.new(self.tensor > _cast_volume_as_tensor(other), keep_labels=False)

    def __ge__(self, other) -> Volume:
        return self.new(self.tensor >= _cast_volume_as_tensor(other), keep_labels=False)

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
        mode: str = 'linear',
        padding_mode: str = 'zeros') -> torch.Tensor:
        """
        Sample volume features at a set of points.

        Args:
            points (Tensor | Mesh): A set of points in world or voxel coordinates with
                shape $(N, 3)$. If the input is a mesh, the vertex positions are used.
            space (Space): The coordinate space of the input points or mesh.
            mode (str, optional): The sampling mode, either 'linear' or 'nearest'.
            padding_mode (str, optional): Padding mode for outside grid values.

        Returns:
            Tensor: The sampled features, with shape $(N, C)$.
        """
        if isinstance(points, vx.Mesh):
            points = points.vertices
        
        # original base shape
        inshape = points.shape[:-1]
        points = points.view(-1, 3)

        # convert to local coordinate space
        if vx.Space(space) == 'world':
            points = self.geometry.inverse().transform(points)
        points = self.geometry.voxel_to_local_coordinates(points)

        # sample the channels
        sampled = torch.nn.functional.grid_sample(
            self.tensor.float().unsqueeze(0),
            points.view(1, len(points), 1, 1, 3),
            align_corners=False,
            mode=('bilinear' if mode == 'linear' else 'nearest'),
            padding_mode=padding_mode)
        
        # if nearest neighbor sampling, convert back to original dtype
        if mode == 'nearest':
            sampled = sampled.type(self.dtype)

        # remove batch and spatial dimensions
        sampled = sampled.squeeze(dim=(0, 3, 4)).swapaxes(0, 1)
        return sampled.view(*inshape, sampled.size(-1))

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
        padded = self.detach().pad(1, 'voxel')
        vertices, faces = marching_cubes(padded.tensor.float(), threshold,
                                         return_local_coords=False)
        if len(vertices[0]) == 0:
            raise ValueError('empty volume - could not tesselate')
        mesh = vx.Mesh(vertices[0].flip(-1) - 1, faces[0])

        # 
        if vx.Space(space) == 'world':
            mesh = mesh.transform(self.geometry)
        return mesh

    def _nonzero_voxel_range(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the inclusive voxel-coordinate range [minc, maxc] enclosing all
        nonzero voxels across the volume channels.

        Returns:
            tuple of Tensor: Minimum and maximum voxel coordinates.
        """
        mask = self.tensor[0] != 0 if self.num_channels == 1 else (self.tensor != 0).any(dim=0)

        # reduce to a boolean profile along each axis instead of materializing
        # all nonzero coordinates, which is much slower for dense volumes
        plane = mask.any(dim=2)
        profiles = (plane.any(dim=1), plane.any(dim=0), mask.any(dim=0).any(dim=0))

        coords = []
        for profile in profiles:
            indices = profile.nonzero()
            if indices.shape[0] == 0:
                raise ValueError('cannot compute nonzero bounds on an empty volume')
            coords.append((indices[0, 0], indices[-1, 0]))

        minc = torch.stack([c[0] for c in coords])
        maxc = torch.stack([c[1] for c in coords])
        return minc, maxc

    def bounds(self,
        nonzero: bool = False,
        margin: float | torch.Tensor | None = None,
        space: vx.Space = 'world') -> vx.BoundingBox:
        """
        Compute a world-space bounding box enclosing the volume grid or the
        non-zero voxels in the image. The box covers the full extent of the
        voxels, i.e. it is padded 0.5 voxels beyond the outermost voxel centers.

        Args:
            nonzero (bool): If True, compute the bounds around all non-zero voxels,
                otherwise use the extent of the image grid.
            margin (float or Tensor, optional): Margin to expand the bounds.
                Can be a positive or negative delta.
            space (Space, optional): Space of the margin values, either 'voxel' or 'world'.

        Returns:
            BoundingBox: Bounding box in world-space coordinates.
        """
        if not nonzero:
            return self.geometry.bounds(margin=margin, space=space)

        # compute the bounding box around all nonzero voxels
        minc, maxc = self._nonzero_voxel_range()
        box = vx.BoundingBox.from_min_max(minc.float() - 0.5, maxc.float() + 0.5)

        # expand (or shrink) margin around border
        if margin is not None:
            box = box.pad(self.geometry.conform_units(margin, space, 'voxel', 2))

        # move the voxel-space box into world coordinates
        return box.transform(self.geometry)

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

    def slice(self,
        point: int | torch.Tensor,
        direction: int | torch.Tensor,
        space: vx.Space) -> Volume:
        """
        Extract a slice from the volume. Note this will still return a volume,
        but with a slice dimension reduced to 1.

        Args:
            point (int or Tensor): A point of the slice plane. If a tensor,
                it should represent a 3D point coordinate. If an int, it should be
                the index of the slice in the specified direction. Note that this requires
                the slice direction axis to be specified as an int as well.
            direction (int or Tensor): The direction of the slice plane. If a tensor,
                it should represent a 3D vector direction. If an int, it should be
                the index of the slice in the specified direction.
            space (Space): The coordinate space of the slice point and direction.

        Returns:
            Volume: The sliced volume instance.
        """
        if vx.Space(space) == 'world':
            raise NotImplementedError('slicing in world space is not yet supported')
        if isinstance(point, torch.Tensor):
            raise NotImplementedError('slicing with a 3d plane point is not yet supported')
        if isinstance(direction, torch.Tensor):
            raise NotImplementedError('slicing with a direction vector is not yet supported')

        if direction < 0 or direction > 2:
            raise ValueError(f'slice direction must be between 0 and 2, got {direction}')
        if point < 0 or point >= self.baseshape[direction]:
            raise ValueError(f'slice index {point} out of bounds for shape {self.baseshape}')

        # create a cropping tuple to extract the slice
        cropping = [slice(None) for _ in range(4)]
        cropping[direction + 1] = slice(point, point + 1)
        return self[tuple(cropping)]

    def crop(self,
        cropping: tuple | vx.BoundingBox,
        margin: float | torch.Tensor | None = None,
        space: vx.Space = 'world') -> Volume:
        """
        Crop the volume to some bounding, either defined by a voxel slicing
        tuple or a world-space bounding box.

        Args:
            cropping (tuple or BoundingBox): Cropping defined by either a tuple
                of slices or a bounding box.
            margin (float or Tensor, optional): Margin to expand the cropping boundary.
                Can be a positive or negative delta. The boundary will be clipped if it
                extends beyond the shape of the volume.
            space (Space): The coordinate space of the margin values, either
                'voxel' or 'world'.

        Returns:
            Volume: The cropped volume instance.
        """
        stride = None
        channels = slice(None)

        if isinstance(cropping, vx.BoundingBox):
            # keep the voxel centers contained in the box, clamped to the grid
            minc, maxc = self.geometry._bounds_voxel_range(cropping, margin, space, clamp=True)

        elif isinstance(cropping, (tuple, int, slice, type(...))):

            # conform single indexing items to a tuple format
            if not isinstance(cropping, tuple):
                cropping = (cropping,)

            # if we get a tuple assume its a tuple of slices
            expanded = vx.slicing.expand_slicing(cropping, 4)
            channels = expanded[0]

            # do not allow cropping to remove a spatial dimension
            if any(isinstance(s, int) for s in expanded[1:]):
                raise ValueError('cannot remove a spatial dimension when cropping a volume')

            # extend the boundary
            minc, maxc, stride = vx.slicing.slicing_to_coordinates(expanded[1:], self.baseshape)
            if margin is not None:
                margin = self.geometry.conform_units(margin, space, 'voxel', 2).cpu().round().int()
                minc = (minc - margin[:, 0]).clamp(min=0)
                maxc = (maxc + margin[:, 1]).clamp(max=torch.tensor(self.baseshape) - 1)

        elif isinstance(cropping, vx.Mesh):
            raise TypeError('cropping by mesh is no longer supported - use a '
                            'bounding box, e.g. volume.crop(mesh.bounds())')
        else:
            raise ValueError(f'unknown cropping item: {type(cropping)}')

        # apply the cropping
        slicing = (channels, *vx.slicing.coordinates_to_slicing(minc, maxc, stride))
        cropped_tensor = self.tensor[slicing]

        # update the geometry based on the voxel shift, scale, and new shape
        geometry = self.geometry.shift(minc, space='voxel')
        if stride is not None:
            geometry = geometry.scale(stride, space='voxel')
        geometry = geometry.reshape(cropped_tensor.shape[-3:], from_origin=True)
        return self.new(cropped_tensor, geometry)

    def crop_to_nonzero(self,
        margin: float | torch.Tensor | None = None,
        *components: float) -> Volume:
        """
        Crop the volume to the bounding box around nonzero voxels.

        Args:
            margin (float or Tensor, optional): Margin (in world units) to expand
                the cropping boundary. Can be a positive or negative delta. The
                boundary will be clipped if it extends beyond the shape of the volume.
            *components (float): Additional components of `margin`, allowing values to
                be passed as separate positional arguments, e.g. `crop_to_nonzero(1, 1, 2)`.

        Returns:
            Volume: The cropped volume instance.
        """
        margin = vx.arguments.merge_components(margin, components)
        # note: we're using the voxel-space range directly here instead of calling
        # self.bounds() to avoid the unnecessary transformation into world space
        # then back again
        minc, maxc = self._nonzero_voxel_range()
        slicing = (slice(None), *vx.slicing.coordinates_to_slicing(minc, maxc))
        return self.crop(slicing, margin=margin)

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
        fill: float = 0,
        antialias: bool = False) -> Volume:
        """
        Resample the volume features to match the geometry of a target volume.

        Args:
            target (Volume | AcquisitionGeometry): Target acquisition geometry.
            mode (str, optional): Interpolation mode.
            padding_mode (str, optional): Padding mode for outside grid values.
            fill (float, optional): Out of bounds value used for fill padding mode.
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
                           localshape=self.baseshape, device=self.device)

        fill_out_of_bounds = padding_mode == 'fill'
        if fill_out_of_bounds:
            padding_mode = 'border'

        resampled = torch.nn.functional.grid_sample(
                        input=self.tensor.float().unsqueeze(0),
                        grid=grid.unsqueeze(0),
                        mode=('bilinear' if mode == 'linear' else mode),
                        padding_mode=padding_mode,
                        align_corners=False).squeeze(0)
    
        if fill_out_of_bounds:
            out_of_bounds = (grid < -1).any(-1) | (grid > 1).any(-1)
            if isinstance(fill, torch.Tensor):
                fill = fill.type(resampled.dtype)
            resampled[out_of_bounds.unsqueeze(0)] = fill

        if antialias:
            kernels = [vx.filters.gaussian_kernel_1d(float(s), truncate, device=resampled.device)
                       for s in sigma]
            resampled = vx.filters._filter_tensor(resampled, kernels, stride=tuple(down_factor),
                                                  padding='valid')

        # probably ideal to keep the data type consistent when using nearest neighbor sampling
        if mode == 'nearest':
            resampled = resampled.type(self.dtype)

        return self.new(resampled, target)

    def resample(self,
        spacing: float | torch.Tensor = None,
        *components: float,
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
            *components (float): Additional components of `spacing`, allowing values to
                be passed as separate positional arguments, e.g. `resample(1, 1, 2)`.
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
        spacing = vx.arguments.merge_components(spacing, components)
        target = self.geometry.resample(spacing=spacing, in_plane_spacing=in_plane_spacing,
                                        slice_spacing=slice_spacing)
        return self.resample_like(target, mode=mode, padding_mode=padding_mode, antialias=antialias)

    def reshape(self, baseshape: int | torch.Size, *components: int) -> Volume:
        """
        Modify the spatial extent of the volume, cropping or padding around the
        center image to fit a given **baseshape**.

        This method is symmetric in that performing a reverse reshape operation
        will always yield the original geometry.

        Args:
            baseshape (int | Size): Target spatial (3D) shape. An isotropic shape
                is assumed if a scalar is provided.
            *components (int): Additional components of `baseshape`, allowing values to
                be passed as separate positional arguments, e.g. `reshape(64, 64, 64)`.

        Returns:
            Volume: Reshaped volume instance.
        """
        return self.resample_like(self.geometry.reshape(baseshape, *components), mode='nearest')

    def pad(self,
        delta: float | torch.Tensor,
        *components: float,
        space: vx.Space = None) -> Volume:
        """
        Pad the spatial extent of the volume by a given delta. Note that
        a negative delta value will result in trimming (cropping).

        Args:
            delta (float or Tensor): Delta of specified units to pad (or crop)
                the volume by in each direction. Can be of size $(1,)$, $(3,)$,
                or $(3, 2)$.
            *components (float): Additional components of `delta`, allowing values to
                be passed as separate positional arguments, e.g. `pad(1, 2, 3, 'voxel')`.
            space (Space): The coordinate space of the delta values, either
                'voxel' or 'world'. Can be provided as the last positional argument.

        Returns:
            Volume: Padded volume instance.
        """
        components, space = vx.arguments.extract_space(components, space)
        delta = vx.arguments.merge_components(delta, components)
        return self.resample_like(self.geometry.pad(delta, space=space), mode='nearest')

    def trim(self,
        delta: float | torch.Tensor,
        *components: float,
        space: vx.Space = None) -> Volume:
        """
        Trim the spatial extent of the volume by a given delta. This is
        equivalent to padding with negative delta values.

        Args:
            delta (float or Tensor): Delta of specified units to trim the volume
                by in each direction. Can be of size $(1,)$, $(3,)$, or $(3, 2)$.
            *components (float): Additional components of `delta`, allowing values to
                be passed as separate positional arguments, e.g. `trim(1, 2, 3, 'voxel')`.
            space (Space): The coordinate space of the delta values, either
                'voxel' or 'world'. Can be provided as the last positional argument.

        Returns:
            Volume: Trimmed volume instance.
        """
        components, space = vx.arguments.extract_space(components, space)
        delta = torch.as_tensor(vx.arguments.merge_components(delta, components))
        return self.pad(-delta, space=space)

    def transform(self,
        transform: vx.AffineVolumeTransform | vx.AffineMatrix,
        resample: bool = False,
        negate: bool = False,
        mode: str = 'linear',
        padding_mode: str = 'zeros') -> Volume:
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
            padding_mode (str, optional): Padding mode for outside grid values if resampling.

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
                           localshape=self.baseshape, device=self.device)

        interpolated = torch.nn.functional.grid_sample(
                        self.tensor.unsqueeze(0).float(),
                        grid.unsqueeze(0),
                        mode=('bilinear' if mode == 'linear' else mode),
                        padding_mode=padding_mode,
                        align_corners=False).squeeze(0)

        if negate:
            # apply inverse transform to the geometry to cancel out world space changes
            target = inverted.convert(space='world') @ target

        return self.new(interpolated, target)

    def pool(self,
        scale: int = 2,
        *components: int,
        mode: str = 'mean',
        space: vx.Space = None,
        spacing_ratio_thresh: float | None = None) -> Volume:
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

        Note that this implementation must mirror the pooling operation used by the geometry
        class. Any changes to the pooling operation in one class must be reflected in the other.

        Args:
            scale (int, optional): The size of the pooling window. Defaults to 2.
            *components (int): Additional components of `scale`, allowing values to
                be passed as separate positional arguments, e.g. `pool(2, 2, 1)`.
            mode (str, optional): Pooling mode - can be 'mean' or 'max'. Defaults to 'mean'.
            space (Space, optional): Space of the scale value. Can be provided as
                the last positional argument. Defaults to 'voxel'.
            spacing_ratio_thresh (float, optional): Slice spacing ratio that determines
                whether the slice dimension is pooled. This is disabled by default.

        Returns:
            Volume: Pooled volume.
        """
        components, space = vx.arguments.extract_space(components, space, default='voxel')
        scale = vx.arguments.merge_components(scale, components)
        scale = self.geometry.conform_units(scale, space, 'voxel').round().int().clamp(min=1)

        factors = [min(d, int(s.item())) for s, d in zip(scale, self.baseshape)]

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
        if not slice_dim_pooling and spacing_ratio < scale[self.geometry.slice_direction]:
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

    def smooth(self,
        sigma: float | torch.Tensor,
        *components: float,
        space: vx.Space = None,
        truncate: float = 2) -> Volume:
        """
        Apply Gaussian smoothing to the image features.

        Args:
            sigma (float | Tensor): Smoothing sigma.
            *components (float): Additional components of `sigma`, allowing values to
                be passed as separate positional arguments, e.g. `smooth(1, 1, 2)`.
            space (Space, optional): The space of the sigma values, either
                'voxel' or 'world'. Can be provided as the last positional
                argument. Defaults to 'world'.
            truncate (float, optional): The number of standard deviations to extend
                the kernel before truncating.

        Returns:
            Volume: Smoothed volume.
        """
        components, space = vx.arguments.extract_space(components, space, default='world')
        sigma = vx.arguments.merge_components(sigma, components)
        return vx.filters.gaussian_filter(self, sigma, space=space, truncate=truncate)

    def dilate(self,
        iterations: int = 1,
        connectivity: int = 1,
        iso_thresh: float | None = None) -> Volume:
        """
        Apply a binary dilation to the nonzero voxels of the volume.

        Args:
            iterations (int, optional): Number of dilation iterations.
            connectivity (int, optional): Neighborhood connectivity between 1 and 3.
            iso_thresh (float, optional): Spacing ratio at or above which the
                operation is applied only in-plane. Disabled by default.

        Returns:
            Volume: Dilated volume of the same data type.
        """
        return vx.morphology.dilate(self, iterations, connectivity, iso_thresh)

    def erode(self,
        iterations: int = 1,
        connectivity: int = 1,
        iso_thresh: float | None = None) -> Volume:
        """
        Apply a binary erosion to the nonzero voxels of the volume.

        Args:
            iterations (int, optional): Number of erosion iterations.
            connectivity (int, optional): Neighborhood connectivity between 1 and 3.
            iso_thresh (float, optional): Spacing ratio at or above which the
                operation is applied only in-plane. Disabled by default.

        Returns:
            Volume: Eroded volume of the same data type.
        """
        return vx.morphology.erode(self, iterations, connectivity, iso_thresh)

    def close(self,
        iterations: int = 1,
        connectivity: int = 1,
        iso_thresh: float | None = None) -> Volume:
        """
        Apply a binary closing (dilation followed by erosion) to the nonzero
        voxels of the volume.

        Args:
            iterations (int, optional): Number of dilation and erosion iterations.
            connectivity (int, optional): Neighborhood connectivity between 1 and 3.
            iso_thresh (float, optional): Spacing ratio at or above which the
                operation is applied only in-plane. Disabled by default.

        Returns:
            Volume: Closed volume of the same data type.
        """
        return vx.morphology.close(self, iterations, connectivity, iso_thresh)

    def open(self,
        iterations: int = 1,
        connectivity: int = 1,
        iso_thresh: float | None = None) -> Volume:
        """
        Apply a binary opening (erosion followed by dilation) to the nonzero
        voxels of the volume.

        Args:
            iterations (int, optional): Number of erosion and dilation iterations.
            connectivity (int, optional): Neighborhood connectivity between 1 and 3.
            iso_thresh (float, optional): Spacing ratio at or above which the
                operation is applied only in-plane. Disabled by default.

        Returns:
            Volume: Opened volume of the same data type.
        """
        return vx.morphology.open(self, iterations, connectivity, iso_thresh)

    # -------------------------------------------------------------------------
    # image visualization
    # -------------------------------------------------------------------------

    def show(self, **kwargs) -> None:
        """
        Show the volume in a Monocle viewer window. This is a convenience
        method for `vx.monocle.show(volume)`.
        """
        vx.monocle.show(self, **kwargs)


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
        transform (AffineMatrix, optional): Grid voxel coordinate transform.
        localshape (Size, optional): If provided, the grid is normalized to the
            range [-1, 1] using this spatial shape and the coordinate order is
            swapped (for torch sampling methods).
        device (device, optional): Device on which to allocate the grid data.

    Returns:
        Tensor: Grid volume tensor.
    """
    ranges = [torch.arange(s, dtype=torch.float32, device=device) for s in baseshape]
    grid = torch.stack(torch.meshgrid(*ranges, indexing='ij'), dim=-1)
    if transform:
        grid = transform.transform(grid)
    if localshape is not None:
        shape = torch.tensor(localshape).to(grid.device)
        grid = ((2 * grid + 1) / shape - 1).flip(-1)
    return grid


def stack(*vols) -> Volume:
    """
    Concatenate (stack) multiple volumes channel-wise. Assumes the volumes are
    in the same image space (with the same base shape).

    Args:
        *vols (Volume): Volumes to merge.

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
        a (Volume): First volume to compare.
        b (Volume): Second volume to compare.
        vol_tol (float, optional): Absolute tolerance for volume tensor comparison.
        geom_tol (float, optional): Absolute tolerance for geometry comparison.

    Returns:
        bool: True if the volumes are equal, False otherwise.
    """
    if a.tensor.shape != b.tensor.shape:
        return False
    if not a.tensor.allclose(b.tensor, atol=vol_tol, rtol=0):
        return False
    if not a.geometry.tensor.allclose(b.geometry.tensor, atol=geom_tol, rtol=0):
        return False
    return True
