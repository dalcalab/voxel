"""
Utilities for image grid filtering and kernel construction.
"""

from __future__ import annotations

import torch
import voxel as vx


def gaussian_kernel_1d(
    sigma: float,
    truncate: float = 2.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None) -> torch.Tensor:
    """
    Generate a 1D Gaussian kernel with a specified standard deviation.

    Args:
        sigma (float): Standard deviation in element (voxel) space.
        truncate (float, optional): The number of standard deviations to extend
            the kernel before truncating.
        device (torch.device, optional): The device on which to create the kernel.
        dtype (torch.dtype, optional): The kernel datatype.

    Returns:
        Tensor: A normalized kernel of length $2 * int(truncate * sigma + 0.5) + 1$.
    """
    r = int(truncate * sigma + 0.5)
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    sigma2 = 1 / torch.clip(torch.as_tensor(sigma), min=1e-5).pow(2)
    pdf = torch.exp(-0.5 * (x.pow(2) * sigma2))
    return pdf / pdf.sum()


def gaussian_filter(
    volume: vx.Volume,
    sigma: float | torch.Tensor,
    *components: float,
    space: vx.Space = None,
    truncate: float = 2.0,
    stride: int | torch.Tensor = 1,
    separable: bool = True,
    padding_mode: str = 'replicate') -> vx.Volume:
    """
    Apply Gaussian smoothing to a volume.

    Args:
        sigma (float or Tensor): Standard deviation(s) of size $(1,)$ or $(3,)$.
        *components (float): Additional components of `sigma`, allowing values to
            be passed as separate positional arguments, e.g. `gaussian_filter(vol, 1, 1, 2, 'voxel')`.
        space (Space): The space of the sigma values, either 'voxel' or 'world'.
            Can be provided as the last positional argument.
        truncate (float, optional): The number of standard deviations to extend
            the kernel before truncating.
        stride (int or Tensor, optional): Downsampling stride(s) in voxel units.
        separable (bool, optional): Whether to filter with three separable 1D
            kernels or a single dense 3D kernel. The results are equivalent.
        padding_mode (str, optional): Border padding mode, e.g. 'replicate' or 'zeros'.

    Returns:
        Volume: Smoothed floating-point volume.
    """
    components, space = vx.arguments.extract_space(components, space)
    sigma = vx.arguments.merge_components(sigma, components)
    sigma = volume.geometry.conform_units(sigma, space, 'voxel')
    kernels = [gaussian_kernel_1d(float(s), truncate, volume.device) if volume.baseshape[i] > 1
               else torch.ones(1, device=volume.device) for i, s in enumerate(sigma)]
    if not separable:
        kernels = _dense_kernel(kernels)
    return apply_filter(volume, kernels, stride=stride, padding_mode=padding_mode)


def box_filter(
    volume: vx.Volume,
    size: float | torch.Tensor,
    *components: float,
    space: vx.Space = None,
    stride: int | torch.Tensor = 1,
    separable: bool = True,
    padding_mode: str = 'replicate') -> vx.Volume:
    """
    Apply mean filtering to a volume with a box kernel.

    The box extent is rounded to the nearest odd number of voxels along
    each dimension.

    Args:
        size (float or Tensor): Box extent(s) of size $(1,)$ or $(3,)$.
        *components (float): Additional components of `size`, allowing values to
            be passed as separate positional arguments, e.g. `box_filter(vol, 3, 3, 1, 'voxel')`.
        space (Space): The space of the size values, either 'voxel' or 'world'.
            Can be provided as the last positional argument.
        stride (int or Tensor, optional): Downsampling stride(s) in voxel units.
        separable (bool, optional): Whether to filter with three separable 1D
            kernels or a single dense 3D kernel. The results are equivalent.
        padding_mode (str, optional): Border padding mode, e.g. 'replicate' or 'zeros'.

    Returns:
        Volume: Filtered floating-point volume.
    """
    components, space = vx.arguments.extract_space(components, space)
    size = vx.arguments.merge_components(size, components)
    size = volume.geometry.conform_units(size, space, 'voxel')
    kernel_size = (torch.round((size - 1) / 2) * 2 + 1).int().clamp(min=1)

    # don't filter volume dimensions of size 1
    for i, s in enumerate(volume.baseshape):
        if s == 1:
            kernel_size[i] = 1

    kernels = [torch.ones(int(k), device=volume.device) / int(k) for k in kernel_size]
    if not separable:
        kernels = _dense_kernel(kernels)
    return apply_filter(volume, kernels, stride=stride, padding_mode=padding_mode)


def apply_filter(
    volume: vx.Volume,
    kernel: torch.Tensor | list,
    stride: int | torch.Tensor = 1,
    padding_mode: str = 'replicate') -> vx.Volume:
    """
    Apply a filter kernel to a volume with 'same'-style output extents.

    Channels are filtered independently. When strided, the grid is downsampled
    to a spatial size of $ceil(baseshape / stride)$ and the volume geometry is
    updated to reflect the new voxel spacing.

    Args:
        kernel (Tensor or list): A single dense 3D kernel of shape $(W, H, D)$
            or a sequence of three 1D kernels applied separably per dimension.
        stride (int or Tensor, optional): Downsampling stride(s) in voxel units.
        padding_mode (str, optional): Border padding mode, e.g. 'replicate' or 'zeros'.

    Returns:
        Volume: Filtered floating-point volume.
    """
    stride = torch.as_tensor(stride)
    stride = [int(s) for s in (stride.repeat(3) if stride.ndim == 0 else stride)]

    kernels = [kernel] if torch.is_tensor(kernel) else list(kernel)
    kernel_size = list(kernels[0].shape) if len(kernels) == 1 else [k.numel() for k in kernels]

    # skip the convolution entirely for an identity kernel
    if all(s == 1 for s in stride) and all(k.numel() == 1 for k in kernels) and \
       all(float(k.flatten()[0]) == 1 for k in kernels):
        return volume

    result = _filter_tensor(volume.tensor, kernel, stride=stride, padding_mode=padding_mode)

    # unstrided, odd-sized kernels preserve the grid and therefore the geometry
    if all(s == 1 for s in stride) and all(k % 2 == 1 for k in kernel_size):
        geometry = volume.geometry
    else:
        # even kernels pad asymmetrically (less on the low side), which shifts
        # the effective window center by half a voxel
        offset = [(k - 1) / 2 - (k - 1) // 2 for k in kernel_size]
        geometry = volume.geometry.shift(offset, 'voxel').scale(stride, 'voxel')
        geometry = geometry.reshape(result.shape[-3:], from_origin=True)

    return volume.new(result, geometry)


def _filter_tensor(
    tensor: torch.Tensor,
    kernel: torch.Tensor | list,
    stride: int | torch.Tensor = 1,
    padding: str = 'same',
    padding_mode: str = 'replicate') -> torch.Tensor:
    """
    Convolve a channeled $(C, W, H, D)$ tensor with a dense 3D kernel or a
    sequence of three separable 1D kernels.

    Padding is applied to the input once up front, so separable and dense
    filtering remain numerically equivalent for any padding mode.

    Args:
        kernel (Tensor or list): A single dense 3D kernel of shape $(W, H, D)$
            or a sequence of three 1D kernels applied separably per dimension.
        stride (int or Tensor, optional): Downsampling stride(s) in voxel units.
        padding (str, optional): Output extent style, either 'same' or 'valid'.
        padding_mode (str, optional): Border padding mode, e.g. 'replicate' or 'zeros'.

    Returns:
        Tensor: Filtered floating-point tensor of shape $(C, W, H, D)$.
    """
    if tensor.ndim != 4:
        raise ValueError(f'expected a 4D (C, W, H, D) tensor, got {tensor.ndim}D')

    stride = torch.as_tensor(stride)
    stride = [int(s) for s in (stride.repeat(3) if stride.ndim == 0 else stride)]
    if len(stride) != 3:
        raise ValueError(f'stride must be a scalar or 3 values, got {len(stride)}')

    kernels = [kernel] if torch.is_tensor(kernel) else list(kernel)
    if len(kernels) == 1 and kernels[0].ndim == 3:
        kernel_size = list(kernels[0].shape)
    elif len(kernels) == 3 and all(k.ndim == 1 for k in kernels):
        kernel_size = [k.numel() for k in kernels]
    else:
        raise ValueError('kernel must be a single 3D tensor or a sequence of three 1D tensors')

    result = tensor.float().unsqueeze(0)
    channels = tensor.shape[0]

    if padding == 'same':
        mode = 'constant' if padding_mode == 'zeros' else padding_mode
        result = torch.nn.functional.pad(result, _compute_padding(kernel_size), mode=mode)
    elif padding != 'valid':
        raise ValueError(f"padding must be 'same' or 'valid', got '{padding}'")

    conv = torch.nn.functional.conv3d
    if len(kernels) == 1:
        weight = kernels[0].to(result.dtype).view(1, 1, *kernel_size).expand(channels, 1, *kernel_size)
        result = conv(result, weight, groups=channels, stride=stride)
    else:
        for dim, k in enumerate(kernels):
            # a single-element unit kernel with no stride is an identity
            if k.numel() == 1 and stride[dim] == 1 and float(k[0]) == 1:
                continue
            shape = [k.numel() if d == dim else 1 for d in range(3)]
            weight = k.to(result.dtype).view(1, 1, *shape).expand(channels, 1, *shape)
            dim_stride = [stride[d] if d == dim else 1 for d in range(3)]
            result = conv(result, weight, groups=channels, stride=dim_stride)

    return result.squeeze(0)


def _dense_kernel(kernels: list) -> torch.Tensor:
    """
    Combine three separable 1D kernels into a dense 3D kernel via outer product.
    """
    return kernels[0][:, None, None] * kernels[1][None, :, None] * kernels[2][None, None, :]


def _compute_padding(kernel_size: list) -> list:
    """
    Compute the per-side input padding that preserves the spatial extents of a
    convolution ('same' padding), in the reversed order expected by `F.pad`.
    """
    padding = [0] * 6
    for i, k in enumerate(reversed(kernel_size)):
        total = int(k) - 1
        padding[2 * i] = total // 2
        padding[2 * i + 1] = total - total // 2
    return padding
