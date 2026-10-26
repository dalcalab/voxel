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
    kernels = _box_kernels(size, volume.baseshape, volume.device)
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
    if all(s == 1 for s in stride) and all(k.numel() == 1 for k in kernels):
        return volume

    result = filter_tensor(volume.tensor, kernel, stride=stride, padding_mode=padding_mode)

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


def filter_tensor(
    tensor: torch.Tensor,
    kernel: torch.Tensor | list,
    stride: int | torch.Tensor = 1,
    padding: str = 'same',
    padding_mode: str = 'replicate') -> torch.Tensor:
    """
    Convolve a channeled $(C, W, H, D)$ tensor with a dense 3D kernel or a
    sequence of three separable 1D kernels.

    Padding is applied to the input once up front, so separable and dense
    filtering remain numerically equivalent for any padding mode. A length-1
    separable kernel is an identity tap and is skipped without reading its value,
    so prebuilt kernels can live on any device without a host sync.

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

    # odd kernels with zero padding can be padded by the convolution itself, which
    # avoids materializing a padded copy of the input (a noticeable cost in hot loops)
    conv_pads = [0, 0, 0]
    if padding == 'same':
        if padding_mode == 'zeros' and all(int(k) % 2 == 1 for k in kernel_size):
            conv_pads = [(int(k) - 1) // 2 for k in kernel_size]
        else:
            mode = 'constant' if padding_mode == 'zeros' else padding_mode
            result = torch.nn.functional.pad(result, _compute_padding(kernel_size), mode=mode)
    elif padding != 'valid':
        raise ValueError(f"padding must be 'same' or 'valid', got '{padding}'")

    conv = torch.nn.functional.conv3d
    if len(kernels) == 1:
        weight = kernels[0].to(result.dtype).view(1, 1, *kernel_size).expand(channels, 1, *kernel_size)
        result = conv(result, weight, groups=channels, stride=stride, padding=conv_pads)
    else:
        for dim, k in enumerate(kernels):
            # a single-element kernel with no stride is an identity
            if k.numel() == 1 and stride[dim] == 1:
                continue
            shape = [k.numel() if d == dim else 1 for d in range(3)]
            weight = k.to(result.dtype).view(1, 1, *shape).expand(channels, 1, *shape)
            dim_stride = [stride[d] if d == dim else 1 for d in range(3)]
            dim_pad = [conv_pads[d] if d == dim else 0 for d in range(3)]
            result = conv(result, weight, groups=channels, stride=dim_stride, padding=dim_pad)

    return result.squeeze(0)


def box_kernel_sizes(size: float | torch.Tensor, shape: torch.Size) -> tuple:
    """
    Voxel extents of a box kernel, as python integers: the requested extents rounded
    to the nearest odd number, and 1 along singleton dimensions, which are never filtered.

    Args:
        size (float or Tensor): Box extent(s) in voxels, one value or one per dimension.
        shape (Size): Spatial shape of the tensor to filter.

    Returns:
        tuple: Three odd integer kernel sizes.
    """
    size = torch.as_tensor(size, dtype=torch.float32)
    size = size.repeat(3) if size.ndim == 0 else size
    kernel_size = (torch.round((size - 1) / 2) * 2 + 1).int().clamp(min=1)
    return tuple(1 if s == 1 else int(k) for k, s in zip(kernel_size, shape))


def box_filter_tensor(tensor: torch.Tensor, sizes: tuple) -> torch.Tensor:
    """
    Separable zero-padded box mean of a channeled $(C, W, H, D)$ tensor, for use in
    hot, differentiable, or compiled code. The kernel sizes are python integers, so no
    kernel tensors are built or inspected and the call never syncs with the host. The
    backward pass is the same filter (an odd zero-padded box filter is self-adjoint),
    which keeps the convolution backward out of compiled graphs, where its lowering
    specializes on the spatial size and recompiles per shape.

    Near the borders the mean includes the zero padding. Divide by `box_coverage`
    to average over the in-bounds voxels only.

    Args:
        tensor (Tensor): Input tensor of shape $(C, W, H, D)$.
        sizes (tuple): Three odd integer kernel sizes, e.g. from `box_kernel_sizes`.

    Returns:
        Tensor: Filtered tensor of the input shape and dtype.
    """
    return _BoxFilter.apply(tensor, tuple(int(k) for k in sizes))


def box_coverage(
    sizes: tuple,
    shape: torch.Size,
    device: torch.device | None = None) -> torch.Tensor:
    """
    Fraction of a zero-padded box window that falls in bounds at every voxel, i.e. the
    box mean of a volume of ones. Dividing a zero-padded box mean by it turns the mean
    into one over the in-bounds voxels only, which keeps windowed statistics correct at
    the borders. Odd kernels always keep the center tap in bounds, so no value is zero.

    Args:
        sizes (tuple): Three odd integer kernel sizes, e.g. from `box_kernel_sizes`.
        shape (Size): Spatial shape of the tensor to filter.
        device (device, optional): Device of the coverage tensor.

    Returns:
        Tensor: Coverage of shape $(1, W, H, D)$.
    """
    axes = []
    for k, extent in zip(sizes, shape):
        ones = torch.ones(1, 1, extent, device=device)
        kernel = torch.ones(1, 1, int(k), device=device) / int(k)
        axes.append(torch.nn.functional.conv1d(ones, kernel, padding='same').view(-1))
    return (axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]).unsqueeze(0)


class _BoxFilter(torch.autograd.Function):
    """
    The zero-padded box mean with its backward spelled out as the same filter.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, sizes: tuple) -> torch.Tensor:
        ctx.sizes = sizes
        return _box_conv(tensor, sizes)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple:
        return _box_conv(grad, ctx.sizes), None


def _box_conv(tensor: torch.Tensor, sizes: tuple) -> torch.Tensor:
    """
    Separable zero-padded box mean with python kernel sizes and conv-side padding.
    """
    channels = tensor.shape[0]
    result = tensor.unsqueeze(0)
    for dim, k in enumerate(sizes):
        if k == 1:
            continue
        shape = [k if d == dim else 1 for d in range(3)]
        weight = torch.full((channels, 1, *shape), 1 / k, dtype=result.dtype, device=result.device)
        result = torch.nn.functional.conv3d(
            result, weight, groups=channels,
            padding=[(k - 1) // 2 if d == dim else 0 for d in range(3)])
    return result.squeeze(0)


def _box_kernels(
    size: torch.Tensor,
    shape: torch.Size,
    device: torch.device | None = None) -> list:
    """
    Build three separable 1D uniform kernels of a given voxel extent, normalized
    so that filtering with them yields a local mean.
    """
    return [torch.ones(k, device=device) / k for k in box_kernel_sizes(size, shape)]


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
