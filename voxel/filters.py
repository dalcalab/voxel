"""
Utilies for image grid filtering and kernel construction.
"""

from __future__ import annotations

import torch


def gaussian_kernel_1d(
    sigma: float,
    truncate: float = 2,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None) -> torch.Tensor:
    """
    Generate a 1D Gaussian kernel with a specified standard deviation.

    Args:
        sigma (float): Standard deviations in element (voxel) space.
        truncate (float, optional): The number of standard deviations to extend
            the kernel before truncating.
        device (torch.device, optional): The device on which to create the kernel.
        device (torch.dtype, optional): The kernel datatype.

    Returns:
        Tensor: A kernel of shape $2 * truncate * sigma + 1$.
    """
    r = int(truncate * sigma + 0.5)
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    sigma2 = 1 / torch.clip(torch.as_tensor(sigma), min=1e-5).pow(2)
    pdf = torch.exp(-0.5 * (x.pow(2) * sigma2))
    return pdf / pdf.sum()


def gaussian_blur(
    image: torch.Tensor,
    sigma: list,
    batched: bool = False,
    truncate: float = 2,
    padding_mode: str = 'replicate') -> torch.Tensor:
    """
    Apply Gaussian blurring to a data grid.

    The Gaussian filter is applied using convolution. The size of the filter kernel
    is determined by the standard deviation and the truncation factor.

    Args:
        image (Tensor): An image tensor with preceding channel dimensions. A
            batch dimension can be included by setting `batched=True`.
        sigma (float): Standard deviations in element (voxel) space.
        batched (bool, optional): If True, assume image has a batch dimension.
        truncate (float, optional): The number of standard deviations to extend
            the kernel before truncating.
        padding_mode (str, optional): The padding mode to use.

    Returns:
        Tensor: The blurred tensor with the same shape as the input tensor.
    """
    ndim = image.ndim - (2 if batched else 1)

    # sanity check for common mistake
    if ndim == 4 and not batched:
        raise ValueError(f'gaussian blur input has {image.ndim} dims, '
                          'but batched option is False')

    # make sure sigmas match the ndim
    sigma = torch.as_tensor(sigma)
    if sigma.ndim == 0:
        sigma = sigma.repeat(ndim)
    if len(sigma) != ndim:
        raise ValueError(f'sigma must be {ndim}D, but got length {len(sigma)}')

    blurred = image.float()
    if not batched:
        blurred = blurred.unsqueeze(0)

    for dim, s in enumerate(sigma):

        # reuse previous kernel if we can
        if dim == 0 or s != sigma[dim - 1]:
            kernel = gaussian_kernel_1d(s, truncate, blurred.device, blurred.dtype)
            kernel_size = len(kernel)

        # kernels are normalized. if the length is one, there's no point in using it
        if kernel_size == 1:
            continue

        # select the kernel for the current dimension
        kernel_shape = [kernel_size if i == dim + 2 else 1 for i in range(ndim + 2)]
        kernel_dim = kernel.view(kernel_shape)

        # expand the kernel for multi-channel images
        num_channels = image.shape[0]
        if num_channels > 1:
            kernel_dim = kernel_dim.expand((num_channels, *kernel_dim.shape[1:]))

        # set the padding
        if padding_mode == 'zeros':
            padding = 'same'
        else:
            pad_size = kernel_size // 2
            pad = [0] * (2 * ndim)
            # note that padding is specified in reverse order
            base = 2 * (ndim - dim - 1)
            pad[base] = pad_size
            pad[base + 1] = pad_size
            blurred = torch.nn.functional.pad(blurred, pad, mode=padding_mode)
            padding = 0

        # apply the convolution
        conv = getattr(torch.nn.functional, f'conv{ndim}d')
        blurred = conv(blurred, kernel_dim, groups=num_channels, padding=padding)

    if not batched:
        blurred = blurred.squeeze(0)

    return blurred


def dilate(image: torch.Tensor, iterations: int = 1, batched: bool = False) -> torch.Tensor:
    """
    Apply a binary dilation operation to a data grid.

    Args:
        image (Tensor): An image tensor with preceding channel dimensions. A
            batch dimension can be included by setting `batched=True`.
        iterations (int, optional): The number of dilation iterations.
        batched (bool, optional): If True, assume image has a batch dimension.

    Returns:
        Tensor: The dilated tensor with the same shape as the input tensor.
    """
    ndim = image.ndim - (2 if batched else 1)

    # sanity check for common mistake
    if ndim == 4 and not batched:
        raise ValueError(f'dilate input has {image.ndim} dims, '
                          'but batched option is False')

    dilated = image.float()
    if not batched:
        dilated = dilated.unsqueeze(0)

    kernel = torch.zeros([3] * ndim, device=dilated.device, dtype=dilated.dtype)
    for dim in range(ndim):
        slices = [slice(None)] * ndim
        slices[dim] = 1
        kernel[tuple(slices)] = 1
    kernel = kernel.view(1, 1, *kernel.shape)

    conv = getattr(torch.nn.functional, f'conv{ndim}d')
    for _ in range(iterations):
        dilated = conv(dilated, kernel, groups=image.shape[0], padding='same')

    if not batched:
        dilated = dilated.squeeze(0)

    return (dilated > 0).to(image.dtype)
