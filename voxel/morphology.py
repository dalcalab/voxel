"""
Binary morphological operations on volume grids.
"""

from __future__ import annotations

import torch
import voxel as vx


def dilate(volume: vx.Volume,
    iterations: int = 1,
    connectivity: int = 1,
    iso_thresh: float | None = None) -> vx.Volume:
    """
    Apply a binary dilation to the nonzero voxels of a volume.

    Args:
        iterations (int, optional): Number of dilation iterations.
        connectivity (int, optional): Neighborhood connectivity: 1 includes
            face neighbors (6), 2 adds edge neighbors (18), and 3 adds corner
            neighbors (26).
        iso_thresh (float, optional): Spacing ratio (slice over in-plane
            spacing) at or above which the operation is applied only in-plane,
            i.e. the kernel is collapsed along the slice direction. If None,
            the operation is always treated as isotropic. Disabled by default.

    Returns:
        Volume: Dilated binary volume of the input data type.
    """
    return _morphological_operation(volume, iterations, connectivity, iso_thresh, mode='dilate')


def erode(volume: vx.Volume,
    iterations: int = 1,
    connectivity: int = 1,
    iso_thresh: float | None = None) -> vx.Volume:
    """
    Apply a binary erosion to the nonzero voxels of a volume. Voxels beyond
    the grid border are considered background.

    Args:
        iterations (int, optional): Number of erosion iterations.
        connectivity (int, optional): Neighborhood connectivity: 1 includes
            face neighbors (6), 2 adds edge neighbors (18), and 3 adds corner
            neighbors (26).
        iso_thresh (float, optional): Spacing ratio (slice over in-plane
            spacing) at or above which the operation is applied only in-plane,
            i.e. the kernel is collapsed along the slice direction. If None,
            the operation is treated as isotropic. Disabled by default.

    Returns:
        Volume: Eroded binary volume of the input data type.
    """
    return _morphological_operation(volume, iterations, connectivity, iso_thresh, mode='erode')


def close(volume: vx.Volume,
    iterations: int = 1,
    connectivity: int = 1,
    iso_thresh: float | None = None) -> vx.Volume:
    """
    Apply a binary closing (dilation followed by erosion) to the nonzero
    voxels of a volume, sealing gaps smaller than the effective kernel.

    Args:
        iterations (int, optional): Number of dilation and erosion iterations.
        connectivity (int, optional): Neighborhood connectivity: 1 includes
            face neighbors (6), 2 adds edge neighbors (18), and 3 adds corner
            neighbors (26).
        iso_thresh (float, optional): Spacing ratio (slice over in-plane
            spacing) at or above which the operation is applied only in-plane,
            i.e. the kernel is collapsed along the slice direction. If None,
            the operation is treated as isotropic. Disabled by default.

    Returns:
        Volume: Closed binary volume of the input data type.
    """
    return erode(dilate(volume, iterations, connectivity, iso_thresh),
                 iterations, connectivity, iso_thresh)


def open(volume: vx.Volume,
    iterations: int = 1,
    connectivity: int = 1,
    iso_thresh: float | None = None) -> vx.Volume:
    """
    Apply a binary opening (erosion followed by dilation) to the nonzero
    voxels of a volume, removing structures smaller than the effective kernel.

    Args:
        iterations (int, optional): Number of erosion and dilation iterations.
        connectivity (int, optional): Neighborhood connectivity: 1 includes
            face neighbors (6), 2 adds edge neighbors (18), and 3 adds corner
            neighbors (26).
        iso_thresh (float, optional): Spacing ratio (slice over in-plane
            spacing) at or above which the operation is applied only in-plane,
            i.e. the kernel is collapsed along the slice direction. If None,
            the operation is treated as isotropic. Disabled by default.

    Returns:
        Volume: Opened binary volume of the input data type.
    """
    return dilate(erode(volume, iterations, connectivity, iso_thresh),
                  iterations, connectivity, iso_thresh)


def _morphological_operation(
    volume: vx.Volume,
    iterations: int,
    connectivity: int,
    iso_thresh: float | None,
    mode: str) -> vx.Volume:
    """
    Apply a binary morphological operation ('dilate' or 'erode') to the
    nonzero voxels of a volume.
    """
    if connectivity not in (1, 2, 3):
        raise ValueError(f'connectivity must be 1, 2, or 3, got {connectivity}')

    # structuring element of neighbors within a manhattan distance
    offsets = torch.stack(torch.meshgrid(*[torch.arange(-1, 2, device=volume.device)] * 3, indexing='ij'))
    kernel = offsets.abs().sum(0) <= connectivity

    # for sufficiently anisotropic volumes, restrict the kernel to the
    # in-plane directions
    if iso_thresh is not None and volume.geometry.spacing_ratio >= iso_thresh:
        kernel &= offsets[volume.geometry.slice_direction] == 0

    kernel = kernel.float().view(1, 1, 3, 3, 3)

    channels = volume.num_channels
    weight = kernel.expand(channels, 1, 3, 3, 3)

    # dilation marks any voxel with a set neighbor, erosion requires all
    # neighbors to be set (zero padding treats the border as background)
    threshold = 0.5 if mode == 'dilate' else float(kernel.sum()) - 0.5

    work = (volume.tensor > 0).float().unsqueeze(0)
    for _ in range(iterations):
        count = torch.nn.functional.conv3d(work, weight, groups=channels, padding=1)
        work = (count > threshold).float()

    return volume.new(work.squeeze(0).type(volume.dtype))


def connected_components(
    volume: vx.Volume,
    connectivity: int = 1,
    largest: bool = False) -> vx.Volume:
    """
    Label the connected components of the nonzero voxels in a volume.

    Components are labeled in order of descending size, so the largest
    component always has label 1. Channels are labeled independently.
    Labeling runs on the CPU and the result is moved back to the device.

    Args:
        connectivity (int, optional): Neighborhood connectivity: 1 includes
            face neighbors (6), 2 adds edge neighbors (18), and 3 adds corner
            neighbors (26).
        largest (bool, optional): If True, only the largest component is
            kept, producing a binary volume with label 1.

    Returns:
        Volume: Integer label map volume.
    """
    ndimage, structure = _binary_structure(connectivity)

    mask = (volume.tensor > 0).cpu().numpy()
    result = torch.zeros(volume.shape, dtype=torch.int32)
    for c in range(volume.num_channels):
        labeled, num = ndimage.label(mask[c], structure=structure)
        labels = torch.from_numpy(labeled).long()
        if num > 0:
            # remap labels so that they are sorted by descending voxel count
            counts = torch.bincount(labels.view(-1), minlength=num + 1)
            mapping = torch.zeros(num + 1, dtype=torch.int32)
            mapping[counts[1:].argsort(descending=True) + 1] = torch.arange(1, num + 1, dtype=torch.int32)
            labels = mapping[labels]
        result[c] = labels

    if largest:
        result = (result == 1).int()

    return volume.new(result.to(volume.device))


def flood_fill(
    volume: vx.Volume,
    point: torch.Tensor,
    space: vx.Space = 'voxel',
    connectivity: int = 1) -> vx.Volume:
    """
    Flood fill a volume from a seed point, extracting the connected region of
    voxels that share the seed's value. Both foreground and background regions
    can be filled. Filling runs on the CPU and the result is moved back to
    the device.

    Args:
        point (Tensor): A 3D seed coordinate.
        space (Space, optional): The coordinate space of the seed point.
        connectivity (int, optional): Neighborhood connectivity: 1 includes
            face neighbors (6), 2 adds edge neighbors (18), and 3 adds corner
            neighbors (26).

    Returns:
        Volume: Binary volume of the input data type marking the filled region.
    """
    if volume.num_channels != 1:
        raise ValueError(f'flood fill expects a single-channel volume, '
                         f'got {volume.num_channels} channels')
    ndimage, structure = _binary_structure(connectivity)

    point = torch.as_tensor(point, dtype=torch.float32, device=volume.device)
    if vx.Space(space) == 'world':
        point = volume.geometry.inverse().transform(point)
    point = tuple(point.round().long().cpu().tolist())
    if any(p < 0 or p >= s for p, s in zip(point, volume.baseshape)):
        raise ValueError(f'seed point {list(point)} is outside the '
                         f'volume extent {tuple(volume.baseshape)}')

    # label the region of voxels matching the seed value and extract the
    # component that contains the seed
    tensor = volume.tensor[0].cpu()
    matching = (tensor == tensor[point]).numpy()
    labeled, _ = ndimage.label(matching, structure=structure)
    region = torch.from_numpy(labeled == labeled[point])

    return volume.new(region.unsqueeze(0).to(volume.device).type(volume.dtype))


def fill_holes(volume: vx.Volume, connectivity: int = 1) -> vx.Volume:
    """
    Fill enclosed background cavities (holes) in the nonzero regions of a
    volume. Background connected to the grid border is not considered a hole.
    Channels are filled independently. Filling runs on the CPU and the result
    is moved back to the device.

    Args:
        connectivity (int, optional): Neighborhood connectivity: 1 includes
            face neighbors (6), 2 adds edge neighbors (18), and 3 adds corner
            neighbors (26).

    Returns:
        Volume: Binary volume of the input data type with holes filled.
    """
    ndimage, structure = _binary_structure(connectivity)

    mask = (volume.tensor > 0).cpu().numpy()
    result = torch.zeros(volume.shape, dtype=torch.bool)
    for c in range(volume.num_channels):
        result[c] = torch.from_numpy(ndimage.binary_fill_holes(mask[c], structure=structure))

    return volume.new(result.to(volume.device).type(volume.dtype))


def _binary_structure(connectivity: int) -> tuple:
    """
    Build a 3D binary structuring element for a neighborhood connectivity,
    lazily importing the scipy ndimage module.
    """
    if connectivity not in (1, 2, 3):
        raise ValueError(f'connectivity must be 1, 2, or 3, got {connectivity}')
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError('connected component morphology requires '
                          'the scipy package') from exc
    return ndimage, ndimage.generate_binary_structure(3, connectivity)
