from __future__ import annotations

import torch
import voxel as vx


# view name (or abbreviation) to the voxel orientation that places the
# through-plane (view) direction on the first axis. the remaining two axes are
# ordered so the rendered 2D slice reads anatomically, e.g. for an axial view
# the rows run anterior -> posterior and the columns run right -> left.
VIEWS = {
    'axial': 'SPL', 'a': 'SPL',
    'coronal': 'AIL', 'c': 'AIL',
    'sagittal': 'LIA', 's': 'LIA',
}

# a categorical palette (RGB in [0, 1]) used to color label overlays when the
# caller does not supply explicit colors
PALETTE = torch.tensor([
    [0.89, 0.10, 0.11],  # red
    [0.22, 0.49, 0.72],  # blue
    [0.30, 0.69, 0.29],  # green
    [0.60, 0.31, 0.64],  # purple
    [1.00, 0.50, 0.00],  # orange
    [0.95, 0.90, 0.20],  # yellow
    [0.12, 0.70, 0.67],  # teal
    [0.97, 0.51, 0.75],  # pink
    [0.65, 0.34, 0.16],  # brown
    [0.60, 0.60, 0.60],  # gray
])

# world-space unit directions of the anatomical axis letters, used to specify
# projection viewpoints and up directions
DIRECTIONS = {
    'R': (1, 0, 0), 'L': (-1, 0, 0),
    'A': (0, 1, 0), 'P': (0, -1, 0),
    'S': (0, 0, 1), 'I': (0, 0, -1),
}


@torch.no_grad()
def snapshot(
    volume: vx.Volume | list[vx.Volume] = None,
    label: vx.Volume | list[vx.Volume] = None,
    view: str = 'axial',
    num_slices: int = 1,
    coord: torch.Tensor | None = None,
    res: int = 256,
    square: bool = False,
    resample: str = 'nearest',
    label_colors: torch.Tensor | list | None = None,
    alpha: float = 0.5,
    outline: bool = False,
    pool: int = 4,
    ) -> torch.Tensor | list[torch.Tensor]:
    """
    Render a volume (or stack of overlaid volumes) into one or more 2D RGB
    snapshot images, optionally blending label masks on top.

    Slices are taken along the view direction and returned as channels-last
    $(H, W, 3)$ uint8 image tensors: a single tensor for one slice, else a list.

    Args:
        volume (Volume or list[Volume]): Volume(s) to render, each with 1
            (grayscale) or 3 (RGB) channels. The first defines the base geometry;
            the rest are composited on top wherever they have in-bounds data.
            Grayscale channels are contrast-normalized to $[0, 1]$. RGB values
            must already lie within $[0, 1]$.
        label (Volume or list[Volume], optional): Label overlay(s). A volume
            with values greater than one is treated as a discrete labelmap:
            values are conformed to integers, each unique nonzero value becomes
            a class, resampled with nearest-neighbor interpolation and colored
            from the volume's `labels` lookup when it defines a color. A volume
            within $[0, 1]$ holds soft mask(s), blending softly for fractional
            values, with each channel a separate class.
        view (str, optional): View plane: 'axial', 'coronal', or 'sagittal' (or
            their first letters). Defaults to 'axial'.
        num_slices (int, optional): Number of evenly spaced slices along the view
            direction. Ignored when `coord` is given. Defaults to 1.
        coord (Tensor, optional): World-space (x, y, z) coordinate. If given, a
            single slice through this point is rendered and `num_slices` ignored.
        res (int, optional): Pixel height of the rendered images. Slices are
            resampled to the isotropic in-plane spacing that yields this height,
            preserving the physical aspect ratio. Defaults to 256.
        square (bool, optional): If True, center-crop or pad the image width to
            match its height, yielding square `(res, res)` images. Defaults to False.
        resample (str, optional): Interpolation mode used when resampling volumes
            and labels onto the snapshot grid, either 'linear' or 'nearest'.
            Defaults to 'nearest'.
        label_colors (Tensor or list, optional): RGB color(s) in $[0, 1]$ for the
            labels, cycled to match the number of label classes. Overrides any
            lookup-defined colors. Defaults to a categorical palette.
        alpha (float, optional): Opacity of the label overlays. Defaults to 0.5.
        outline (bool, optional): If True, draw a fully opaque one-pixel outline
            just inside the boundary of each label in its color, unaffected by
            `alpha`. Defaults to False.
        pool (int, optional): Pooling window used to make grayscale normalization
            robust to outlier voxels. Set to 1 or None to disable. Defaults to 4.

    Returns:
        Tensor or list[Tensor]: A single $(H, W, 3)$ uint8 RGB image, or a
            list of them.
    """
    if volume is None:
        raise ValueError('must provide at least one volume to snapshot')

    # normalize the volume argument into a list of Volumes, wrapping raw tensors
    if isinstance(volume, (vx.Volume, torch.Tensor)):
        volume = [volume]
    volumes = [v if isinstance(v, vx.Volume) else vx.Volume(v) for v in volume]

    view = str(view).lower()
    if view not in VIEWS:
        raise ValueError(f'unknown view \'{view}\', expected one of axial, coronal, or sagittal')

    if res < 1:
        raise ValueError(f'res must be positive, got {res}')

    # reorient the base geometry so the view (through-plane) direction is axis 0
    geometry = volumes[0].geometry.reorient(VIEWS[view])
    spacing = geometry.spacing
    baseshape = torch.tensor(geometry.baseshape, device=geometry.device)

    if coord is not None:
        # render a single slice at the plane passing through the world coordinate
        coord = torch.as_tensor(coord, dtype=torch.float32, device=geometry.device)
        index = int(geometry.inverse().map(coord)[0].round().clamp(0, baseshape[0] - 1))
        target = geometry.shift([index, 0, 0], space='voxel')
        target = target.reshape((1, *geometry.baseshape[1:]), from_origin=True)
    else:
        # resample the view axis so it holds num_slices + 2 evenly spaced slices,
        # then trim the outer two to avoid the (often empty) extremes
        if num_slices < 1:
            raise ValueError(f'num_slices must be positive, got {num_slices}')
        target_spacing = spacing.clone()
        target_spacing[0] = spacing[0] * baseshape[0] / (num_slices + 2)
        target = geometry.resample(spacing=target_spacing)
        target = target.trim((1, 0, 0), space='voxel')

    # resample the in-plane axes to the isotropic spacing that yields an image
    # height of res, preserving the physical aspect ratio of the slice, then
    # lock the exact shape (absorbing rounding in the resampled grid extent)
    span = target.spacing[1] * target.baseshape[1]
    if square:
        span = max(span, target.spacing[2] * target.baseshape[2])
    in_plane = span / res
    target = target.resample(target.spacing[0], in_plane, in_plane)
    shape = list(target.baseshape)
    target = target.reshape(shape[0], res, res if square else shape[2])

    # composite the base image, as RGB (3, S, H, W), across all input volumes.
    # later volumes overlay the earlier ones wherever they carry valid data
    image = None
    for i, vol in enumerate(volumes):
        if vol.num_channels not in (1, 3):
            raise ValueError(f'snapshot volumes must have 1 (grayscale) or 3 (RGB) '
                             f'channels, got {vol.num_channels}')
        resampled = vol.resample_like(target, mode=resample)

        if vol.num_channels == 1:
            # rescale a grayscale volume to [0, 1], deriving the bounds from a
            # pooled copy so isolated outlier voxels do not dominate the contrast.
            # nearest resampling preserves integer dtypes, so cast to float first
            resampled = resampled.float()
            # TODO: make this a utility function
            reference = resampled.pool(pool) if pool and pool > 1 else resampled
            lower = reference.min()
            upper = reference.max()
            resampled = ((resampled.tensor - lower) / (upper - lower + 1e-6)).clamp(0, 1)
            resampled = resampled.repeat(3, 1, 1, 1)
        else:
            # an RGB volume is shown as-is and must already lie within [0, 1]
            if vol.tensor.min() < 0 or vol.tensor.max() > 1:
                raise ValueError('RGB (3-channel) snapshot volumes must have '
                                 'values within [0, 1]')
            resampled = resampled.tensor.float()

        if i == 0:
            image = resampled
        else:
            foreground = vol.ones_like().resample_like(target, mode=resample).tensor > 0.99
            image = torch.where(foreground, resampled, image)

    # gather the label masks along with an optional preassigned color per mask
    masks = []
    mask_colors = []
    if label is not None:
        if isinstance(label, (vx.Volume, torch.Tensor)):
            label = [label]
        for lab in label:
            if lab.tensor.max() > 1:
                # values beyond one indicate a discrete labelmap: resample with
                # nearest to preserve values, conform to integer labels, and
                # split each unique nonzero value into its own binary mask,
                # colored by the label lookup
                resampled = lab.resample_like(target, mode='nearest')
                lut = resampled.labels
                tensor = resampled.tensor.round().int()
                for c in range(resampled.num_channels):
                    channel = tensor[c:c + 1]
                    for value in channel.unique().tolist():
                        if value == 0:
                            continue
                        masks.append((channel == value).float())
                        entry = lut.get(value) if lut is not None else None
                        mask_colors.append(None if entry is None else entry.color)
            else:
                # a volume within [0, 1] holds soft (potentially probabilistic)
                # masks, one class per channel
                resampled = lab.resample_like(target, mode=resample).tensor.clamp(0, 1)
                masks.extend(resampled[c:c + 1] for c in range(resampled.shape[0]))
                mask_colors.extend([None] * resampled.shape[0])

    # blend the masks on top using their assigned colors. explicit label_colors
    # override any lookup colors, and the palette is cycled for the rest
    if masks:
        if label_colors is not None:
            palette = torch.as_tensor(label_colors, dtype=torch.float32)
            if palette.ndim == 1:
                palette = palette.unsqueeze(0)
            colors = palette[torch.arange(len(masks)) % palette.shape[0]]
        else:
            colors = torch.stack([PALETTE[i % PALETTE.shape[0]] if c is None else c.cpu()
                                  for i, c in enumerate(mask_colors)])
        colors = colors.to(image.device)
        for mask, color in zip(masks, colors):
            blend = mask * float(alpha)
            image = image * (1 - blend) + color.view(3, 1, 1, 1) * blend
            if outline:
                # the outline is the in-plane erosion shell of the binarized
                # mask, drawn fully opaque. eroding keeps the outline inside the
                # label, so adjacent label outlines never overlap, and labels
                # touching the image edge draw no outline along the border
                binary = (mask > 0.5).float()
                eroded = 1 - torch.nn.functional.max_pool2d(1 - binary, 3, stride=1, padding=1)
                image = torch.where((binary - eroded) > 0.5, color.view(3, 1, 1, 1), image)

    # quantize to 8-bit and split the stack into per-slice 2D images
    image = (image.clamp(0, 1).detach() * 255).round().to(torch.uint8)
    slices = list(image.movedim(0, -1).unbind(dim=0))
    return slices[0] if len(slices) == 1 else slices


@torch.no_grad()
def projection(
    volume: vx.Volume,
    mode: str,
    exclude: vx.Volume | None = None,
    center: torch.Tensor | None = None,
    viewpoint: str | torch.Tensor = 'A',
    up: str | torch.Tensor | None = None,
    viewport: float | None = None,
    depth: float | None = None,
    res: int = 256,
    ) -> torch.Tensor:
    """
    Project a volume onto a 2D image plane along a view direction,
    for example to generate a maximum intensity projection (MIP).

    Args:
        volume (Volume): Volume to project.
        mode (str, optional): Reduction applied along each ray, either 'max',
            'min', or 'mean'. Defaults to 'max'.
        exclude (Volume, optional): Mask whose nonzero voxels are ignored by the
            projection. A raw tensor is assumed to share the volume geometry.
        center (Tensor, optional): World-space (x, y, z) coordinate at the center
            of the image. Defaults to the center of the volume geometry.
        viewpoint (str or Tensor, optional): World-space direction of the camera
            relative to the center, either an anatomical axis letter ('R', 'L',
            'A', 'P', 'S', or 'I') or a vector. Defaults to 'A', which views
            from anterior.
        up (str or Tensor, optional): World-space direction pointing to the top
            of the image, as an axis letter or a vector. It is made orthogonal
            to the view direction, so it only needs to be roughly upward. If
            None, it is superior, or anterior when viewing along the
            superior-inferior axis.
        viewport (float, optional): Width and height of the square image plane in
            world units. Defaults to the extent of the volume from the viewpoint.
        depth (float, optional): Thickness of the projected slab in world units,
            centered at `center` along the view direction. Defaults to the full
            extent of the volume.
        res (int, optional): Pixel height and width of the image. Defaults to 256.

    Returns:
        Tensor: A $(H, W, C)$ float image of projected intensities.
    """
    if isinstance(volume, torch.Tensor):
        volume = vx.Volume(volume)

    if mode not in ('max', 'min', 'mean'):
        raise ValueError(f'unknown projection mode \'{mode}\', expected max, min, or mean')
    if res < 1:
        raise ValueError(f'res must be positive, got {res}')
    if viewport is not None and viewport <= 0:
        raise ValueError(f'viewport must be positive, got {viewport}')
    if depth is not None and depth <= 0:
        raise ValueError(f'depth must be positive, got {depth}')

    geometry = volume.geometry
    device = geometry.device

    if center is None:
        center = geometry.center
    center = torch.as_tensor(center, dtype=torch.float32, device=device)

    # build the camera basis. the forward direction points from the camera into
    # the scene, and up is superior by default unless the view runs along that
    # axis. the up direction is made orthogonal to the forward direction
    forward = -_unit_direction(viewpoint, 'viewpoint', device)
    if up is None:
        up = 'A' if forward[2].abs() > 0.99 else 'S'
    up = _unit_direction(up, 'up', device)
    up = up - forward.dot(up) * forward
    if up.norm() < 1e-3:
        raise ValueError('up direction must not be parallel to the viewpoint')
    up = up / up.norm()
    right = torch.cross(forward, up, dim=0)

    # measure the extent of the volume along each camera axis
    corners = geometry.bounds().corner_points() - center
    distance = corners @ forward
    near, far = distance.min(), distance.max()
    if viewport is None:
        rows = corners @ up
        cols = corners @ right
        viewport = max(float(rows.max() - rows.min()), float(cols.max() - cols.min()))

    # build a view-aligned grid with depth along the first axis, sampled at the
    # finest input spacing, and image rows and columns running down and right.
    # the grid spans the volume along the depth axis, optionally restricted to
    # a slab around the center, with a small tolerance so float error does not
    # add an extra sample
    pixel = viewport / res
    step = geometry.spacing.min()
    if depth is not None:
        near = near.clamp(min=-depth / 2)
        far = far.clamp(max=depth / 2)
    num = int(((far - near) / step - 1e-3).ceil().clamp(min=1))
    matrix = torch.eye(4, device=device)
    matrix[:3, :3] = torch.stack((forward * step, -up * pixel, right * pixel), dim=1)
    target = vx.AcquisitionGeometry((num, res, res), matrix)
    target = target.shift_to_point(center + forward * (near + far) / 2)

    # out-of-bounds and excluded samples take a fill value that is neutral to
    # the reduction, so rays that miss the volume read as background
    fill = float({'max': volume.min(), 'min': volume.max(), 'mean': 0}[mode])
    reduce, combine = {
        'max': (torch.amax, torch.maximum),
        'min': (torch.amin, torch.minimum),
        'mean': (torch.sum, torch.add),
    }[mode]

    if exclude is not None:
        if isinstance(exclude, torch.Tensor):
            exclude = vx.Volume(exclude, geometry)
        exclude = (exclude != 0).float()

    # the mean is normalized by the number of in-bounds, non-excluded samples
    # along each ray, tracked by resampling a mask of ones
    ones = geometry.ones_like() if mode == 'mean' else None

    # number of grid samples resampled at once when computing projections. the
    # view-aligned grid is processed in depth slabs of this size to bound memory
    slab_samples = 2 ** 24

    # resample and reduce the grid in depth slabs to bound peak memory
    image = torch.full((volume.num_channels, res, res), fill, device=device)
    count = torch.zeros((1, res, res), device=device)
    slab = max(1, slab_samples // (res * res))
    for start in range(0, num, slab):
        chunk = target.shift((start, 0, 0), space='voxel')
        chunk = chunk.reshape((min(slab, num - start), res, res), from_origin=True)
        values = volume.resample_like(chunk, padding_mode='fill', fill=fill).tensor
        if exclude is not None:
            excluded = exclude.resample_like(chunk).tensor > 0.5
            values = values.masked_fill(excluded, fill)
        image = combine(image, reduce(values, 1))
        if ones is not None:
            include = ones.resample_like(chunk, padding_mode='fill', fill=0).tensor
            if exclude is not None:
                include = include * ~excluded
            count = count + include.sum(1)

    if mode == 'mean':
        image = image / count.clamp(min=1e-6)

    return image.movedim(0, -1)


@torch.no_grad()
def sliding_projection(
    volume: vx.Volume,
    depth: float,
    mode: str,
    dim: int | str | None = None,
    ) -> vx.Volume:
    """
    Compute a sliding window projection along a voxel axis, for example a
    sliding maximum intensity projection (MIP).

    Each slice is replaced by the reduction over a window of neighboring slices
    centered on it, so the output matches the input geometry. Windows are
    truncated at the edges of the volume.

    Args:
        volume (Volume): Volume to project.
        depth (float): Thickness of the window in world units, rounded to the
            nearest whole number of slices (at least one).
        mode (str, optional): Reduction applied over each window, either 'max',
            'min', or 'mean'. Defaults to 'max'.
        dim (int or str, optional): Voxel axis to slide along, either an index or
            a view name ('axial', 'coronal', or 'sagittal', or their first
            letters) selecting the voxel axis closest to that view direction.
            Defaults to the slice direction of the geometry.

    Returns:
        Volume: Projected volume with the same geometry as the input.
    """
    if isinstance(volume, torch.Tensor):
        volume = vx.Volume(volume)

    if mode not in ('max', 'min', 'mean'):
        raise ValueError(f'unknown projection mode \'{mode}\', expected max, min, or mean')
    if depth <= 0:
        raise ValueError(f'depth must be positive, got {depth}')

    geometry = volume.geometry
    if dim is None:
        dim = int(geometry.slice_direction)
    elif isinstance(dim, str):
        # the first letter of the view orientation names its through-plane
        # direction, which is matched (either sign) against the voxel axes
        view = dim.lower()
        if view not in VIEWS:
            raise ValueError(f'unknown view \'{dim}\', expected one of axial, coronal, or sagittal')
        letter = VIEWS[view][0]
        opposite = dict(S='I', A='P', L='R')[letter]
        dim = next(i for i, c in enumerate(geometry.orientation.name) if c in (letter, opposite))
    if dim not in (0, 1, 2):
        raise ValueError(f'dim must be a spatial axis 0, 1, or 2, got {dim}')

    # the window covers the slices at offsets [lower, upper] around each slice,
    # with the extra slice of an even window falling on the upper side
    num = max(1, round(depth / float(geometry.spacing[dim])))
    lower, upper = -((num - 1) // 2), num // 2

    tensor = volume.tensor.float() if mode == 'mean' else volume.tensor
    axis = dim + 1
    length = tensor.shape[axis]
    combine = {'max': torch.maximum, 'min': torch.minimum, 'mean': torch.add}[mode]

    # accumulate the window in place, combining the output with a copy of the
    # input shifted by each offset over the range where the two overlap. this
    # avoids materializing a stack of shifted copies
    result = tensor.clone()
    count = torch.ones(length, device=tensor.device)
    for offset in range(lower, upper + 1):
        size = length - abs(offset)
        if offset == 0 or size <= 0:
            continue
        start = max(0, -offset)
        target = result.narrow(axis, start, size)
        combine(target, tensor.narrow(axis, max(0, offset), size), out=target)
        count[start:start + size] += 1

    # the mean divides by the number of slices in each (possibly truncated) window
    if mode == 'mean':
        shape = [1] * result.ndim
        shape[axis] = length
        result = result / count.view(shape)

    return volume.new(result)


def _unit_direction(direction: str | torch.Tensor, name: str, device: torch.device) -> torch.Tensor:
    """
    Convert an anatomical axis letter or a vector into a world-space unit vector.
    """
    if isinstance(direction, str):
        if direction.upper() not in DIRECTIONS:
            raise ValueError(f'unknown {name} direction \'{direction}\', expected one of R, L, A, P, S, or I')
        direction = DIRECTIONS[direction.upper()]
    direction = torch.as_tensor(direction, dtype=torch.float32, device=device)
    if direction.shape != (3,) or direction.norm() == 0:
        raise ValueError(f'{name} must be a nonzero 3D vector')
    return direction / direction.norm()
