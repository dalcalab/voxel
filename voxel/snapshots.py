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
