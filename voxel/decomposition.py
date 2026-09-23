from __future__ import annotations

import torch
import voxel as vx


def pca(
    volumes: vx.Volume | list[vx.Volume],
    n_components: int = 3,
    mask: vx.Volume | list[vx.Volume] | None = None,
    center: bool = True,
    standardize: bool = False,
    whiten: bool = False,
    normalize: str | None = 'quantile',
    quantile: float = 0.01,
    return_basis: bool = False,
    ) -> vx.Volume | list[vx.Volume] | tuple:
    """
    Project the feature channels of one or more volumes onto their principal
    components using PCA.

    Given a list of volumes, a single shared basis is fit across all of them so
    the resulting colormaps are comparable.

    Args:
        volumes (Volume or list[Volume]): Volume(s) to project.
        n_components (int, optional): Number of components to keep, i.e. the
            output channel count. Defaults to 3 (RGB).
        mask (Volume or list[Volume], optional): Foreground mask(s) restricting
            which voxels are used to fit the basis and normalization. The basis
            is still applied to every voxel. A single mask is applied to all
            inputs, a list must match the number of inputs.
        center (bool, optional): Subtract the per-channel mean before fitting.
        standardize (bool, optional): Scale each input channel to unit variance
            before fitting (correlation-based PCA).
        whiten (bool, optional): Scale each output component to unit variance,
            balancing their contribution to the colormap.
        normalize (str, optional): Per-component output normalization: 'minmax'
            (rescale to $[0, 1]$), 'quantile' (robust rescaling that clips the
            `quantile` tails), or None. Stats are computed from foreground voxels.
            Defaults to 'quantile'.
        quantile (float, optional): Tail fraction clipped when using 'quantile`
            normalization. Defaults to 0.01.
        return_basis (bool, optional): If True, also return the fit basis dict.

    Returns:
        Volume or list[Volume]: The projected volume(s) with `n_components`
            channels, matching the single-vs-list structure of the input. If
            `return_basis` is True, an `(output, basis)` tuple is returned instead.
    """
    single = isinstance(volumes, vx.Volume)
    volumes = [volumes] if single else list(volumes)
    if len(volumes) == 0:
        raise ValueError('no input volumes provided')

    # all volumes must share the same feature dimensionality
    channels = volumes[0].num_channels
    if any(v.num_channels != channels for v in volumes):
        counts = [v.num_channels for v in volumes]
        raise ValueError(f'all input volumes must have a matching number of '
                         f'channels, got {counts}')
    if n_components > channels:
        raise ValueError(f'n_components ({n_components}) cannot exceed the number '
                         f'of input channels ({channels})')

    # resolve the mask argument into one entry per volume
    if mask is None:
        masks = [None] * len(volumes)
    elif isinstance(mask, vx.Volume):
        masks = [mask] * len(volumes)
    else:
        masks = list(mask)
        if len(masks) != len(volumes):
            raise ValueError(f'expected one mask per volume, got {len(masks)} '
                             f'masks for {len(volumes)} volumes')

    # gather the foreground feature vectors (N, C) used to fit the basis. masks
    # are resampled onto each volume grid, which is a cheap no-op when the
    # geometries already match (see Volume.resample_like)
    features = [v.tensor.reshape(channels, -1).movedim(0, 1).float() for v in volumes]
    foreground = [
        None if m is None else m.resample_like(v, mode='nearest').tensor[:1].reshape(-1).bool()
        for m, v in zip(masks, volumes)
    ]
    fit_features = torch.cat([f if m is None else f[m] for f, m in zip(features, foreground)], dim=0)
    if fit_features.shape[0] < n_components:
        raise ValueError(f'not enough foreground voxels ({fit_features.shape[0]}) '
                         f'to fit {n_components} components')

    # fit the PCA basis on the foreground features
    mean = fit_features.mean(dim=0) if center else fit_features.new_zeros(channels)
    scale = fit_features.std(dim=0).clamp(min=1e-6) if standardize else fit_features.new_ones(channels)
    normalized = (fit_features - mean) / scale

    # PCA via exact SVD; the rows of vh are the principal directions
    _, s, vh = torch.linalg.svd(normalized, full_matrices=False)
    components = vh[:n_components].mT

    # resolve the SVD sign ambiguity deterministically so that, for each
    # component, the entry of largest magnitude is positive
    peak = components[components.abs().argmax(dim=0), torch.arange(n_components, device=components.device)]
    components = components * torch.where(peak < 0, -1.0, 1.0)

    # per-component standard deviation used for optional whitening
    component_std = (s[:n_components] / max(fit_features.shape[0] - 1, 1) ** 0.5).clamp(min=1e-6)

    def project(feats: torch.Tensor) -> torch.Tensor:
        projected = ((feats - mean) / scale) @ components
        return projected / component_std if whiten else projected

    # compute per-component normalization bounds from the foreground projections
    lower = upper = None
    if normalize == 'minmax':
        fit_projected = project(fit_features)
        lower, upper = fit_projected.amin(dim=0), fit_projected.amax(dim=0)
    elif normalize == 'quantile':
        if quantile <= 0 or quantile >= 0.5:
            raise ValueError(f'quantile must be in the range (0, 0.5), got {quantile}')
        # quantile caps the reduced dimension size, so strided-subsample if needed
        fit_projected = project(fit_features)[:: fit_features.shape[0] // 2 ** 24 + 1]
        qs = torch.tensor([quantile, 1.0 - quantile], device=fit_projected.device)
        bounds = torch.quantile(fit_projected, qs, dim=0)
        lower, upper = bounds[0], bounds[1]
    elif normalize is not None:
        raise ValueError(f'unknown normalization mode \'{normalize}\'')

    # project every voxel of each volume and rescale to the output range
    outputs = []
    for v, feats in zip(volumes, features):
        projected = project(feats)
        if normalize is not None:
            projected = ((projected - lower) / (upper - lower).clamp(min=1e-6)).clamp(0, 1)
        tensor = projected.movedim(1, 0).reshape(n_components, *v.baseshape)
        outputs.append(v.new(tensor))

    output = outputs[0] if single else outputs
    if return_basis:
        basis = {
            'mean': mean,
            'scale': scale,
            'components': components,
            'component_std': component_std,
            'lower': lower,
            'upper': upper,
        }
        return output, basis
    return output
