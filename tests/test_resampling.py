import pytest
import torch
import voxel as vx


def test_resample(brain) -> None:

    # resampling to the same resolution should not change the underlying data
    assert brain.tensor is brain.resample_like(brain.geometry).tensor

    # simple test to make sure linear and nearest interpolation are different
    assert not vx.volumes_equal(brain.resample(2, mode='linear'),
                                brain.resample(2, mode='nearest').float())

    # shifted volume should be identical to a rolled volume
    resampled = brain.resample_like(brain.geometry.shift((1, 0, 0), 'voxel'))
    rolled = resampled.new(brain.tensor.roll(-1, 1))
    assert vx.volumes_equal(rolled[:, :-1], resampled[:, :-1])

    # 2x trilinear resampling should be (nearly) identical to average pooling
    resampled = brain.resample(2)
    pool = torch.nn.functional.avg_pool3d
    pooled = resampled.new(pool(brain.tensor.unsqueeze(0).float(), 2, 2).squeeze(0))
    assert vx.volumes_equal(resampled, pooled, vol_tol=3e-3)


def test_antialiasing(brain) -> None:

    # verify that antialiasing works as expected by assuming the result
    # is within a certain (manually computed) bound of the non-antialiased result.
    # use source and target volumes with different non-isotropic resolutions and orientations
    vol = brain.reorient('LIA').resample((0.8, 0.9, 0.8))
    target = vol.geometry.reorient('SPL').resample((1.5, 1.5, 5)).rotate((-10, 15, 10), 'world')
    aa = vol.resample_like(target, antialias=True)
    noaa = vol.resample_like(target, antialias=False)

    # this is a reasonable error bound for this specific case
    error = (aa - noaa).abs().quantile(0.99)
    assert error > 35 and error < 37

    # a larger sigma proportion should blur further, without moving the sampled grid
    errors = []
    for proportion in (0.25, 0.5, 1.0):
        scaled = vol.resample_like(target, antialias=proportion)
        assert scaled.baseshape == noaa.baseshape
        assert torch.allclose(scaled.geometry.tensor, noaa.geometry.tensor)
        errors.append((scaled - noaa).abs().quantile(0.99))
    assert errors[0] < error < errors[1] < errors[2]

    # make sure smoothing is not applied in situations where it should not be
    # i.e. when the downsampling factor is less than 2
    vol = brain.reorient('IRP').resample((1.8, 1.8, 2))
    target = vol.geometry.reorient('ALS').resample((2.5, 0.6, 0.6))
    noaa = vol.resample_like(target, antialias=False)
    for proportion in (True, 0.5, 1.0):
        assert vx.volumes_equal(vol.resample_like(target, antialias=proportion), noaa)


def test_resample_multichannel(brain) -> None:

    # channels are resampled independently: a stacked volume must match the
    # per-channel results of its single-channel components
    a = brain.float()
    b = a * 0.5
    target = a.geometry.resample(2).rotate((5, -5, 5), 'world')
    resampled = vx.volume.stack(a, b).resample_like(target)
    assert resampled.num_channels == 2
    assert vx.volumes_equal(resampled[[0]], a.resample_like(target), vol_tol=1e-4)
    assert vx.volumes_equal(resampled[[1]], b.resample_like(target), vol_tol=1e-4)

    # the out-of-bounds fill value applies to every channel
    target = target.shift((50, 0, 0), 'world')
    filled = vx.volume.stack(a, b).resample_like(target, padding_mode='fill', fill=-1)
    assert vx.volumes_equal(filled[[0]], a.resample_like(target, padding_mode='fill', fill=-1), vol_tol=1e-4)
    assert vx.volumes_equal(filled[[1]], b.resample_like(target, padding_mode='fill', fill=-1), vol_tol=1e-4)
    assert (filled.tensor == -1).any()


def test_resample_roundtrip(brain) -> None:

    # downsampling and resampling back only loses high-frequency detail, so
    # the interior should stay close to the original
    vol = brain.float()
    back = vol.resample(2).resample_like(vol.geometry)
    assert vx.geometries_equal(back.geometry, vol.geometry)
    inner = (slice(None), *(slice(40, -40),) * 3)
    error = (back.tensor[inner] - vol.tensor[inner]).abs().mean()
    assert error < 15  # mean abs error on a [0, 255] image


def test_world_point_sampling(brain) -> None:

    # linear sampling at exact grid points reproduces the voxel values
    vol = brain.float()
    voxels = torch.tensor([[50.0, 60, 70], [100, 120, 90], [128, 128, 128]])
    sampled = vol.sample(vol.geometry.map(voxels), space='world')
    expected = torch.stack([vol.tensor[:, int(v[0]), int(v[1]), int(v[2])] for v in voxels])
    assert sampled.shape == (3, 1)
    assert torch.allclose(sampled, expected, atol=5e-3)

    # nearest sampling of an integer volume preserves dtype and label values
    labels = (vol > 100).int()
    points = vol.geometry.map(torch.rand(100, 3) * 200 + 20)
    nearest = labels.sample(points, space='world', mode='nearest')
    assert nearest.dtype == labels.dtype
    assert bool(torch.isin(nearest.unique(), labels.unique()).all())


def test_point_sampling(brain) -> None:

    # make sure that sampling a volume at its own grid points and reshaping
    # the result back to the original shape gives the original volume
    vol = brain.float()
    points = vx.volume.volume_grid(vol.baseshape).view(-1, 3)
    sampled = vol.sample(points, space='voxel').swapaxes(0, 1)
    reshaped = vol.new(sampled.view(vol.shape))
    assert vx.volumes_equal(vol, reshaped, vol_tol=5e-3)


def test_resample_variadic(brain) -> None:

    # unpacked spacing components match the scalar and sequence forms
    resampled = brain.resample(2)
    assert vx.volumes_equal(brain.resample(2, 2, 2), resampled)
    assert vx.volumes_equal(brain.resample((2, 2, 2)), resampled)


# the triton kernel needs a cuda device, or the triton interpreter for cpu-only testing
_TRITON = vx.resampling.triton is not None and (torch.cuda.is_available() or vx.resampling._INTERPRET)
_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _voxel_to_voxel(source_shape, output_shape) -> vx.AffineMatrix:
    """
    A rotated, anisotropically scaled, and fractionally shifted mapping from
    output voxels to source voxels, roughly centered on the source volume.
    """
    rotation = vx.affine.angles_to_rotation_matrix(torch.tensor([13.7, -21.3, 8.9])).tensor[:3, :3]
    matrix = torch.eye(4)
    matrix[:3, :3] = rotation @ torch.diag(torch.tensor([1.13, 0.71, 1.37]))
    matrix[:3, 3] = torch.tensor(source_shape) / 2 - matrix[:3, :3] @ (torch.tensor(output_shape) / 2) \
                  + torch.tensor([0.31, -1.27, 0.53])
    return vx.AffineMatrix(matrix.to(_DEVICE))


@pytest.mark.skipif(not _TRITON, reason='requires triton on cuda, or TRITON_INTERPRET=1')
@pytest.mark.parametrize('mode', ['linear', 'nearest'])
@pytest.mark.parametrize('padding_mode', ['zeros', 'border', 'fill'])
def test_triton_resampling(mode, padding_mode) -> None:
    triton, torch_ = vx.resampling._resample_triton, vx.resampling._resample_torch
    shape, outshape = (10, 12, 14), (11, 9, 16)
    matrix = _voxel_to_voxel(shape, outshape)
    source = torch.rand(3, *shape, device=_DEVICE)

    # the kernel reproduces grid_sample up to float32 round-off in linear mode and exactly
    # in nearest mode (ties have measure zero for this matrix), for float and integer sources
    for tensor in (source, (source * 255).to(torch.uint8)):
        a = triton(tensor, outshape, matrix, mode, padding_mode, -7.0)
        b = torch_(tensor, outshape, matrix, mode, padding_mode, -7.0)
        assert a.dtype == b.dtype == torch.float32
        assert torch.allclose(a, b, atol=(1e-5 * b.abs().max().item() if mode == 'linear' else 0), rtol=0)
    if padding_mode == 'fill':
        assert (a == -7).any() and not (a == -7).all()

    # gradients with respect to the source and the matrix match the grid_sample path,
    # for every combination of which inputs require gradients
    weights = torch.randn(3, *outshape, device=_DEVICE)
    for need_source, need_matrix in ((True, True), (True, False), (False, True)):
        grads = []
        for func in (triton, torch_):
            s = source.clone().requires_grad_(need_source)
            m = vx.AffineMatrix(matrix.tensor.clone().requires_grad_(need_matrix))
            (func(s, outshape, m, mode, padding_mode, -7.0) * weights).sum().backward()
            grads.append((s.grad, m.tensor.grad))
        (sa, ma), (sb, mb) = grads
        if need_source:
            assert torch.allclose(sa, sb, atol=1e-5 * sb.abs().max().item(), rtol=0)
        if need_matrix and mode == 'linear':
            assert torch.allclose(ma, mb, atol=1e-3 * mb.abs().max().item(), rtol=0)
        elif need_matrix:
            # nearest interpolation has no coordinate gradient (grid_sample gives zeros)
            assert ma is None and not mb.any()


@pytest.mark.skipif(not torch.cuda.is_available() or vx.resampling.triton is None, reason='requires cuda')
def test_triton_no_shape_recompile() -> None:
    kernel = vx.resampling._resample_forward
    compiled = lambda: sum(len(entry[0]) for entry in kernel.device_caches.values())
    args = dict(mode='linear', padding_mode='zeros', fill=0.0)
    vx.resampling._resample_triton(torch.rand(1, 8, 8, 8, device='cuda'), (8, 8, 8), _voxel_to_voxel((8, 8, 8), (8, 8, 8)), **args)
    count = compiled()
    for shape, outshape in (((1, 16, 32, 48), (7, 1, 9)), ((3, 5, 7, 9), (32, 16, 64)), ((16, 1, 1, 17), (1, 1, 1))):
        source = torch.rand(*shape, device='cuda')
        vx.resampling._resample_triton(source, outshape, _voxel_to_voxel(shape[1:], outshape), **args)
    assert compiled() == count
