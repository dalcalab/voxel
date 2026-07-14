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
    assert error > 34 and error < 36

    # make sure smoothing is not applied in situations where it should not be
    # i.e. when the downsampling factor is less than 2
    vol = brain.reorient('IRP').resample((1.8, 1.8, 2))
    target = vol.geometry.reorient('ALS').resample((2.5, 0.6, 0.6))
    aa = vol.resample_like(target, antialias=True)
    noaa = vol.resample_like(target, antialias=False)
    assert vx.volumes_equal(aa, noaa)


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
    sampled = vol.sample(vol.geometry.transform(voxels), space='world')
    expected = torch.stack([vol.tensor[:, int(v[0]), int(v[1]), int(v[2])] for v in voxels])
    assert sampled.shape == (3, 1)
    assert torch.allclose(sampled, expected, atol=5e-3)

    # nearest sampling of an integer volume preserves dtype and label values
    labels = (vol > 100).int()
    points = vol.geometry.transform(torch.rand(100, 3) * 200 + 20)
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
