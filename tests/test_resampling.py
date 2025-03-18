import torch
import voxel as vx

from . import utility


# TODO: add a mult-frame case to all of these tests


def test_resample():
    
    # resampling to the same resolution should not change the underlying data
    vol = utility.brain_t1w()
    assert vol.tensor is vol.resample_like(vol.geometry).tensor

    # simple test to make sure linear and nearest interpolation are different
    assert not vx.volumes_equal(vol.resample(2, mode='linear'),
                                vol.resample(2, mode='nearest').float())

    # shifted volume should be identical to a rolled volume
    resampled = vol.resample_like(vol.geometry.shift((1, 0, 0), 'voxel'))
    rolled = resampled.new(vol.tensor.roll(-1, 1))
    assert vx.volumes_equal(rolled[:, :-1], resampled[:, :-1])

    # 2x trilinear resampling should be (nearly) identical to average pooling
    resampled = vol.resample(2)
    pool = torch.nn.functional.avg_pool3d
    pooled = resampled.new(pool(vol.tensor.unsqueeze(0).float(), 2, 2).squeeze(0))
    assert vx.volumes_equal(resampled, pooled, vol_tol=3e-3)


def test_antialiasing():

    # verify that antialiasing works as expected by assuming the result
    # is within a certain (manually computed) bound of the non-antialiased result.
    # use source and target volumes with different non-isotropic resolutions and orientations
    vol = utility.brain_t1w().reorient('LIA').resample((0.8, 0.9, 0.8))
    target = vol.geometry.reorient('SPL').resample((1.5, 1.5, 5)).rotate((-10, 15, 10), 'world')
    aa = vol.resample_like(target, antialias=True)
    noaa = vol.resample_like(target, antialias=False)
    error = (aa - noaa).abs().quantile(0.99)
    assert error > 35 and error < 36

    # make sure smoothing is not applied in situations where it should not be
    # i.e. when the downsampling factor is less than 2
    vol = utility.brain_t1w().reorient('IRP').resample((1.8, 1.8, 2))
    target = vol.geometry.reorient('ALS').resample((2.5, 0.6, 0.6))
    aa = vol.resample_like(target, antialias=True)
    noaa = vol.resample_like(target, antialias=False)
    assert vx.volumes_equal(aa, noaa)


def test_point_sampling():

    # make sure that sampling a volume at its own grid points and reshaping
    # the result back to the original shape gives the original volume
    vol = utility.brain_t1w()
    points = vx.volume.volume_grid(vol.baseshape).view(-1, 3)
    sampled = vol.sample(points, space='voxel', mode='nearest').swapaxes(0, 1)
    reshaped = vol.new(sampled.view(vol.shape))
    assert (reshaped == vol).all()
