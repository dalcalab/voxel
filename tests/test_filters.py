import pytest
import torch
import voxel as vx

from conftest import nontrivial_geometry


def delta_volume(baseshape: tuple = (11, 13, 15)) -> vx.Volume:
    """
    Volume with a unit impulse at the grid center and a 1mm-isotropic geometry.
    """
    volume = vx.AcquisitionGeometry(baseshape).zeros_like()
    center = [s // 2 for s in baseshape]
    volume.tensor[0, center[0], center[1], center[2]] = 1
    return volume


def test_gaussian_kernel_1d() -> None:
    for sigma, truncate in ((1.0, 2.0), (2.5, 3.0)):
        kernel = vx.filters.gaussian_kernel_1d(sigma, truncate)
        assert len(kernel) == 2 * int(truncate * sigma + 0.5) + 1
        assert torch.allclose(kernel.sum(), torch.tensor(1.0))
        assert torch.equal(kernel, kernel.flip(0))
        assert kernel.argmax() == len(kernel) // 2

    # a zero sigma yields the length-1 identity kernel
    assert torch.equal(vx.filters.gaussian_kernel_1d(0), torch.ones(1))


@pytest.mark.parametrize('separable', [True, False])
def test_gaussian_delta_response(separable) -> None:

    # the impulse response of the filter is the kernel itself
    volume = delta_volume()
    filtered = vx.filters.gaussian_filter(volume, 1.0, 'voxel', separable=separable)
    kernel = vx.filters.gaussian_kernel_1d(1.0)
    expected = kernel[:, None, None] * kernel[None, :, None] * kernel[None, None, :]
    r = len(kernel) // 2
    center = [s // 2 for s in volume.baseshape]
    block = filtered.tensor[0][tuple(slice(c - r, c + r + 1) for c in center)]
    assert torch.allclose(block, expected, atol=1e-6)
    assert torch.allclose(filtered.tensor.sum(), torch.tensor(1.0), atol=1e-5)


@pytest.mark.parametrize('padding_mode', ['replicate', 'zeros'])
def test_gaussian_separable_equals_dense(multichannel_volume, padding_mode) -> None:
    separable = vx.filters.gaussian_filter(multichannel_volume, 1.5, 'voxel',
                                           separable=True, padding_mode=padding_mode)
    dense = vx.filters.gaussian_filter(multichannel_volume, 1.5, 'voxel',
                                       separable=False, padding_mode=padding_mode)
    assert torch.allclose(separable.tensor, dense.tensor, atol=1e-5)

    # channels are filtered independently
    single = multichannel_volume[[0]]
    filtered = vx.filters.gaussian_filter(single, 1.5, 'voxel', padding_mode=padding_mode)
    assert torch.allclose(separable.tensor[:1], filtered.tensor, atol=1e-6)


@pytest.mark.parametrize('padding_mode', ['replicate', 'zeros'])
def test_box_separable_equals_dense(multichannel_volume, padding_mode) -> None:
    separable = vx.filters.box_filter(multichannel_volume, 3, 'voxel',
                                      separable=True, padding_mode=padding_mode)
    dense = vx.filters.box_filter(multichannel_volume, 3, 'voxel',
                                  separable=False, padding_mode=padding_mode)
    assert torch.allclose(separable.tensor, dense.tensor, atol=1e-5)


def test_box_uniform_identity() -> None:

    # mean filtering a constant volume (with replicate padding) is an identity
    volume = nontrivial_geometry().full_like(2.5)
    filtered = vx.filters.box_filter(volume, 3, 'voxel')
    assert torch.allclose(filtered.tensor, volume.tensor.float(), atol=1e-6)
    assert filtered.geometry is volume.geometry


def test_box_kernel_size_rounding() -> None:

    # box extents round to the nearest odd voxel count: a size of 4 spreads
    # an impulse over a 5-voxel support along each axis
    volume = delta_volume()
    filtered = vx.filters.box_filter(volume, 4, 'voxel')
    assert int((filtered.tensor[0].sum((1, 2)) > 0).sum()) == 5
    assert torch.allclose(filtered.tensor.max(), torch.tensor(1 / 5 ** 3))

    # sub-voxel sizes are an identity returning the same instance
    assert vx.filters.box_filter(volume, 0.5, 'voxel') is volume


def test_integer_input_returns_float() -> None:
    volume = vx.Volume((torch.rand(10, 12, 14) * 255).to(torch.uint8), nontrivial_geometry())
    for filtered in (vx.filters.box_filter(volume, 3, 'voxel'),
                     vx.filters.gaussian_filter(volume, 1.0, 'voxel')):
        assert torch.is_floating_point(filtered.tensor)
        assert filtered.tensor.max() > 0
        assert torch.allclose(filtered.tensor.mean(), volume.tensor.float().mean(), rtol=0.05)


def test_world_voxel_space_consistency(small_volume) -> None:

    # on a 1mm-isotropic geometry, world and voxel units are interchangeable
    iso = vx.Volume(small_volume.tensor)
    assert torch.equal(vx.filters.gaussian_filter(iso, 1.5, 'world').tensor,
                       vx.filters.gaussian_filter(iso, 1.5, 'voxel').tensor)
    assert torch.equal(vx.filters.box_filter(iso, 3, 'world').tensor,
                       vx.filters.box_filter(iso, 3, 'voxel').tensor)

    # on an anisotropic geometry, a world-space sigma matches the equivalent
    # per-axis voxel-space sigma
    world = vx.filters.gaussian_filter(small_volume, 1.5, 'world')
    voxel = vx.filters.gaussian_filter(small_volume, 1.5 / small_volume.geometry.spacing, 'voxel')
    assert torch.allclose(world.tensor, voxel.tensor, atol=1e-6)


def test_strided_filter_geometry(small_volume) -> None:
    filtered = vx.filters.gaussian_filter(small_volume, 1.0, 'voxel', stride=2)

    # shape is ceil(baseshape / stride) and spacing scales with the stride
    assert tuple(filtered.baseshape) == tuple(-(-s // 2) for s in small_volume.baseshape)
    assert torch.allclose(filtered.geometry.spacing, 2 * small_volume.geometry.spacing, atol=1e-5)

    # voxel v of the strided output sits at voxel 2v of the input grid
    v = torch.tensor([1.0, 2, 3])
    assert torch.allclose(filtered.geometry.map(v),
                          small_volume.geometry.map(2 * v), atol=1e-4)

    # the strided tensor equals the unstrided result subsampled
    unstrided = vx.filters.gaussian_filter(small_volume, 1.0, 'voxel')
    assert torch.allclose(filtered.tensor, unstrided.tensor[:, ::2, ::2, ::2], atol=1e-6)

    # per-axis strides only downsample the requested dimensions
    partial = vx.filters.gaussian_filter(small_volume, 1.0, 'voxel', stride=(1, 2, 1))
    assert tuple(partial.baseshape) == (10, 6, 14)


def test_apply_filter_custom_kernel(small_volume) -> None:

    # a dense kernel and its separable factors give the same result
    kernel = vx.filters.gaussian_kernel_1d(1.0)
    kernels = [kernel, kernel, kernel]
    dense = kernel[:, None, None] * kernel[None, :, None] * kernel[None, None, :]
    a = vx.filters.apply_filter(small_volume, kernels)
    b = vx.filters.apply_filter(small_volume, dense)
    assert torch.allclose(a.tensor, b.tensor, atol=1e-6)

    # unstrided filtering preserves the geometry instance, and identity
    # kernels skip the filtering entirely
    assert a.geometry is small_volume.geometry
    assert vx.filters.apply_filter(small_volume, [torch.ones(1)] * 3) is small_volume

    with pytest.raises(ValueError):
        vx.filters.apply_filter(small_volume, [torch.ones(3), torch.ones(3)])


def test_volume_wrappers(small_volume) -> None:
    assert torch.equal(small_volume.smooth(1.5).tensor,
                       vx.filters.gaussian_filter(small_volume, 1.5, 'world').tensor)
    assert torch.equal(small_volume.smooth(1.5, space='voxel').tensor,
                       vx.filters.gaussian_filter(small_volume, 1.5, 'voxel').tensor)


def test_variadic_components() -> None:
    volume = delta_volume()

    # unpacked positional components match the sequence form
    assert vx.volumes_equal(vx.filters.gaussian_filter(volume, 1, 1, 2, 'voxel'),
                            vx.filters.gaussian_filter(volume, (1, 1, 2), 'voxel'))
    assert vx.volumes_equal(vx.filters.box_filter(volume, 3, 3, 1, 'voxel'),
                            vx.filters.box_filter(volume, (3, 3, 1), space='voxel'))
