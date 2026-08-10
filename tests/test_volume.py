import pytest
import torch
import voxel as vx

from conftest import nontrivial_geometry


def blob_volume(baseshape: tuple = (24, 24, 24), sigma: float = 1.5) -> vx.Volume:
    """
    Volume containing a smooth Gaussian blob at the grid center.
    """
    geometry = nontrivial_geometry(baseshape)
    grid = vx.volume.volume_grid(baseshape)
    center = (torch.tensor(baseshape) - 1) / 2
    tensor = torch.exp(-((grid - center) ** 2).sum(-1) / (2 * sigma ** 2))
    return vx.Volume(tensor, geometry)


def test_construction() -> None:

    # 3D tensors gain a channel dimension, 4D tensors are kept as-is
    volume = vx.Volume(torch.rand(10, 12, 14))
    assert volume.shape == (1, 10, 12, 14)
    assert volume.baseshape == (10, 12, 14)
    assert volume.num_channels == 1
    assert vx.Volume(torch.rand(3, 10, 12, 14)).num_channels == 3

    with pytest.raises(ValueError):
        vx.Volume(torch.rand(10, 12))
    with pytest.raises(ValueError):
        vx.Volume(torch.rand(2, 3, 10, 12, 14))

    # the default geometry centers the volume at the world origin
    assert torch.allclose(volume.geometry.center, torch.zeros(3))

    # a geometry with a mismatched baseshape is rejected
    with pytest.raises(ValueError):
        vx.Volume(torch.rand(10, 12, 14), vx.AcquisitionGeometry((8, 12, 14)))


def test_new_and_copy(small_volume) -> None:

    # new() replaces the tensor and propagates the geometry instance
    new = small_volume.new(small_volume.tensor + 1)
    assert new.geometry is small_volume.geometry

    # copy() clones the tensor but shares the (immutable) geometry
    copy = small_volume.copy()
    assert copy.geometry is small_volume.geometry
    copy.tensor[0, 0, 0, 0] = 100
    assert small_volume.tensor[0, 0, 0, 0] != 100


def test_operators_propagate_geometry(small_volume) -> None:
    for result in (small_volume + 1, small_volume * 2, -small_volume,
                   small_volume > 0.5, small_volume ** 2, 1 - small_volume,
                   small_volume + small_volume):
        assert isinstance(result, vx.Volume)
        assert result.geometry is small_volume.geometry

    assert torch.allclose((small_volume + 1).tensor, small_volume.tensor + 1)
    assert torch.allclose((2 * small_volume).tensor, 2 * small_volume.tensor)

    # note: binary ops between volumes do not validate geometry agreement,
    # the result simply takes the left operand's geometry
    other = small_volume.new(small_volume.tensor, small_volume.geometry.shift((1, 1, 1), 'world'))
    assert (small_volume + other).geometry is small_volume.geometry


def test_inplace_operators(small_volume) -> None:

    # in-place arithmetic mutates the existing tensor storage
    tensor = small_volume.tensor
    reference = tensor.clone()
    small_volume += 1
    assert small_volume.tensor.data_ptr() == tensor.data_ptr()
    assert torch.allclose(small_volume.tensor, reference + 1)
    small_volume *= 2
    assert torch.allclose(small_volume.tensor, (reference + 1) * 2)


@pytest.mark.parametrize('name', ['abs', 'exp', 'log', 'sqrt', 'square', 'floor', 'ceil', 'isnan'])
def test_elementwise_wrappers(small_volume, name) -> None:
    result = getattr(small_volume, name)()
    assert isinstance(result, vx.Volume)
    assert result.geometry is small_volume.geometry
    assert torch.equal(result.tensor, getattr(small_volume.tensor, name)())


def test_clamp_and_pow(small_volume) -> None:
    clamped = small_volume.clamp(min=0.2, max=0.8)
    assert torch.equal(clamped.tensor, small_volume.tensor.clamp(0.2, 0.8))

    # in-place clamping mutates the source tensor
    small_volume.clamp(max=0.5, inplace=True)
    assert small_volume.tensor.max() <= 0.5

    assert torch.equal(small_volume.pow(2).tensor, small_volume.tensor.pow(2))


def test_reductions(multichannel_volume) -> None:

    # a full reduction returns a plain tensor
    assert isinstance(multichannel_volume.mean(), torch.Tensor)
    assert torch.allclose(multichannel_volume.mean(), multichannel_volume.tensor.mean())
    assert torch.allclose(multichannel_volume.max(), multichannel_volume.tensor.amax())

    # reducing the channel dimension returns a single-channel volume
    summed = multichannel_volume.sum(dim=0)
    assert isinstance(summed, vx.Volume)
    assert summed.num_channels == 1
    assert torch.allclose(summed.tensor.squeeze(0), multichannel_volume.tensor.sum(0))

    # reducing a spatial dimension returns a plain tensor
    assert isinstance(multichannel_volume.sum(dim=1), torch.Tensor)

    mask = multichannel_volume > 0.5
    assert isinstance(mask.any(dim=0), vx.Volume)
    assert isinstance(mask.all(), torch.Tensor)


def test_type_conversions(small_volume) -> None:
    assert small_volume.int().dtype == torch.int32
    assert small_volume.bool().dtype == torch.bool
    assert small_volume.half().dtype == torch.float16
    assert small_volume.type(torch.float64).dtype == torch.float64

    # converting to the current dtype returns the same instance
    assert small_volume.type(torch.float32) is small_volume


def test_indexing(multichannel_volume) -> None:

    # boolean tensor and volume masks index the raw tensor
    mask = multichannel_volume.tensor > 0.5
    assert torch.equal(multichannel_volume[mask], multichannel_volume.tensor[mask])

    # a single-channel volume mask broadcasts across channels
    volume_mask = multichannel_volume.new(mask[:1])
    assert multichannel_volume[volume_mask].numel() == 3 * int(mask[:1].sum())

    # channel-list indexing reshuffles channels and keeps the geometry
    shuffled = multichannel_volume[[2, 0]]
    assert shuffled.num_channels == 2
    assert shuffled.geometry is multichannel_volume.geometry
    assert torch.equal(shuffled.tensor[0], multichannel_volume.tensor[2])
    with pytest.raises(ValueError):
        multichannel_volume[[0.5, 1]]

    # mask assignment mutates the tensor in-place
    multichannel_volume[multichannel_volume.new(mask)] = 0.0
    assert multichannel_volume.tensor[mask].max() == 0


def test_setitem_mask_broadcast(multichannel_volume) -> None:
    # like __getitem__, a single-channel volume mask is broadcast
    # across channels during assignment
    volume_mask = multichannel_volume.new(multichannel_volume.tensor[:1] > 0.5)
    multichannel_volume[volume_mask] = 0.0
    assert multichannel_volume[volume_mask].max() == 0


def test_crop_world_invariance(small_volume) -> None:

    # slice-indexing crops the grid and shifts the geometry so that world
    # coordinates of retained voxels are unchanged
    cropped = small_volume[:, 2:8, 3:9, 4:10]
    assert cropped.baseshape == (6, 6, 6)
    assert torch.equal(cropped.tensor, small_volume.tensor[:, 2:8, 3:9, 4:10])
    original_coord = small_volume.geometry.transform(torch.tensor([2.0, 3, 4]))
    assert torch.allclose(cropped.geometry.origin, original_coord, atol=1e-5)

    # cropping to the volume's own bounds is an identity
    assert vx.volumes_equal(small_volume.crop(small_volume.bounds()), small_volume, geom_tol=1e-4)

    # margins are clamped to the volume extent
    clamped = small_volume.crop((slice(None), slice(2, 8)), margin=(100, 100, 100), space='voxel')
    assert vx.volumes_equal(clamped, small_volume)


def test_crop_invalid(small_volume) -> None:
    with pytest.raises(ValueError):
        small_volume[:, 3]
    with pytest.raises(ValueError):
        small_volume.crop('bad cropping')

    # meshes are no longer accepted as bounds, and a box entirely outside
    # the grid cannot be cropped to
    with pytest.raises(TypeError):
        small_volume.crop(small_volume.bounds().mesh())
    with pytest.raises(ValueError):
        small_volume.crop(small_volume.bounds().shift(1e5))


def test_crop_by_bounding_box(small_volume) -> None:

    # cropping one volume by the nonzero bounds of another retains exactly
    # the voxels covered by the nonzero block
    other = small_volume.geometry.zeros_like()
    other.tensor[0, 3:6, 4:7, 5:8] = 1
    cropped = small_volume.crop(other.bounds(nonzero=True))
    assert cropped.baseshape == (3, 3, 3)
    assert torch.equal(cropped.tensor, small_volume.tensor[:, 3:6, 4:7, 5:8])
    block_corner = small_volume.geometry.transform(torch.tensor([3.0, 4, 5]))
    assert torch.allclose(cropped.geometry.origin, block_corner, atol=1e-5)


def test_crop_asymmetric_margin(small_volume) -> None:

    # a (3, 2) margin expands each side independently
    margin = torch.tensor([[1, 0], [0, 2], [0, 0]])
    cropping = (slice(None), slice(2, 8), slice(3, 9), slice(4, 10))
    cropped = small_volume.crop(cropping, margin=margin, space='voxel')
    assert torch.equal(cropped.tensor, small_volume.tensor[:, 1:8, 3:11, 4:10])


def test_bounds_nonzero() -> None:
    geometry = nontrivial_geometry()

    # nonzero bounds span the union of nonzero voxels across all channels,
    # covering the full voxel extents (0.5 beyond the outermost centers)
    tensor = torch.zeros(3, 10, 12, 14)
    tensor[0, 3:6, 4:7, 5:8] = 1
    tensor[2, 2:4, 6:9, 5:8] = 1
    volume = vx.Volume(tensor, geometry)
    bounds = volume.bounds(nonzero=True)
    voxels = geometry.inverse().transform(bounds.corner_points())
    assert torch.allclose(voxels.amin(0), torch.tensor([1.5, 3.5, 4.5]), atol=1e-4)
    assert torch.allclose(voxels.amax(0), torch.tensor([5.5, 8.5, 7.5]), atol=1e-4)

    with pytest.raises(ValueError):
        vx.Volume(torch.zeros(1, 10, 12, 14), geometry).bounds(nonzero=True)


def test_crop_to_nonzero() -> None:
    geometry = nontrivial_geometry()
    volume = geometry.zeros_like()
    volume.tensor[0, 3:6, 4:7, 5:8] = 1

    cropped = volume.crop_to_nonzero()
    assert cropped.baseshape == (3, 3, 3)
    assert bool((cropped.tensor == 1).all())
    block_corner = geometry.transform(torch.tensor([3.0, 4, 5]))
    assert torch.allclose(cropped.geometry.origin, block_corner, atol=1e-5)

    # a voxel margin expands the crop around the block
    assert cropped.baseshape < volume.crop_to_nonzero(margin=1).baseshape


@pytest.mark.parametrize('space', ['voxel', 'world'])
def test_pad_trim_roundtrip(small_volume, space) -> None:
    margin = 2.4 if space == 'world' else 2
    padded = small_volume.pad(margin, space)
    assert all(p > b for p, b in zip(padded.baseshape, small_volume.baseshape))
    assert vx.volumes_equal(padded.trim(margin, space), small_volume, geom_tol=1e-5)


def test_transform_geometry(small_volume) -> None:

    # by default only the geometry is transformed, the tensor is untouched
    trf = vx.affine.compose_affine(translation=(2, -1, 3), rotation=(5, 0, 0))
    moved = small_volume.transform(trf)
    assert moved.tensor is small_volume.tensor
    assert torch.allclose(moved.geometry.tensor, (trf @ small_volume.geometry).tensor, atol=1e-5)

    # an explicit None matches the header-only default
    assert vx.volumes_equal(small_volume.transform(trf, resample=None), moved)

    # applying the inverse transform restores the original geometry
    restored = moved.transform(trf.inverse())
    assert vx.volumes_equal(restored, small_volume, geom_tol=1e-4)


def test_transform_resample() -> None:

    # when resampling, image features move in world space per the transform:
    # the world centroid of a blob should track the transformed centroid
    blob = blob_volume()
    trf = vx.affine.compose_affine(translation=(2, -1, 3), rotation=(5, 0, 0))
    centroid = blob.centroids('world')[0]
    moved = blob.transform(trf, resample=True)
    assert moved.baseshape == blob.baseshape
    assert torch.allclose(moved.centroids('world')[0], trf.transform(centroid), atol=0.05)


def test_transform_warp() -> None:

    # a warp input always resamples, pinned to the warp grid
    blob = blob_volume()
    offset = torch.tensor([3.0, -2.0, 1.5])
    grid = blob.geometry.transform(vx.volume.volume_grid(blob.baseshape))
    warp = vx.Warp(grid + offset, blob.geometry)

    centroid = blob.centroids('world')[0]
    moved = blob.transform(warp)
    assert vx.volumes_equal(moved, warp.map(blob))
    assert vx.geometries_equal(moved.geometry, blob.geometry)

    # the warp is a pull-back, so features move opposite the coordinate offset
    assert torch.allclose(moved.centroids('world')[0], centroid - offset, atol=0.05)

    # explicitly enabling resampling matches the default, disabling it fails
    assert vx.volumes_equal(blob.transform(warp, resample=True), moved)
    with pytest.raises(ValueError):
        blob.transform(warp, resample=False)


def test_pool(small_volume) -> None:
    pooled = small_volume.pool(2)
    reference = torch.nn.functional.avg_pool3d(small_volume.tensor, 2, ceil_mode=True)
    assert torch.allclose(pooled.tensor, reference)
    assert torch.allclose(pooled.geometry.spacing, 2 * small_volume.geometry.spacing, atol=1e-5)

    pooled = small_volume.pool(2, mode='max')
    reference = torch.nn.functional.max_pool3d(small_volume.tensor, 2, ceil_mode=True)
    assert torch.allclose(pooled.tensor, reference)

    with pytest.raises(ValueError):
        small_volume.pool(2, mode='median')


def test_stack(small_volume, multichannel_volume) -> None:
    stacked = vx.volume.stack(small_volume, multichannel_volume)
    assert stacked.num_channels == 4
    assert stacked.geometry is small_volume.geometry
    assert torch.equal(stacked.tensor[:1], small_volume.tensor)

    # a list input and a single-volume input are also supported
    assert vx.volume.stack([small_volume, small_volume]).num_channels == 2
    assert vx.volume.stack(small_volume) is small_volume


def test_volumes_equal(small_volume) -> None:
    assert vx.volumes_equal(small_volume, small_volume)
    assert not vx.volumes_equal(small_volume, small_volume + 1e-3)
    assert vx.volumes_equal(small_volume, small_volume + 1e-3, vol_tol=1e-2)
    shifted = small_volume.new(small_volume.tensor, small_volume.geometry.shift((1, 1, 1), 'world'))
    assert not vx.volumes_equal(small_volume, shifted)
    assert vx.volumes_equal(small_volume, shifted, geom_tol=2)


def test_volumes_equal_shape_mismatch(small_volume) -> None:
    # differently-shaped volumes should compare as unequal, not raise
    assert not vx.volumes_equal(small_volume, small_volume[:, 2:])


def test_onehot_argmax_roundtrip() -> None:
    geometry = nontrivial_geometry()
    labels = vx.Volume(torch.randint(0, 4, (10, 12, 14)), geometry)

    onehot = labels.onehot()
    assert onehot.num_channels == 4
    assert onehot.geometry is geometry

    recovered = onehot.argmax(dim=0)
    assert isinstance(recovered, vx.Volume)
    assert torch.equal(recovered.tensor.squeeze(0), labels.tensor.squeeze(0).long())

    # one-hot encoding requires a single-channel integer volume
    with pytest.raises(AssertionError):
        labels.float().onehot()
    with pytest.raises(AssertionError):
        vx.volume.stack(labels, labels).onehot()


def test_quantile(small_volume) -> None:
    reference = torch.quantile(small_volume.tensor.flatten(), 0.3)
    assert torch.allclose(small_volume.quantile(0.3), reference, atol=1e-2)
    assert small_volume.quantile(0) == small_volume.tensor.min()
    assert small_volume.quantile(1) == small_volume.tensor.max()
    with pytest.raises(ValueError):
        small_volume.quantile(1.5)


def test_isin_unique() -> None:
    labels = vx.Volume(torch.randint(0, 4, (10, 12, 14)))
    mask = labels.isin([1, 2])
    assert isinstance(mask, vx.Volume)
    assert torch.equal(mask.tensor, torch.isin(labels.tensor, torch.tensor([1, 2])))
    assert labels.unique().tolist() == [0, 1, 2, 3]


def test_slice(small_volume) -> None:
    sliced = small_volume.slice(3, 1, 'voxel')
    assert sliced.baseshape == (10, 1, 14)
    assert torch.equal(sliced.tensor, small_volume.tensor[:, :, 3:4])
    coord = small_volume.geometry.transform(torch.tensor([0.0, 3, 0]))
    assert torch.allclose(sliced.geometry.origin, coord, atol=1e-5)

    with pytest.raises(NotImplementedError):
        small_volume.slice(3, 1, 'world')
    with pytest.raises(ValueError):
        small_volume.slice(3, 4, 'voxel')
    with pytest.raises(ValueError):
        small_volume.slice(100, 1, 'voxel')


def test_variadic_components(small_volume) -> None:

    # unpacked positional components match the sequence form
    assert vx.volumes_equal(small_volume.pad(1, 2, 3, 'voxel'),
                            small_volume.pad((1, 2, 3), 'voxel'))
    assert vx.volumes_equal(small_volume.trim(1, 2, 3, 'voxel'),
                            small_volume.trim((1, 2, 3), 'voxel'))
    assert vx.volumes_equal(small_volume.reshape(12, 12, 12),
                            small_volume.reshape((12, 12, 12)))
    assert vx.volumes_equal(small_volume.reshape(12), small_volume.reshape((12, 12, 12)))
    assert vx.volumes_equal(small_volume.crop_to_nonzero(1, 1, 2),
                            small_volume.crop_to_nonzero((1, 1, 2)))
    assert vx.volumes_equal(small_volume.pool(2, 2, 2), small_volume.pool(2))
    assert vx.volumes_equal(small_volume.smooth(1, 1, 2, 'voxel'),
                            small_volume.smooth((1, 1, 2), space='voxel'))
    assert vx.volumes_equal(small_volume.resample(1, 1, 2),
                            small_volume.resample((1, 1, 2)))

    # the pooling mode is keyword-only and no longer parses positionally
    with pytest.raises(ValueError):
        small_volume.pool(2, 'max')
    assert vx.volumes_equal(small_volume.pool(2, mode='max'),
                            small_volume.pool((2, 2, 2), mode='max'))
