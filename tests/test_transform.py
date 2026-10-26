import pytest
import torch

import voxel as vx


def _volume(shape=(8, 8, 8)) -> vx.Volume:
    return vx.Volume(torch.rand(1, *shape))


def _field(volume: vx.Volume, scale: float = 0.0) -> vx.VectorField:
    grid = volume.geometry.map(vx.volume.volume_grid(volume.baseshape))
    vectors = scale * torch.sin(grid).permute(3, 0, 1, 2)
    return vx.VectorField(vectors, volume.geometry, space='world')


def test_vector_field_pair_inverse() -> None:
    forward, reverse = _field(_volume()), _field(_volume())
    pair = vx.VectorFieldPair(forward, reverse).inverse()
    assert pair.forward is reverse and pair.reverse is forward


def test_transform_series_condense() -> None:
    a = vx.affine.compose_affine(translation=torch.tensor([1.0, 2, 3]), rotation=torch.tensor([5.0, -3, 8]))
    b = vx.affine.compose_affine(translation=torch.tensor([0.5, 0, 0]))
    volume = _volume((16, 16, 16))
    field = _field(volume, scale=0.5)

    # the normal form is one affine followed by one deformation, the trailing affine
    # carrying the field along with it
    series = vx.TransformSeries([a, b, field, a]).condense()
    assert len(series) == 2
    assert torch.allclose(series[0].tensor, (a @ b @ a).tensor, atol=1e-6)
    carried = series[1]
    assert isinstance(carried, vx.VectorField) and carried.space == field.space
    assert torch.allclose(carried.geometry.tensor, (a @ field.geometry).tensor, atol=1e-5)
    assert torch.allclose(carried.tensor, field.transform(a).tensor, atol=1e-5)

    # affine-only series come back as the bare product, an empty one as the identity
    assert isinstance(vx.TransformSeries([a, b]).condense(), vx.AffineMatrix)
    assert torch.equal(vx.TransformSeries([]).condense().tensor, vx.AffineMatrix().tensor)
    assert vx.TransformSeries([field]).condense() is field


def test_transform_series_condense_deformations() -> None:
    volume = _volume((16, 16, 16))
    first, second = _field(volume, scale=0.3), _field(volume, scale=-0.2)
    pair = vx.VectorFieldPair(first, second)

    # two fields compose onto the grid of the later one
    composed = vx.TransformSeries([first, second]).condense()
    assert isinstance(composed, vx.VectorField)
    assert torch.allclose(composed.tensor, vx.compose_vector_fields(first, second).tensor)

    # pairs stay pairs, inverting as the reversed series, a lone field breaks the pair
    both = vx.TransformSeries([pair, pair.inverse()]).condense()
    assert isinstance(both, vx.VectorFieldPair)
    assert torch.allclose(both.forward.tensor, vx.compose_vector_fields(first, second).tensor)
    assert torch.allclose(both.reverse.tensor, vx.compose_vector_fields(first, second).tensor)
    assert isinstance(vx.TransformSeries([pair, first]).condense(), vx.VectorField)

    # a last warp keeps the result pinned to its grid
    coarse = volume.geometry.resample(spacing=2.0)
    warp = first.as_warp(coarse)
    pinned = vx.TransformSeries([second, warp]).condense()
    assert isinstance(pinned, vx.Warp) and pinned.baseshape == coarse.baseshape
    assert isinstance(vx.TransformSeries([warp, second]).condense(), vx.VectorField)


def test_transform_series_inverse() -> None:
    a = vx.affine.compose_affine(translation=torch.tensor([1.0, 2, 3]))
    field = _field(_volume())
    pair = vx.VectorFieldPair(field, _field(_volume()))

    inverse = vx.TransformSeries([a, pair]).inverse()
    assert isinstance(inverse[0], vx.VectorFieldPair) and inverse[0].forward is pair.reverse
    assert torch.allclose(inverse[1].tensor, a.inverse().tensor)

    # lone fields and warps cannot be inverted
    with pytest.raises(ValueError):
        vx.TransformSeries([a, field]).inverse()
    with pytest.raises(ValueError):
        vx.TransformSeries([field.as_warp()]).inverse()


def test_volume_transform_affine_is_header_only() -> None:
    volume = _volume()
    a = vx.affine.compose_affine(translation=torch.tensor([1.0, 2, 3]))
    moved = volume.transform(a)
    assert torch.equal(moved.tensor, volume.tensor)
    assert torch.allclose(moved.geometry.tensor, (a @ volume.geometry).tensor)


def test_volume_transform_field_keeps_grid() -> None:
    volume = _volume()

    # a zero field on a coarser grid resamples onto the volume's own grid unchanged
    coarse = vx.Volume(torch.zeros(1, 4, 4, 4), volume.geometry.resample(spacing=2.0))
    zero = _field(coarse)
    moved = volume.transform(zero)
    assert moved.baseshape == volume.baseshape
    assert torch.equal(moved.geometry.tensor, volume.geometry.tensor)
    assert torch.allclose(moved.tensor, volume.tensor, atol=1e-5)

    # a pair applies its forward field
    field = _field(volume, scale=0.5)
    pair = vx.VectorFieldPair(field, _field(volume))
    assert torch.equal(volume.transform(pair).tensor, volume.transform(field).tensor)

    # a warp pins the result to its own grid
    assert volume.transform(zero.as_warp()).baseshape == coarse.baseshape


def test_volume_transform_series_is_condensed() -> None:
    volume = _volume()
    a = vx.affine.compose_affine(translation=torch.tensor([1.0, 2, 3]))
    field = _field(volume, scale=0.5)

    # an affine then a field is already the normal form, so the steps match exactly
    moved = volume.transform(vx.TransformSeries([a, field]))
    by_hand = volume.transform(a).transform(field)
    assert torch.equal(moved.tensor, by_hand.tensor)
    assert torch.equal(moved.geometry.tensor, by_hand.geometry.tensor)

    # a longer series is interpolated once, landing where the steps would have, and
    # agreeing with them away from the edges, where the steps pad with zeros
    ramp = vx.Volume(volume.geometry.grid('world').sum(-1, keepdim=True).permute(3, 0, 1, 2), volume.geometry)
    series = vx.TransformSeries([field, a, field])
    direct = ramp.transform(series)
    stepwise = ramp.transform(field).transform(a).transform(field)
    assert torch.allclose(direct.geometry.tensor, stepwise.geometry.tensor, atol=1e-5)
    inner = (slice(None), slice(4, -4), slice(4, -4), slice(4, -4))
    assert torch.allclose(direct.tensor[inner], stepwise.tensor[inner], atol=1e-2)


def test_volume_transform_like() -> None:
    volume = _volume((16, 16, 16))
    a = vx.affine.compose_affine(translation=torch.tensor([1.0, 2, 3]))
    field = _field(volume, scale=0.5)
    target = volume.geometry.resample(spacing=0.5).shift((0.3, -0.2, 0.1), space='world')

    # an affine moves the header, then resamples onto the target
    moved = volume.transform(a, like=target)
    assert torch.equal(moved.geometry.tensor, target.tensor)
    assert torch.allclose(moved.tensor, volume.transform(a).resample_like(target).tensor)

    # a field is sampled on the target grid directly
    deformed = volume.transform(field, like=target)
    assert torch.equal(deformed.geometry.tensor, target.tensor)
    assert torch.allclose(deformed.tensor, field.as_warp(target).map(volume).tensor)

    # a warp keeps its grid when it matches, and is re-expressed otherwise
    warp = field.as_warp()
    assert torch.equal(volume.transform(warp, like=volume.geometry).tensor, warp.map(volume).tensor)
    assert torch.allclose(volume.transform(warp, like=target).tensor, deformed.tensor, atol=1e-5)

    # a series lands on the target in one interpolation, a volume stands in for a geometry
    series = volume.transform(vx.TransformSeries([a, field]), like=vx.Volume(torch.zeros(1, *target.baseshape), target))
    assert torch.equal(series.geometry.tensor, target.tensor)
    assert torch.allclose(series.tensor, field.as_warp(target).map(volume.transform(a)).tensor)


def test_vector_field_transform_like() -> None:
    volume = _volume((16, 16, 16))
    a = vx.affine.compose_affine(translation=torch.tensor([1.0, 2, 3]), rotation=torch.tensor([0.0, 0, 90]))
    target = volume.geometry.resample(spacing=0.5)
    for space in ('world', 'voxel'):
        field = _field(volume, scale=0.5).in_space(space)
        moved = field.transform(a, like=target)
        assert isinstance(moved, vx.VectorField) and moved.space == field.space
        assert torch.equal(moved.geometry.tensor, target.tensor)
        expected = field.transform(a).in_space('world').resample_like(target).in_space(space)
        assert torch.allclose(moved.tensor, expected.tensor, atol=1e-5)


def test_vector_field_as_warp_geometry() -> None:
    volume = _volume()
    field = _field(volume, scale=0.5)
    fine = volume.geometry.resample(spacing=0.5)

    warp = field.as_warp(fine)
    assert warp.baseshape == fine.baseshape
    grid = fine.map(vx.volume.volume_grid(fine.baseshape))
    assert torch.allclose(warp.coordinates - grid, field.sample(grid, space='world'), atol=1e-6)


def test_any_transform_alias() -> None:
    a = vx.AffineMatrix()
    assert isinstance(a, vx.AnyTransform)
    assert isinstance(vx.TransformSeries([a]), vx.AnyTransform)
    assert not isinstance(torch.eye(4), vx.AnyTransform)
