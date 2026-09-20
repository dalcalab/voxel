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
    a = vx.affine.compose_affine(translation=torch.tensor([1.0, 2, 3]))
    b = vx.affine.compose_affine(translation=torch.tensor([0.5, 0, 0]))
    field = _field(_volume())

    # neighboring affines merge in application order, deformations are kept
    series = vx.TransformSeries([a, b, field, a]).condense()
    assert len(series) == 3
    assert torch.allclose(series[0].tensor, (b @ a).tensor)
    assert series[1] is field

    # a single remaining transform comes back bare
    assert isinstance(vx.TransformSeries([a, b]).condense(), vx.AffineMatrix)


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


def test_volume_transform_series_is_stepwise() -> None:
    volume = _volume()
    a = vx.affine.compose_affine(translation=torch.tensor([1.0, 2, 3]))
    field = _field(volume, scale=0.5)

    moved = volume.transform(vx.TransformSeries([a, field]))
    by_hand = volume.transform(a).transform(field)
    assert torch.equal(moved.tensor, by_hand.tensor)
    assert torch.equal(moved.geometry.tensor, by_hand.geometry.tensor)


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
