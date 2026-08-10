"""
Tests for dense deformation structures: VectorField and Warp.

Expected values are always analytic or cross-validated between independent
code paths, never captured from the implementation itself. Geometries are
deliberately hard: anisotropic non-unit spacing, non-RAS orientation,
rotation, and translation.
"""

import pytest
import torch
import voxel as vx

from conftest import nontrivial_geometry


def hard_geometry(orientation: str = 'LIA',
                  spacing: tuple = (0.7, 1.3, 2.4),
                  baseshape: tuple = (12, 14, 10)) -> vx.AcquisitionGeometry:
    """
    An anisotropic, non-1mm, non-RAS, rotated, and translated geometry.
    """
    geometry = vx.geometry_from_spacing(baseshape, spacing, orientation, space='voxel')
    return geometry.rotate((10, -8, 5), space='world').shift((3, -2, 5), space='world')


def linear_ramp_volume(geometry: vx.AcquisitionGeometry,
                       weights: tuple = (0.3, -0.2, 0.5),
                       bias: float = 1.0) -> vx.Volume:
    """
    A volume whose voxel values are a linear function of world coordinates,
    which trilinear interpolation reproduces exactly in the grid interior.
    """
    coords = geometry.map(vx.volume.volume_grid(geometry.baseshape))
    values = coords @ torch.tensor(weights) + bias
    return vx.Volume(values, geometry)


def sinusoidal_field(geometry: vx.AcquisitionGeometry,
                     space: vx.Space,
                     amplitude: float = 1.0,
                     period: float = 20.0) -> vx.VectorField:
    """
    A smooth, deterministic, world-space sinusoidal velocity field, converted
    to the requested vector value space.
    """
    coords = geometry.map(vx.volume.volume_grid(geometry.baseshape))
    phases = torch.tensor([0.0, 2.0, 4.0])
    vectors = amplitude * torch.sin(2 * torch.pi * coords / period + phases)
    field = vx.VectorField(vectors.permute(3, 0, 1, 2), geometry, space='world')
    return field.in_space(space)


def interior(margin: int) -> tuple:
    """
    Slices selecting voxels at least `margin` away from every volume face.
    """
    return (slice(margin, -margin),) * 3


def translation_warp(geometry: vx.AcquisitionGeometry, offset: tuple) -> vx.Warp:
    """
    A warp mapping grid points to their world position plus a constant offset,
    whose displacement field trilinear interpolation reproduces exactly.
    """
    grid = geometry.map(vx.volume.volume_grid(geometry.baseshape))
    return vx.Warp(grid + torch.tensor(offset), geometry)


# ---------------------------------------------------------------------------
# construction and typing
# ---------------------------------------------------------------------------


def test_vector_field_validation() -> None:

    # tensors must be 4D with exactly 3 channels
    with pytest.raises(ValueError):
        vx.VectorField(torch.rand(10, 12, 14), space='world')
    with pytest.raises(ValueError):
        vx.VectorField(torch.rand(2, 10, 12, 14), space='world')

    # a vector value space is required
    with pytest.raises(ValueError):
        vx.VectorField(torch.rand(3, 10, 12, 14))

    # the space is stored as a Space instance and accepts aliases
    field = vx.VectorField(torch.rand(3, 10, 12, 14), space='world')
    assert isinstance(field.space, vx.Space)
    assert field.space == 'world'
    assert vx.VectorField(torch.rand(3, 10, 12, 14), space='image').space == 'voxel'


def test_vector_field_from_volume() -> None:
    geometry = hard_geometry()
    volume = vx.Volume(torch.rand(3, *geometry.baseshape), geometry)

    # a 3-channel volume is adopted along with its geometry
    field = vx.VectorField(volume, space='voxel')
    assert torch.allclose(field.tensor, volume.tensor)
    assert vx.geometries_equal(field.geometry, geometry)

    # an explicitly provided geometry takes precedence
    other = nontrivial_geometry(tuple(geometry.baseshape))
    field = vx.VectorField(volume, other, space='voxel')
    assert vx.geometries_equal(field.geometry, other)

    # channel count and space are still validated
    with pytest.raises(ValueError):
        vx.VectorField(vx.Volume(torch.rand(2, *geometry.baseshape), geometry), space='voxel')
    with pytest.raises(ValueError):
        vx.VectorField(volume)


def test_vector_field_new() -> None:
    geometry = hard_geometry()
    field = vx.VectorField(torch.rand(3, *geometry.baseshape), geometry, space='voxel')

    # a 3-channel tensor stays a vector field and preserves the space
    replaced = field.new(torch.rand(3, *geometry.baseshape))
    assert isinstance(replaced, vx.VectorField)
    assert replaced.space == 'voxel'
    assert vx.geometries_equal(replaced.geometry, geometry)

    # any other channel count decays to a plain volume
    decayed = field.new(torch.rand(2, *geometry.baseshape))
    assert type(decayed) is vx.Volume


def test_warp_validation() -> None:
    geometry = hard_geometry()
    coords = torch.rand(*geometry.baseshape, 3)

    # tensors must be channel-last with exactly 3 components
    with pytest.raises(ValueError):
        vx.Warp(torch.rand(*geometry.baseshape))
    with pytest.raises(ValueError):
        vx.Warp(torch.rand(*geometry.baseshape, 2))
    with pytest.raises(ValueError):
        vx.Warp(vx.Volume(torch.rand(2, *geometry.baseshape), geometry))

    # tensor-based construction
    warp = vx.Warp(coords, geometry)
    assert warp.baseshape == geometry.baseshape
    assert warp.device == coords.device
    assert torch.allclose(warp.coordinates, coords)

    # volume-based construction adopts the volume geometry
    warp = vx.Warp(vx.Volume(coords.permute(3, 0, 1, 2), geometry))
    assert torch.allclose(warp.coordinates, coords)
    assert vx.geometries_equal(warp.geometry, geometry)

    # volume conversion roundtrip is exact
    rebuilt = vx.Warp(warp.as_volume())
    assert torch.allclose(rebuilt.coordinates, warp.coordinates)
    assert vx.geometries_equal(rebuilt.geometry, warp.geometry)


# ---------------------------------------------------------------------------
# spacing and space conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_in_space_spacing(axis) -> None:
    geometry = hard_geometry()
    spacing = (0.7, 1.3, 2.4)

    # a constant voxel-space unit vector along a grid axis
    tensor = torch.zeros(3, *geometry.baseshape)
    tensor[axis] = 1
    field = vx.VectorField(tensor, geometry, space='voxel')

    # in world space it must match the corresponding matrix column everywhere
    world = field.in_space('world')
    column = geometry.tensor[:3, axis].view(3, 1, 1, 1).expand_as(tensor)
    assert torch.allclose(world.tensor, column, atol=1e-5)

    # and its length must equal the voxel spacing along that axis
    norms = world.tensor.norm(dim=0)
    assert torch.allclose(norms, torch.full_like(norms, spacing[axis]), atol=1e-5)
    assert torch.allclose(geometry.spacing[axis], torch.tensor(spacing[axis]), atol=1e-5)


@pytest.mark.parametrize('orientation', ['RAS', 'LIA', 'PSR', 'AIL'])
def test_in_space_roundtrip(orientation) -> None:
    geometry = hard_geometry(orientation)
    field = vx.VectorField(torch.randn(3, *geometry.baseshape), geometry, space='voxel')

    # conversion to the current space is a no-op returning the same instance
    assert field.in_space('voxel') is field

    # a voxel-world-voxel roundtrip recovers the original vectors
    roundtrip = field.in_space('world').in_space('voxel')
    assert roundtrip.space == 'voxel'
    assert torch.allclose(roundtrip.tensor, field.tensor, atol=1e-5)


# ---------------------------------------------------------------------------
# warp and displacement conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('space', ['voxel', 'world'])
def test_as_warp_zero_field(space) -> None:
    geometry = hard_geometry()

    # a zero field warps to the identity world coordinate grid
    field = vx.VectorField(torch.zeros(3, *geometry.baseshape), geometry, space=space)
    warp = field.as_warp()
    grid = geometry.map(vx.volume.volume_grid(geometry.baseshape))
    assert torch.allclose(warp.coordinates, grid, atol=1e-5)

    # and mapping a volume on the same grid reproduces it
    volume = vx.Volume(torch.rand(*geometry.baseshape), geometry)
    mapped = warp.map(volume)
    assert vx.volumes_equal(mapped, volume.float(), vol_tol=1e-4)


@pytest.mark.parametrize('space', ['voxel', 'world'])
def test_warp_displacement_roundtrip(space) -> None:
    geometry = hard_geometry()
    field = vx.VectorField(0.5 * torch.randn(3, *geometry.baseshape), geometry, space=space)

    # warp conversion and back must reproduce the world-space displacements
    recovered = field.as_warp().as_displacement_field()
    assert recovered.space == 'world'
    expected = field.in_space('world')
    assert torch.allclose(recovered.tensor, expected.tensor, atol=1e-4)


# ---------------------------------------------------------------------------
# warp mapping
# ---------------------------------------------------------------------------


def test_map_linear_ramp_translation() -> None:
    fixed = hard_geometry()
    weights, bias = (0.3, -0.2, 0.5), 1.0

    # moving volume with a linear world-space ramp on a large, distinct geometry
    moving_geometry = nontrivial_geometry((48, 48, 48)).shift_to_point(fixed.center)
    moving = linear_ramp_volume(moving_geometry, weights, bias)

    # warp fixed grid points by a constant world translation
    offset = torch.tensor([1.5, -2.0, 1.0])
    grid = fixed.map(vx.volume.volume_grid(fixed.baseshape))
    warp = vx.Warp(grid + offset, fixed)
    mapped = warp.map(moving)

    # only compare where sample points fall inside the moving voxel-center hull
    voxels = moving_geometry.inverse().map(warp.coordinates)
    limits = torch.tensor(moving.baseshape) - 1.5
    inside = ((voxels > 0.5) & (voxels < limits)).all(dim=-1)
    assert inside.float().mean() > 0.9

    # linear interpolation of a linear ramp is exact: check analytic values
    expected = (grid + offset) @ torch.tensor(weights) + bias
    assert torch.allclose(mapped.tensor[0][inside], expected[inside], atol=1e-3)
    assert vx.geometries_equal(mapped.geometry, fixed)


def test_map_matches_resample_like() -> None:
    fixed = hard_geometry()
    moving = vx.Volume(torch.rand(20, 22, 18), nontrivial_geometry((20, 22, 18)))

    # an identity warp must reproduce plain resampling onto the fixed grid
    grid = fixed.map(vx.volume.volume_grid(fixed.baseshape))
    mapped = vx.Warp(grid, fixed).map(moving)
    resampled = moving.resample_like(fixed)
    assert vx.volumes_equal(mapped, resampled.float(), vol_tol=1e-4)


# ---------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('method', ['euler', 'rk2'])
@pytest.mark.parametrize('space', ['voxel', 'world'])
def test_integrate_constant_field(method, space) -> None:
    geometry = hard_geometry()

    # constant velocity, kept small so trajectories stay inside the grid
    velocity = torch.tensor([0.8, -0.6, 0.5])
    tensor = velocity.view(3, 1, 1, 1).expand(3, *geometry.baseshape)
    field = vx.VectorField(tensor, geometry, space=space)

    # interior starting points expressed in the field's space
    points = vx.volume.volume_grid(geometry.baseshape)[interior(3)].reshape(-1, 3)
    if vx.Space(space) == 'world':
        points = geometry.map(points)

    # a constant field is integrated exactly regardless of the step size
    for dt in (1.0, 0.25):
        result = field.integrate(points, dt=dt, method=method)
        assert torch.allclose(result, points + velocity, atol=1e-4)

    # total time scales the displacement, and dt > time still takes one step
    result = field.integrate(points, dt=0.5, method=method, time=2)
    assert torch.allclose(result, points + 2 * velocity, atol=1e-4)
    result = field.integrate(points, dt=5.0, method=method)
    assert torch.allclose(result, points + velocity, atol=1e-4)


def test_integrate_space_argument() -> None:
    geometry = hard_geometry()
    field = sinusoidal_field(geometry, 'world', amplitude=1.0)

    # the same interior points expressed in both coordinate spaces
    voxel_points = vx.volume.volume_grid(geometry.baseshape)[interior(3)].reshape(-1, 3)
    world_points = geometry.map(voxel_points)

    # integrating in either space must trace the same world trajectories
    from_voxel = field.integrate(voxel_points, dt=0.25, space='voxel')
    from_world = field.integrate(world_points, dt=0.25, space='world')
    assert torch.allclose(geometry.map(from_voxel), from_world, atol=1e-3)


@pytest.mark.parametrize('space', ['voxel', 'world'])
def test_integrate_linear_field(space) -> None:
    geometry = hard_geometry(baseshape=(16, 18, 14))
    rate = 0.3

    # a linear expansion field about the grid center, sampled exactly by
    # trilinear interpolation, with the analytic solution x(t) = c + (x - c) e^t
    grid = geometry.map(vx.volume.volume_grid(geometry.baseshape))
    center = geometry.center
    field = vx.VectorField((rate * (grid - center)).permute(3, 0, 1, 2),
                           geometry, space='world')
    field = field.in_space(space)

    # start from points near the center so trajectories stay in the grid
    points = grid[interior(5)].reshape(-1, 3)
    expected = center + (points - center) * torch.exp(torch.tensor(rate))

    # euler converges at first order, rk2 at second: both must land near the
    # analytic solution, with rk2 strictly more accurate at equal step size
    euler = field.integrate(points, dt=0.05, method='euler', space='world')
    rk2 = field.integrate(points, dt=0.05, method='rk2', space='world')
    euler_error = (euler - expected).norm(dim=-1).max()
    rk2_error = (rk2 - expected).norm(dim=-1).max()
    assert euler_error < 0.05
    assert rk2_error < 2e-3
    assert rk2_error < euler_error / 2


def test_integrate_exact_gradient() -> None:
    geometry = hard_geometry()
    field = sinusoidal_field(geometry, 'world', amplitude=1.0)
    points = geometry.map(vx.volume.volume_grid(geometry.baseshape))[interior(3)]

    # approximate gradients must not change the forward result at all
    exact = field.integrate(points, dt=0.25, method='rk2')
    approx = field.integrate(points, dt=0.25, method='rk2', exact_gradient=False)
    assert torch.equal(approx, exact)

    # and gradients still flow back to the field tensor
    tensor = field.tensor.clone().requires_grad_(True)
    tracked = vx.VectorField(tensor, geometry, space='world')
    result = tracked.integrate(points, dt=0.25, method='rk2', exact_gradient=False)
    result.sum().backward()
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()
    assert tensor.grad.abs().sum() > 0


def test_integrate_validation() -> None:
    geometry = hard_geometry()
    field = vx.VectorField(torch.rand(3, *geometry.baseshape), geometry, space='voxel')
    with pytest.raises(ValueError):
        field.integrate(torch.zeros(1, 3), dt=0.1, method='rk4')


# ---------------------------------------------------------------------------
# exponentiation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('space', ['voxel', 'world'])
def test_exponentiate_constant_field(space) -> None:
    geometry = hard_geometry()

    # a constant stationary velocity flows to the same constant displacement
    velocity = torch.tensor([0.4, -0.3, 0.5])
    tensor = velocity.view(3, 1, 1, 1).expand(3, *geometry.baseshape)
    field = vx.VectorField(tensor, geometry, space=space)
    displacement = field.exponentiate(steps=6)
    assert isinstance(displacement, vx.VectorField)
    assert displacement.space == space

    # compare away from the boundary, where zero padding bleeds in
    result = displacement.tensor[(slice(None), *interior(3))]
    expected = velocity.view(3, 1, 1, 1).expand_as(result)
    assert torch.allclose(result, expected, atol=1e-3)


def test_exponentiate_matches_integration() -> None:
    geometry = hard_geometry(baseshape=(16, 18, 14))
    field = sinusoidal_field(geometry, 'world', amplitude=1.0)

    # scaling-and-squaring and stepwise integration approximate the same flow
    displacement = field.exponentiate(steps=6)
    points = geometry.map(vx.volume.volume_grid(geometry.baseshape))[interior(3)]
    integrated = field.integrate(points, dt=1 / 64, method='rk2') - points

    # the two independent numerical paths must agree in the interior
    exponentiated = displacement.tensor.permute(1, 2, 3, 0)[interior(3)]
    assert torch.allclose(exponentiated, integrated, atol=0.05)


def test_exponentiate_space_consistency() -> None:
    geometry = hard_geometry(baseshape=(16, 18, 14))
    world_field = sinusoidal_field(geometry, 'world', amplitude=1.0)
    voxel_field = world_field.in_space('voxel')

    # exponentiating in either space must describe the same world displacement
    from_world = world_field.exponentiate(steps=6)
    from_voxel = voxel_field.exponentiate(steps=6).in_space('world')
    result = from_voxel.tensor[(slice(None), *interior(3))]
    expected = from_world.tensor[(slice(None), *interior(3))]
    assert torch.allclose(result, expected, atol=1e-3)


def test_exponentiate_inverse() -> None:
    geometry = hard_geometry(baseshape=(16, 18, 14))

    # the composition stacks two exponentials and a displacement sampling, so
    # use a smoother field to keep trilinear interpolation error subdominant
    field = sinusoidal_field(geometry, 'world', amplitude=1.0, period=40.0)

    # exponentiate the field and its negation
    forward = field.exponentiate(steps=6)
    backward = field.new(-field.tensor).exponentiate(steps=6)

    # composing the two displacements must cancel to the identity
    points = geometry.map(vx.volume.volume_grid(geometry.baseshape))[interior(3)]
    moved = points + backward.sample(points, space='world')
    composed = moved + forward.sample(moved, space='world')
    assert torch.allclose(composed, points, atol=0.05)


# ---------------------------------------------------------------------------
# transform composition
# ---------------------------------------------------------------------------


def test_compose_transforms_affine() -> None:
    a1 = vx.affine.compose_affine(translation=(1.5, -1.0, 0.5), rotation=(3, -2, 4))
    a2 = vx.affine.compose_affine(translation=(-0.5, 1.0, 2.0), rotation=(-2, 5, 1))

    # affine-only composition stays an affine equal to the matrix product
    merged = vx.compose_transforms(a1, a2)
    assert isinstance(merged, vx.AffineMatrix)
    assert not isinstance(merged, vx.Warp)
    assert torch.allclose(merged.tensor, (a2 @ a1).tensor, atol=1e-5)

    # applying the merged transform matches applying each in order
    volume = vx.Volume(torch.rand(10, 12, 14), nontrivial_geometry())
    sequential = volume.transform(a1).transform(a2)
    direct = volume.transform(merged)
    assert torch.equal(sequential.tensor, direct.tensor)
    assert vx.geometries_equal(sequential.geometry, direct.geometry, tol=1e-4)


def test_compose_transforms_mixed() -> None:

    # linear-ramp moving volume large enough that every sample lands interior,
    # so each interpolation in both application paths is exact
    moving_geometry = nontrivial_geometry((64, 64, 80)).shift_to_point(torch.zeros(3))
    volume = linear_ramp_volume(moving_geometry)

    # affines and constant-displacement warps interleaved, with the first warp
    # grid comfortably covering the domain of the second
    a1 = vx.affine.compose_affine(translation=(1.0, -0.5, 0.8), rotation=(2, -1, 3))
    a2 = vx.affine.compose_affine(translation=(-0.8, 0.6, -1.2), rotation=(-1, 2, -2))
    w1 = translation_warp(vx.geometry_from_spacing((40, 40, 40), 1.0), (1.2, -0.7, 0.9))
    w2 = translation_warp(hard_geometry(), (-0.6, 1.1, -0.4))

    # merged application must match applying each transform in order
    sequential = w2.map(w1.map(volume.transform(a1)).transform(a2))
    merged = vx.compose_transforms(a1, w1, a2, w2)
    assert isinstance(merged, vx.Warp)
    assert vx.geometries_equal(merged.geometry, w2.geometry, tol=1e-4)
    direct = merged.map(volume)
    assert torch.allclose(direct.tensor, sequential.tensor, atol=1e-3)


def test_compose_transforms_trailing_affine() -> None:
    volume = vx.Volume(torch.rand(24, 26, 22), nontrivial_geometry((24, 26, 22)))
    w1 = translation_warp(hard_geometry(), (0.8, -0.5, 0.6))
    a2 = vx.affine.compose_affine(translation=(2.0, -1.0, 0.5), rotation=(4, 2, -3))

    # a trailing affine keeps the mapping and only moves the output domain
    merged = vx.compose_transforms(w1, a2)
    assert isinstance(merged, vx.Warp)
    assert torch.allclose(merged.coordinates, w1.coordinates)
    sequential = w1.map(volume).transform(a2)
    direct = merged.map(volume)
    assert torch.equal(direct.tensor, sequential.tensor)
    assert vx.geometries_equal(direct.geometry, sequential.geometry, tol=1e-4)


def test_compose_transforms_validation() -> None:
    with pytest.raises(ValueError):
        vx.compose_transforms()
    with pytest.raises(TypeError):
        vx.compose_transforms(torch.eye(4))
    single = vx.affine.compose_affine(translation=(1.0, 2.0, 3.0))
    assert isinstance(vx.compose_transforms(single), vx.AffineMatrix)
