import pytest
import torch
import voxel as vx

from conftest import all_orientations, nontrivial_geometry


@pytest.fixture
def geometry() -> vx.AcquisitionGeometry:
    """
    Anisotropic, rotated, and translated geometry with baseshape (10, 12, 14).
    """
    return nontrivial_geometry()


def test_default_geometry() -> None:

    # the default matrix centers the grid at the world origin
    geometry = vx.AcquisitionGeometry((10, 12, 14))
    assert torch.allclose(geometry.center, torch.zeros(3))
    assert geometry.baseshape == (10, 12, 14)
    assert torch.allclose(geometry.spacing, torch.ones(3))
    assert geometry.is_isotropic()

    # the matrix tensor is read-only
    with pytest.raises(AttributeError):
        geometry[0, 0] = 2.0


def test_cast_acquisition_geometry(small_volume) -> None:
    assert vx.cast_acquisition_geometry(small_volume) is small_volume.geometry
    geometry = small_volume.geometry
    assert vx.cast_acquisition_geometry(geometry) is geometry
    with pytest.raises(ValueError):
        vx.cast_acquisition_geometry('not a geometry')


def test_spacing_decomposition(geometry) -> None:

    # spacing is recovered from the rotated matrix via QR decomposition
    assert torch.allclose(geometry.spacing, torch.tensor([1.0, 1.2, 0.8]), atol=1e-5)
    assert not geometry.is_isotropic()

    # spacing should be invariant to any additional world-space rotation
    rotated = geometry.rotate((25, -10, 40), 'world')
    assert torch.allclose(rotated.spacing, geometry.spacing, atol=1e-5)


def test_slice_direction(geometry) -> None:

    # inferred from the largest spacing when not explicit
    assert not geometry.slice_direction_is_explicit
    assert geometry.slice_direction == 1
    assert geometry.in_plane_directions == [0, 2]
    assert torch.allclose(geometry.slice_spacing, torch.tensor(1.2), atol=1e-5)
    assert torch.allclose(geometry.spacing_ratio, torch.tensor(1.2 / 0.9), atol=1e-5)

    # an explicit direction overrides the heuristic and survives geometry ops
    explicit = vx.AcquisitionGeometry(geometry.baseshape, geometry.tensor, slice_direction=0)
    assert explicit.slice_direction_is_explicit
    assert explicit.slice_direction == 0
    assert explicit.shift((2, 2, 2), 'voxel').slice_direction == 0
    assert explicit.scale(2, 'voxel').slice_direction == 0

    # resampling only propagates it when spacing is set per-direction
    assert not explicit.resample(2).slice_direction_is_explicit
    assert explicit.resample(slice_spacing=2).slice_direction == 0


def test_origin_center_fov(geometry) -> None:
    assert torch.allclose(geometry.origin, geometry.map(torch.zeros(3)))
    center_voxel = (torch.tensor(geometry.baseshape) - 1) / 2
    assert torch.allclose(geometry.center, geometry.map(center_voxel))
    expected_fov = geometry.spacing * torch.tensor(geometry.baseshape)
    assert torch.allclose(geometry.fov, expected_fov, atol=1e-5)


def test_unit_conversion_roundtrip(geometry) -> None:
    units = torch.tensor([2.0, 3.0, 4.0])
    world = geometry.voxel_to_world_units(units)
    assert torch.allclose(geometry.world_to_voxel_units(world), units, atol=1e-5)

    # scalars are repeated to length 3 and num expands the second dimension
    assert geometry.conform_units(2.0, 'voxel', 'voxel').shape == (3,)
    conformed = geometry.conform_units(2.0, 'voxel', 'voxel', 2)
    assert conformed.shape == (3, 2)
    assert torch.allclose(conformed, torch.full((3, 2), 2.0))


@pytest.mark.parametrize('space', ['voxel', 'world'])
def test_shift_roundtrip(geometry, space) -> None:
    delta = torch.tensor([1.5, -2.0, 3.0])
    shifted = geometry.shift(delta, space)
    assert not vx.geometries_equal(shifted, geometry)
    assert vx.geometries_equal(shifted.shift(-delta, space), geometry, tol=1e-5)

    # a world-space shift moves the origin by exactly the delta
    if space == 'world':
        assert torch.allclose(shifted.origin, geometry.origin + delta, atol=1e-5)


def test_shift_to_point(geometry) -> None:
    target = torch.tensor([10.0, -4.0, 2.0])
    assert torch.allclose(geometry.shift_to_point(target).center, target, atol=1e-5)
    assert torch.allclose(geometry.shift_to_point(target, center=False).origin, target, atol=1e-5)


def test_scale(geometry) -> None:

    # a voxel-space scale changes spacing but anchors the grid origin
    scaled = geometry.scale(2, 'voxel')
    assert torch.allclose(scaled.spacing, 2 * geometry.spacing, atol=1e-5)
    assert torch.allclose(scaled.origin, geometry.origin, atol=1e-5)

    # a world-space scale is applied to world coordinates, moving the origin
    scaled = geometry.scale(2, 'world')
    assert torch.allclose(scaled.spacing, 2 * geometry.spacing, atol=1e-5)
    assert torch.allclose(scaled.origin, 2 * geometry.origin, atol=1e-5)


def test_rotate(geometry) -> None:

    # single-axis rotations invert exactly
    rotated = geometry.rotate((0, 15, 0), 'voxel')
    assert vx.geometries_equal(rotated.rotate((0, -15, 0), 'voxel'), geometry, tol=1e-5)

    # a voxel-space rotation (corner=False) pivots about the grid center
    assert torch.allclose(rotated.center, geometry.center, atol=1e-5)
    corner = geometry.rotate((0, 15, 0), 'voxel', corner=True)
    assert torch.allclose(corner.origin, geometry.origin, atol=1e-5)

    # a world-space rotation left-composes the rotation matrix
    trf = vx.affine.angles_to_rotation_matrix(torch.tensor([0.0, 15, 0]))
    assert torch.allclose(geometry.rotate((0, 15, 0), 'world').tensor, (trf @ geometry).tensor, atol=1e-6)


def test_shear(geometry) -> None:

    # a zero shear is an identity and pivots about the grid center
    assert vx.geometries_equal(geometry.shear(torch.zeros(3, 2), 'voxel'), geometry)
    sheared = geometry.shear(torch.full((3, 2), 0.1), 'voxel')
    assert torch.allclose(sheared.center, geometry.center, atol=1e-4)
    with pytest.raises(ValueError):
        geometry.shear(torch.zeros(3), 'voxel')


@pytest.mark.parametrize('space', ['voxel', 'world'])
def test_pad_trim_inverse(geometry, space) -> None:
    margin = 2.4 if space == 'world' else 2
    padded = geometry.pad(margin, space)
    assert all(p > b for p, b in zip(padded.baseshape, geometry.baseshape))
    assert vx.geometries_equal(padded.trim(margin, space), geometry, tol=1e-5)

    # world coordinates of retained voxels are unchanged
    voxel_margin = geometry.conform_units(margin, space, 'voxel', 2).round()[:, 0]
    assert torch.allclose(padded.map(voxel_margin), geometry.origin, atol=1e-5)


def test_reshape(geometry) -> None:

    # centered reshaping preserves the grid center (for even shape deltas)
    grown = geometry.reshape((14, 16, 18))
    assert torch.allclose(grown.center, geometry.center, atol=1e-5)

    # from_origin reshaping preserves the origin instead
    assert torch.allclose(geometry.reshape((14, 16, 18), from_origin=True).origin, geometry.origin)

    # reshaping is symmetric: the reverse operation restores the original
    assert vx.geometries_equal(grown.reshape(geometry.baseshape), geometry, tol=1e-5)


def test_resample(geometry) -> None:
    resampled = geometry.resample(2)
    assert torch.allclose(resampled.spacing, torch.full((3,), 2.0), atol=1e-5)

    # shape rounds up to preserve the field of view
    expected = (geometry.spacing * torch.tensor(geometry.baseshape) / 2).ceil().int()
    assert tuple(resampled.baseshape) == tuple(expected.tolist())

    with pytest.raises(ValueError):
        geometry.resample()
    with pytest.raises(ValueError):
        geometry.resample(2, in_plane_spacing=1)


def test_pool(geometry) -> None:
    pooled = geometry.pool(2)
    assert torch.allclose(pooled.spacing, 2 * geometry.spacing, atol=1e-5)
    expected = [-(-s // 2) for s in geometry.baseshape]
    assert tuple(pooled.baseshape) == tuple(expected)

    # with a spacing ratio threshold, the slice dimension is not pooled and is
    # instead resampled to a roughly isotropic spacing
    pooled = geometry.pool(2, spacing_ratio_thresh=1.2)
    assert pooled.spacing_ratio >= 0.99


def test_bounds_box(geometry) -> None:

    # bounds cover the full voxel extents: corners land 0.5 voxels beyond
    # the outermost voxel centers
    bounds = geometry.bounds()
    assert isinstance(bounds, vx.BoundingBox)
    voxels = geometry.inverse().map(bounds.corner_points())
    shape = torch.tensor(geometry.baseshape).float()
    assert torch.allclose(voxels.amin(0), torch.full((3,), -0.5), atol=1e-4)
    assert torch.allclose(voxels.amax(0), shape - 0.5, atol=1e-4)


def test_bounds_crop(geometry) -> None:

    # cropping or fitting a geometry to its own bounds is an identity, since
    # only the voxel centers inside the box are kept
    assert vx.geometries_equal(geometry.crop(geometry.bounds()), geometry, tol=1e-4)
    assert vx.geometries_equal(geometry.fit_to_bounds(geometry.bounds()), geometry, tol=1e-4)

    # a box extending past the grid grows the grid under fit_to_bounds,
    # while crop stays clamped to the current extent
    expanded = geometry.bounds(margin=2, space='voxel')
    fit = geometry.fit_to_bounds(expanded)
    assert vx.geometries_equal(fit, geometry.pad(2, 'voxel'), tol=1e-4)
    assert vx.geometries_equal(geometry.crop(expanded), geometry, tol=1e-4)

    # margins expand the range directly, with the same clamping distinction
    fit = geometry.fit_to_bounds(geometry.bounds(), margin=2, space='voxel')
    assert vx.geometries_equal(fit, geometry.pad(2, 'voxel'), tol=1e-4)
    crop = geometry.crop(geometry.bounds(), margin=2, space='voxel')
    assert vx.geometries_equal(crop, geometry, tol=1e-4)

    # a smaller box shrinks both to the contained voxel centers
    shrunk = geometry.bounds(margin=-2, space='voxel')
    assert vx.geometries_equal(geometry.crop(shrunk), geometry.pad(-2, 'voxel'), tol=1e-4)
    assert vx.geometries_equal(geometry.fit_to_bounds(shrunk), geometry.pad(-2, 'voxel'), tol=1e-4)

    # meshes are no longer accepted as bounds, and a box entirely outside
    # the grid cannot be cropped to
    with pytest.raises(TypeError):
        geometry.crop(geometry.bounds().mesh())
    with pytest.raises(ValueError):
        geometry.crop(geometry.bounds().shift(1e5))


def test_geometry_from_spacing() -> None:

    # the default is an isotropic RAS grid centered at the world origin
    geometry = vx.geometry_from_spacing((10, 12, 14), 1)
    assert geometry.baseshape == (10, 12, 14)
    assert torch.allclose(geometry.spacing, torch.ones(3))
    assert geometry.orientation.name == 'RAS'
    assert torch.allclose(geometry.center, torch.zeros(3))

    # world-space spacing components are ordered along the world axes and
    # permuted to the corresponding grid dimensions
    geometry = vx.geometry_from_spacing((10, 12, 14), (1, 2, 3), orientation='ASR', space='world')
    assert geometry.orientation.name == 'ASR'
    assert torch.allclose(geometry.spacing, torch.tensor([2.0, 3.0, 1.0]))

    # voxel-space components map directly to the grid dimensions
    geometry = vx.geometry_from_spacing((10, 12, 14), (1, 2, 3), orientation='ASR', space='voxel')
    assert geometry.orientation.name == 'ASR'
    assert torch.allclose(geometry.spacing, torch.tensor([1.0, 2.0, 3.0]))

    # the device defaults to that of the spacing tensor unless provided
    spacing = torch.tensor([1.0, 1.0, 2.0])
    assert vx.geometry_from_spacing((10, 12, 14), spacing).device == spacing.device
    geometry = vx.geometry_from_spacing((10, 12, 14), spacing, device='cpu')
    assert geometry.device == torch.device('cpu')

    with pytest.raises(ValueError):
        vx.geometry_from_spacing((10, 12, 14), (1, 2))


@pytest.mark.parametrize('orientation', all_orientations())
def test_geometry_from_spacing_orientations(orientation) -> None:
    spacing = torch.tensor([1.0, 2.0, 3.0])
    geometry = vx.geometry_from_spacing((10, 12, 14), spacing, orientation=orientation)
    assert geometry.orientation.name == orientation
    assert torch.allclose(geometry.center, torch.zeros(3))

    # the input spacing is recovered when reordered along the world axes
    world_spacing = geometry.spacing[geometry.orientation.dims.argsort()]
    assert torch.allclose(world_spacing, spacing)

    # constructing in world space matches reorienting an origin-centered RAS geometry
    reoriented = vx.geometry_from_spacing((10, 12, 14), spacing).reorient(orientation)
    constructed = vx.geometry_from_spacing(reoriented.baseshape, spacing, orientation=orientation)
    assert vx.geometries_equal(constructed, reoriented, tol=1e-5)


def test_geometries_equal(geometry) -> None:
    assert vx.geometries_equal(geometry, geometry)
    assert not vx.geometries_equal(geometry, geometry.reshape((12, 12, 14)))
    shifted = geometry.shift((1e-3, 1e-3, 1e-3), 'world')
    assert not vx.geometries_equal(geometry, shifted)
    assert vx.geometries_equal(geometry, shifted, tol=1e-2)


def test_shift_scalar(geometry) -> None:
    # a scalar delta is ambiguous for a translation, so it is rejected
    # rather than broadcast to length 3
    with pytest.raises(ValueError):
        geometry.shift(2.0, 'world')


@pytest.mark.parametrize('name', ['zeros_like', 'ones_like', 'rand_like', 'randn_like'])
def test_like_factories(geometry, name) -> None:
    volume = getattr(geometry, name)(channels=2, dtype=torch.float64)
    assert isinstance(volume, vx.Volume)
    assert volume.num_channels == 2
    assert volume.dtype == torch.float64
    assert volume.geometry is geometry


def test_full_like(geometry) -> None:
    volume = geometry.full_like(3.5)
    assert isinstance(volume, vx.Volume)
    assert bool((volume.tensor == 3.5).all())


def test_variadic_components(geometry) -> None:

    # unpacked positional components match the sequence form
    assert vx.geometries_equal(geometry.shift(1, 2, 3, 'voxel'),
                               geometry.shift((1, 2, 3), 'voxel'))
    assert vx.geometries_equal(geometry.shift(1, 2, 3, space='voxel'),
                               geometry.shift((1, 2, 3), space='voxel'))
    assert vx.geometries_equal(geometry.scale(1, 1, 2, 'world'),
                               geometry.scale((1, 1, 2), 'world'))
    assert vx.geometries_equal(geometry.rotate(10, 0, -5, 'world'),
                               geometry.rotate((10, 0, -5), 'world'))
    assert vx.geometries_equal(geometry.shear(0.1, 0, 0, 0.2, 0, 0, 'voxel'),
                               geometry.shear(((0.1, 0), (0, 0.2), (0, 0)), 'voxel'))
    assert vx.geometries_equal(geometry.resample(1, 1, 2), geometry.resample((1, 1, 2)))
    assert vx.geometries_equal(geometry.pool(2, 2, 1), geometry.pool((2, 2, 1)))
    assert vx.geometries_equal(geometry.pad(1, 2, 3, 'voxel'),
                               geometry.pad((1, 2, 3), 'voxel'))
    assert vx.geometries_equal(geometry.trim(1, 2, 3, 'voxel'),
                               geometry.trim((1, 2, 3), 'voxel'))
    assert vx.geometries_equal(geometry.reshape(14, 16, 18), geometry.reshape((14, 16, 18)))

    # a scalar reshape is expanded to an isotropic shape
    assert vx.geometries_equal(geometry.reshape(16), geometry.reshape((16, 16, 16)))

    # bounding boxes computed from unpacked margins match as well
    a, b = geometry.bounds(1, 2, 3), geometry.bounds((1, 2, 3))
    assert torch.allclose(a.center, b.center) and torch.allclose(a.extent, b.extent)
    a, b = geometry.bounds(2, 'voxel'), geometry.bounds(2, space='voxel')
    assert torch.allclose(a.center, b.center) and torch.allclose(a.extent, b.extent)

    # a required space cannot be omitted or provided twice
    with pytest.raises(TypeError):
        geometry.pad(1, 2, 3)
    with pytest.raises(TypeError):
        geometry.shift(1, 2, 3, 'voxel', space='world')
