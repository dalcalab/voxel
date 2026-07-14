import pytest
import torch
import voxel as vx


def seed_volume(baseshape: tuple = (9, 9, 9)) -> vx.Volume:
    """
    Volume with a single nonzero seed voxel at the grid center.
    """
    volume = vx.AcquisitionGeometry(baseshape).zeros_like()
    center = [s // 2 for s in baseshape]
    volume.tensor[0, center[0], center[1], center[2]] = 1
    return volume


def two_blob_volume() -> vx.Volume:
    """
    Volume with a small 8-voxel blob (first in scan order) and a large
    125-voxel blob.
    """
    volume = vx.AcquisitionGeometry((12, 12, 12)).zeros_like()
    volume.tensor[0, 1:3, 1:3, 1:3] = 1
    volume.tensor[0, 5:10, 5:10, 5:10] = 1
    return volume


def hollow_cube_volume() -> vx.Volume:
    """
    Volume with a 5^3 cube shell enclosing a 3^3 cavity.
    """
    volume = vx.AcquisitionGeometry((9, 9, 9)).zeros_like()
    volume.tensor[0, 2:7, 2:7, 2:7] = 1
    volume.tensor[0, 3:6, 3:6, 3:6] = 0
    return volume


def test_dilate_grows_seed() -> None:
    seed = seed_volume()

    # one iteration marks the neighborhood defined by the connectivity
    for connectivity, count in ((1, 7), (2, 19), (3, 27)):
        dilated = vx.morphology.dilate(seed, connectivity=connectivity)
        assert int(dilated.tensor.sum()) == count

    # two 6-connected iterations produce a manhattan ball of radius 2
    assert int(vx.morphology.dilate(seed, iterations=2).tensor.sum()) == 25

    # data type and geometry are preserved
    for cast in (seed.bool(), seed.int()):
        dilated = vx.morphology.dilate(cast)
        assert dilated.dtype == cast.dtype
        assert dilated.geometry is cast.geometry

    with pytest.raises(ValueError):
        vx.morphology.dilate(seed, connectivity=4)


def test_erode_shrinks_and_border() -> None:

    # a 5^3 cube erodes to its inner 3^3 core
    cube = vx.AcquisitionGeometry((9, 9, 9)).zeros_like()
    cube.tensor[0, 2:7, 2:7, 2:7] = 1
    eroded = vx.morphology.erode(cube)
    assert int(eroded.tensor.sum()) == 27
    assert bool((eroded.tensor[0, 3:6, 3:6, 3:6] == 1).all())

    # voxels beyond the grid are background, so a full volume erodes inward
    full = vx.AcquisitionGeometry((5, 5, 5)).ones_like()
    assert int(vx.morphology.erode(full).tensor.sum()) == 27


def test_erode_dilate_roundtrip() -> None:
    cube = vx.AcquisitionGeometry((9, 9, 9)).zeros_like()
    cube.tensor[0, 2:7, 2:7, 2:7] = 1
    for connectivity in (1, 2, 3):
        roundtrip = vx.morphology.erode(vx.morphology.dilate(cube, connectivity=connectivity),
                                        connectivity=connectivity)
        assert torch.equal(roundtrip.tensor, cube.tensor)


def test_close_fills_gaps() -> None:

    # closing fills a small internal hole and restores the original shape
    holey = vx.AcquisitionGeometry((9, 9, 9)).zeros_like()
    holey.tensor[0, 2:7, 2:7, 2:7] = 1
    holey.tensor[0, 4, 4, 4] = 0
    closed = vx.morphology.close(holey)
    assert int(closed.tensor.sum()) == 125
    assert bool((closed.tensor[0, 2:7, 2:7, 2:7] == 1).all())


def test_open_removes_specks() -> None:

    # opening removes an isolated voxel while preserving the cube (corner
    # connectivity keeps the cube edges exact through the round trip)
    speckled = vx.AcquisitionGeometry((11, 11, 11)).zeros_like()
    speckled.tensor[0, 3:8, 3:8, 3:8] = 1
    speckled.tensor[0, 0, 0, 0] = 1
    opened = vx.morphology.open(speckled, connectivity=3)
    assert int(opened.tensor.sum()) == 125
    assert opened.tensor[0, 0, 0, 0] == 0


def test_iso_thresh() -> None:

    # anisotropic geometry with a slice spacing 4x the in-plane spacing
    geometry = vx.AcquisitionGeometry((9, 9, 9), vx.affine.compose_affine(scale=(1.0, 1, 4)))
    assert geometry.slice_direction == 2
    seed = geometry.zeros_like()
    seed.tensor[0, 4, 4, 4] = 1

    # at or above the threshold, the kernel collapses along the slice
    # direction and the seed only grows in-plane
    inplane = vx.morphology.dilate(seed, iso_thresh=3)
    assert int(inplane.tensor.sum()) == 5
    assert int(inplane.tensor[0, :, :, 3].sum()) == 0

    # below the threshold (or when disabled) the volume is treated as isotropic
    assert int(vx.morphology.dilate(seed, iso_thresh=5).tensor.sum()) == 7
    assert int(vx.morphology.dilate(seed).tensor.sum()) == 7

    # an in-plane erosion preserves a single-slice plate that a full 3D
    # erosion would remove entirely
    plate = geometry.zeros_like()
    plate.tensor[0, 3:6, 3:6, 4] = 1
    assert int(vx.morphology.erode(plate).tensor.sum()) == 0
    assert int(vx.morphology.erode(plate, iso_thresh=3).tensor.sum()) == 1


def test_connected_components() -> None:
    pytest.importorskip('scipy')

    # components are labeled by descending size, regardless of scan order
    volume = two_blob_volume()
    labels = vx.morphology.connected_components(volume)
    assert labels.dtype == torch.int32
    assert labels.unique().tolist() == [0, 1, 2]
    assert int((labels == 1).tensor.sum()) == 125
    assert int((labels == 2).tensor.sum()) == 8
    assert labels.geometry is volume.geometry

    # the largest option keeps only label 1
    largest = vx.morphology.connected_components(volume, largest=True)
    assert largest.unique().tolist() == [0, 1]
    assert int(largest.tensor.sum()) == 125

    # diagonal neighbors merge only with corner connectivity
    diagonal = vx.AcquisitionGeometry((4, 4, 4)).zeros_like()
    diagonal.tensor[0, 0, 0, 0] = 1
    diagonal.tensor[0, 1, 1, 1] = 1
    assert int(vx.morphology.connected_components(diagonal).max()) == 2
    assert int(vx.morphology.connected_components(diagonal, connectivity=3).max()) == 1

    with pytest.raises(ValueError):
        vx.morphology.connected_components(volume, connectivity=0)


def test_flood_fill() -> None:
    pytest.importorskip('scipy')
    volume = two_blob_volume()

    # a seed inside a blob extracts only that blob, preserving the data type
    filled = vx.morphology.flood_fill(volume, (6, 6, 6))
    assert filled.dtype == volume.dtype
    assert int(filled.tensor.sum()) == 125
    assert filled.tensor[0, 1, 1, 1] == 0

    # world-space seed points are supported
    world = volume.geometry.transform(torch.tensor([6.0, 6, 6]))
    assert torch.equal(vx.morphology.flood_fill(volume, world, space='world').tensor, filled.tensor)

    # a background seed fills the connected background region
    background = vx.morphology.flood_fill(volume, (0, 0, 0))
    assert int(background.tensor.sum()) == 12 ** 3 - 125 - 8

    with pytest.raises(ValueError):
        vx.morphology.flood_fill(volume, (20, 0, 0))
    with pytest.raises(ValueError):
        vx.morphology.flood_fill(vx.volume.stack(volume, volume), (6, 6, 6))


def test_fill_holes() -> None:
    pytest.importorskip('scipy')

    # an enclosed cavity is filled into a solid cube
    shell = hollow_cube_volume()
    filled = vx.morphology.fill_holes(shell)
    assert int(filled.tensor.sum()) == 125
    assert bool((filled.tensor[0, 2:7, 2:7, 2:7] == 1).all())
    assert vx.morphology.fill_holes(shell.bool()).dtype == torch.bool

    # a cavity opened to the outside is not a hole
    notched = shell.copy()
    notched.tensor[0, 4, 4, 2] = 0
    assert torch.equal(vx.morphology.fill_holes(notched).tensor.bool(), notched.tensor.bool())


def test_volume_wrappers() -> None:
    cube = vx.AcquisitionGeometry((9, 9, 9)).zeros_like()
    cube.tensor[0, 2:7, 2:7, 2:7] = 1
    cube.tensor[0, 4, 4, 4] = 0
    assert torch.equal(cube.dilate(2, connectivity=2).tensor,
                       vx.morphology.dilate(cube, 2, connectivity=2).tensor)
    assert torch.equal(cube.erode().tensor, vx.morphology.erode(cube).tensor)
    assert torch.equal(cube.close().tensor, vx.morphology.close(cube).tensor)
    assert torch.equal(cube.open().tensor, vx.morphology.open(cube).tensor)
    assert cube.erode().geometry is cube.geometry
