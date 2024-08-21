import itertools
import torch
import random
import voxel as vx


def all_orientations() -> list:
    """
    Returns all 48 possible grid orientations in world space.
    """
    data = ['LR', 'PA', 'IS']
    choices = [(item[0], item[1]) for item in data]
    orders = list(itertools.permutations(choices))
    orientations = []
    for order in orders:
        perms = list(itertools.product(*order))
        orientations.extend([''.join(p) for p in perms])
    return orientations


def test_orientation_init() -> None:

    # test construction from a few hardcoded orientation matrices

    assert 'RAS' == vx.Orientation(vx.AffineMatrix(torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])))

    assert 'ILA' == vx.Orientation(vx.AffineMatrix(torch.tensor([
        [ 0, -1, 0],
        [ 0,  0, 1],
        [-1,  0, 0]])))

    assert 'PSR' == vx.Orientation(vx.AffineMatrix(torch.tensor([
        [ 0,  0, 1],
        [-1,  0, 0],
        [ 0,  1, 0]])))


def test_orientation_equality() -> None:

    # test that all orientations are equal to themselves
    for name in all_orientations():
        orientation = vx.Orientation(name)
        assert orientation.name == name
        assert orientation == vx.Orientation(name)
        assert orientation == name
        assert name == orientation
    
    # double check that inequality works
    assert vx.Orientation('RAS') != vx.Orientation('LPI')


def test_reorientation() -> None:

    # random volume
    vol = vx.Volume(torch.rand(10, 20, 30), torch.diag(torch.tensor([1, 1.2, 0.8, 1])))

    # cycle through all orientations
    reoriented = vol
    orientations = all_orientations()
    random.shuffle(orientations)
    for name in orientations:

        # reorienting back should give the same original volume
        assert vx.volumes_equal(vol, vol.reorient(name).reorient('RAS'))

        # reorient the volume
        reoriented = reoriented.reorient(name)
        assert reoriented.geometry.orientation == name
    
    # make sure we get back to the original volume
    assert vx.volumes_equal(vol, reoriented.reorient('RAS'))


def test_orientation_slice_direction() -> None:

    shape = (10, 20, 30)
    geometry = vx.AcquisitionGeometry(shape, slice_direction=1)
    vol = vx.Volume(torch.ones(shape), geometry)

    # make sure the explicitly-set slice direction is propagated correctly
    assert vol.reorient('RAS').geometry._explicit_slice_direction == 1
    assert vol.reorient('SRA').geometry._explicit_slice_direction == 2
    assert vol.reorient('PIL').geometry._explicit_slice_direction == 0
