"""
Shared fixtures and helpers for the voxel test suite.
"""

import itertools
import pathlib
import random

import pytest
import torch
import voxel as vx


DATA = pathlib.Path(__file__).parent / 'data'


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


def nontrivial_geometry(baseshape: tuple = (10, 12, 14)) -> vx.AcquisitionGeometry:
    """
    An anisotropic, rotated, and translated acquisition geometry.
    """
    matrix = vx.affine.compose_affine(translation=(3, -2, 5),
                                      rotation=(10, 0, -5),
                                      scale=(1, 1.2, 0.8))
    return vx.AcquisitionGeometry(baseshape, matrix)


@pytest.fixture(autouse=True)
def _seed() -> None:
    """
    Seed random generators before each test for determinism.
    """
    torch.manual_seed(0)
    random.seed(0)


@pytest.fixture(scope='session')
def brain() -> vx.Volume:
    """
    Example 1mm-isotropic, T1-weighted brain volume (stored as uint8).
    Session-scoped, so treat as read-only.
    """
    return vx.load_volume(str(DATA / 'brain-t1w.nii.gz'))


@pytest.fixture
def small_volume() -> vx.Volume:
    """
    Small random volume with a non-trivial acquisition geometry.
    """
    return vx.Volume(torch.rand(10, 12, 14), nontrivial_geometry())


@pytest.fixture
def multichannel_volume() -> vx.Volume:
    """
    Three-channel random volume with a non-trivial acquisition geometry.
    """
    return vx.Volume(torch.rand(3, 10, 12, 14), nontrivial_geometry())


@pytest.fixture
def box_mesh() -> vx.Mesh:
    """
    Axis-aligned box mesh spanning [-0.5, 1.5] along each axis.
    """
    return vx.BoundingBox(center=torch.full((3,), 0.5), extent=torch.ones(3)).mesh()
