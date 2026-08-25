import numpy as np
import pytest
import torch
import voxel as vx

from conftest import DATA, nontrivial_geometry


def test_nifti_load_geometry(tmp_path) -> None:
    pytest.importorskip('nibabel')
    vol = vx.Volume(torch.rand(10, 12, 14), nontrivial_geometry())
    path = str(tmp_path / 'vol.nii.gz')
    vol.save(path)

    geom = vx.load_geometry(path)
    assert vx.geometries_equal(geom, vx.load_volume(path).geometry)
    assert 'nii' in geom.reference


def test_nifti_load_geometry_real_file() -> None:
    pytest.importorskip('nibabel')
    path = str(DATA / 'brain-t1w.nii.gz')
    assert vx.geometries_equal(vx.load_geometry(path), vx.load_volume(path).geometry)


def test_nrrd_load_geometry(tmp_path) -> None:
    nrrd = pytest.importorskip('nrrd')
    path = str(tmp_path / 'vol.nrrd')
    header = {
        'space': 'left-posterior-superior',
        'space directions': [[1.0, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 0.8]],
        'space origin': [3.0, -2.0, 5.0],
    }
    nrrd.write(path, np.random.rand(10, 12, 14), header)

    geom = vx.load_geometry(path)
    assert geom.baseshape == (10, 12, 14)
    assert vx.geometries_equal(geom, vx.load_volume(path).geometry)


def test_load_geometry_unsupported_format(tmp_path) -> None:
    vol = vx.Volume(torch.rand(4, 4, 4))
    path = str(tmp_path / 'vol.pth')
    vol.save(path)

    with pytest.raises(NotImplementedError):
        vx.load_geometry(path)
