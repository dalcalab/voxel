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


def test_affine_text_roundtrip(tmp_path) -> None:
    matrix = torch.tensor([
        [0.1, -0.2, 0.3, 12.345678901234567],
        [0.4, 0.5, -0.6, -3.0],
        [0.7, -0.8, 0.9, 1e-10],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float64)
    affine = vx.AffineMatrix(matrix, dtype=torch.float64)
    path = tmp_path / 'affine.txt'
    vx.save_affine(affine, path)
    loaded = vx.load_affine(path)
    assert loaded.tensor.dtype == torch.float64
    assert torch.equal(loaded.tensor, matrix)


def test_affine_text_load_variants(tmp_path) -> None:
    path = tmp_path / 'affine.mat'
    path.write_text('# comment\n1 0 0 5\n0 1 0 6\n0 0 1 7\n')
    loaded = vx.load_affine(path)
    expected = torch.eye(4, dtype=torch.float64)
    expected[:3, 3] = torch.tensor([5.0, 6.0, 7.0])
    assert torch.equal(loaded.tensor, expected)

    path.write_text('1 0 0\n0 1 0\n0 0 1\n')
    with pytest.raises(ValueError):
        vx.load_affine(path)


def test_transform_file_roundtrip(tmp_path) -> None:
    geometry = nontrivial_geometry()
    affine = vx.AffineMatrix(torch.rand(4, 4, dtype=torch.float64), dtype=torch.float64)
    field = vx.VectorField(torch.rand(3, *geometry.baseshape), geometry, space='world')
    pair = vx.VectorFieldPair(field, vx.VectorField(-field.tensor, geometry, space='voxel'))
    warp = field.as_warp()
    path = tmp_path / 'series.vxx'
    vx.TransformSeries([affine, pair, warp]).save(path)

    loaded = vx.load_transform(path)
    assert isinstance(loaded, vx.TransformSeries) and len(loaded) == 3
    assert loaded[0].tensor.dtype == torch.float64 and torch.equal(loaded[0].tensor, affine.tensor)
    assert isinstance(loaded[1], vx.VectorFieldPair)
    assert torch.equal(loaded[1].forward.tensor, field.tensor) and loaded[1].forward.space == vx.Space('world')
    assert torch.equal(loaded[1].reverse.tensor, -field.tensor) and loaded[1].reverse.space == vx.Space('voxel')
    assert vx.geometries_equal(loaded[1].forward.geometry, geometry)
    assert isinstance(loaded[2], vx.Warp) and torch.equal(loaded[2].coordinates, warp.coordinates)
    assert vx.geometries_equal(loaded[2].geometry, geometry)

    # a single step loads as the bare transform, and a .vxx is a plain zip of npy arrays
    for transform, cls in ((affine, vx.AffineMatrix), (field, vx.VectorField), (pair, vx.VectorFieldPair), (warp, vx.Warp)):
        transform.save(path)
        assert isinstance(vx.load_transform(path), cls)
    import zipfile
    with zipfile.ZipFile(path) as archive:
        assert 'manifest.json' in archive.namelist()
        assert archive.compression == zipfile.ZIP_STORED
        with archive.open('0.coordinates.npy') as file:
            assert np.lib.format.read_array(file).shape == (*geometry.baseshape, 3)


def test_transform_file_errors(tmp_path) -> None:
    geometry = nontrivial_geometry()
    field = vx.VectorField(torch.rand(3, *geometry.baseshape), geometry, space='world')
    with pytest.raises(TypeError):
        field.save(tmp_path / 'field.txt')
    with pytest.raises(TypeError):
        vx.save_transform(torch.eye(4), tmp_path / 'tensor.vxx')
    field.save(tmp_path / 'field.vxx')
    with pytest.raises(TypeError):
        vx.load_affine(tmp_path / 'field.vxx')
    with pytest.raises(ValueError):
        vx.load_transform(tmp_path / 'field.vxx', fmt='unknown')
    with pytest.raises(ValueError):
        vx.save_transform(field, tmp_path / 'field.xyz')


def test_vector_field_saves_as_volume(tmp_path) -> None:
    pytest.importorskip('nibabel')
    field = vx.VectorField(torch.rand(3, 5, 6, 7), nontrivial_geometry((5, 6, 7)), space='world')
    path = tmp_path / 'field.nii.gz'
    field.save(path)
    loaded = vx.load_volume(path)
    assert torch.allclose(loaded.tensor, field.tensor)
    assert not isinstance(loaded, vx.VectorField)
