import pytest
import torch
import voxel as vx

from conftest import nontrivial_geometry


def test_pca_output() -> None:
    volume = nontrivial_geometry((16, 18, 20)).rand_like(channels=4)
    output = vx.pca(volume)
    assert isinstance(output, vx.Volume)
    assert output.num_channels == 3
    assert output.baseshape == volume.baseshape
    assert output.tensor.min() >= 0 and output.tensor.max() <= 1

    # results are deterministic (the SVD sign ambiguity is resolved)
    assert torch.equal(vx.pca(volume).tensor, output.tensor)

    # a list input fits a single shared basis and returns a list
    outputs = vx.pca([volume, volume])
    assert isinstance(outputs, list) and len(outputs) == 2
    assert torch.allclose(outputs[0].tensor, outputs[1].tensor)

    # basis contents are returned on request
    _, basis = vx.pca(volume, return_basis=True)
    assert basis['components'].shape == (4, 3)


def test_pca_masked() -> None:
    volume = nontrivial_geometry((16, 18, 20)).rand_like(channels=4)
    mask = volume.geometry.zeros_like(dtype=torch.bool)
    mask.tensor[0, 4:12] = True
    output = vx.pca(volume, mask=mask)
    assert output.num_channels == 3


def test_pca_validation() -> None:
    volume = nontrivial_geometry((16, 18, 20)).rand_like(channels=4)
    other = volume.geometry.rand_like(channels=2)

    with pytest.raises(ValueError):
        vx.pca([])
    with pytest.raises(ValueError):
        vx.pca([volume, other])
    with pytest.raises(ValueError):
        vx.pca(volume, n_components=5)
    with pytest.raises(ValueError):
        vx.pca(volume, quantile=0.6)
    with pytest.raises(ValueError):
        vx.pca(volume, normalize='sigmoid')
    with pytest.raises(ValueError):
        vx.pca([volume, volume], mask=[volume.geometry.ones_like(dtype=torch.bool)] * 3)
