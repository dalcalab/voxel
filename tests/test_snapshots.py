import pytest
import torch
import voxel as vx

from conftest import nontrivial_geometry


@pytest.fixture
def volume() -> vx.Volume:
    return nontrivial_geometry((16, 18, 20)).rand_like()


def test_snapshot_output(volume) -> None:

    # a single slice is a channels-last (H, W, 3) uint8 RGB image tensor
    image = vx.snapshot(volume)
    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.uint8
    assert image.ndim == 3 and image.shape[-1] == 3
    assert image.shape[0] == 256

    # multiple slices are returned as a list
    images = vx.snapshot(volume, num_slices=3)
    assert isinstance(images, list) and len(images) == 3
    assert all(i.shape == images[0].shape for i in images)

    # a world coordinate selects a single slice
    coord = vx.snapshot(volume, coord=volume.geometry.center)
    assert isinstance(coord, torch.Tensor) and coord.shape[-1] == 3
    assert coord.shape[0] == 256


def test_snapshot_resolution(volume) -> None:

    # the image height matches res and the width preserves the physical
    # aspect ratio of the slice, regardless of the input voxel spacing
    image = vx.snapshot(volume, res=64)
    assert image.shape[0] == 64
    geometry = volume.geometry.reorient(vx.snapshots.VIEWS['axial'])
    extent = geometry.spacing * torch.tensor(geometry.baseshape, dtype=torch.float32)
    assert abs(image.shape[1] - 64 * extent[2] / extent[1]) <= 1

    # square output crops or pads the width to match the height
    square = vx.snapshot(volume, res=64, square=True)
    assert square.shape == (64, 64, 3)

    # nearest-neighbor resampling is supported
    nearest = vx.snapshot(volume, res=64, resample='nearest')
    assert nearest.shape == image.shape

    # nearest resampling preserves integer dtypes, which must still pass
    # through pooling and contrast normalization
    quantized = (volume * 255).int()
    image = vx.snapshot(quantized, res=64, resample='nearest')
    assert image.dtype == torch.uint8


def test_snapshot_labels(volume) -> None:
    label = (volume > 0.5).float()
    plain = vx.snapshot(volume)
    labeled = vx.snapshot(volume, label=label)
    assert labeled.shape == plain.shape
    assert not torch.equal(labeled, plain)

    # explicit label colors are accepted
    colored = vx.snapshot(volume, label=label, label_colors=[1.0, 0, 0])
    assert colored.shape == plain.shape


def test_snapshot_multilabel(volume) -> None:

    def contains(image, color):
        color = (torch.as_tensor(color, dtype=torch.float32) * 255).round().byte()
        return (image == color.view(1, 1, 3)).all(-1).any()

    # an integer labelmap with two foreground classes
    labelmap = torch.zeros(1, *volume.baseshape, dtype=torch.int64)
    labelmap[0, 2:8] = 17
    labelmap[0, 10:14] = 42

    # lookup colors are used when the volume carries a label set. at full
    # opacity the fill regions hold the exact class colors
    lut = vx.LabelLookup()
    lut[17] = vx.Label('one', [0.8, 0.2, 0.0])
    lut[42] = vx.Label('two', [0.0, 0.4, 0.6])
    seg = vx.Volume(labelmap, volume.geometry, labels=lut)
    image = vx.snapshot(volume, label=seg, alpha=1.0)
    assert contains(image, [0.8, 0.2, 0.0]) and contains(image, [0.0, 0.4, 0.6])

    # without a lookup, the palette is cycled across the class values
    seg = vx.Volume(labelmap, volume.geometry)
    image = vx.snapshot(volume, label=seg, alpha=1.0)
    assert contains(image, vx.snapshots.PALETTE[0]) and contains(image, vx.snapshots.PALETTE[1])

    # a colorless lookup entry falls back to the palette
    lut = vx.LabelLookup()
    lut[17] = vx.Label('one', [0.8, 0.2, 0.0])
    lut[42] = vx.Label('two')
    seg = vx.Volume(labelmap, volume.geometry, labels=lut)
    image = vx.snapshot(volume, label=seg, alpha=1.0)
    assert contains(image, [0.8, 0.2, 0.0]) and contains(image, vx.snapshots.PALETTE[1])

    # explicit label colors override the lookup
    image = vx.snapshot(volume, label=seg, alpha=1.0, label_colors=[[1.0, 0, 0], [0, 1.0, 0]])
    assert contains(image, [1.0, 0, 0]) and contains(image, [0, 1.0, 0])
    assert not contains(image, [0.8, 0.2, 0.0])

    # a floating-point volume with values beyond one is conformed to an
    # integer labelmap and renders identically
    assert torch.equal(vx.snapshot(volume, label=seg.float(), alpha=1.0),
                       vx.snapshot(volume, label=seg, alpha=1.0))

    # a volume within [0, 1] remains a single soft mask, merging the classes
    merged = vx.snapshot(volume, label=(seg > 0).float(), alpha=1.0)
    assert not torch.equal(merged, vx.snapshot(volume, label=seg, alpha=1.0))


def test_snapshot_outline(volume) -> None:
    label = (volume > 0.5).float()
    plain = vx.snapshot(volume)

    # an outline changes the image on top of the alpha-blended fill
    labeled = vx.snapshot(volume, label=label)
    outlined = vx.snapshot(volume, label=label, outline=True)
    assert not torch.equal(outlined, labeled)

    # the outline is drawn fully opaque even when the fill is invisible, so
    # every differing pixel is exactly the label color
    outlined = vx.snapshot(volume, label=label, alpha=0.0, outline=True,
                           label_colors=[1.0, 0, 0])
    diff = (outlined != plain).any(dim=-1)
    assert diff.any()
    assert torch.all(outlined[diff] == torch.tensor([255, 0, 0], dtype=torch.uint8))

    # the outline lies within the label region, so at full fill opacity it is
    # indistinguishable from the fill itself
    assert torch.equal(vx.snapshot(volume, label=label, alpha=1.0, outline=True),
                       vx.snapshot(volume, label=label, alpha=1.0))

    # a label covering the whole image is not eroded at the borders, so the
    # outline disappears at the image edges
    plain = vx.snapshot(volume, resample='nearest')
    covered = vx.snapshot(volume, label=volume.ones_like(), alpha=0.0,
                          outline=True, resample='nearest')
    assert torch.equal(covered, plain)


def test_snapshot_validation(volume) -> None:
    with pytest.raises(ValueError):
        vx.snapshot()
    with pytest.raises(ValueError):
        vx.snapshot(volume, view='oblique')
    with pytest.raises(ValueError):
        vx.snapshot(volume, num_slices=0)
    with pytest.raises(ValueError):
        vx.snapshot(volume, res=0)

    # only 1 (grayscale) or 3 (RGB) channel volumes are supported
    with pytest.raises(ValueError):
        vx.snapshot(vx.volume.stack(volume, volume))

    # RGB inputs must lie within [0, 1]
    rgb = vx.volume.stack(volume, volume, volume) * 10
    with pytest.raises(ValueError):
        vx.snapshot(rgb)


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
