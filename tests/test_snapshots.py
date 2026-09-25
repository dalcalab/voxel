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


@pytest.fixture
def aligned() -> vx.Volume:
    # an isotropic RAS volume, so that with res matching the grid size the
    # projection samples land exactly on voxel centers
    return vx.AcquisitionGeometry((12, 12, 12)).rand_like(channels=2)


def to_image(reduced: torch.Tensor) -> torch.Tensor:
    # convert a (C, A, B) axis-aligned reduction into a (B, A, C) image with
    # both image axes flipped, matching the upright projection layout
    return reduced.transpose(1, 2).flip(1, 2).movedim(0, -1)


def test_projection_output(volume) -> None:

    # the projection is a channels-last (H, W, C) float image
    image = vx.projection(volume, res=64)
    assert image.shape == (64, 64, 1)
    assert image.dtype == torch.float32

    # channels are projected independently
    stacked = vx.volume.stack(volume, volume * 2)
    image = vx.projection(stacked, 'mean', res=32)
    assert image.shape == (32, 32, 2)
    assert torch.allclose(image[..., 1], image[..., 0] * 2, atol=1e-5)


def test_projection_aligned(aligned) -> None:

    # an anterior view reduces along y, with superior at the top of the image
    # and the subject's left on the right
    tensor = aligned.tensor
    reductions = dict(max=tensor.amax(2), min=tensor.amin(2), mean=tensor.mean(2))
    for mode, reduced in reductions.items():
        image = vx.projection(aligned, mode, res=12)
        assert torch.allclose(image, to_image(reduced), atol=1e-5)

    # the viewpoint does not need to be a unit vector
    image = vx.projection(aligned, res=12)
    assert torch.allclose(vx.projection(aligned, viewpoint=(0, 5, 0), res=12), image, atol=1e-5)

    # an inferior view reduces along z, with anterior at the top of the image
    inferior = vx.projection(aligned, viewpoint=(0, 0, -1), res=12)
    assert torch.allclose(inferior, to_image(tensor.amax(3)), atol=1e-5)


def test_projection_framing(aligned) -> None:
    image = vx.projection(aligned, res=12)

    # moving the center along image-right shifts the image content left
    center = aligned.geometry.center + torch.tensor([-3.0, 0, 0])
    shifted = vx.projection(aligned, center=center, res=12)
    assert torch.allclose(shifted[:, :-3], image[:, 3:], atol=1e-5)

    # a larger viewport at the same pixel size pads the image with the fill
    # value, which is the volume minimum for a maximum projection
    padded = vx.projection(aligned, viewport=24, res=24)
    assert torch.allclose(padded[6:18, 6:18], image, atol=1e-5)
    assert torch.all(padded[:6] == aligned.min())


def test_projection_depth(aligned) -> None:
    tensor = aligned.tensor

    # a depth restricts the projection to a slab around the center, here the
    # four voxel planes nearest the center along y
    image = vx.projection(aligned, depth=4, res=12)
    assert torch.allclose(image, to_image(tensor[:, :, 4:8].amax(2)), atol=1e-5)

    # the slab follows the center along the view direction
    center = aligned.geometry.center + torch.tensor([0, 3.0, 0])
    image = vx.projection(aligned, center=center, depth=4, res=12)
    assert torch.allclose(image, to_image(tensor[:, :, 7:11].amax(2)), atol=1e-5)

    # a depth beyond the volume extent covers the whole volume
    assert torch.allclose(vx.projection(aligned, depth=100, res=12),
                          vx.projection(aligned, res=12), atol=1e-5)


def test_projection_viewpoint(volume) -> None:

    # opposite viewpoints sample the same rays, mirroring the image
    viewpoint = torch.tensor([0.3, -0.5, 0.8])
    front = vx.projection(volume, viewpoint=viewpoint, res=32)
    back = vx.projection(volume, viewpoint=-viewpoint, res=32)
    assert torch.allclose(back, front.flip(1), atol=1e-4)


def test_projection_exclude(aligned) -> None:
    tensor = aligned.tensor
    image = vx.projection(aligned, res=12)

    # excluding the brightest voxel lowers the maximum along its ray
    exclude = tensor[:1] == tensor[0].max()
    excluded = vx.projection(aligned, exclude=exclude, res=12)
    assert excluded[..., 0].max() < image[..., 0].max()

    # the mean is taken over the remaining samples along each ray
    exclude = aligned.zeros_like(channels=1)
    exclude.tensor[:, :, 6:] = 1
    mean = vx.projection(aligned, 'mean', exclude=exclude, res=12)
    assert torch.allclose(mean, to_image(tensor[:, :, :6].mean(2)), atol=1e-5)

    # excluding everything leaves only the fill value
    everything = aligned.ones_like(channels=1)
    assert torch.all(vx.projection(aligned, exclude=everything, res=12) == aligned.min())
    assert torch.all(vx.projection(aligned, 'mean', exclude=everything, res=12) == 0)


def test_projection_directions(aligned) -> None:
    tensor = aligned.tensor

    # axis letters are equivalent to their world-space vectors
    assert torch.allclose(vx.projection(aligned, viewpoint='I', res=12),
                          vx.projection(aligned, viewpoint=(0, 0, -1), res=12), atol=1e-5)

    # flipping the up direction rotates the image by 180 degrees
    inferior = vx.projection(aligned, viewpoint='I', res=12)
    flipped = vx.projection(aligned, viewpoint='I', up='P', res=12)
    assert torch.allclose(flipped, inferior.flip(0, 1), atol=1e-5)

    # with right as up, the image rows run toward the left and the columns
    # toward superior
    image = vx.projection(aligned, viewpoint='A', up='R', res=12)
    assert torch.allclose(image, tensor.amax(2).flip(1).movedim(0, -1), atol=1e-5)

    # the up direction only needs to be roughly upward
    tilted = vx.projection(aligned, up=(0, 0.5, 1), res=12)
    assert torch.allclose(tilted, vx.projection(aligned, res=12), atol=1e-5)


def test_projection_validation(volume) -> None:
    with pytest.raises(ValueError):
        vx.projection(volume, mode='median')
    with pytest.raises(ValueError):
        vx.projection(volume, viewpoint=(0, 0, 0))
    with pytest.raises(ValueError):
        vx.projection(volume, viewpoint='X')
    with pytest.raises(ValueError):
        vx.projection(volume, viewpoint=(0, 1))
    with pytest.raises(ValueError):
        vx.projection(volume, viewpoint='A', up='P')
    with pytest.raises(ValueError):
        vx.projection(volume, res=0)
    with pytest.raises(ValueError):
        vx.projection(volume, viewport=0)
    with pytest.raises(ValueError):
        vx.projection(volume, depth=0)


def test_sliding_projection(volume) -> None:
    tensor = volume.tensor
    reductions = dict(max=torch.amax, min=torch.amin, mean=torch.mean)

    def naive(dim, num, reduce):
        # reduce a truncated window of num slices around each slice, with the
        # extra slice of an even window on the upper side
        slices = []
        for i in range(tensor.shape[dim + 1]):
            lo, hi = max(0, i - (num - 1) // 2), min(tensor.shape[dim + 1], i + num // 2 + 1)
            slices.append(reduce(tensor.narrow(dim + 1, lo, hi - lo), dim + 1))
        return torch.stack(slices, dim + 1)

    # odd and even windows along every axis, with depth in world units
    for dim in range(3):
        spacing = float(volume.geometry.spacing[dim])
        for num in (1, 3, 4):
            for mode, reduce in reductions.items():
                result = vx.sliding_projection(volume, depth=num * spacing, mode=mode, dim=dim)
                assert vx.geometries_equal(result.geometry, volume.geometry)
                assert torch.allclose(result.tensor, naive(dim, num, reduce), atol=1e-6)

    # the default axis is the slice direction
    dim = int(volume.geometry.slice_direction)
    spacing = float(volume.geometry.spacing[dim])
    assert torch.equal(vx.sliding_projection(volume, depth=3 * spacing).tensor,
                       vx.sliding_projection(volume, depth=3 * spacing, dim=dim).tensor)

    # view names select the voxel axis closest to the view direction, in any
    # voxel orientation
    views = dict(S='axial', I='axial', A='coronal', P='coronal', L='sagittal', R='sagittal')
    for vol in (volume, volume.reorient('PSL')):
        for dim, letter in enumerate(vol.geometry.orientation.name):
            spacing = float(vol.geometry.spacing[dim])
            expected = vx.sliding_projection(vol, depth=3 * spacing, dim=dim).tensor
            for view in (views[letter], views[letter][0]):
                assert torch.equal(vx.sliding_projection(vol, depth=3 * spacing, dim=view).tensor, expected)

    # a window thinner than a slice leaves the volume unchanged, and max
    # projections preserve integer data types
    quantized = (volume * 255).int()
    assert torch.equal(vx.sliding_projection(quantized, depth=0.1).tensor, quantized.tensor)
    assert vx.sliding_projection(quantized, depth=5).dtype == torch.int32

    with pytest.raises(ValueError):
        vx.sliding_projection(volume, depth=5, mode='median')
    with pytest.raises(ValueError):
        vx.sliding_projection(volume, depth=0)
    with pytest.raises(ValueError):
        vx.sliding_projection(volume, depth=5, dim=3)
    with pytest.raises(ValueError):
        vx.sliding_projection(volume, depth=5, dim='oblique')
