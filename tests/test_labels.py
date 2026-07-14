import pytest
import torch
import voxel as vx

from conftest import nontrivial_geometry


def example_lookup() -> vx.LabelLookup:
    lut = vx.LabelLookup()
    lut[0] = 'Unknown'
    lut[17] = vx.Label('Left-Hippocampus', [0.9, 0.2, 0.2])
    lut[42] = ('Right-Amygdala', [0.1, 0.4, 0.8])
    return lut


def test_label_construction() -> None:
    label = vx.Label('Hippocampus', [0.9, 0.2, 0.2])
    assert label.name == 'Hippocampus'
    assert torch.allclose(label.color, torch.tensor([0.9, 0.2, 0.2]))

    assert vx.Label('Background').color is None

    with pytest.raises(ValueError):
        vx.Label('bad', [0.1, 0.2])
    with pytest.raises(ValueError):
        vx.Label('bad', [1.5, 0.2, 0.3])


def test_lookup_setget_and_coercion() -> None:
    lut = example_lookup()
    assert isinstance(lut[17], vx.Label)
    assert lut[17].name == 'Left-Hippocampus'
    assert lut[0].color is None
    assert torch.allclose(lut[42].color, torch.tensor([0.1, 0.4, 0.8]))

    lut.add(99, 'Extra', [0.0, 0.0, 0.0])
    assert lut[99].name == 'Extra'

    with pytest.raises(ValueError):
        lut['bad'] = 'nope'
    with pytest.raises(ValueError):
        lut[True] = 'no bools'


def test_lookup_search() -> None:
    lut = example_lookup()
    matches = lut.search('hippo')
    assert len(matches) == 1
    assert matches[0][0] == 17

    assert lut.search('AMYG')[0][1].name == 'Right-Amygdala'
    assert lut.search('Unknown', exact=True)[0][0] == 0
    assert lut.search('nknow', exact=True) == []


def test_lookup_indices_and_colors() -> None:
    lut = example_lookup()
    assert lut.indices == [0, 17, 42]

    colors = lut.colors(default=(1, 1, 1))
    assert colors.shape == (3, 3)
    assert torch.allclose(colors[0], torch.tensor([1.0, 1.0, 1.0]))  # default for no-color label
    assert torch.allclose(colors[1], torch.tensor([0.9, 0.2, 0.2]))


def test_recode_tensor_util_forward_reverse() -> None:
    # forward maps index positions to mapping values (the documented example)
    tensor = torch.tensor([0, 3, 0, 2, 0, 1])
    forward = vx.labels.recode(tensor, [0, 2, 4, 8])
    assert forward.tolist() == [0, 8, 0, 4, 0, 2]

    # reverse maps values back to their index positions
    reverse = vx.labels.recode(forward, [0, 2, 4, 8], reverse=True)
    assert torch.equal(reverse, tensor)

    # forward values may be of any dtype
    floated = vx.labels.recode(tensor, [0.0, 0.5, 1.0, 1.5])
    assert torch.is_floating_point(floated)

    # reverse values not present in the mapping fall back to the default
    dropped = vx.labels.recode(torch.tensor([0, 7, 2]), [0, 2, 4], reverse=True, default=0)
    assert dropped.tolist() == [0, 0, 1]

    with pytest.raises(ValueError):
        vx.labels.recode(tensor, [0.0, 0.5], reverse=True)  # reverse needs integers

    with pytest.raises(ValueError):
        vx.labels.recode(tensor.float(), [0, 2, 4, 8])  # forward needs an integer index map


def test_volume_recode_roundtrip() -> None:
    geometry = nontrivial_geometry()
    lut = vx.LabelLookup()
    lut[0] = 'Unknown'
    lut[5] = 'A'
    lut[17] = 'B'
    lut[42] = 'C'
    values = torch.tensor(lut.indices)
    seg = vx.Volume(values[torch.randint(0, 4, (10, 12, 14))], geometry, labels=lut)

    # real label values -> sparse channel indices
    sparse = seg.recode(lut, reverse=True)
    assert int(sparse.tensor.max()) == 3
    assert sparse.labels is None

    # sparse indices -> real label values (attaches the lookup)
    restored = sparse.recode(lut)
    assert torch.equal(restored.tensor, seg.tensor)
    assert restored.labels is lut


def test_onehot_collapse_roundtrip() -> None:
    geometry = nontrivial_geometry()
    lut = vx.LabelLookup()
    lut[0] = 'Unknown'
    lut[5] = 'A'
    lut[17] = 'B'
    lut[42] = 'C'
    values = torch.tensor(lut.indices)
    seg = vx.Volume(values[torch.randint(0, 4, (10, 12, 14))], geometry, labels=lut)

    encoded = seg.onehot(labels=lut)
    assert encoded.num_channels == 4
    assert encoded.labels is None

    recovered = encoded.collapse(labels=lut)
    assert torch.equal(recovered.tensor.squeeze(0), seg.tensor.squeeze(0))
    assert recovered.labels is lut

    # integer labels behave like plain one-hot / argmax
    plain = vx.Volume(torch.randint(0, 4, (10, 12, 14)), geometry)
    assert plain.onehot().num_channels == 4
    assert torch.equal(plain.onehot().collapse().tensor.squeeze(0), plain.tensor.squeeze(0).long())


def test_background_flag() -> None:
    # foreground-only labels; background 0 is not listed
    fg = [17, 42]

    # reverse recode without background sends unlisted 0 to the default (0),
    # colliding with label 17's index; with background, 0 gets its own slot
    tensor = torch.tensor([0, 17, 42])
    assert vx.labels.recode(tensor, fg, reverse=True).tolist() == [0, 0, 1]
    assert vx.labels.recode(tensor, fg, reverse=True, background=True).tolist() == [0, 1, 2]

    # onehot reserves a leading background channel
    assert vx.labels.onehot(tensor, labels=fg).shape[0] == 2
    encoded = vx.labels.onehot(tensor, labels=fg, background=True)
    assert encoded.shape[0] == 3
    assert encoded[:, 0].tolist() == [1, 0, 0]  # background voxel -> channel 0

    # collapse round-trips through the background-aware channels
    recovered = vx.labels.collapse(encoded, labels=fg, background=True)
    assert recovered.tolist() == [0, 17, 42]

    # if 0 is already present it is not duplicated
    assert vx.labels.onehot(tensor, labels=[0, 17, 42], background=True).shape[0] == 3


def test_volume_background_roundtrip() -> None:
    geometry = nontrivial_geometry()
    lut = vx.LabelLookup()
    lut[17] = 'B'
    lut[42] = 'C'
    seg = vx.Volume(torch.tensor([0, 17, 42])[torch.randint(0, 3, (10, 12, 14))], geometry)

    encoded = seg.onehot(labels=lut, background=True)
    assert encoded.num_channels == 3
    recovered = encoded.collapse(labels=lut, background=True)
    assert torch.equal(recovered.tensor.squeeze(0), seg.tensor.squeeze(0))


def test_lookup_device_and_caching() -> None:
    lut = example_lookup()

    # the ordered-index tensor is cached until the keys change
    first = lut.indices_tensor
    assert first is lut.indices_tensor
    assert first.tolist() == [0, 17, 42]
    lut.add(99, 'Extra')
    assert lut.indices_tensor is not first
    assert lut.indices_tensor.tolist() == [0, 17, 42, 99]

    # device transfer returns a copy carrying the colors and device
    moved = lut.cpu()
    assert moved is not lut
    assert moved.device == torch.device('cpu')
    assert moved[17].name == 'Left-Hippocampus'
    assert torch.allclose(moved[42].color, lut[42].color)
    assert moved.indices_tensor.device == torch.device('cpu')

    # every key-changing mutator invalidates the cache
    cached = lut.indices_tensor
    del lut[99]
    assert lut.indices_tensor is not cached
    assert lut.indices_tensor.tolist() == [0, 17, 42]

    lut.pop(42)
    after_pop = lut.indices_tensor
    assert after_pop.tolist() == [0, 17]
    lut.pop(12345, None)  # missing key + default: no mutation, cache preserved
    assert lut.indices_tensor is after_pop

    lut.clear()
    assert lut.indices_tensor.tolist() == []


def test_labels_propagation_kept() -> None:
    geometry = nontrivial_geometry()
    lut = example_lookup()
    seg = vx.Volume(torch.randint(0, 43, (10, 12, 14)), geometry, labels=lut)

    assert seg.copy().labels is lut
    assert seg.crop((slice(0, 5), slice(0, 6), slice(0, 7))).labels is lut
    assert seg.reorient('LIA').labels is lut
    assert seg.resample_like(seg).labels is lut
    assert seg.int().labels is lut
    assert seg.argmax(dim=0).labels is lut


def test_labels_propagation_dropped() -> None:
    geometry = nontrivial_geometry()
    lut = example_lookup()
    seg = vx.Volume(torch.randint(0, 5, (10, 12, 14)), geometry, labels=lut)

    assert seg.onehot().labels is None
    assert seg.float().softmax(dim=0).labels is None
    assert (seg == 1).labels is None
    assert (seg > 2).labels is None

    # *_like factories build from the geometry and carry no labels
    assert seg.zeros_like().labels is None


def test_labels_setter_typecheck() -> None:
    geometry = nontrivial_geometry()
    seg = vx.Volume(torch.randint(0, 5, (10, 12, 14)), geometry)
    assert seg.labels is None
    with pytest.raises(TypeError):
        seg.labels = {17: 'Hippocampus'}


def test_csv_io_roundtrip(tmp_path) -> None:
    lut = example_lookup()
    path = tmp_path / 'labels.csv'
    lut.save(str(path))

    # colors are stored as hex strings in the color column
    assert '#' in path.read_text()

    reloaded = vx.load_labels(str(path))
    assert reloaded.indices == lut.indices
    assert reloaded[17].name == 'Left-Hippocampus'
    assert reloaded[0].color is None
    assert torch.allclose(reloaded[42].color, lut[42].color, atol=1 / 255)


def test_tsv_io_roundtrip(tmp_path) -> None:
    lut = example_lookup()
    path = tmp_path / 'labels.tsv'
    lut.save(str(path))

    # tab-separated on disk
    assert '\t' in path.read_text()

    reloaded = vx.load_labels(str(path))
    assert reloaded.indices == lut.indices
    assert reloaded[42].name == 'Right-Amygdala'
    assert torch.allclose(reloaded[42].color, lut[42].color, atol=1 / 255)


def test_tabular_io_fmt_and_load(tmp_path) -> None:
    lut = example_lookup()
    path = tmp_path / 'labels.tbl'
    vx.save_labels(lut, str(path), fmt='tabular')  # extension enforced to .csv
    reloaded = vx.LabelLookup.load(str(path.with_suffix('.csv')))
    assert reloaded[17].name == 'Left-Hippocampus'


def _labeled_volume() -> vx.Volume:
    return vx.Volume(torch.randint(0, 43, (10, 12, 14)).int(), nontrivial_geometry(), labels=example_lookup())


def test_nifti_label_embedding(tmp_path) -> None:
    nib = pytest.importorskip('nibabel')
    seg = _labeled_volume()
    path = tmp_path / 'seg.nii.gz'
    seg.save(str(path))

    reloaded = vx.load_volume(str(path))
    assert reloaded.labels is not None
    assert reloaded.labels.indices == seg.labels.indices
    assert reloaded.labels[17].name == 'Left-Hippocampus'
    assert reloaded.labels[0].color is None  # uncolored label stays colorless
    assert torch.allclose(reloaded.labels[42].color, seg.labels[42].color, atol=1 / 255)

    # the header is tagged as a label map
    assert nib.load(str(path)).header.get_intent()[0] == 'label'


def test_nifti_no_labels(tmp_path) -> None:
    pytest.importorskip('nibabel')
    vol = vx.Volume(torch.rand(10, 12, 14), nontrivial_geometry())
    path = tmp_path / 'vol.nii.gz'
    vol.save(str(path))
    assert vx.load_volume(str(path)).labels is None


def test_nifti_foreign_comment_not_misread(tmp_path) -> None:
    nib = pytest.importorskip('nibabel')
    img = nib.Nifti1Image(torch.zeros(4, 4, 4).numpy(), torch.eye(4).numpy())
    img.header.extensions.append(nib.nifti1.Nifti1Extension('comment', b'just a plain comment'))
    path = tmp_path / 'foreign.nii.gz'
    nib.save(img, str(path))
    assert vx.load_volume(str(path)).labels is None


def test_mgh_label_embedding(tmp_path) -> None:
    pytest.importorskip('surfa')
    seg = _labeled_volume()
    path = tmp_path / 'seg.mgz'
    seg.save(str(path))

    reloaded = vx.load_volume(str(path))
    assert reloaded.labels is not None
    assert reloaded.labels[17].name == 'Left-Hippocampus'
    assert torch.allclose(reloaded.labels[42].color, seg.labels[42].color, atol=1 / 255)
