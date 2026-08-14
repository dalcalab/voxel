---
description: Work with segmentation label maps in PyTorch — lookup tables that pair label values with names and display colors and follow the volume through its operations.
---

# Label Maps

Label maps, or segmentations, are volumes of integer label values. A [`LabelLookup`](reference/classes/label-lookup.md) annotates those values with names and display colors and travels with the volume through its operations.

## Lookup tables

A `LabelLookup` is an ordered dict mapping integer values to [`Label`](reference/classes/label.md) entries, each a name with an optional RGB color. Assignments accept a name, a `(name, color)` pair, or a `Label`.

```python
import voxel as vx

labels = vx.LabelLookup()
labels[0] = 'Unknown'
labels[10] = vx.Label('Left Hippocampus', color=(0.9, 0.2, 0.2))
labels[11] = ('Right Hippocampus', (0.1, 0.4, 0.8))

labels.search('hippo')     # [(10, Label), (11, Label)] substring match
labels.colors()            # (N, 3) RGB tensor in table order
```

Attach a lookup at construction with `vx.Volume(tensor, labels=labels)` or through the `labels` property.

```python
seg = vx.load_volume('seg.nii.gz')
seg.labels = labels
```

Labels follow the volume through operations that preserve its meaning as a label map, such as cropping, reorientation, and dtype casts. Operations that break that meaning, such as one-hot encoding and comparisons, drop them.

## Recoding and encoding

Label values are often sparse, drawn from a larger convention like the anatomical codes 10 and 11 above. `recode` converts between these sparse values and the compact indices 0 to N-1, following the entry order of a lookup. `onehot` expands a label map into one binary channel per class, and `collapse` reduces a channel encoding back to a single map of label values. Each accepts a lookup, so the value-to-index mapping happens in one step.

```python
compact = seg.recode(labels, reverse=True)          # sparse values to indices 0..N-1
restored = compact.recode(labels)                   # back to values, lookup re-attached

hot = seg.onehot(labels=labels, background=True)    # (N+1, W, H, D), channel 0 background
merged = probs.collapse(labels=labels)              # argmax channels to label values
```

Probability maps reduce with the usual channel operations.

```python
probs = logits.softmax()          # softmax over channels
seg = probs.argmax()              # discrete label map
ventricles = seg.isin(torch.tensor([4, 43]))
seg.unique()                      # values present in the map
```

The same operations exist for plain tensors as `vx.labels.recode`, `vx.labels.onehot`, and `vx.labels.collapse`.

## Masks and morphology

The binary morphology operations `dilate`, `erode`, `close`, and `open` act on the nonzero voxels of a volume and are available directly as `Volume` methods.

```python
mask = seg > 0
mask = mask.close(iterations=2)              # seal small gaps
mask = mask.dilate(iterations=1, connectivity=3)
```

`connectivity` selects the neighborhood, where 1 includes faces (6 neighbors), 2 adds edges (18), and 3 adds corners (26). For thick-slice volumes, `iso_thresh` restricts the kernel to in-plane operation when the slice spacing is disproportionate.

The `voxel.morphology` module also provides labeling utilities. For now, these run on the CPU via scipy and return results to the original device.

```python
comps = vx.morphology.connected_components(mask)   # labeled, largest is 1
main = vx.morphology.connected_components(mask, largest=True)
filled = vx.morphology.fill_holes(mask)
region = vx.morphology.flood_fill(vol, point, space='world')
```

## Reading and writing

Lookups serialize to CSV or TSV tables of index, name, and hex color columns.

```python
labels.save('labels.csv')
labels = vx.load_labels('labels.csv')
```

When a volume with labels is saved to NIfTI, the lookup is embedded directly in the file. It is serialized as a small JSON block, a `labels` list of index, name, and hex color entries, stored in a header comment extension, and the header intent code is set to `label`. On load, any header extension holding such a block is parsed back into the volume's lookup, so the annotation survives a round trip. MGH files embed the lookup through the format's native lookup support, which requires a color for every entry (black is assigned to labels without one). The torch volume format does not store labels.

For rendering colored segmentation overlays, see [Visualization](visualization.md).
