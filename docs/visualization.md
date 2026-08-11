# Visualization

Voxel ships with *Monocle*, an interactive browser viewer for volumes, segmentations, and meshes, along with utilities that render 2D snapshot images for quick checks, experimental logging, or figure rendering.

## Monocle

Monocle is a browser-based viewer bundled with voxel for interactive inspection of imaging data. `vol.show()` opens a volume directly, and `vx.monocle.show` is the one-shot form for multiple sources, including volumes, numpy arrays, nibabel images, and file paths.

```python
import voxel as vx

vol.show()
vx.monocle.show(vol, seg, names=['image', 'segmentation'])
```

The `vx.Monocle` builder composes a session of images, segmentations, and meshes with per-item display options. Segmentations fall back to the volume's own label lookup for names and colors, and meshes draw as contours where they cut each slice plane, with vertices expected in world coordinates.

```python
viewer = vx.Monocle(title='qc', linked=True)
viewer.image(vol, name='t1w', window=(100, 200))
viewer.segmentation(seg, opacity=0.5)
viewer.mesh(surface, color='orange')
viewer.show()
```

`viewer.write('scene.html')` saves the session as a standalone HTML file for sharing, and `viewer.html()` returns the markup directly, e.g. for notebook embedding.

## Snapshots

`vx.snapshot` renders volume slices as channels-last $(H, W, 3)$ uint8 image tensors. The default call returns a single slice through the middle of the volume, along the axial view direction.

```python
image = vx.snapshot(vol)
```

`view` selects the slicing direction, either `'axial'`, `'coronal'`, or `'sagittal'`, and `num_slices` renders multiple slices, evenly spaced along that direction while skipping the often empty extremes of the volume. A single tensor is returned for one slice and a list otherwise.

```python
images = vx.snapshot(vol, view='coronal', num_slices=5)
```

To target a specific location, `coord` renders the one slice passing through a world-space $(x, y, z)$ coordinate.

```python
image = vx.snapshot(vol, coord=point)
```

Grayscale volumes are contrast-normalized automatically with pooled statistics for outlier robustness. Three-channel volumes are treated as RGB and must already lie in $[0, 1]$. Passing a list of volumes composites them in order on the first volume's grid. `res` sets the output resolution, and `square=True` yields square images.

## Label overlays

Segmentations overlay through the `label` argument. A label volume with values greater than one is treated as a discrete label map, resampled with nearest neighbor and colored per unique value, using the colors of its own [label lookup](labels.md) when one is attached. Values within $[0, 1]$ are treated as soft masks with one class per channel.

```python
image = vx.snapshot(vol, label=seg, alpha=0.4)
image = vx.snapshot(vol, label=mask, label_colors=[[1, 0, 0]])
image = vx.snapshot(vol, label=seg, outline=True)
```

`label_colors` overrides any lookup colors and cycles to match the class count, and `alpha` sets the overlay opacity.

## Feature volumes

`vx.pca` projects high-dimensional feature channels, such as network activations, onto principal components for RGB display. Given multiple volumes, one shared basis is fit so their colormaps are comparable, and `mask` restricts the fit to foreground voxels.

```python
rgb = vx.pca(features)  # (64, W, H, D) to (3, W, H, D)
rgb_a, rgb_b = vx.pca([feats_a, feats_b], mask=brain_mask)
image = vx.snapshot(rgb)
```
