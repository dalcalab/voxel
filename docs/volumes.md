# Volumes & Geometry

A [`Volume`](reference/classes/volume.md) is the core representation of volumetric imaging data, such as MR and CT images. It pairs a multi-channel 3D image tensor with an [`AcquisitionGeometry`](reference/classes/acquisition-geometry.md) that anchors the voxel grid in world coordinates, the physical space of the scanner. As operations like resampling, reshaping, and cropping are applied to a volume, the geometry is updated in step, so the image always stays correctly placed in world space.

## The Volume object

A volume stores a channels-first torch tensor of shape $(C, W, H, D)$, with the spatial dimensions $(W, H, D)$ referred to as the *baseshape*.

```python
import voxel as vx

vol = vx.load_volume('image.nii.gz')

vol.tensor         # (C, W, H, D) torch tensor
vol.baseshape      # spatial shape (W, H, D)
vol.num_channels   # C
vol.geometry       # voxel to world affine relationship
```

A volume can be loaded from file, as above, or constructed directly from a 3D or 4D (multi-channel) tensor. When no geometry is given, the default is a shifted identity that centers the grid on the world origin.

```python
vol = vx.Volume(torch.rand(128, 128, 128), geometry=geometry)
```

Volumes move across devices and dtypes just like tensors.

```python
vol = vol.cuda().float()
vol = vol.to('cpu').type(torch.int32)
```

Beyond geometry handling, volumes are duck-typed like torch tensors, supporting arithmetic, reductions, indexing, and autograd operations like `detach`, as detailed in [Indexing and math](#indexing-and-math).

## Acquisition geometry

Volumes carry an [`AcquisitionGeometry`](reference/classes/acquisition-geometry.md) that comprises, at its core, a 4×4 voxel-to-world [affine](transforms.md#affine-matrices) paired with the spatial grid shape it describes. From this it derives the physical layout of the acquisition.

```python
geom = vol.geometry
geom.spacing        # voxel size in mm, e.g. (1., 1., 8.)
geom.orientation    # anatomical orientation, e.g. RAS
geom.origin         # world position of voxel (0, 0, 0)
geom.center         # world position of the grid center
geom.fov            # field of view extent in mm
geom.is_isotropic() # True if spacing is uniform
```

An [`Orientation`](reference/classes/orientation.md) describes how the stored voxel data is laid out relative to the anatomy of the subject. The world frame follows the RAS convention, in which $x$ points to the subject's right (R), $y$ anterior (A), and $z$ superior (S), so a three-character string like `'RAS'` names the anatomical direction that each grid axis traverses.

The acquisition geometry therefore defines the relationship between two coordinate systems. *Voxel space* is the image grid, integer indices $(i, j, k)$ into the data tensor. *World space* is the physical space of the scanner, continuous $(x, y, z)$ coordinates, typically in millimeters. The affine encodes where the image sits in physical space, the spacing between grid points, and how its axes are oriented. 3D coordinates move between the frames through the geometry or its inverse.

```python
world_coords = vol.geometry.map(voxel_coords)
voxel_coords = vol.geometry.inverse().map(world_coords)
```

Many operations can be parameterized in either voxel or world units through a [`Space`](reference/classes/space.md) argument, so smoothing sigmas, padding margins, and distances can be given in physical units that are invariant to the underlying voxel resolution.

## Resampling and reorientation

Volume data can be resampled onto the grid of a new geometry, for instance to change the voxel spacing.

```python
iso = vol.resample(1.0, antialias=True)       # 1 mm isotropic
aniso = vol.resample(1, 1, 2)                 # per-axis spacing
thick = vol.resample(slice_spacing=3.0)       # change slice spacing only
```

An image can also be conformed to the grid of another image or geometry. Since equal tensor shapes do not imply spatial alignment, this is the standard way to align inputs across images.

```python
vol = vol.resample_like(target)  # target can be a volume or geometry
```

Resampling runs on torch grid interpolation, so it is device-accelerated, and it avoids unnecessary work, skipping interpolation entirely when the grids already coincide or differ only by an integer voxel shift. The interpolation method is set with `mode`, either `'linear'` or `'nearest'` (use nearest for label maps), and `antialias=True` applies a Gaussian prefilter when downsampling.

Grid growth and reduction follow the same pattern, with the geometry updated so content stays fixed in world coordinates.

```python
padded = vol.pad(10, space='world')        # extend grid 10 mm per side
trimmed = vol.trim(2, space='voxel')       # remove 2 voxels per side
low = vol.pool(scale=2, mode='mean')       # fast 2x downsample
big = vol.reshape((256, 256, 256))         # symmetric pad or crop to a shape
```

Operations that change the grid, like cropping, resampling, reorienting, and pooling, return a new volume with an updated geometry, so world coordinates always stay correct, and a mesh or point set defined in world space will still line up with the result.

### Geometries as resampling targets

Every grid operation on a volume is also available on its geometry (`shift`, `scale`, `rotate`, `resample`, `reorient`, `pad`, `reshape`, and others), each returning a new geometry. A geometry edit only updates a matrix and a shape, so it is essentially free, while every volume-level resample interpolates the full image. Rather than chaining expensive resampling operations, compose the target grid on the geometry and resample once.

```python
target = vol.geometry.resample(1.0).reorient('RAS').reshape((192, 192, 192))
conformed = vol.resample_like(target)   # one interpolation instead of three
```

Geometry edits without a volume counterpart, like rotation, build transformed or augmented grids, and a geometry can synthesize new volumes on its grid directly.

```python
rotated = vol.geometry.rotate(0, 0, 10, 'world')   # rotate the grid itself
augmented = vol.resample_like(rotated)             # e.g. rotation augmentation

noise = target.randn_like(channels=4)   # random Volume on the target grid
```

## Indexing and math

Volumes are duck-typed like torch tensors. Operators, reductions, and indexing all work, and every result remains a `Volume` with correct world-space geometry.

```python
diff = (a - b).abs()                     # element-wise math
mask = (vol > 50) & (vol < 500)          # boolean Volume
brain = vol * (seg > 0)                  # zero out background
vol += noise                             # in-place ops supported
```

Operands can be volumes, tensors, or scalars. Reductions follow torch semantics, and reducing over the channel dimension returns a per-voxel `Volume`.

```python
vol.mean()                # scalar tensor over all elements
vol.max(dim=0)            # per-voxel max over channels, still a Volume
vol.quantile(0.95)        # robust intensity statistics
```

Boolean volumes index like boolean tensors, for both reading and writing.

```python
values = vol[mask]        # 1D tensor of masked intensities
vol[vol < 0] = 0          # in-place masked assignment
```

Slice indexing works on the $(C, W, H, D)$ grid, and the geometry shifts automatically so the crop stays put in world space.

```python
sub = vol[:, 32:160, 32:192, :]           # geometry updated internally
tight = vol.crop_to_nonzero(margin=5)     # bounding crop with a margin
roi = vol.crop(box)                       # crop to a world-space BoundingBox
```

Because geometry follows the data, a crop followed by `resample_like` back onto the original grid lands exactly where it started.

## Filtering

Filtering is implemented with torch convolutions and runs on the GPU. Sigmas and kernel extents can be given in world units, so filtering behaves consistently across resolutions.

```python
smooth = vol.smooth(sigma=2, space='world')   # 2 mm Gaussian blur
```

The `voxel.filters` module exposes anisotropic sigmas, strided filtering that downsamples the grid, box kernels, and arbitrary custom kernels.

```python
smooth = vx.filters.gaussian_filter(vol, sigma=[2, 2, 0], space='world')
low = vx.filters.gaussian_filter(vol, sigma=1, space='voxel', stride=2)
local_mean = vx.filters.box_filter(vol, size=5, space='world')
custom = vx.filters.apply_filter(vol, kernel)  # separable 1D or dense 3D kernel
```

## Reading and writing

Volumes load and save through `vx.load_volume` and `vol.save`, with the file format inferred from the extension or forced with the `fmt` parameter.

```python
vol = vx.load_volume('image.nii.gz')
vol.save('image.nii.gz')              # or vx.save_volume(vol, ...)
```

I/O backend libraries are imported lazily, so voxel only requires them when a format needs one. NIfTI files (`.nii`, `.nii.gz`) are read through [nibabel](https://nipy.org/nibabel/), and MGH files (`.mgz`, `.mgh`) through [surfa](https://github.com/freesurfer/surfa). NIfTI and MGH loads keep a reference to the original header, so a load and save round trip preserves it without floating-point drift in the affine.
