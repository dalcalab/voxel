---
description: Apply affine matrices and dense nonlinear deformations to medical images, meshes, and 3D coordinates in PyTorch — differentiable spatial transforms for image registration.
---

# Transforms

Voxel provides representations for both linear (or affine) matrix transforms and nonlinear (or dense) deformations. By convention they act on world coordinates, and they apply uniformly to volumes, meshes, bounding boxes, and bare 3D coordinate points.

## Affine matrices

An [`AffineMatrix`](reference/classes/affine-matrix.md) is a 4×4 transform for 3D coordinates with no assumptions about what it maps between. Compose with `@`, invert with `inverse`, and apply to $(..., 3)$ coordinates with `map`.

```python
import voxel as vx

aff = vx.affine.compose_affine(rotation=[0, 0, 15], translation=[2, 0, 0])
pts = aff.map(pts)                  # transform (N, 3) points
combined = aff @ other.inverse()    # matrix composition
```

## Building affines

The `voxel.affine` module provides constructors for common parameterizations.

```python
aff = vx.affine.compose_affine(translation=t, rotation=r, scale=s, shear=h)
aff = vx.affine.angles_to_rotation_matrix([0, 0, 15])   # Euler angles, degrees
aff = vx.affine.quaternion_to_rotation_matrix(q)        # scalar-first (w, x, y, z)
aff = vx.affine.random_affine(max_translation=10, max_rotation=15)
```

`compose_affine` combines its components in translation, rotation, scale, shear order, accepts a rotation of length 3 (Euler angles) or 4 (quaternion), and is *differentiable* with respect to its inputs. `least_squares_alignment` fits an affine between paired point sets or meshes.

```python
aff = vx.affine.least_squares_alignment(moving_points, fixed_points)
```

## Dense warps

A `Warp` is a dense nonlinear deformation, stored as a $(W, H, D, 3)$ tensor of absolute world coordinates over a fixed-side grid defined by its geometry. Each grid point holds the world coordinate at which a moving volume is sampled, a pull-back mapping.

```python
warp = vx.Warp(coordinates, geometry)
warped = warp.map(moving)  # sample moving at the mapped coordinates
```

The `as_displacement_field` method returns the world-space displacements relative to the identity grid as a `VectorField`, described below.

## Vector fields

A `VectorField` is a 3-channel `Volume` subclass holding a field of 3D vectors, such as displacements or flow velocities. The required `space` argument declares whether the vector values are in `'world'` or `'voxel'` units, a property distinct from the geometry of the grid the field is sampled on.

```python
field = vx.VectorField(tensor, geometry, space='world')
field = field.in_space('voxel')     # convert vector units through the geometry
warp = field.as_warp()              # identity grid plus world displacements
```

Since it is a volume, a vector field supports the full volume API, including `resample_like`, indexing, and device moves.

Velocity fields integrate 3D points through the flow and exponentiate into displacement fields.

```python
ends = field.integrate(points, dt=0.1, method='rk2')  # flow points
disp = field.exponentiate(steps=8)                    # scaling & squaring
```

`integrate` uses fixed-step forward Euler or midpoint rk2, and points outside the field extent see zero velocity. Setting `exact_gradient=False` drops the rk2 midpoint from the autograd graph, roughly halving backward cost in exchange for a small step-dependent gradient bias. `exponentiate` applies scaling and squaring to a stationary velocity field and returns displacements in the same vector space.

Mesh vertices deform by sampling a displacement field or integrating through a velocity field.

```python
moved = mesh.new(mesh.vertices + disp.sample(mesh, space='world'))
moved = mesh.new(field.integrate(mesh.vertices, dt=0.1, space='world'))
```

## Composition

`compose_transforms` merges any sequence of affines and warps into a single equivalent transform. Arguments are given in application order, the order in which they would be applied to a volume. The result is an `AffineMatrix` when all inputs are affine and a `Warp` otherwise.

```python
merged = vx.compose_transforms(aff, warp)
moved = moving.transform(merged)
```

Composing before applying avoids compounding interpolation error. An affine following a warp only moves the output domain of the warp, and points a warp maps beyond the extent of a preceding warp see zero displacement from it.

## Applying transforms

`Volume.transform` applies a world-space affine or a dense warp. By default an affine only updates the geometry, so the volume tensor is untouched and the image simply moves in world space. With `resample=True` the data is instead interpolated in place on its grid. A warp always resamples and pins the result to the warp's geometry.

```python
rot = vx.affine.compose_affine(rotation=[0, 0, 10])   # world-space rotation
moved = vol.transform(rot)                            # geometry only, lossless
resampled = vol.transform(rot, resample=True)         # interpolate voxels
warped = vol.transform(warp)                          # dense warp, always resamples
```

Meshes and boxes transform their coordinates directly. `Mesh.transform` accepts affines, and `BoundingBox.transform` is exact under rotation and axis-aligned scaling but approximate under shear. Bare points go through `map`.

```python
moved_mesh = mesh.transform(aff)
moved_box = box.transform(aff)
moved_pts = aff.map(pts)
```
