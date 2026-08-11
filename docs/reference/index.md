# API Reference

Voxel is conventionally imported as:

```python
import voxel as vx
```

Core classes and I/O functions are available at the top level (`vx.Volume`, `vx.load_volume`, ...), while specialized functions live in module namespaces (`vx.filters.gaussian_filter`, `vx.morphology.dilate`, ...).

## Classes

| Class | Description |
| --- | --- |
| [Volume](classes/volume.md) | Multi-channel 3D image with a world-space geometry |
| [AcquisitionGeometry](classes/acquisition-geometry.md) | Voxel-to-world transform of an image grid |
| [Orientation](classes/orientation.md) | Anatomical orientation of the voxel axes |
| [AffineMatrix](classes/affine-matrix.md) | 4×4 transform for 3D coordinates |
| [BoundingBox](classes/bounding-box.md) | Oriented 3D bounding box |
| [LabelLookup](classes/label-lookup.md) | Table mapping label values to names and colors |
| [Label](classes/label.md) | Single named, colored entry of a label lookup |
| [Mesh](classes/mesh.md) | Triangular mesh in world space |
| [Space](classes/space.md) | Voxel vs. world coordinate frame designator |
| [Warp](classes/warp.md) | Dense world-coordinate deformation |
| [VectorField](classes/vector-field.md) | Volume of 3D displacement or velocity vectors |

## Functions

| Group | Contents |
| --- | --- |
| [io](functions/io.md) | Loading and saving volumes, meshes, affines, boxes, and labels |
| [volume](functions/volume.md) | Volume stacking, grids, and comparison |
| [acquisition](functions/acquisition.md) | Geometry and orientation casting and comparison |
| [affine](functions/affine.md) | Transform constructors and fitting |
| [warp](functions/warp.md) | Transform composition |
| [bounds](functions/bounds.md) | Oriented bounding box fitting |
| [labels](functions/labels.md) | Label map recoding and one-hot encoding |
| [filters](functions/filters.md) | Convolution filtering |
| [morphology](functions/morphology.md) | Binary morphology and labeling |
| [snapshots](functions/snapshots.md) | 2D visualization |
