# Meshes

A [`Mesh`](reference/classes/mesh.md) is a triangular surface, a $(V, 3)$ vertex tensor and an $(F, 3)$ face-index tensor living in world coordinates. Like volumes, meshes are GPU-compatible and differentiable through their vertex data.

```python
import voxel as vx

mesh = vx.load_mesh('lh.white')       # FreeSurfer, OBJ, PLY, STL, ...
mesh.num_vertices, mesh.num_faces
mesh = mesh.cuda()
```

`mesh.new(vertices)` builds a mesh with updated vertices and the same faces, carrying over cached properties that remain valid.

## Geometric properties

Differential and connectivity properties are computed on demand and cached until the vertices change.

```python
mesh.triangles           # (F, 3, 3) vertex coordinates per face
mesh.face_normals        # (F, 3) unit normals
mesh.face_areas          # (F,) triangle areas
mesh.face_angles         # (F, 3) interior angles in radians
mesh.vertex_normals      # (V, 3) angle-weighted unit normals
mesh.vertex_areas        # (V,) surface area attributed to each vertex
mesh.unique_edges        # (E, 2) undirected edge list
mesh.adjacent_faces      # face adjacency across shared edges
mesh.uniform_laplacian   # sparse uniform Laplacian operator
```

## Editing

```python
relaxed = mesh.smooth_mesh(alpha=0.5, iterations=10)   # Laplacian smoothing
sub = mesh.extract_submesh(vertex_mask)                # keep marked vertices
mesh = mesh.flip_faces()                               # reverse winding and normals
main = mesh.extract_submesh(mesh.largest_connected_components())
```

`extract_submesh` remaps face indices to the retained vertices. `largest_connected_components(k)` returns a boolean vertex mask of the k largest components and runs on the CPU via scipy.

## Vertex features

Per-vertex feature tensors pool and diffuse along the surface.

```python
pooled = mesh.gather(feats, reduce='mean')             # reduce across edge neighbors
smoothed = mesh.smooth_features(feats, iterations=5)   # diffuse along the surface
```

## Meshes and volumes

Meshes live in world space, so they compose naturally with any volume through its geometry. Sample image intensities at vertices, crop a volume to a surface, or move a mesh into voxel coordinates.

```python
feats = vol.sample(mesh, space='world')            # (V, C) sampled features
roi = vol.crop(mesh.bounds())                      # crop to the vertex bounds
vox_mesh = mesh.transform(vol.geometry.inverse())  # world to voxel coordinates
```

In the other direction, `tesselate` extracts a surface around thresholded voxel components. It requires `pytorch3d` and is not differentiable.

```python
surface = seg.tesselate(threshold=0.5)   # world-space Mesh
surface = surface.smooth_mesh(iterations=5)
surface.save('surface.ply')
```

To deform a mesh nonlinearly, see [vector fields](transforms.md#vector-fields).

## Reading and writing

```python
mesh = vx.load_mesh('lh.white')    # FreeSurfer surface
mesh.save('surface.ply')
```

Supported formats include OBJ, PLY, STL, GLTF/GLB, OFF, 3MF, DXF, and 3DXML through `trimesh`, FreeSurfer surfaces through `surfa`, and GIFTI through `nibabel`.
