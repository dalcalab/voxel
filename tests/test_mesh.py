import pytest
import torch
import voxel as vx


def test_construction() -> None:
    with pytest.raises(ValueError):
        vx.Mesh(torch.rand(8, 2), torch.zeros(2, 3).int())
    with pytest.raises(ValueError):
        vx.Mesh(torch.rand(8, 3), torch.zeros(2, 2).int())

    # faces are always cast to integer type
    mesh = vx.Mesh(torch.rand(3, 3), torch.tensor([[0.0, 1, 2]]))
    assert not torch.is_floating_point(mesh.faces)


def test_box_mesh(box_mesh) -> None:
    assert box_mesh.num_vertices == 8
    assert box_mesh.num_faces == 12
    assert torch.allclose(box_mesh.vertices.amin(0), torch.full((3,), -0.5))
    assert torch.allclose(box_mesh.vertices.amax(0), torch.full((3,), 1.5))


def test_edge_topology(box_mesh) -> None:
    assert box_mesh.edges.shape == (36, 2)
    assert box_mesh.unique_edges.shape == (18, 2)
    assert box_mesh.adjacent_faces.shape == (18, 2)

    # Euler characteristic of a closed genus-0 surface: V - E + F = 2
    euler = box_mesh.num_vertices - len(box_mesh.unique_edges) + box_mesh.num_faces
    assert euler == 2


def test_face_properties(box_mesh) -> None:

    # the box spans an edge length of 2, so the surface area is 6 * 2^2
    assert torch.allclose(box_mesh.face_areas.sum(), torch.tensor(24.0))
    assert torch.allclose(box_mesh.face_normals.norm(dim=1), torch.ones(12))

    # triangle angles always sum to pi
    assert torch.allclose(box_mesh.face_angles.sum(1), torch.full((12,), torch.pi))

    # vertex areas redistribute the total face area
    assert torch.allclose(box_mesh.vertex_areas.sum(), torch.tensor(24.0))
    assert torch.allclose(box_mesh.vertex_normals.norm(dim=1), torch.ones(8), atol=1e-5)


def test_box_mesh_winding(box_mesh) -> None:
    # consistent outward face winding implies the signed volume matches the
    # enclosed volume of the 2-unit cube
    triangles = box_mesh.triangles
    cross = torch.cross(triangles[:, 1], triangles[:, 2], dim=1)
    signed = (triangles[:, 0] * cross).sum() / 6
    assert torch.allclose(signed, torch.tensor(8.0))


def test_flip_faces(box_mesh) -> None:
    flipped = box_mesh.flip_faces()
    assert torch.allclose(flipped.face_normals, -box_mesh.face_normals)
    assert torch.allclose(flipped.face_areas, box_mesh.face_areas)


def test_transform(box_mesh) -> None:

    # a rigid transform preserves areas and angles, and rotates normals
    trf = vx.affine.compose_affine(translation=(4, -2, 1), rotation=(0, 30, 0))
    moved = box_mesh.transform(trf)
    assert torch.allclose(moved.face_areas, box_mesh.face_areas, atol=1e-5)
    assert torch.allclose(moved.face_angles, box_mesh.face_angles, atol=1e-5)
    rotated = box_mesh.face_normals @ trf.tensor[:3, :3].T
    assert torch.allclose(moved.face_normals, rotated, atol=1e-5)

    # a pure translation moves vertices exactly
    shifted = box_mesh.transform(vx.affine.translation_matrix(torch.tensor([1.0, 2, 3])))
    assert torch.allclose(shifted.vertices, box_mesh.vertices + torch.tensor([1.0, 2, 3]))


def test_bounds(box_mesh) -> None:
    bounds = box_mesh.bounds()
    assert isinstance(bounds, vx.BoundingBox)

    # the bounds fit the vertices exactly, with no implicit padding
    min_coord, max_coord = bounds.min_max_coords()
    assert torch.allclose(min_coord, box_mesh.vertices.amin(0))
    assert torch.allclose(max_coord, box_mesh.vertices.amax(0))
    expanded = box_mesh.bounds(margin=1)
    assert torch.allclose(expanded.min_max_coords()[0], box_mesh.vertices.amin(0) - 1)


def test_smooth_mesh(box_mesh) -> None:

    # laplacian smoothing contracts the box, and a zero alpha is an identity
    smoothed = box_mesh.smooth_mesh(alpha=0.5, iterations=2)
    original_extent = box_mesh.vertices.amax(0) - box_mesh.vertices.amin(0)
    smoothed_extent = smoothed.vertices.amax(0) - smoothed.vertices.amin(0)
    assert bool((smoothed_extent < original_extent).all())
    assert torch.equal(box_mesh.smooth_mesh(alpha=0.0).vertices, box_mesh.vertices)


def test_gather(box_mesh) -> None:

    # gathering a constant feature over neighbors returns the same constant
    features = torch.ones(box_mesh.num_vertices, 2)
    assert torch.allclose(box_mesh.gather(features), features)

    # summed gathering scales with the (directional) vertex degree
    degrees = torch.bincount(box_mesh.edges[:, 1].long(), minlength=8).float()
    summed = box_mesh.gather(torch.ones(box_mesh.num_vertices, 1), reduce='sum')
    assert torch.allclose(summed.squeeze(1), degrees)


def test_extract_submesh(box_mesh) -> None:

    # keep the four vertices of one box side: only faces fully contained
    # in the mask survive, with reindexed face indices
    mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0]).bool()
    submesh = box_mesh.extract_submesh(mask)
    assert submesh.num_vertices == 4
    assert submesh.num_faces == 2
    assert submesh.faces.max() < submesh.num_vertices
    assert torch.equal(submesh.vertices, box_mesh.vertices[mask])


def test_cache_transfer(box_mesh) -> None:

    # transferable topology caches computed before new() carry over to a
    # vertex-modified mesh, while geometry-dependent caches are recomputed
    edges = box_mesh.edges
    unique = box_mesh.unique_edges
    areas = box_mesh.face_areas
    scaled = box_mesh.new(box_mesh.vertices * 2)
    assert scaled.edges is edges
    assert scaled.unique_edges is unique
    assert torch.allclose(scaled.face_areas, 4 * areas)


def test_largest_connected_components(box_mesh) -> None:
    pytest.importorskip('scipy')

    # combine the 8-vertex box with a lone 3-vertex triangle
    triangle = vx.Mesh(torch.rand(3, 3) + 10, torch.tensor([[0, 1, 2]]))
    combined = vx.Mesh(
        torch.cat([box_mesh.vertices, triangle.vertices]),
        torch.cat([box_mesh.faces, triangle.faces + box_mesh.num_vertices]))

    mask = combined.largest_connected_components(k=1)
    assert torch.equal(mask, torch.tensor([True] * 8 + [False] * 3))
    assert bool(combined.largest_connected_components(k=2).all())


def test_bounds_variadic(box_mesh) -> None:

    # unpacked margin components match the sequence form
    a, b = box_mesh.bounds(1, 2, 3), box_mesh.bounds((1, 2, 3))
    assert torch.allclose(a.center, b.center)
    assert torch.allclose(a.extent, b.extent)
