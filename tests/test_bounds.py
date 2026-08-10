import pytest
import torch
import voxel as vx


def rotated_cloud(num: int = 4000) -> tuple:
    """
    Uniform points filling a known rotated and translated box, returned
    along with the true box volume.
    """
    extent = torch.tensor([3.0, 2, 1])
    rotation = vx.affine.angles_to_rotation_matrix(torch.tensor([20.0, -10, 35]))[:3, :3]
    points = (torch.rand(num, 3) * 2 - 1) * extent
    points = points @ rotation.T + torch.tensor([5.0, -3, 2])
    return points, (2 * extent).prod().item()


def box_volume(box: vx.BoundingBox) -> float:
    return (2 * box.extent).prod().item()


def box_contains(box: vx.BoundingBox, points: torch.Tensor) -> bool:
    projected = (points - box.center) @ box.rotation
    return bool((projected.abs() <= box.extent + 1e-4).all())


def mesh_signed_volume(mesh: vx.Mesh) -> float:
    v0, v1, v2 = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
    return (torch.cross(v0, v1, dim=-1) * v2).sum().item() / 6


def test_corner_points() -> None:
    box = vx.BoundingBox(center=torch.tensor([1.0, 2, 3]), extent=torch.tensor([1.0, 2, 3]))
    points = box.corner_points()
    assert points.shape == (8, 3)
    min_coord, max_coord = box.min_max_coords()
    assert torch.allclose(min_coord, torch.tensor([0.0, 0, 0]))
    assert torch.allclose(max_coord, torch.tensor([2.0, 4, 6]))

    # all corners are unique
    assert points.unique(dim=0).shape == (8, 3)


def test_corner_cache() -> None:
    box = vx.BoundingBox(center=torch.tensor([1.0, 2, 3]), extent=torch.tensor([1.0, 2, 3]))

    # corners are computed once and reused
    assert box.corner_points() is box.corner_points()
    assert torch.equal(box.cpu().corner_points(), box.corner_points())

    # operations return fresh instances with correctly recomputed corners
    shifted = box.shift(torch.tensor([1.0, 0, 0]))
    assert shifted is not box
    assert torch.allclose(shifted.corner_points(), box.corner_points() + torch.tensor([1.0, 0, 0]))
    padded = box.pad(1)
    assert torch.allclose(padded.min_max_coords()[0], box.min_max_coords()[0] - 1)


def test_validation() -> None:
    with pytest.raises(ValueError):
        vx.BoundingBox(center=torch.zeros(2))
    with pytest.raises(ValueError):
        vx.BoundingBox(rotation=torch.eye(4))
    with pytest.raises(ValueError):
        vx.BoundingBox(extent=torch.ones(3, 1))
    with pytest.raises(ValueError):
        vx.BoundingBox(extent=torch.tensor([1.0, -1, 1]))

    # integer inputs are upcast to float
    box = vx.BoundingBox(center=torch.tensor([1, 2, 3]), extent=torch.tensor([1, 2, 3]))
    assert box.center.is_floating_point()
    assert box.extent.is_floating_point()

    # mismatched parameter devices are rejected rather than silently moved
    if torch.cuda.is_available():
        with pytest.raises(ValueError):
            vx.BoundingBox(center=torch.zeros(3).cuda(), extent=torch.ones(3))


def test_from_min_max() -> None:
    box = vx.BoundingBox.from_min_max(torch.tensor([0.0, 1, 2]), torch.tensor([2.0, 5, 8]))
    assert torch.allclose(box.center, torch.tensor([1.0, 3, 5]))
    assert torch.allclose(box.extent, torch.tensor([1.0, 2, 3]))
    assert torch.allclose(box.rotation, torch.eye(3))
    min_coord, max_coord = box.min_max_coords()
    assert torch.allclose(min_coord, torch.tensor([0.0, 1, 2]))
    assert torch.allclose(max_coord, torch.tensor([2.0, 5, 8]))


def test_from_points() -> None:
    points = torch.tensor([[0.0, 1, 2], [2.0, 5, 8], [1.0, 3, 5]])
    box = vx.BoundingBox.from_points(points)
    min_coord, max_coord = box.min_max_coords()
    assert torch.allclose(min_coord, points.amin(0))
    assert torch.allclose(max_coord, points.amax(0))
    with pytest.raises(ValueError):
        vx.BoundingBox.from_points(torch.zeros(3))


def test_mesh() -> None:
    box = vx.BoundingBox(extent=torch.tensor([1.0, 2, 3]))
    mesh = box.mesh()
    assert mesh.num_vertices == 8
    assert mesh.num_faces == 12
    assert torch.allclose(mesh.vertices, box.corner_points())

    # windings are outward-facing, so the signed volume matches the box volume
    assert mesh_signed_volume(mesh) == pytest.approx(box_volume(box))


def test_mesh_winding_reflected() -> None:
    # a reflected frame (negative determinant) must not flip the windings inward
    box = vx.BoundingBox(rotation=torch.diag(torch.tensor([-1.0, 1, 1])),
                         extent=torch.tensor([1.0, 2, 3]))
    mesh = box.mesh()
    assert mesh_signed_volume(mesh) == pytest.approx(box_volume(box))

    # every face normal points away from the box center
    v0, v1, v2 = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    centroids = (v0 + v1 + v2) / 3
    assert ((normals * (centroids - box.center)).sum(-1) > 0).all()


def test_geometry() -> None:
    box = vx.BoundingBox(center=torch.tensor([4.0, -1, 2]), extent=torch.tensor([1.0, 2, 3]))
    geometry = box.geometry(0.5)
    assert tuple(geometry.baseshape) == (4, 8, 12)
    assert torch.allclose(geometry.spacing, torch.full((3,), 0.5), atol=1e-5)
    assert torch.allclose(geometry.center, box.center, atol=1e-5)
    with pytest.raises(ValueError):
        box.geometry(torch.ones(2))


def test_pad_scale_trim() -> None:
    box = vx.BoundingBox(center=torch.tensor([1.0, 1, 1]), extent=torch.tensor([1.0, 2, 3]))

    # symmetric padding grows the extent without moving the center
    padded = box.pad(1)
    assert torch.allclose(padded.extent, box.extent + 1)
    assert torch.allclose(padded.center, box.center)

    scaled = box.scale(2)
    assert torch.allclose(scaled.extent, 2 * box.extent)
    assert torch.allclose(scaled.center, box.center)

    # asymmetric per-side margins shift the center and grow by the mean margin
    margin = torch.tensor([[1.0, 0], [0, 2], [0, 0]])
    padded = box.pad(margin)
    assert torch.allclose(padded.extent, box.extent + torch.tensor([0.5, 1, 0]))
    assert torch.allclose(padded.center, box.center + torch.tensor([-0.5, 1, 0]))
    min_coord, max_coord = padded.min_max_coords()
    assert torch.allclose(min_coord, box.min_max_coords()[0] - torch.tensor([1.0, 0, 0]))
    assert torch.allclose(max_coord, box.min_max_coords()[1] + torch.tensor([0.0, 2, 0]))

    # trim inverts pad
    trimmed = box.pad(margin).trim(margin)
    assert torch.allclose(trimmed.center, box.center)
    assert torch.allclose(trimmed.extent, box.extent)

    # shrinking beyond the extent is invalid
    with pytest.raises(ValueError):
        box.trim(4)


def test_shift() -> None:
    box = vx.BoundingBox(center=torch.tensor([1.0, 2, 3]))
    shifted = box.shift(torch.tensor([1.0, -2, 0.5]))
    assert torch.allclose(shifted.center, torch.tensor([2.0, 0, 3.5]))
    assert torch.allclose(shifted.extent, box.extent)
    assert torch.allclose(box.center, torch.tensor([1.0, 2, 3]))


def test_rotate() -> None:
    box = vx.BoundingBox(center=torch.tensor([1.0, 2, 3]), extent=torch.tensor([1.0, 2, 3]))
    rotated = box.rotate((0, 0, 45))
    assert torch.allclose(rotated.extent, box.extent)
    assert torch.allclose(rotated.center, box.center)

    # the rotation stays orthogonal, preserving the box volume
    identity = rotated.rotation @ rotated.rotation.T
    assert torch.allclose(identity, torch.eye(3), atol=1e-5)
    assert box_volume(rotated) == pytest.approx(box_volume(box))


def test_transform() -> None:
    box = vx.BoundingBox(center=torch.tensor([1.0, 2, 3]),
                         extent=torch.tensor([1.0, 2, 3])).rotate((0, 0, 30))
    matrix = vx.affine.angles_to_rotation_matrix(torch.tensor([10.0, 20, -15]))
    matrix = vx.AffineMatrix(matrix.tensor @ vx.affine.translation_matrix(torch.tensor([4.0, -1, 2])).tensor)

    # for a rigid transform, the box corners map exactly
    transformed = box.transform(matrix)
    expected = matrix.map(box.corner_points())
    assert torch.allclose(transformed.corner_points(), expected, atol=1e-4)
    assert box_volume(transformed) == pytest.approx(box_volume(box), rel=1e-4)

    # axis-aligned anisotropic scaling of an axis-aligned box is exact
    aligned = vx.BoundingBox(extent=torch.tensor([1.0, 2, 3]))
    scaling = vx.AffineMatrix(torch.diag(torch.tensor([2.0, 3, 4, 1])))
    scaled = aligned.transform(scaling)
    assert torch.allclose(scaled.extent, torch.tensor([2.0, 6, 12]))


def test_fit_extent() -> None:
    points, _ = rotated_cloud()
    box = vx.BoundingBox().fit_extent(points)
    assert box_contains(box, points)

    # fitting also accepts a bounding box, mesh, or acquisition geometry
    other = vx.BoundingBox().fit_extent(vx.BoundingBox(extent=torch.tensor([2.0, 2, 2])))
    assert torch.allclose(other.extent, torch.full((3,), 2.0))
    geometry = vx.AcquisitionGeometry((10, 12, 14))
    fitted = vx.BoundingBox().fit_extent(geometry)
    assert box_contains(fitted, geometry.bounds().corner_points())
    meshed = vx.BoundingBox().fit_extent(vx.BoundingBox(extent=torch.tensor([2.0, 2, 2])).mesh())
    assert torch.allclose(meshed.extent, torch.full((3,), 2.0))


def test_obbox_pca() -> None:
    points, true_volume = rotated_cloud()
    box = vx.bounds.obbox_pca(points)
    assert box_contains(box, points)
    assert box_volume(box) < 1.1 * true_volume


def test_obbox_fine_tune() -> None:
    points, true_volume = rotated_cloud()
    pca_box = vx.bounds.obbox_pca(points)
    tuned = vx.bounds.obbox(points)
    assert box_contains(tuned, points)

    # fine-tuning should never do worse than its PCA initialization
    assert box_volume(tuned) <= 1.01 * box_volume(pca_box)
    assert box_volume(tuned) < 1.1 * true_volume

    with pytest.raises(AssertionError):
        vx.bounds.obbox(points, initialize=False, fine_tune=False)


def test_save_load(tmp_path) -> None:
    box = vx.BoundingBox(center=torch.tensor([1.0, 2, 3]), extent=torch.tensor([1.0, 2, 3]))
    path = tmp_path / 'box.pt'
    box.save(path)
    loaded = vx.load_bounding_box(path)
    assert torch.allclose(loaded.center, box.center)
    assert torch.allclose(loaded.rotation, box.rotation)
    assert torch.allclose(loaded.extent, box.extent)


def test_variadic_components() -> None:
    box = vx.BoundingBox(center=torch.tensor([1.0, 2, 3]), extent=torch.tensor([1.0, 2, 3]))

    def boxes_equal(a: vx.BoundingBox, b: vx.BoundingBox) -> bool:
        return bool(torch.allclose(a.center, b.center)
                    and torch.allclose(a.extent, b.extent)
                    and torch.allclose(a.rotation, b.rotation))

    # unpacked positional components match the sequence form
    assert boxes_equal(box.shift(1, 2, 3), box.shift((1, 2, 3)))
    assert boxes_equal(box.scale(1, 1, 2), box.scale((1, 1, 2)))
    assert boxes_equal(box.pad(1, 2, 3), box.pad((1, 2, 3)))
    assert boxes_equal(box.trim(0.2, 0.3, 0.1), box.trim((0.2, 0.3, 0.1)))
    assert boxes_equal(box.rotate(0, 0, 45), box.rotate((0, 0, 45)))
    assert vx.geometries_equal(box.geometry(1, 1, 2), box.geometry((1, 1, 2)))

    # stray trailing strings are rejected on methods without a space argument
    with pytest.raises(TypeError):
        box.pad(1, 2, 'voxel')
