import math

import pytest
import torch

import voxel as vx


def test_affine_matrix_construction() -> None:

    # default is the 4x4 identity
    assert torch.equal(vx.AffineMatrix().tensor, torch.eye(4))

    # a 3x3 linear block is padded with zero translation and a homogeneous row
    linear = torch.tensor([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])
    padded = vx.AffineMatrix(linear).tensor
    assert padded.shape == (4, 4)
    assert torch.equal(padded[:3, :3], linear)
    assert torch.equal(padded[:3, 3], torch.zeros(3))
    assert torch.equal(padded[3], torch.tensor([0.0, 0, 0, 1]))

    # a 3x4 block just gets the homogeneous row
    block = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert torch.equal(vx.AffineMatrix(block).tensor[:3], block)

    # constructing from another AffineMatrix copies the underlying tensor
    a = vx.AffineMatrix(linear)
    assert torch.equal(vx.AffineMatrix(a).tensor, a.tensor)

    # unsupported shapes are rejected
    with pytest.raises(ValueError):
        vx.AffineMatrix(torch.zeros(2, 2))


def test_affine_matrix_dtype() -> None:

    # defaults to float32, integer input is coerced to float
    assert vx.AffineMatrix().tensor.dtype == torch.float32
    assert vx.AffineMatrix(torch.eye(4, dtype=torch.int64)).tensor.dtype == torch.float32

    # an explicit floating dtype is preserved
    assert vx.AffineMatrix(torch.eye(4), dtype=torch.float64).tensor.dtype == torch.float64


def test_transform() -> None:

    matrix = vx.AffineMatrix(torch.tensor([
        [2.0, 0, 0, 1],
        [0, 3, 0, 2],
        [0, 0, 4, 3]]))

    # known scaling + translation applied to a single point
    point = torch.tensor([1.0, 1, 1])
    assert torch.allclose(matrix.transform(point), torch.tensor([3.0, 5, 7]))

    # leading dimensions are preserved for batched coordinates
    for shape in [(3,), (5, 3), (4, 6, 3)]:
        coords = torch.rand(shape)
        assert matrix.transform(coords).shape == coords.shape

    # the identity leaves coordinates unchanged
    coords = torch.rand(7, 3)
    assert torch.allclose(vx.AffineMatrix().transform(coords), coords)

    # the trailing dimension must be of size 3
    with pytest.raises(ValueError):
        matrix.transform(torch.zeros(5, 4))


def test_inverse() -> None:

    matrix = vx.affine.compose_affine(translation=[3.0, -2, 5], rotation=[10.0, 20, 30], scale=[1.2, 0.8, 1.1])

    # a matrix composed with its inverse is the identity
    identity = (matrix @ matrix.inverse()).tensor
    assert torch.allclose(identity, torch.eye(4), atol=1e-5)

    # inverting the transform inverts the coordinate mapping
    coords = torch.rand(10, 3)
    assert torch.allclose(matrix.inverse().transform(matrix.transform(coords)), coords, atol=1e-4)


def test_matmul() -> None:

    a = vx.affine.compose_affine(translation=[1.0, 2, 3], rotation=[5.0, 0, 0])
    b = vx.affine.compose_affine(scale=[2.0, 0.5, 1.5])

    # composition matches applying the two transforms in sequence
    coords = torch.rand(8, 3)
    assert torch.allclose((a @ b).transform(coords), a.transform(b.transform(coords)), atol=1e-5)

    # a 4x4 result stays an AffineMatrix, other shapes fall back to a raw tensor
    assert isinstance(a @ b, vx.AffineMatrix)
    product = a @ torch.ones(4, 2)
    assert isinstance(product, torch.Tensor) and product.shape == (4, 2)


def test_translation_matrix() -> None:

    matrix = vx.affine.translation_matrix(torch.tensor([1.0, 2, 3]))
    assert torch.equal(matrix.tensor[:3, 3], torch.tensor([1.0, 2, 3]))

    # translation shifts points by the given vector
    assert torch.allclose(matrix.transform(torch.zeros(3)), torch.tensor([1.0, 2, 3]))

    # only a 3-vector is accepted
    with pytest.raises(ValueError):
        vx.affine.translation_matrix(torch.zeros(4))


def test_angles_to_rotation_matrix() -> None:

    # zero rotation is the identity
    assert torch.allclose(vx.affine.angles_to_rotation_matrix(torch.zeros(3)).tensor, torch.eye(4), atol=1e-6)

    rotation = vx.affine.angles_to_rotation_matrix(torch.tensor([30.0, -45, 60])).tensor[:3, :3]

    # the result is a proper rotation (orthonormal with determinant +1)
    assert torch.allclose(rotation @ rotation.T, torch.eye(3), atol=1e-5)
    assert torch.allclose(torch.det(rotation), torch.tensor(1.0), atol=1e-5)

    # degrees and radians are consistent
    degrees = vx.affine.angles_to_rotation_matrix(torch.tensor([90.0, 0, 0]), degrees=True).tensor
    radians = vx.affine.angles_to_rotation_matrix(torch.tensor([math.pi / 2, 0, 0]), degrees=False).tensor
    assert torch.allclose(degrees, radians, atol=1e-6)


def test_quaternion_to_rotation_matrix() -> None:

    # the identity quaternion is the identity rotation
    identity = vx.affine.quaternion_to_rotation_matrix(torch.tensor([1.0, 0, 0, 0])).tensor
    assert torch.allclose(identity, torch.eye(4), atol=1e-6)

    # a non-unit quaternion is normalized into a proper rotation
    rotation = vx.affine.quaternion_to_rotation_matrix(torch.tensor([0.5, 1.0, -2.0, 0.3])).tensor[:3, :3]
    assert torch.allclose(rotation @ rotation.T, torch.eye(3), atol=1e-5)
    assert torch.allclose(torch.det(rotation), torch.tensor(1.0), atol=1e-5)

    # a 90 degree rotation about x maps +y to +z (scalar-first convention)
    quaternion = torch.tensor([math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0])
    mapped = vx.affine.quaternion_to_rotation_matrix(quaternion).transform(torch.tensor([0.0, 1, 0]))
    assert torch.allclose(mapped, torch.tensor([0.0, 0, 1]), atol=1e-6)

    # a quaternion must have four entries
    with pytest.raises(ValueError):
        vx.affine.quaternion_to_rotation_matrix(torch.zeros(3))


def test_compose_affine() -> None:

    # no components yields the identity
    assert torch.allclose(vx.affine.compose_affine().tensor, torch.eye(4))

    # translation alone matches the dedicated translation matrix
    translation = torch.tensor([1.0, 2, 3])
    assert torch.allclose(
        vx.affine.compose_affine(translation=translation).tensor,
        vx.affine.translation_matrix(translation).tensor,
        atol=1e-6)

    # components compose as T @ R @ Z @ S: scale then translate a point
    composed = vx.affine.compose_affine(translation=[1.0, 2, 3], scale=[2.0, 3, 4])
    assert torch.allclose(composed.transform(torch.ones(3)), torch.tensor([3.0, 5, 7]), atol=1e-6)

    # a length-4 rotation is interpreted as a quaternion
    quaternion = torch.tensor([0.5, 1.0, -2.0, 0.3])
    assert torch.allclose(
        vx.affine.compose_affine(rotation=quaternion).tensor,
        vx.affine.quaternion_to_rotation_matrix(quaternion).tensor,
        atol=1e-6)

    # rotation must be a 3-vector (angles) or 4-vector (quaternion)
    with pytest.raises(ValueError):
        vx.affine.compose_affine(rotation=torch.zeros(2))


def test_compose_affine_differentiable() -> None:

    # gradients flow back to the transform parameters
    translation = torch.zeros(3, requires_grad=True)
    quaternion = torch.tensor([1.0, 0, 0, 0], requires_grad=True)
    matrix = vx.affine.compose_affine(translation=translation, rotation=quaternion)
    matrix.tensor.sum().backward()
    assert translation.grad is not None
    assert quaternion.grad is not None


def test_least_squares_alignment() -> None:

    torch.manual_seed(0)

    # a known affine relating two point sets should be recovered
    source = torch.rand(30, 3) * 100
    matrix = vx.affine.compose_affine(translation=[3.0, -2, 5], rotation=[10.0, 20, 30], scale=[1.1, 0.9, 1.2])
    target = matrix.transform(source)

    solved = vx.affine.least_squares_alignment(source, target)
    assert torch.allclose(solved.transform(source), target, atol=1e-2)
    assert torch.allclose(solved.tensor, matrix.tensor, atol=1e-2)


def test_least_squares_alignment_float64() -> None:

    torch.manual_seed(0)

    # float64 point sets must be handled without a dtype mismatch
    source = torch.rand(30, 3, dtype=torch.float64) * 100
    matrix = vx.affine.compose_affine(
        translation=[3.0, -2, 5], rotation=[10.0, 20, 30], scale=[1.1, 0.9, 1.2], dtype=torch.float64)
    target = matrix.transform(source)

    solved = vx.affine.least_squares_alignment(source, target)
    assert torch.allclose(solved.transform(source), target, atol=1e-2)


def test_random_affine() -> None:

    # all-zero limits gives the identity
    assert torch.allclose(vx.affine.random_affine().tensor, torch.eye(4))

    # negative limits are rejected
    with pytest.raises(ValueError):
        vx.affine.random_affine(max_translation=-1)

    # sampled translation stays within the requested bound
    translation = vx.affine.random_affine(max_translation=5).tensor[:3, 3]
    assert bool((translation.abs() <= 5).all())


def test_affine_volume_transform() -> None:

    shape = (8, 9, 10)
    source = vx.AcquisitionGeometry(shape, vx.affine.compose_affine(translation=[1.0, 2, 3], scale=[1.0, 1.2, 0.8]))
    target = vx.AcquisitionGeometry(shape, vx.affine.compose_affine(translation=[-2.0, 1, 4], rotation=[5.0, 0, 0]))
    matrix = vx.affine.compose_affine(translation=[2.0, -1, 3], rotation=[3.0, -4, 5])
    transform = vx.AffineVolumeTransform(matrix, 'world', source, target)

    # inverting swaps source and target and is self-undoing
    inverse = transform.inverse()
    assert torch.allclose(inverse.source.tensor, target.tensor)
    assert torch.allclose(inverse.target.tensor, source.tensor)
    assert torch.allclose(inverse.inverse().tensor, transform.tensor, atol=1e-5)

    # converting to another space and back is a round trip
    roundtrip = transform.convert(space='voxel').convert(space='world')
    assert torch.allclose(roundtrip.tensor, transform.tensor, atol=1e-4)
