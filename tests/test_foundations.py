import pytest
import torch
import voxel as vx


# slicing utilities

def test_coordinates_to_slicing() -> None:
    slicing = vx.slicing.coordinates_to_slicing(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))
    assert slicing == (slice(1, 5), slice(2, 6), slice(3, 7))

    strided = vx.slicing.coordinates_to_slicing(
        torch.tensor([0, 0, 0]), torch.tensor([7, 7, 7]), torch.tensor([1, 2, 3]))
    assert strided == (slice(0, 8, 1), slice(0, 8, 2), slice(0, 8, 3))


@pytest.mark.parametrize('slicing', [
    (slice(1, 5), slice(2, 6), slice(3, 7)),
    (slice(None), slice(2, None), slice(None, 7)),
    (slice(-4, -1), slice(None), slice(None)),
    (slice(1, 5, 2), slice(None), slice(None)),
])
def test_slicing_coordinate_roundtrip(slicing) -> None:

    # converting a slicing to coordinates and back must select the same voxels
    shape = (10, 12, 14)
    tensor = torch.arange(torch.Size(shape).numel()).view(shape)
    coords = vx.slicing.slicing_to_coordinates(slicing, shape)
    roundtrip = vx.slicing.coordinates_to_slicing(coords[0], coords[1], coords[2])
    assert torch.equal(tensor[slicing], tensor[roundtrip])


def test_slicing_to_coordinates_values() -> None:
    minc, maxc, stride = vx.slicing.slicing_to_coordinates((slice(2, 8), 3, slice(None)), (10, 12, 14))
    assert minc.tolist() == [2, 3, 0]
    assert maxc.tolist() == [7, 3, 13]
    assert stride is None

    # negative indices wrap around the shape
    minc, maxc, _ = vx.slicing.slicing_to_coordinates((slice(-4, -1),), (10,))
    assert minc.tolist() == [6]
    assert maxc.tolist() == [8]


def test_expand_slicing() -> None:
    assert vx.slicing.expand_slicing((slice(1, 2),), 3) == \
        (slice(1, 2), slice(None), slice(None))
    assert vx.slicing.expand_slicing((slice(1, 2), Ellipsis, slice(3, 4)), 4) == \
        (slice(1, 2), slice(None), slice(None), slice(3, 4))

    with pytest.raises(ValueError):
        vx.slicing.expand_slicing((Ellipsis, Ellipsis), 4)
    with pytest.raises(ValueError):
        vx.slicing.expand_slicing((slice(None),) * 5, 4)


def test_conform_coordinates() -> None:
    assert vx.slicing.conform_coordinates(torch.tensor(2.0)).shape == (3,)
    assert vx.slicing.conform_coordinates(torch.tensor(2.0), 2).shape == (3, 2)
    assert vx.slicing.conform_coordinates(torch.ones(3), 2).shape == (3, 2)
    assert vx.slicing.conform_coordinates(torch.ones(3, 2), 2).shape == (3, 2)

    with pytest.raises(ValueError):
        vx.slicing.conform_coordinates(torch.ones(2))
    with pytest.raises(ValueError):
        vx.slicing.conform_coordinates(torch.ones(3, 3), 2)


def test_conform_coordinates_num() -> None:
    # a (3,) input should expand to (3, num) for any num, not just 2
    assert vx.slicing.conform_coordinates(torch.ones(3), 3).shape == (3, 3)


# coordinate spaces

def test_space() -> None:
    assert vx.Space('world') == 'world'
    assert vx.Space('voxel') == 'voxel'
    assert vx.Space('image') == 'voxel'
    assert vx.Space('world') != 'voxel'
    assert vx.Space('world') == vx.Space('world')
    assert vx.Space(vx.Space('voxel')) == 'voxel'
    assert repr(vx.Space('image')) == "Space('voxel')"

    with pytest.raises(ValueError):
        vx.Space('scanner')


def test_space_equality_unknown_type() -> None:
    # comparing against an unrelated type should be False, not an error
    assert (vx.Space('world') == 3.14) is False


# property caching

class Cached:
    """
    Minimal class exercising the cached property decorators.
    """

    def __init__(self) -> None:
        vx.caching.init_property_cache(self)
        self.calls = 0

    @vx.caching.cached
    def prop(self) -> int:
        self.calls += 1
        return self.calls

    @vx.caching.cached_transferable
    def transferable(self) -> int:
        self.calls += 1
        return self.calls


def test_cached_properties() -> None:
    obj = Cached()
    assert obj.prop == 1
    assert obj.prop == 1
    assert obj.calls == 1

    # cached properties are read-only
    with pytest.raises(AttributeError):
        obj.prop = 10

    # clearing the cache forces recomputation
    vx.caching.empty_property_cache(obj)
    assert obj.prop == 2


def test_transferable_cache() -> None:
    source = Cached()
    assert source.prop == 1
    assert source.transferable == 2

    # only the transferable cache carries over to a new instance
    target = Cached()
    vx.caching.transfer_property_cache(source, target)
    assert target.transferable == 2
    assert target.prop == 1
