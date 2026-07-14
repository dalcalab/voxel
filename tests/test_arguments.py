import pytest
import torch
import voxel as vx


def test_merge_components() -> None:

    # without components the value passes through unchanged
    assert vx.arguments.merge_components(2, ()) == 2
    assert vx.arguments.merge_components((1, 2, 3), ()) == (1, 2, 3)
    assert vx.arguments.merge_components(None, ()) is None
    tensor = torch.rand(3)
    assert vx.arguments.merge_components(tensor, ()) is tensor

    # positional components are merged into a single tuple
    assert vx.arguments.merge_components(1, (2, 3)) == (1, 2, 3)

    # components cannot follow an empty leading value
    with pytest.raises(TypeError):
        vx.arguments.merge_components(None, (1, 2))

    # stray strings are rejected (e.g. a space passed to a spaceless method)
    with pytest.raises(TypeError):
        vx.arguments.merge_components(1, (2, 'voxel'))


def test_extract_space() -> None:

    # a trailing string or Space instance is popped as the space
    assert vx.arguments.extract_space((1, 2, 'voxel'), None) == ((1, 2), 'voxel')
    components, space = vx.arguments.extract_space((1, vx.Space('world')), None)
    assert components == (1,)
    assert vx.Space(space) == 'world'

    # a keyword space passes through untouched
    assert vx.arguments.extract_space((1, 2, 3), 'world') == ((1, 2, 3), 'world')

    # the default applies only when no space is given
    assert vx.arguments.extract_space((), None, default='voxel') == ((), 'voxel')
    assert vx.arguments.extract_space(('world',), None, default='voxel') == ((), 'world')

    # space cannot be provided both positionally and as a keyword
    with pytest.raises(TypeError):
        vx.arguments.extract_space((1, 'voxel'), 'world')

    # space is required when no default exists
    with pytest.raises(TypeError):
        vx.arguments.extract_space((1, 2, 3), None)
