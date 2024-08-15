"""
Image and world coordinate space representations.
"""

from __future__ import annotations


space_lookup = {
    'voxel': 'V',
    'vox': 'V',
    'image': 'V',
    'world': 'W',
}


class Space:
    """
    An object designating either the voxel (image grid) or world coordinate space.
    """

    def __init__(self, space: Space | str) -> None:
        """
        Args:
            space (Space | str): Coordinate space. If string, can be 'voxel' or 'world'.
        """
        if isinstance(space, Space):
            self.code = space.code
        else:
            match = space_lookup.get(space)
            if match is None:
                raise ValueError(f'unknown space: {space}')
            self.code = match

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self._categories[self.code]}')"

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Space):
            return self.code == value.code
        elif isinstance(value, str):
            return self.code == space_lookup[value]
        else:
            raise ValueError(f'cannot cast {type(value)} to space')
