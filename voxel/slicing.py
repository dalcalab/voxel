"""
Management of slicing tuples and indexing coordinates for tensors.
"""

import torch


def coordinates_to_slicing(
    minc: torch.Tensor,
    maxc: torch.Tensor,
    stride: torch.Tensor = None) -> tuple:
    """
    Converts min, max, and stride coordinates to a slicing tuple.

    Args:
        minc (Tensor): The minimum integer coordinate.
        maxc (Tensor): The maximum integer coordinate (inclusive).
        stride (Tensor, optional): The sampling strides (or steps).

    Returns:
        tuple: Slicing tuple.
    """
    if stride is None:
        stride = [None for _ in minc]
    make = lambda a, b, c: slice(int(a), int(b + 1), c if c is None else int(c))
    return tuple([make(*params) for params in zip(minc, maxc, stride)])


def expand_slicing(slicing: tuple, length: int) -> tuple:
    """
    Expands a slicing tuple to specified length, accounting for
    missing items and ellipses.

    Args:
        slicing (tuple): The original slicing tuple.
        length (int): The target slicing length.

    Returns:
        tuple: The expanded slicing.
    """
    if Ellipsis in slicing:
        # if an ellipsis is in the slicing, we need to be smart
        # about how this expands given the target length
        expanded = [slice(None) for _ in range(length)]
        for i, s in enumerate(slicing):
            if s == Ellipsis:
                if i < len(slicing) - 1:
                    remaining = slicing[i+1:]
                    expanded[-len(remaining):] = remaining
                break
            else:
                expanded[i] = s
        expanded = tuple(expanded)

        # its impossible to evaluate the slicing when multiple ellipsis are used
        if Ellipsis in expanded:
            raise ValueError('cannot use more than one ellipsis when indexing')

    else:
        # fill in any remaining elements with open slices
        missing = length - len(slicing)
        if missing > 0:
            remaining = [slice(None) for _ in range(missing)]
            expanded = (*slicing, *remaining)
        elif missing < 0:
            raise ValueError(f'slicing dimensions {len(slicing)} exceed '
                             f'limit of {length}')
        else:
            expanded = slicing

    return expanded


def slicing_to_coordinates(slicing: tuple, shape: tuple) -> tuple:
    """
    Converts a slicing tuple to min, max (inclusive), and stride coordinates.

    Args:
        slicing (tuple): The slicing tuple.
        shape (tuple of int): The shape of the tensor.

    Returns:
        tuple of Tensor: Minimum coordinates, maximum coordinates, and strides.
    """
    shape = torch.tensor(shape, dtype=torch.int)

    # make sure slicing is conformed to the right size
    slicing = expand_slicing(slicing, len(shape))

    # extract the minimum coordinate based on the slice start index
    minc = torch.zeros(len(shape), dtype=torch.int)
    for i, s in enumerate(slicing):
        if isinstance(s, slice) and s.start is not None:
            minc[i] = s.start
        elif isinstance(s, (int, torch.Tensor)):
            minc[i] = s

    # wrap any negative coordinates
    negative = minc < 0
    minc[negative] = shape[negative] + minc[negative]

    # extract the minimum coordinate based on the slice
    # stop index (subtracted by 1)
    maxc = shape - 1
    for i, s in enumerate(slicing):
        if isinstance(s, slice) and s.stop is not None:
            maxc[i] = s.stop - 1
        elif isinstance(s, (int, torch.Tensor)):
            maxc[i] = s

    # again, wrap any negative coordinates
    negative = maxc < 0
    maxc[negative] = shape[negative] + maxc[negative]

    # extract stride (if exists) from the slice step
    stride = [s.step if isinstance(s, slice) else None for s in slicing]
    stride = torch.tensor([x or 1 for x in stride]) if any(stride) else None

    return (minc, maxc, stride)


def conform_coordinates(coords: torch.Tensor, num: int = None) -> torch.Tensor:
    """
    Conform coordinates to certain shape and dimensionality.

    Args:
        coords (Tensor): Coordinate tensor of size $(1,)$ or $(3,)$ or $(3, N)$.
        num (int, optional): Number of coordinates $N$ represented in the second dimension
            of the output tensor. If None, the output tensor will have shape $(3,)$.
    
    Returns:
        Tensor: Coordinates of shape $(3, N)$ or $(3,)$.
    """
    coords = torch.as_tensor(coords)
    if coords.ndim == 0:
        coords = coords.repeat((3, num)) if num is not None else coords.repeat(3)
    elif coords.shape == (3,) and num is not None:
        coords = coords.unsqueeze(1).repeat(1, 2)
    if num is not None and coords.shape != (3, num):
        raise ValueError(f'tensor must be of size (3, {num}) or (3,) or (1,), got {coords.shape}')
    if num is None and coords.shape != (3,):
        raise ValueError(f'tensor must be of size (3,) or (1,), got {coords.shape}')
    return coords
