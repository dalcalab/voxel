"""
Label lookup tables that pair integer label values with names and display colors.
"""

from __future__ import annotations

import os
import torch
import voxel as vx


def _conform_color(color) -> torch.Tensor:
    """
    Conform a color specification into a float RGB tensor in the range [0, 1].
    """
    tensor = torch.as_tensor(color, dtype=torch.float32).squeeze()
    if tensor.ndim != 1 or tensor.numel() != 3:
        raise ValueError(f'label color must have 3 (RGB) elements, got shape {tuple(tensor.shape)}')
    if float(tensor.min()) < 0 or float(tensor.max()) > 1:
        raise ValueError('label color values must be in the range [0, 1]')
    return tensor


class Label:
    """
    A single entry of a `LabelLookup`, describing one label value.

    The integer value itself is the key in the enclosing `LabelLookup`, so it is
    not stored on the label.
    """

    def __init__(self, name: str, color: torch.Tensor | list | tuple | None = None) -> None:
        """
        Args:
            name (str): The human-readable name of the label.
            color (Tensor or list or tuple, optional): An RGB color with three
                elements in the range [0, 1]. If None, the label has no color.
        """
        self.name = str(name)
        self.color = None if color is None else _conform_color(color)

    def __repr__(self) -> str:
        if self.color is None:
            return f"Label(name='{self.name}')"
        color = [round(float(c), 3) for c in self.color]
        return f"Label(name='{self.name}', color={color})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Label):
            return NotImplemented
        if self.name != other.name:
            return False
        if (self.color is None) != (other.color is None):
            return False
        return self.color is None or bool(torch.equal(self.color, other.color))


def _conform_label(value) -> Label:
    """
    Coerce a value into a `Label`, accepting a `Label`, a name string, or a
    `(name, color)` tuple.
    """
    if isinstance(value, Label):
        return value
    if isinstance(value, str):
        return Label(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return Label(value[0], value[1])
    raise ValueError('label value must be a Label, a name string, or a (name, color) tuple')


class LabelLookup(dict):
    """
    An ordered lookup table mapping integer label values to `Label` entries.
    """

    def __init__(self, mapping: dict | None = None) -> None:
        """
        Args:
            mapping (dict, optional): Initial integer-to-label entries.
        """
        super().__init__()
        self._device = torch.device('cpu')
        self._indices_tensor = None
        if mapping is not None:
            self.update(mapping)

    def _invalidate(self) -> None:
        # drop the cached index tensor; a None cache is the dirty signal, so any
        # method that changes the set of keys must call this
        self._indices_tensor = None

    def __setitem__(self, key: int, value) -> None:
        if isinstance(key, bool) or not isinstance(key, int):
            raise ValueError(f'label lookup keys must be integers, got {type(key).__name__}')
        label = _conform_label(value)
        if label.color is not None:
            label.color = label.color.to(self._device)
        super().__setitem__(int(key), label)
        self._invalidate()

    def __delitem__(self, key: int) -> None:
        super().__delitem__(key)
        self._invalidate()

    def update(self, mapping) -> None:
        for key, value in dict(mapping).items():
            self[key] = value

    def setdefault(self, key: int, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key: int, *args):
        present = key in self
        value = super().pop(key, *args)
        if present:
            self._invalidate()
        return value

    def popitem(self):
        item = super().popitem()
        self._invalidate()
        return item

    def clear(self) -> None:
        super().clear()
        self._invalidate()

    @property
    def device(self) -> torch.device:
        """
        The device of the label colors and cached lookup tensor.
        """
        return self._device

    @property
    def indices_tensor(self) -> torch.Tensor:
        """
        A cached 1D tensor of the ordered label values (keys), on the table's
        device. Rebuilt automatically when the keys change.
        """
        if self._indices_tensor is None:
            self._indices_tensor = torch.tensor(tuple(self.keys()), dtype=torch.long, device=self._device)
        return self._indices_tensor

    def to(self, device: torch.device) -> LabelLookup:
        """
        Return a copy of the lookup table with colors and cache on a new device.

        Args:
            device (device): The target device.

        Returns:
            LabelLookup: The moved lookup table.
        """
        moved = LabelLookup()
        moved._device = torch.device(device)
        for key, label in self.items():
            entry = Label(label.name)
            entry.color = None if label.color is None else label.color.to(moved._device)
            dict.__setitem__(moved, key, entry)
        return moved

    def cpu(self) -> LabelLookup:
        """
        Return a copy of the lookup table on the CPU.
        """
        return self.to('cpu')

    def cuda(self) -> LabelLookup:
        """
        Return a copy of the lookup table on the current CUDA device.
        """
        return self.to('cuda')

    def add(self, index: int, name: str, color: torch.Tensor | list | tuple | None = None) -> None:
        """
        Add a label entry to the lookup table.

        Args:
            index (int): The integer label value.
            name (str): The human-readable name of the label.
            color (Tensor or list or tuple, optional): An RGB color in [0, 1].
        """
        self[index] = Label(name, color)

    def search(self, query: str, exact: bool = False) -> list[tuple[int, Label]]:
        """
        Search for label entries by name.

        Args:
            query (str): The name (or substring) to search for, case-insensitive.
            exact (bool, optional): If True, require an exact (case-insensitive)
                name match rather than a substring match.

        Returns:
            list of (int, Label): The matching (index, label) pairs.
        """
        query = query.lower()
        results = []
        for index, label in self.items():
            name = label.name.lower()
            if (name == query) if exact else (query in name):
                results.append((index, label))
        return results

    @property
    def indices(self) -> list[int]:
        """
        The ordered list of integer label values (keys) in the table.
        """
        return list(self.keys())

    def colors(self, default: torch.Tensor | list | tuple = (0, 0, 0)) -> torch.Tensor:
        """
        Stack the label colors into an $(N, 3)$ tensor in insertion order.

        Args:
            default (Tensor or list or tuple, optional): The RGB color to
                substitute for any label that has no color.

        Returns:
            Tensor: An $(N, 3)$ float tensor of RGB colors in [0, 1].
        """
        default = torch.as_tensor(default, dtype=torch.float32)
        rows = [label.color if label.color is not None else default for label in self.values()]
        if not rows:
            return torch.empty((0, 3))
        return torch.stack(rows)

    def save(self, filename: os.PathLike, fmt: str | None = None) -> None:
        """
        Save the lookup table to a file.

        Args:
            filename (PathLike): The path to the file to save.
            fmt (str, optional): The format of the file. If None, the format is
                determined by the file extension.
        """
        vx.save_labels(self, filename, fmt=fmt)

    @classmethod
    def load(cls, filename: os.PathLike, fmt: str | None = None) -> LabelLookup:
        """
        Load a lookup table from a file.

        Args:
            filename (PathLike): The path to the file to load.
            fmt (str, optional): The format of the file. If None, the format is
                determined by the file extension.

        Returns:
            LabelLookup: The loaded lookup table.
        """
        return vx.load_labels(filename, fmt=fmt)


def _as_mapping_tensor(mapping, device: torch.device, background: bool = False) -> torch.Tensor:
    """
    Coerce a recode mapping (a `LabelLookup`, tensor, or sequence) into a 1D
    tensor of values on the given device. If `background` is set, the background
    label 0 is prepended to the mapping unless it is already present, reserving
    the first index position for it.
    """
    if isinstance(mapping, LabelLookup):
        mapping = mapping.indices_tensor
    mapping = torch.as_tensor(mapping, device=device)
    if background and not bool((mapping == 0).any()):
        zero = torch.zeros(1, dtype=mapping.dtype, device=device)
        mapping = torch.cat([zero, mapping])
    return mapping


def recode(tensor: torch.Tensor,
    mapping: torch.Tensor | list | LabelLookup,
    reverse: bool = False,
    default: int = 0,
    background: bool = False) -> torch.Tensor:
    """
    Remap the values of a label tensor through an ordered mapping.

    In the forward direction, `tensor` is treated as an index map and each value
    `i` is replaced with `mapping[i]` (a lookup-table gather); the mapping values
    may be of any dtype. In the reverse direction, each value is replaced with its
    *position* within `mapping`, which requires an integer mapping; values not
    present in `mapping` become `default`.

    A `LabelLookup` is treated as its ordered integer `indices`.

    Args:
        tensor (Tensor): An integer label tensor.
        mapping (Tensor or list or LabelLookup): The ordered values to map index
            positions to.
        reverse (bool, optional): Map values back to their index positions.
        default (int, optional): The index assigned to reverse-mapped values that
            are not present in `mapping`.
        background (bool, optional): Prepend the background label 0 to `mapping`
            (unless already present) so it occupies the first index position.

    Returns:
        Tensor: The recoded tensor, with the same shape as `tensor`.
    """
    if torch.is_floating_point(tensor):
        raise ValueError(f'recode requires an integer tensor, got {tensor.dtype}')
    mapping = _as_mapping_tensor(mapping, tensor.device, background=background)
    if not reverse:
        return mapping[tensor.long()]

    if torch.is_floating_point(mapping):
        raise ValueError(f'reverse recode requires an integer mapping, got {mapping.dtype}')
    mapping = mapping.long()
    size = int(mapping.max()) + 1 if mapping.numel() > 0 else 0
    if tensor.numel() > 0:
        size = max(size, int(tensor.max()) + 1)

    inverse = torch.full((size,), default, dtype=torch.long, device=tensor.device)
    inverse[mapping] = torch.arange(mapping.numel(), device=tensor.device)
    return inverse[tensor.long()]


def onehot(tensor: torch.Tensor,
    labels: int | torch.Tensor | LabelLookup = -1,
    background: bool = False) -> torch.Tensor:
    """
    One-hot encode an integer label tensor, placing classes on a leading dimension.

    Args:
        tensor (Tensor): An integer label tensor.
        labels (int or Tensor or LabelLookup, optional): The classes to encode. If
            an integer, it is the total number of classes (-1 infers one greater
            than the largest value). If a tensor or lookup of label values, the
            tensor is first reverse-recoded so those values map to the channels.
        background (bool, optional): When `labels` is a tensor or lookup, reserve
            the first channel for the background label 0 (unless already present).

    Returns:
        Tensor: The one-hot encoded tensor with the class dimension inserted first.
    """
    if torch.is_floating_point(tensor):
        raise ValueError(f'one hot requires an integer tensor, got {tensor.dtype}')
    if isinstance(labels, int):
        indices = tensor.long()
        num_classes = labels
    else:
        mapping = _as_mapping_tensor(labels, tensor.device, background=background)
        indices = recode(tensor, mapping, reverse=True)
        num_classes = mapping.numel()
    encoded = torch.nn.functional.one_hot(indices, num_classes=num_classes)
    return encoded.movedim(-1, 0)


def collapse(tensor: torch.Tensor,
    labels: torch.Tensor | LabelLookup = None,
    background: bool = False) -> torch.Tensor:
    """
    Collapse a one-hot (or probabilistic) tensor into a label tensor.

    This is the inverse of `onehot`: the leading class dimension is reduced with
    an argmax, and the resulting channel indices are optionally recoded into label
    values.

    Args:
        tensor (Tensor): A tensor with classes on the leading dimension.
        labels (Tensor or LabelLookup, optional): The label values the
            channels correspond to. If None, the channel indices are returned
            directly, otherwise they are forward-recoded into label values.
        background (bool, optional): When `labels` is provided, treat the first
            channel as the background label 0 (unless already present), matching
            `onehot`.

    Returns:
        Tensor: The label tensor with the class dimension removed.
    """
    indices = tensor.argmax(dim=0)
    if labels is None:
        return indices
    mapping = _as_mapping_tensor(labels, tensor.device, background=background)
    return recode(indices, mapping, reverse=False)
