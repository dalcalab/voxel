"""
Reading and writing label lookup tables to various file formats.
"""

from __future__ import annotations

import csv
import os
import torch
import voxel as vx

from .utility import IOProtocol


def load_labels(filename: os.PathLike, fmt: str | None = None) -> vx.LabelLookup:
    """
    Load a label lookup table from a file.

    Args:
        filename (PathLike): The path to the file to load.
        fmt (str, optional): The format of the file. If None, the format is
            determined by the file extension.

    Returns:
        LabelLookup: The loaded lookup table.
    """
    vx.io.utility.check_file_readability(filename)

    if fmt is None:
        proto = vx.io.utility.find_protocol_by_extension(label_io_protocols, filename)
        if proto is None:
            raise ValueError(f'cannot determine file format from extension for {filename}')
    else:
        proto = vx.io.utility.find_protocol_by_name(label_io_protocols, fmt)
        if proto is None:
            raise ValueError(f'unknown file format {fmt}')

    return proto().load(filename)


def save_labels(lookup: vx.LabelLookup, filename: os.PathLike, fmt: str | None = None, **kwargs) -> None:
    """
    Save a label lookup table to a file.

    Args:
        lookup (LabelLookup): The lookup table to save.
        filename (PathLike): The path to the file to save.
        fmt (str, optional): The format of the file. If None, the format is
            determined by the file extension.
        **kwargs (Any): Additional arguments to pass to the file writing method.
    """
    if fmt is None:
        proto = vx.io.utility.find_protocol_by_extension(label_io_protocols, filename)
        if proto is None:
            raise ValueError(f'cannot determine file format from extension for {filename}')
    else:
        proto = vx.io.utility.find_protocol_by_name(label_io_protocols, fmt)
        if proto is None:
            raise ValueError(f'unknown file format {fmt}')
        filename = proto.enforce_extension(filename)

    proto().save(lookup, filename, **kwargs)


class LabelTabularIO(IOProtocol):
    """
    IO protocol for storing a label lookup table as a delimited text table.

    The file has a header row `index,name,color` followed by one row per label.
    Colors are stored as `#rrggbb` hex strings and converted to and from the
    in-memory RGB float representation in [0, 1]; a label with no color is written
    with an empty color field. The delimiter is chosen from the file extension:
    a comma for `.csv` and a tab for `.tsv`.
    """
    name = 'tabular'
    extensions = ('.csv', '.tsv')

    def _delimiter(self, filename: os.PathLike) -> str:
        return '\t' if str(filename).lower().endswith('.tsv') else ','

    def _color_to_hex(self, color: torch.Tensor) -> str:
        r, g, b = (color * 255).round().to(torch.int).tolist()
        return f'#{r:02x}{g:02x}{b:02x}'

    def _color_from_hex(self, text: str) -> torch.Tensor:
        text = text.strip().lstrip('#')
        rgb = [int(text[i:i + 2], 16) for i in (0, 2, 4)]
        return torch.tensor(rgb, dtype=torch.float32) / 255.0

    def load(self, filename: os.PathLike) -> vx.LabelLookup:
        lookup = vx.LabelLookup()
        with open(filename, newline='') as f:
            reader = csv.reader(f, delimiter=self._delimiter(filename))
            next(reader, None)  # skip the header row
            for row in reader:
                if not row or row[0].strip().startswith('#'):
                    continue
                index = int(row[0])
                name = row[1]
                color = None
                if len(row) >= 3 and row[2].strip() != '':
                    color = self._color_from_hex(row[2])
                lookup[index] = vx.Label(name, color)
        return lookup

    def save(self, lookup: vx.LabelLookup, filename: os.PathLike) -> None:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=self._delimiter(filename))
            writer.writerow(['index', 'name', 'color'])
            for index, label in lookup.items():
                color = '' if label.color is None else self._color_to_hex(label.color)
                writer.writerow([index, label.name, color])


label_io_protocols = [
    LabelTabularIO,
]
