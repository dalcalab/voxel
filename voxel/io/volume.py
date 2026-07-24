"""
Reading and writing image volumes to various file formats.
"""

from __future__ import annotations

import os
import json
import torch
import numpy as np
import voxel as vx

from .utility import IOProtocol


def load_volume(filename: os.PathLike, fmt: str | None = None) -> vx.Volume:
    """
    Load a volume from a file.

    Args:
        filename (PathLike): The path to the file to load.
        fmt (str, optional): The format of the file. If None, the format is
            determined by the file extension.

    Returns:
        Volume: The loaded volume.
    """
    vx.io.utility.check_file_readability(filename)

    if fmt is None:
        proto = vx.io.utility.find_protocol_by_extension(volume_io_protocols, filename)
        if proto is None:
            raise ValueError(f'cannot determine file format from extension for {filename}')
    else:
        proto = vx.io.utility.find_protocol_by_name(volume_io_protocols, fmt)
        if proto is None:
            raise ValueError(f'unknown file format {fmt}')

    return proto().load(filename)


def save_volume(volume: vx.Volume, filename: os.PathLike, fmt: str | None = None) -> None:
    """
    Save a volume to a file.

    Args:
        volume (Volume): The volume to save.
        filename (PathLike): The path to the file to save.
        fmt (str, optional): The format of the file. If None, the format is
            determined by the file extension.
    """
    if fmt is None:
        proto = vx.io.utility.find_protocol_by_extension(volume_io_protocols, filename)
        if proto is None:
            raise ValueError(f'cannot determine file format from extension for {filename}')
    else:
        proto = vx.io.utility.find_protocol_by_name(volume_io_protocols, fmt)
        if proto is None:
            raise ValueError(f'unknown file format {fmt}')
        filename = proto.enforce_extension(filename)

    proto().save(volume, filename)


class NiftiHeaderReference:
    """
    Caches the parameters of a nifti file header. This is carried in the
    metadata of an acquisition geometry so that it can be used as a reference
    (if needed) when resaving a volume, avoiding corruptions to the original
    file header.
    """

    def __init__(self, nii) -> None:
        """
        Args:
            nii (Nifti1Image): The nifti image to cache.
        """
        self.qform_code = int(nii.header['qform_code'])
        self.sform_code = int(nii.header['sform_code'])
        self.xyzt_units = nii.header['xyzt_units']
        self.channel_spacing = nii.header['pixdim'][4]
        self.baseshape = tuple(nii.header['dim'][1:4])
        self.spacing = nii.header['pixdim'][1:4]
        self.affine = nii.affine


class NiftiArrayIO(IOProtocol):
    """
    Array IO protocol for nifti files.
    """
    name = 'nifti'
    extensions = ('.nii.gz', '.nii')

    def __init__(self) -> None:
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError('the `nibabel` python package must be installed for nifti volume IO')
        self.nib = nib

    def _color_to_hex(self, color: torch.Tensor) -> str:
        r, g, b = (color * 255).round().to(torch.int).tolist()
        return f'#{r:02x}{g:02x}{b:02x}'

    def _color_from_hex(self, text: str) -> torch.Tensor:
        text = text.strip().lstrip('#')
        rgb = [int(text[i:i + 2], 16) for i in (0, 2, 4)]
        return torch.tensor(rgb, dtype=torch.float32) / 255.0

    def _labels_to_json(self, lookup: vx.LabelLookup) -> str:
        """
        Serialize a label lookup table into a JSON string. Colors are stored as
        `#rrggbb` hex strings and omitted for labels that have no color.
        """
        entries = []
        for index, label in lookup.items():
            entry = {'index': int(index), 'name': label.name}
            if label.color is not None:
                entry['color'] = self._color_to_hex(label.color)
            entries.append(entry)
        return json.dumps({'vxlabels': entries})

    def _labels_from_json(self, text: str | bytes) -> vx.LabelLookup | None:
        """
        Parse a label lookup table from a JSON string, returning None if the text
        is not a recognized `vxlabels` block.
        """
        if isinstance(text, bytes):
            try:
                text = text.decode('utf-8')
            except UnicodeDecodeError:
                return None
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict) or 'vxlabels' not in data:
            return None

        lookup = vx.LabelLookup()
        for entry in data['vxlabels']:
            color = entry.get('color')
            if color is not None:
                color = self._color_from_hex(color)
            lookup[int(entry['index'])] = vx.Label(entry['name'], color)
        return lookup

    def load(self, filename: os.PathLike) -> vx.Volume:
        """
        Read array from a Nifti file.

        Args:
            filename (PathLike): The path to the Nifti file to read.

        Returns:
            Volume: The loaded volume.
        """
        nii = self.nib.load(filename)
        array = np.asanyarray(nii.dataobj)

        # not supported by torch
        if array.dtype in (np.uint16, np.uint32):
            array = array.astype(np.int32)

        features = torch.from_numpy(array)
        if features.ndim == 4:
            features = features.moveaxis(-1, 0)

        # 
        spacing = torch.from_numpy(nii.header['pixdim'][1:4])
        affine = torch.from_numpy(nii.affine)
        volume = vx.Volume(features, affine)

        # 
        if not torch.allclose(volume.geometry.spacing, spacing, atol=0.01, rtol=0.2):

            explicit_spacing = ', '.join([f'{s:.2f}' for s in spacing])
            affine_spacing = ', '.join([f'{s:.2f}' for s in volume.geometry.spacing])
            warning = f'warning: explicit voxel spacing in the nifti header ({explicit_spacing}) ' \
                      f'does not match scanner affine spacing ({affine_spacing})'

            if torch.allclose(volume.geometry.tensor.abs(), torch.eye(4), atol=1e-5):
                volume = vx.Volume(features, volume.geometry.scale(spacing, space='world'))
                warning = f'{warning} - overwriting with explicit spacing'

            print(warning)

        #
        volume.geometry.reference['nii'] = NiftiHeaderReference(nii)

        # read an embedded label lookup table, if present
        for ext in nii.header.extensions:
            labels = self._labels_from_json(ext.get_content())
            if labels is not None:
                volume.labels = labels
                break

        #
        return volume

    def save(self, volume: vx.Volume, filename: os.PathLike) -> None:
        """
        Write volume to a Nifti file.

        Args:
            volume (Volume): The volume to save.
            filename (PathLike): The path to the Nifti file to write.
        """

        # 
        volume_array = volume.tensor.movedim(0, -1).detach().cpu().numpy()
        if volume_array.shape[-1] == 1:
            volume_array = np.squeeze(volume_array, -1)

        # convert to a valid output type (for now this is only bool but there are probably more)
        type_map = {np.dtype('bool'): np.uint8}
        dtype_id = next((i for dt, i in type_map.items() if np.issubdtype(volume_array.dtype, dt)), None)
        if dtype_id is not None:
            volume_array = volume_array.astype(dtype_id)

        # 
        nii = self.nib.Nifti1Image(volume_array, np.eye(4))

        # 
        spacing = volume.geometry.spacing.detach().cpu().numpy().astype(np.float64)
        affine = volume.geometry.tensor.detach().cpu().numpy().astype(np.float64)

        # 
        ref = volume.geometry.reference.get('nii')
        matches_original = ref is not None and \
            ref.baseshape == tuple(volume.baseshape) and \
            np.isclose(ref.affine, affine, rtol=0, atol=1e-3).all()

        # 
        if matches_original:
            affine = ref.affine
            spacing = ref.spacing

        # reset pixdim; qfac and spatial spacing are (re)written after set_qform below
        nii.header['pixdim'][:] = 1

        # set units - fallback to mm and seconds
        default = np.asarray(2, dtype=np.uint8) | np.asarray(8, dtype=np.uint8)
        nii.header['xyzt_units'] = default if not matches_original else ref.xyzt_units

        # geometry-specific header data
        nii.set_sform(affine, 1 if not matches_original else ref.sform_code)
        nii.set_qform(affine, 1 if not matches_original else ref.qform_code)

        # set pixdim after qform, to avoid it clobbering the spacing (when there is shear)
        nii.header['pixdim'][1:4] = spacing
        nii.header['pixdim'][4] = 1 if not matches_original else ref.channel_spacing

        # embed the label lookup table as a json comment extension
        if volume.labels:
            payload = self._labels_to_json(volume.labels).encode('utf-8')
            nii.header.extensions.append(self.nib.nifti1.Nifti1Extension('comment', payload))
            nii.header.set_intent('label')

        # write
        self.nib.save(nii, filename)


class MghArrayIO(IOProtocol):
    """
    Array IO protocol for mgh files.
    """
    name = 'mgh'
    extensions = ('.mgz', '.mgh')

    def __init__(self) -> None:
        try:
            import surfa as sf
        except ImportError:
            raise ImportError('the `surfa` python package must be installed for mgh volume IO')
        self.sf = sf

    def _labels_to_surfa(self, lookup: vx.LabelLookup):
        """
        Convert a label lookup table into a surfa `LabelLookup`. surfa requires a
        color per entry, so labels without one are assigned black.
        """
        sf_lookup = self.sf.LabelLookup()
        for index, label in lookup.items():
            if label.color is None:
                color = [0, 0, 0]
            else:
                color = [int(round(float(c) * 255)) for c in label.color]
            sf_lookup[int(index)] = (label.name, color)
        return sf_lookup

    def _labels_from_surfa(self, sf_lookup) -> vx.LabelLookup:
        """
        Convert a surfa `LabelLookup` into a voxel label lookup table.
        """
        lookup = vx.LabelLookup()
        for index, element in sf_lookup.items():
            color = None
            if getattr(element, 'color', None) is not None:
                color = [float(c) / 255.0 for c in element.color[:3]]
            lookup[int(index)] = vx.Label(element.name, color)
        return lookup

    def load(self, filename: os.PathLike) -> vx.Volume:
        """
        Read array from a MGH file.

        Args:
            filename (PathLike): The path to the MGH file to read.

        Returns:
            Volume: The loaded volume.
        """
        sv = self.sf.load_volume(filename)

        data = vx.io.numpy_to_tensor(sv.framed_data).movedim(-1, 0)
        matrix = vx.io.numpy_to_tensor(sv.geom.vox2world.matrix, copy=True)
        volume = vx.Volume(data, matrix)

        volume.geometry.reference['mgh'] = sv.geom

        if sv.labels is not None:
            volume.labels = self._labels_from_surfa(sv.labels)

        return volume

    def save(self, volume: vx.Volume, filename: os.PathLike) -> None:
        """
        Write volume to a MGH file.

        Args:
            volume (Volume): The volume to save.
            filename (PathLike): The path to the MGH file to write.
        """
        volume_array = volume.tensor.movedim(0, -1).detach().cpu().numpy()
        affine = volume.geometry.tensor.detach().cpu().numpy()

        ref = volume.geometry.reference.get('mgh')
        matches_original = ref is not None and \
            tuple(ref.shape) == tuple(volume.baseshape) and \
            np.isclose(ref.vox2world.matrix, affine, rtol=0, atol=1e-3).all()

        geometry = ref if matches_original else self.sf.ImageGeometry(volume.baseshape, vox2world=affine)

        labels = self._labels_to_surfa(volume.labels) if volume.labels else None

        self.sf.Volume(volume_array, geometry=geometry, labels=labels).save(filename)


class PytorchVolumeIO(IOProtocol):
    """
    Array IO protocol for storing a simple volume in a pytorch file.
    The only data stored is the features tensor and the world affine.
    This is useful for fast data loading during training.
    """
    name = 'torch'
    extensions = ('.pth', '.pt')

    def load(self, filename: os.PathLike) -> vx.Volume:
        """
        Read array from a pytorch file.

        Args:
            filename (PathLike): The path to the pytorch file to read.

        Returns:
            Volume: The loaded volume.
        """
        items = torch.load(filename, weights_only=False)
        if 'v' not in items or 'm' not in items:
            raise RuntimeError(f'could not find `v` or `m` data keys in {filename}')
        return vx.Volume(items['v'], items['m'])

    def save(self, volume: vx.Volume, filename: os.PathLike) -> None:
        """
        Write volume to a pytorch file.

        Args:
            volume (Volume): The volume to save.
            filename (PathLike): The path to the pytorch file to write.
        """
        features = volume.tensor.detach().cpu().contiguous()
        matrix = volume.geometry.tensor.detach().cpu().contiguous()
        torch.save({'v': features, 'm': matrix}, filename)


# enabled volume IO protocol classes
volume_io_protocols = [
    NiftiArrayIO,
    MghArrayIO,
    PytorchVolumeIO,
]
