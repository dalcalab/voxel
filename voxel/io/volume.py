"""
Reading and writing image volumes to various file formats.
"""

import os
import torch
import numpy as np
import voxel as vx

from .utility import IOProtocol


def load_volume(filename: os.PathLike, fmt: str = None) -> vx.Volume:
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
        proto = vx.io.protocol.find_protocol_by_name(volume_io_protocols, fmt)
        if proto is None:
            raise ValueError(f'unknown file format {fmt}')

    return proto().load(filename)


def save_volume(volume: vx.Volume, filename: os.PathLike, fmt: str = None) -> None:
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
        proto = vx.io.protocol.find_protocol_by_name(volume_io_protocols, fmt)
        if proto is None:
            raise ValueError(f'unknown file format {fmt}')
        filename = proto.enforce_extension(filename)

    proto().save(volume, filename)


class NiftiHeaderReference:
    """
    A reference to cache parameters of a nifti file header.
    This is passed around in the metadata of an acquisition
    geometry so that use it a reference (if needed) for resaving
    a volume without introducing any corruptions to the original
    file header.v
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
        if array.dtype == np.uint16:
            array = array.astype(np.int32)

        features = torch.from_numpy(array)
        if features.ndim == 4:
            features = features.moveaxis(-1, 0)

        # 
        spacing = torch.from_numpy(nii.header['pixdim'][1:4])
        affine = torch.from_numpy(nii.affine)
        volume = vx.Volume(features, affine)

        # 
        volume.geometry.metadata['nii_reference'] = NiftiHeaderReference(nii)

        # 
        if not torch.isclose(volume.geometry.spacing, spacing, atol=0.01, rtol=0.2).all():
            explicit_spacing = ', '.join([f'{s}:.2f' for s in spacing])
            affine_spacing = ', '.join([f'{s}:.2f' for s in volume.geometry.spacing])
            print('Warning: There is a substantial difference between the explicit voxel '
                  f'spacing in the nifti header ({explicit_spacing}) and the '
                  f'computed spacing from the scanner affine ({affine_spacing})')

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
        type_map = {
            np.bool8: np.uint8,
        }
        dtype_id = next((i for dt, i in type_map.items() if np.issubdtype(volume_array.dtype, dt)), None)
        if dtype_id is not None:
            volume_array = volume_array.astype(dtype_id)

        # 
        nii = self.nib.Nifti1Image(volume_array, np.eye(4))

        # 
        spacing = volume.geometry.spacing.detach().cpu().numpy().astype(np.float64)
        affine = volume.geometry.tensor.detach().cpu().numpy().astype(np.float64)

        # 
        ref = volume.geometry.metadata.get('nii_reference')
        matches_original = ref is not None and \
            ref.baseshape == volume.baseshape and \
            np.isclose(ref.affine, affine, rtol=0, atol=1e-4).all()

        # 
        if matches_original:
            affine = ref.affine
            spacing = ref.spacing

        # set spatial and temporal spacing
        nii.header['pixdim'][:] = 1
        nii.header['pixdim'][4] = 1 if not matches_original else ref.channel_spacing
        nii.header['pixdim'][1:4] = spacing

        # set units - fallback to mm and seconds
        default = np.asarray(2, dtype=np.uint8) | np.asarray(8, dtype=np.uint8)
        nii.header['xyzt_units'] = default if not matches_original else ref.xyzt_units

        # geometry-specific header data
        nii.set_sform(affine, 1 if not matches_original else ref.sform_code)
        nii.set_qform(affine, 1 if not matches_original else ref.qform_code)

        # write
        self.nib.save(nii, filename)


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
        items = torch.load(filename)
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
        features = volume.tensor.detach().cpu()
        matrix = volume.geometry.tensor.detach().cpu()
        torch.save({'v': features, 'm': matrix}, filename)


# enabled volume IO protocol classes
volume_io_protocols = [
    NiftiArrayIO,
    PytorchVolumeIO,
]
