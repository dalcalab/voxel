"""
Reading and writing meshes to various file formats.
"""

import os
import torch
import voxel as vx

from .utility import IOProtocol


def load_mesh(filename: os.PathLike, fmt: str = None) -> vx.Mesh:
    """
    Load a mesh from a file.

    Args:
        filename (PathLike): The path to the file to load.
        fmt (str, optional): The format of the file. If None, the format is
            determined by the file extension.

    Returns:
        Mesh: The loaded mesh.
    """
    vx.io.utility.check_file_readability(filename)

    if fmt is None:
        proto = vx.io.utility.find_protocol_by_extension(mesh_io_protocols, filename)
        if proto is None:
            proto = FreesurferIO
    else:
        proto = vx.io.protocol.find_protocol_by_name(mesh_io_protocols, fmt)
        if proto is None:
            raise ValueError(f'unknown file format {fmt}')

    return proto().load(filename)


def save_mesh(mesh: vx.Mesh, filename: os.PathLike, fmt: str = None, **kwargs) -> None:
    """
    Save a mesh to a file.

    Args:
        mesh (Mesh): The mesh to save.
        filename (PathLike): The path to the file to save.
        fmt (str, optional): The format of the file. If None, the format is
            determined by the file extension.
        kwargs: Additional arguments to pass to the file writing method.
    
    Notes:
        By default the mesh will write to 
    """
    if fmt is None:
        proto = vx.io.utility.find_protocol_by_extension(mesh_io_protocols, filename)
        if proto is None:
            proto = FreesurferIO
    else:
        proto = vx.io.protocol.find_protocol_by_name(mesh_io_protocols, fmt)
        if proto is None:
            raise ValueError(f'unknown file format {fmt}')
        filename = proto.enforce_extension(filename)

    proto().save(mesh, filename, **kwargs)


class WavefrontIO(IOProtocol):
    """
    Mesh IO protocol for wavefront obj files.
    """

    name = 'obj'
    extensions = ('.obj',)

    def __init__(self):
        try:
            import trimesh
        except ImportError:
            raise ImportError('the trimesh python package must be installed for '
                              'wavefront surface IO')
        self.trimesh = trimesh

    def load(self, filename: os.PathLike) -> vx.Mesh:
        """
        Read mesh from a wavefront object file.

        Args:
            filename (PathLike): The path to the wavefront file to read.

        Returns:
            Mesh: The loaded mesh.
        """
        tmesh = self.trimesh.exchange.load.load(filename, process=False)
        return vx.Mesh(torch.as_tensor(tmesh.vertices), torch.as_tensor(tmesh.faces))

    def save(self, mesh: vx.Mesh, filename: os.PathLike) -> None:
        """
        Write mesh to a wavefront object file.

        Args:
            mesh (Mesh): The mesh to save.
            filename (PathLike): Output path.
        """
        parameters = dict(
            header='SPACE=RAS\n',
            include_color=False,
            include_texture=False,
            return_texture=False,
            write_texture=True)
        mesh = mesh.cpu()
        tmesh = self.trimesh.Trimesh(mesh.vertices.detach().numpy(),
                                     mesh.faces.detach().numpy(), process=False)
        tmesh.export(filename, **parameters)


class StanfordPolygonIO(IOProtocol):
    """
    Mesh IO protocol for the Polygon File Format, a.k.a. Stanford PLY.
    """

    name = 'ply'
    extensions = ('.ply',)

    def __init__(self):
        try:
            import trimesh
        except ImportError:
            raise ImportError('the trimesh python package must be installed '
                              'for polygon surface IO')
        self.trimesh = trimesh

    def load(self, filename: os.PathLike) -> vx.Mesh:
        """
        Read mesh from a polygon file.

        Args:
            filename (PathLike): The path to the wavefront file to read.

        Returns:
            Mesh: The loaded mesh.
        """
        tmesh = self.trimesh.exchange.load.load(filename, process=False)
        return vx.Mesh(torch.as_tensor(tmesh.vertices), torch.as_tensor(tmesh.faces))

    def save(self,
        mesh: vx.Mesh,
        filename: os.PathLike,
        vertex_attributes: torch.Tensor = None,
        face_attributes: torch.Tensor = None) -> None:
        """
        Write mesh to a polygon file.

        Args:
            mesh (Mesh): The mesh to save.
            filename (PathLike): Output path.
            vertex_attributes (Tensor, optional): Dictionary of vertex attribute
                tensors to save with the mesh.
            face_attributes (Tensor, optional): Dictionary of face attribute tensors
                to save with the mesh.
        """
        def prep_attributes(attributes):
            if not isinstance(attributes, dict):
                attributes = {'attribute': attributes}
            return {k: v.detach().cpu().numpy() for k, v in attributes.items()}

        mesh = mesh.cpu()
        tmesh = self.trimesh.Trimesh(mesh.vertices.detach().numpy(),
                                     mesh.faces.detach().numpy(), process=False)
        if vertex_attributes is not None:
            tmesh.vertex_attributes = prep_attributes(vertex_attributes)
        if face_attributes is not None:
            tmesh.face_attributes = prep_attributes(face_attributes)
        tmesh.export(filename, include_attributes=True)


class AbstractTrimeshIO(IOProtocol):
    """
    Mesh IO protocol for the Polygon File Format, a.k.a. Stanford PLY.
    """
    def __init__(self):
        try:
            import trimesh
        except ImportError:
            raise ImportError('the trimesh python package must be installed '
                              f'for {self.name} surface IO')
        self.trimesh = trimesh

    def load(self, filename: os.PathLike) -> vx.Mesh:
        """
        Read mesh from a file.

        Args:
            filename (PathLike): The path to the file to read.

        Returns:
            Mesh: The loaded mesh.
        """
        tmesh = self.trimesh.exchange.load.load(filename, process=False)
        return vx.Mesh(torch.as_tensor(tmesh.vertices), torch.as_tensor(tmesh.faces))

    def save(self, mesh: vx.Mesh, filename: os.PathLike) -> None:
        """
        Write mesh to a file.

        Args:
            mesh (Mesh): The mesh to save.
            filename (PathLike): Output path.
        """
        mesh = mesh.cpu()
        tmesh = self.trimesh.Trimesh(mesh.vertices.detach().numpy(),
                                     mesh.faces.detach().numpy(), process=False)
        tmesh.export(filename)


class GltfIO(AbstractTrimeshIO):
    name = 'gltf'
    extensions = ('.gltf', '.glb')


class Stl3D(AbstractTrimeshIO):
    name = 'stl'
    extensions = ('.stl',)


class ThreeDxmlIO(AbstractTrimeshIO):
    name = '3dxml'
    extensions = ('.3dxml',)


class ThreeMfIO(AbstractTrimeshIO):
    name = '3mf'
    extensions = ('.3mf',)


class DrawingExchangeIO(AbstractTrimeshIO):
    name = 'dxf'
    extensions = ('.dxf',)


class ObjectIO(AbstractTrimeshIO):
    name = 'off'
    extensions = ('.off',)


class FreesurferIO(IOProtocol):
    """
    Mesh IO protocol for freesurfer surface files.
    """

    name = 'fs'
    extensions = ('.surf', '.srf', '.fs')

    def __init__(self):
        try:
            import surfa
        except ImportError:
            raise ImportError('the surfa python package must be installed for '
                              'freesurfer surface IO')
        self.surfa = surfa

    def load(self, filename: os.PathLike) -> vx.Mesh:
        """
        Read mesh from a freesurfer surface file.

        Args:
            filename (PathLike): The path to the file to read.

        Returns:
            Mesh: The loaded mesh.
        """
        smesh = self.surfa.load_mesh(filename).convert(space='world')
        return vx.Mesh(torch.as_tensor(smesh.vertices), torch.as_tensor(smesh.faces))

    def save(self, mesh: vx.Mesh, filename: os.PathLike) -> None:
        """
        Write mesh to a freesurfer surface file.

        Args:
            mesh (Mesh): The mesh to save.
            filename (PathLike): Output path.
        """
        mesh = mesh.cpu()
        smesh = self.surfa.Mesh(mesh.vertices.detach().numpy(),
                                mesh.faces.detach().numpy(), space='world')
        smesh.save(filename)


mesh_io_protocols = [
    WavefrontIO,
    StanfordPolygonIO,
    GltfIO,
    Stl3D,
    ThreeDxmlIO,
    ThreeMfIO,
    DrawingExchangeIO,
    FreesurferIO,
]
