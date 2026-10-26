"""
Spatial transforms family
"""

from __future__ import annotations

import os
import voxel as vx

from .affine import AffineMatrix
from .warp import VectorField, VectorFieldPair, Warp, compose_vector_fields


class TransformSeries:
    """
    An ordered list of transforms (affine matrices, displacement fields or pairs, warps),
    listed in the order they are applied to a volume. Affines are world-space transforms.
    A volume applies the condensed series, at most one affine moving its geometry and
    one deformation resampling it (see `Volume.transform`), so the data is interpolated
    once however many steps there are.
    """

    def __init__(self, transforms: list[AnyTransform]) -> None:
        self.transforms = list(transforms)

    def __len__(self) -> int:
        return len(self.transforms)

    def __iter__(self):
        return iter(self.transforms)

    def __getitem__(self, index):
        return self.transforms[index]

    def __repr__(self) -> str:
        return f'TransformSeries({[type(t).__name__ for t in self.transforms]})'

    def condense(self) -> AnyTransform:
        """
        Reduce the series to its normal form: one affine, one deformation, or an affine
        followed by a deformation. Affines multiply, a deformation followed by an affine
        is carried along by it, and deformations compose onto the grid of the later one.
        The deformation stays a pair when every one is a pair, so the result inverts, and
        is a warp when the last one is a warp, so the result stays pinned to its grid.

        Returns:
            The bare transform, a two-step TransformSeries of affine then deformation, or
            the identity affine for an empty series.
        """
        affine, dense = None, None
        for transform in self.transforms:
            if isinstance(transform, AffineMatrix):
                affine = transform if affine is None else transform @ affine
                dense = None if dense is None else _carried(dense, transform)
            else:
                dense = transform if dense is None else _composed(dense, transform)
        if dense is None:
            return AffineMatrix() if affine is None else affine
        return dense if affine is None else TransformSeries([affine, dense])

    def to(self, device) -> TransformSeries:
        """
        Move every transform of the series to a device.
        """
        return TransformSeries([transform.to(device) for transform in self.transforms])

    def save(self, filename: os.PathLike, fmt: str | None = None) -> None:
        """
        Save the series, step by step, to a file.

        Args:
            filename (PathLike): The path to the file to save.
            fmt (str, optional): The format of the file. If None, the format is
                determined by the file extension.
        """
        vx.save_transform(self, filename, fmt=fmt)

    def inverse(self) -> TransformSeries:
        """
        The series undoing this one: each transform inverted, in reverse order. Affines and
        field pairs invert; a lone displacement field or warp cannot.
        """
        inverted = []
        for transform in reversed(self.transforms):
            if isinstance(transform, (AffineMatrix, VectorFieldPair)):
                inverted.append(transform.inverse())
            else:
                raise ValueError(f'cannot invert a series holding a lone {type(transform).__name__}')
        return TransformSeries(inverted)


# anything a volume can be transformed by (see Volume.transform)
AnyTransform = AffineMatrix | VectorField | VectorFieldPair | Warp | TransformSeries


def _carried(dense: VectorField | VectorFieldPair | Warp, affine: AffineMatrix):
    """
    A deformation moved along with the anatomy it deforms by an affine.
    """
    if isinstance(dense, VectorFieldPair):
        return VectorFieldPair(dense.forward.transform(affine), dense.reverse.transform(affine))
    if isinstance(dense, Warp):
        return dense.as_displacement_field().transform(affine).as_warp()
    return dense.transform(affine)


def _composed(first: VectorField | VectorFieldPair | Warp, second: VectorField | VectorFieldPair | Warp):
    """
    The deformation of applying `first` then `second`, on the grid of `second`.
    """
    if isinstance(first, VectorFieldPair) and isinstance(second, VectorFieldPair):
        return VectorFieldPair(compose_vector_fields(first.forward, second.forward),
                               compose_vector_fields(second.reverse, first.reverse))

    def field(dense):
        if isinstance(dense, VectorFieldPair):
            return dense.forward
        return dense.as_displacement_field() if isinstance(dense, Warp) else dense

    composed = compose_vector_fields(field(first), field(second))
    return composed.as_warp() if isinstance(second, Warp) else composed
