"""
Spatial transforms family
"""

from __future__ import annotations

from .affine import AffineMatrix
from .warp import VectorField, VectorFieldPair, Warp


class TransformSeries:
    """
    An ordered list of transforms (affine matrices, displacement fields or pairs, warps),
    listed in the order they are applied to a volume. Affines are world-space transforms.
    A volume is transformed step by step: affines move its geometry, deformations
    resample it (see `Volume.transform`), so no single composite is ever formed.
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

    def condense(self):
        """
        Merge every run of neighboring affines into a single matrix, leaving deformations
        as they are.

        Returns:
            The bare transform if a single one remains, otherwise a new TransformSeries.
        """
        condensed = []
        for transform in self.transforms:
            if isinstance(transform, AffineMatrix) and condensed and isinstance(condensed[-1], AffineMatrix):
                condensed[-1] = transform @ condensed[-1]
            else:
                condensed.append(transform)
        return condensed[0] if len(condensed) == 1 else TransformSeries(condensed)

    def to(self, device) -> TransformSeries:
        """
        Move every transform of the series to a device.
        """
        return TransformSeries([transform.to(device) for transform in self.transforms])

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
