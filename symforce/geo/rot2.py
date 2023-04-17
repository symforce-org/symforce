# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import symforce.internal.symbolic as sf
from symforce import typing as T
from symforce.ops.interfaces.lie_group import LieGroup

from .complex import Complex
from .matrix import Matrix12
from .matrix import Matrix21
from .matrix import Matrix22
from .matrix import Vector2


class Rot2(LieGroup):
    """
    Group of two-dimensional orthogonal matrices with determinant +1, representing rotations
    in 2D space. Backed by a complex number.
    """

    def __init__(self, z: Complex = None) -> None:
        """
        Construct from a unit complex number, or identity if none provided.

        Args:
            z (Complex):
        """
        self.z = z if z is not None else Complex.identity()
        assert isinstance(self.z, Complex)

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return "<Rot2 {}>".format(repr(self.z))

    @classmethod
    def storage_dim(cls) -> int:
        return 2

    def to_storage(self) -> T.List[T.Scalar]:
        return self.z.to_storage()

    @classmethod
    def from_storage(cls, vec: T.Sequence[T.Scalar]) -> Rot2:
        return cls(Complex.from_storage(vec))

    @classmethod
    def symbolic(cls, name: str, **kwargs: T.Any) -> Rot2:
        return cls(Complex.symbolic(name, **kwargs))

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls) -> Rot2:
        return cls(Complex.identity())

    def compose(self, other: Rot2) -> Rot2:
        return self.__class__(self.z * other.z)

    def inverse(self) -> Rot2:
        # In general, the inverse of a complex number z is z.conj()/|z|^2. But since a Rot2
        # is represented by a unit complex number with |z| = 1, the inverse is just z.conj()
        return self.__class__(self.z.conj())

    # -------------------------------------------------------------------------
    # Lie group implementation
    # -------------------------------------------------------------------------

    @classmethod
    def tangent_dim(cls) -> int:
        return 1

    @classmethod
    def from_tangent(cls, v: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()) -> Rot2:
        assert len(v) == 1
        theta = v[0]
        return Rot2(Complex(sf.cos(theta), sf.sin(theta)))

    def to_tangent(self, epsilon: T.Scalar = sf.epsilon()) -> T.List[T.Scalar]:
        return [sf.atan2(self.z.imag, self.z.real, epsilon=epsilon)]

    @classmethod
    def hat(cls, vec: T.Sequence[T.Scalar]) -> Matrix22:
        assert len(vec) == 1
        theta = vec[0]
        return Matrix22([[0, -theta], [theta, 0]])

    def storage_D_tangent(self) -> Matrix21:
        """
        Note: generated from symforce/notebooks/storage_D_tangent.ipynb
        """
        return Matrix21([[-self.z.imag], [self.z.real]])

    def tangent_D_storage(self) -> Matrix12:
        """
        Note: generated from symforce/notebooks/tangent_D_storage.ipynb
        """
        return T.cast(Matrix12, self.storage_D_tangent().T)

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @T.overload
    def __mul__(self, right: Vector2) -> Vector2:  # pragma: no cover
        pass

    @T.overload
    def __mul__(self, right: Rot2) -> Rot2:  # pragma: no cover
        pass

    def __mul__(self, right: T.Union[Rot2, Vector2]) -> T.Union[Rot2, Vector2]:
        """
        Left-multiplication. Either rotation concatenation or point transform.
        """
        if isinstance(right, Vector2):
            return T.cast(Vector2, self.to_rotation_matrix() * right)
        elif isinstance(right, Rot2):
            return self.compose(right)
        else:
            raise NotImplementedError('Unsupported type: "{}"'.format(type(right)))

    @classmethod
    def from_angle(cls, theta: T.Scalar) -> Rot2:
        """
        Create a Rot2 from an angle `theta` in radians

        This is equivalent to from_tangent([theta])
        """
        return cls.from_tangent([theta])

    def to_rotation_matrix(self) -> Matrix22:
        """
        A matrix representation of this element in the Euclidean space that contains it.
        """
        return Matrix22([[self.z.real, -self.z.imag], [self.z.imag, self.z.real]])

    @classmethod
    def random(cls) -> Rot2:
        """
        Generate a random element of SO3.
        """
        return Rot2(Complex.unit_random())

    @classmethod
    def random_from_uniform_sample(cls, u1: T.Scalar, pi: T.Scalar = sf.pi) -> Rot2:
        """
        Generate a random element of SO2 from a variable uniformly sampled on [0, 1].
        """
        return Rot2(Complex.unit_random_from_uniform_sample(u1, pi))
