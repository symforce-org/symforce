# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

import symforce.internal.symbolic as sf
from symforce import typing as T
from symforce.ops.interfaces import LieGroup

from .matrix import Matrix23
from .matrix import Matrix32
from .matrix import Matrix33
from .matrix import Vector2
from .matrix import Vector3


class Unit3(LieGroup):
    """
    Direction in R^3 represented as a unit vector on the S^2 sphere manifold.

    Storage is three dimensional, and tangent space is two dimensional. Due to the nature of the
    manifold, the unit X vector is handled as a singularity.

    The implementation of the retract and local_coordinates functions are based on Appendix B.2 :

    [Hertzberg 2013] Integrating Generic Sensor Fusion Algorithms with Sound State Representations
    through Encapsulation of Manifolds

    The retract operation performs a perturbation to the desired unit X vector, which is then rotated to
    desired position along the actual stored unit vector through a Householder-reflection + relection
    across the XZ plane.

        x.retract(delta) = x [+] delta = Rx * Exp(delta), where
        Exp(delta) = [cos(||delta||), sinc(||delta||) * delta], and
        Rx = (I - 2 vv^T / (v^Tv))X, v = x - e_x != 0, X is a matrix negating 2nd vector component
           = diag(1, -1, -1)       , x = e_x

    See: `unit3_visualization.ipynb` for a visualization of the Unit3 manifold.
    """

    E_X = Vector3.unit_x()
    E_Z = Vector3.unit_z()
    FLIP_Y = Matrix33.diag([1, -1, 1])

    def __init__(self, x: Vector3) -> None:
        """
        Construct from a normalized :class:`Vector3 <symforce.geo.matrix.Vector3>`.
        """
        self.x = x
        assert isinstance(self.x, Vector3)

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return "<Unit3 xyz=[{}, {}, {}]>".format(repr(self.x[0]), repr(self.x[1]), repr(self.x[2]))

    @classmethod
    def storage_dim(cls) -> int:
        return Vector3.storage_dim()

    def to_storage(self) -> T.List[T.Scalar]:
        return self.x.to_storage()

    @classmethod
    def from_storage(cls, vec: T.Sequence[T.Scalar]) -> Unit3:
        return cls(Vector3.from_storage(vec))

    @classmethod
    def symbolic(cls, name: str, **kwargs: T.Any) -> Unit3:
        return cls(Vector3.symbolic(name, **kwargs))

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls) -> Unit3:
        raise NotImplementedError("Unit3 does not have an identity element")

    def compose(self, other: Unit3) -> Unit3:
        raise NotImplementedError("Unit3 does not have a composition operation")

    def inverse(self) -> Unit3:
        raise NotImplementedError("Unit3 does not have an inverse operation")

    # -------------------------------------------------------------------------
    # Lie group implementation
    # -------------------------------------------------------------------------

    @classmethod
    def tangent_dim(cls) -> int:
        return 2

    @classmethod
    def from_tangent(cls, v: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()) -> Unit3:
        raise NotImplementedError()

    def to_tangent(self, epsilon: T.Scalar = sf.epsilon()) -> T.List[T.Scalar]:
        raise NotImplementedError()

    def storage_D_tangent(self, epsilon: T.Scalar = sf.epsilon()) -> Matrix32:
        return Matrix32(self._Rx(epsilon=epsilon)[:, 1:])

    def tangent_D_storage(self, epsilon: sf.Scalar = sf.epsilon()) -> Matrix23:
        return Matrix23(self.storage_D_tangent(epsilon=epsilon).T)

    # -------------------------------------------------------------------------
    # Tangent space methods
    # -------------------------------------------------------------------------

    def _Rx(self, epsilon: T.Scalar = sf.epsilon()) -> Matrix33:
        v = self.x - self.E_X

        # Calculate standard rotation matrix, that will be used in most
        # locations across the spherical manifold.
        v_safe = v + epsilon * sf.sign_no_zero(v[2]) * Vector3.unit_z()
        Rx_0 = (
            Matrix33.eye() - 2 * v_safe * v_safe.transpose() / v_safe.squared_norm()
        ) * self.FLIP_Y

        # If the vector is close to the singularity, we need to use a slightly different calculation.
        # This calculation is specifically meant to address situations where the internal storage
        # is :
        #  - Closely aligned to the singularity direction.
        #  - Slightly off of the spherical manifold. (i.e. storage is not a perfect unit vector)
        v_singularity = Vector3(0, v_safe[1], v_safe[2])
        Rx_1 = (
            Matrix33.eye()
            - 2 * v_singularity * v_singularity.transpose() / v_singularity.squared_norm()
        ) * self.FLIP_Y

        # We are close to the singularity if the angle between self.x and E_X is close to 0.
        # is_close = 0 -- if self.x is pointing in the positive X direction, and the distance
        #                 from the X axis is small.
        # is_close = 1 -- if self.x is pointing at all in the negative X direction, or if the
        #                 distance from the X axis is larger than 10 * epsilon.
        is_close = sf.less(self.x[1:].squared_norm(), 10 * epsilon * sf.sign_no_zero(self.x[0]))

        # Use Rx_0 for most Unit3 directions. Use Rx_1 if we are close to the singularity.
        return (1 - is_close) * Rx_0 + is_close * Rx_1

    @staticmethod
    def _exp(delta: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()) -> Vector3:
        delta_vector = Vector2.from_storage(delta)

        def sinc_safe(x: sf.Scalar, epsilon: sf.Scalar) -> sf.Scalar:
            x_safe = x + epsilon * sf.sign_no_zero(x)
            return sf.sin(x_safe) / x_safe

        norm = delta_vector.norm()
        return Vector3(
            sf.cos(norm),
            sinc_safe(norm, epsilon) * delta_vector[0],
            sinc_safe(norm, epsilon) * delta_vector[1],
        )

    @staticmethod
    def _logarithm(vector: Vector3, epsilon: T.Scalar = sf.epsilon()) -> Vector2:
        w = vector[0]
        w_safe = sf.clamp(w, -1 + epsilon, 1 - epsilon)
        v = vector[1:]
        return Vector2(sf.acos(w_safe) / sf.sqrt(1 - w_safe**2) * v)

    def retract(self, delta: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()) -> Unit3:
        return self.from_unit_vector(Vector3(self._Rx(epsilon) * self._exp(delta, epsilon)))

    def local_coordinates(
        self, vector: Unit3, epsilon: T.Scalar = sf.epsilon()
    ) -> T.List[T.Scalar]:
        return self._logarithm(
            Vector3(self._Rx(epsilon).transpose() * vector.to_unit_vector()), epsilon
        ).to_flat_list()

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def to_unit_vector(self) -> Vector3:
        """
        Returns a :class:`Vector3` version of the unit direction.
        """
        return self.x

    def basis(self, epsilon: T.Scalar = sf.epsilon()) -> Matrix32:
        """
        Returns a :class:`Matrix32` with the basis vectors of the tangent space (in R^3) at the
        current Unit3 direction.
        """
        return self.storage_D_tangent(epsilon=epsilon)

    @classmethod
    def from_vector(cls, a: Vector3, epsilon: T.Scalar = sf.epsilon()) -> Unit3:
        """
        Return a :class:`Unit3` that points along the direction of vector ``a``

        ``a`` will be normalized.
        """
        return cls(a.normalized(epsilon=epsilon))

    @classmethod
    def from_unit_vector(cls, a: Vector3) -> Unit3:
        """
        Return a :class:`Unit3` that points along the direction of vector ``a``

        ``a`` is expected to be a unit vector.
        """
        return cls(a)

    @classmethod
    def random(cls, epsilon: T.Scalar = sf.epsilon()) -> Unit3:
        """
        Generate a random :class:`Unit3` direction.
        """
        u1, u2 = np.random.uniform(low=0.0, high=1.0, size=(2,))
        return cls.random_from_uniform_samples(u1, u2, epsilon=epsilon)

    @classmethod
    def random_from_uniform_samples(
        cls, u1: T.Scalar, u2: T.Scalar, epsilon: T.Scalar = sf.epsilon()
    ) -> Unit3:
        """
        Generate a random :class:`Unit3` direction from two variables uniformly sampled in [0, 1].
        """
        theta = 2 * sf.pi * u1
        phi = sf.acos(2 * u2 - 1)
        return cls.from_vector(
            Vector3([sf.sin(phi) * sf.cos(theta), sf.sin(phi) * sf.sin(theta), sf.cos(phi)]),
            epsilon=epsilon,
        )
