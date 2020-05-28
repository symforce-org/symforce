# mypy: disallow-untyped-defs

import numpy as np

from symforce import sympy as sm
from symforce import types as T

from .base import LieGroup
from .complex import Complex
from .matrix import Matrix


class Rot2(LieGroup):
    """
    Group of two-dimensional orthogonal matrices with determinant +1, representing rotations
    in 2D space. Backed by a complex number.
    """

    TANGENT_DIM = 1
    MATRIX_DIMS = (2, 2)
    STORAGE_DIM = 2
    STORAGE_TYPE = Complex

    def __init__(self, z=None):
        # type: (Complex) -> None
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

    def __repr__(self):
        # type: () -> str
        return "<Rot2 {}>".format(repr(self.z))

    def to_storage(self):
        # type: () -> T.List[T.Scalar]
        return self.z.to_storage()

    @classmethod
    def from_storage(cls, vec):
        # type: (T.List[T.Scalar]) -> Rot2
        return cls(Complex.from_storage(vec))

    @classmethod
    def symbolic(cls, name, **kwargs):
        # type: (str, T.Any) -> Rot2
        return cls(Complex.symbolic(name, **kwargs))

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls):
        # type: () -> Rot2
        return cls(Complex.identity())

    def compose(self, other):
        # type: (Rot2) -> Rot2
        return self.__class__(self.z * other.z)

    def inverse(self):
        # type: () -> Rot2
        return self.__class__(self.z.inverse())

    # -------------------------------------------------------------------------
    # Lie group implementation
    # -------------------------------------------------------------------------
    @classmethod
    def expmap(cls, v, epsilon=0):
        # type: (T.List[T.Scalar], T.Scalar) -> Rot2
        assert len(v) == 1
        theta = v[0]
        return Rot2(Complex(sm.cos(theta), sm.sin(theta)))

    def logmap(self, epsilon=0):
        # type: (T.Scalar) -> T.List[T.Scalar]
        return [sm.atan2_safe(self.z.imag, self.z.real, epsilon=epsilon)]

    @classmethod
    def hat(cls, vec):
        # type: (T.List[T.Scalar]) -> T.List[T.List[T.Scalar]]
        assert len(vec) == 1
        theta = vec[0]
        return [[0, -theta], [theta, 0]]

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @T.overload
    def __mul__(self, right):  # pragma: no cover
        # type: (Matrix) -> Matrix
        pass

    @T.overload
    def __mul__(self, right):  # pragma: no cover
        # type: (Rot2) -> Rot2
        pass

    def __mul__(self, right):
        # type: (T.Union[Rot2, Matrix]) -> T.Union[Rot2, Matrix]
        """
        Left-multiplication. Either rotation concatenation or point transform.

        Args:
            right (Rot2 or Matrix):

        Returns:
            Rot2 or Matrix:
        """
        if isinstance(right, sm.Matrix):
            assert right.shape == (2, 1), right.shape
            return self.to_rotation_matrix() * right
        elif isinstance(right, Rot2):
            return self.compose(right)
        else:
            raise NotImplementedError('Unsupported type: "{}"'.format(type(right)))

    def to_rotation_matrix(self):
        # type: () -> Matrix
        """
        A matrix representation of this element in the Euclidean space that contains it.

        Returns:
            Matrix: Matrix of shape given by self.MATRIX_DIMS
        """
        return Matrix([[self.z.real, -self.z.imag], [self.z.imag, self.z.real]])

    @classmethod
    def random(cls):
        # type: () -> Rot2
        """
        Generate a random element of SO3.
        """
        return Rot2(Complex.unit_random())

    @classmethod
    def random_from_uniform_sample(cls, u1, pi=sm.pi):
        # type: (T.Scalar, T.Scalar) -> Rot2
        """
        Generate a random element of SO2 from a variable uniformly sampled on [0, 1].
        """
        return Rot2(Complex.unit_random_from_uniform_sample(u1, pi))
