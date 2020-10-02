import numpy as np

from symforce.ops.interfaces.lie_group import LieGroup
from symforce import sympy as sm
from symforce import types as T

from .complex import Complex
from .matrix import Matrix
from .matrix import Matrix12
from .matrix import Matrix22
from .matrix import Matrix21


class Rot2(LieGroup):
    """
    Group of two-dimensional orthogonal matrices with determinant +1, representing rotations
    in 2D space. Backed by a complex number.
    """

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

    @classmethod
    def storage_dim(cls):
        # type: () -> int
        return 2

    def to_storage(self):
        # type: () -> T.List[T.Scalar]
        return self.z.to_storage()

    @classmethod
    def from_storage(cls, vec):
        # type: (T.Sequence[T.Scalar]) -> Rot2
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
    def tangent_dim(cls):
        # type: () -> int
        return 1

    @classmethod
    def from_tangent(cls, v, epsilon=0):
        # type: (T.Sequence[T.Scalar], T.Scalar) -> Rot2
        assert len(v) == 1
        theta = v[0]
        return Rot2(Complex(sm.cos(theta), sm.sin(theta)))

    def to_tangent(self, epsilon=0):
        # type: (T.Scalar) -> T.List[T.Scalar]
        return [sm.atan2(self.z.imag, self.z.real)]

    @classmethod
    def hat(cls, vec):
        # type: (T.Sequence[T.Scalar]) -> Matrix22
        assert len(vec) == 1
        theta = vec[0]
        return Matrix22([[0, -theta], [theta, 0]])

    def storage_D_tangent(self):
        # type: () -> Matrix21
        """
        Note: generated from symforce/notebooks/storage_D_tangent.ipynb
        """
        return Matrix21([[-self.z.imag], [self.z.real]])

    def tangent_D_storage(self):
        # type: () -> Matrix12
        """
        Note: generated from symforce/notebooks/tangent_D_storage.ipynb
        """
        return T.cast(Matrix12, self.storage_D_tangent().T)

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @T.overload
    def __mul__(self, right):  # pragma: no cover
        # type: (Matrix21) -> Matrix21
        pass

    @T.overload
    def __mul__(self, right):  # pragma: no cover
        # type: (Rot2) -> Rot2
        pass

    @T.overload
    def __mul__(self, right):  # pragma: no cover
        # type: (sm.Matrix) -> sm.Matrix
        pass

    def __mul__(self, right):
        # type: (T.Union[Rot2, Matrix21]) -> T.Union[Rot2, Matrix21]
        """
        Left-multiplication. Either rotation concatenation or point transform.
        """
        if isinstance(right, (sm.Matrix, Matrix)):
            assert right.shape == (2, 1), right.shape
            return T.cast(Matrix21, self.to_rotation_matrix() * right)
        elif isinstance(right, Rot2):
            return self.compose(right)
        else:
            raise NotImplementedError('Unsupported type: "{}"'.format(type(right)))

    def to_rotation_matrix(self):
        # type: () -> Matrix22
        """
        A matrix representation of this element in the Euclidean space that contains it.
        """
        return Matrix22([[self.z.real, -self.z.imag], [self.z.imag, self.z.real]])

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
