import functools
from symforce import sympy as sm

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
        return "<Rot2 {}>".format(repr(self.z))

    def to_storage(self):
        return self.z.to_storage()

    @classmethod
    def from_storage(cls, vec):
        return cls(Complex.from_storage(vec))

    @classmethod
    def symbolic(cls, name, **kwargs):
        return cls(Complex.symbolic(name, **kwargs))

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls):
        return cls(Complex.identity())

    def compose(self, other):
        return self.__class__(self.z * other.z)

    def inverse(self):
        return self.__class__(self.z.inverse())

    # -------------------------------------------------------------------------
    # Lie group implementation
    # -------------------------------------------------------------------------
    @classmethod
    def expmap(cls, v, epsilon=0):
        assert len(v) == 1
        theta = v[0]
        return Rot2(Complex(sm.cos(theta), sm.sin(theta)))

    def logmap(self, epsilon=0):
        return [sm.atan2_safe(self.z.imag, self.z.real, epsilon=epsilon)]

    @classmethod
    def hat(cls, vec):
        assert len(vec) == 1
        theta = vec[0]
        return [[0, -theta], [theta, 0]]

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def __mul__(self, right):
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
        """
        A matrix representation of this element in the Euclidean space that contains it.

        Returns:
            Matrix: Matrix of shape given by self.MATRIX_DIMS
        """
        return Matrix([[self.z.real, -self.z.imag], [self.z.imag, self.z.real]])
