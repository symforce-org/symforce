from symforce import sympy as sm
from symforce.ops import StorageOps

from .base import LieGroup

import numpy as np


class Matrix(sm.Matrix, LieGroup):
    """
    Matrix type that inherits from the Sympy Matrix class.

    References:

        https://docs.sympy.org/latest/tutorial/matrices.html
        https://en.wikipedia.org/wiki/Vector_space

    It is also treated as lie group that represents the linear space of two dimensional matrices
    under the *addition* operation. This causes some confusion between the naming of methods, such
    as `.identity()` and `.inverse()`. The linear algebra equivalents are available at
    `.matrix_identity()` and `.matrix_inverse()`. Splitting the matrix type and the lie group ops
    is a possible action to make this better.

    NOTE(hayk): Unlike the other group types, this inherits from the storage type (sympy.Matrix)
    rather than keeping it an attribute. The reason is to be compatible with general sympy
    operations and to avoid a huge amount of wrapping. Doing this means that the type is of
    dynamic dimensions rather than fixed size. For that reason, the interface is somewhat abused
    in that things that typically only require the type require the instance.
    """

    @property
    def MATRIX_DIMS(self):
        return self.shape

    @property
    def TANGENT_DIM(self):
        return self.MATRIX_DIMS[0] * self.MATRIX_DIMS[1]

    @property
    def STORAGE_DIM(self):
        return self.TANGENT_DIM

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    def from_storage(self, vec):
        return self.__class__(vec)

    def to_storage(self):
        return self.to_tangent()

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    def identity(self):
        return self.__class__.zeros(*self.MATRIX_DIMS)

    def compose(self, other):
        return self + other

    def inverse(self):
        return -self

    # -------------------------------------------------------------------------
    # Lie group concept - see symforce.ops.lie_group_ops
    # -------------------------------------------------------------------------

    def from_tangent(self, vec, epsilon=0):
        if isinstance(vec, (list, tuple)):
            vec = self.__class__(vec)
        return self.__class__(vec.reshape(*self.MATRIX_DIMS).tolist())

    def to_tangent(self, epsilon=0):
        return list(self.reshape(self.TANGENT_DIM, 1))

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def to_matrix(self):
        """
        A matrix representation of this element in the Euclidean space that contains it.

        Returns:
            (Matrix) Matrix of shape given by self.MATRIX_DIMS
        """
        return self

    def zero(self):
        """
        Matrix of zeros.

        Returns:
            Matrix:
        """
        return self.identity()

    @classmethod
    def zeros(cls, rows, cols):  # pylint: disable=signature-differs
        """
        Matrix of zeros.

        Args:
            rows (int):
            cols (int):

        Returns:
            Matrix:
        """
        return cls([[sm.S.Zero] * cols for _ in range(rows)])

    def one(self):
        """
        Matrix of ones.

        Returns:
            Matrix:
        """
        return self.__class__.ones(*self.MATRIX_DIMS)

    @classmethod
    def ones(cls, rows, cols):  # pylint: disable=signature-differs
        """
        Matrix of ones.

        Args:
            rows (int):
            cols (int):

        Returns:
            Matrix:
        """
        return cls([[sm.S.One] * cols for _ in range(rows)])

    @classmethod
    def diag(cls, diagonal):  # pylint: disable=arguments-differ
        """
        Construct a square matrix from the diagonal.

        Args:
            diagonal (Matrix): Diagonal vector

        Returns:
            Matrix:
        """
        mat = cls.zeros(len(diagonal), len(diagonal))
        for i in range(len(diagonal)):
            mat[i, i] = diagonal[i]
        return mat

    @classmethod
    def eye(cls, rows, cols=None):  # pylint: disable=arguments-differ
        """
        Construct an identity matrix of the given dimensions

        Args:
            rows (int):
            cols (int):  constructs a rows x rows square matrix if cols in None

        Returns:
            Matrix:
        """
        if cols is None:
            cols = rows
        mat = cls.zeros(rows, cols)
        for i in range(min(rows, cols)):
            mat[i, i] = sm.S.One
        return mat

    def matrix_identity(self):
        """
        Identity matrix - ones on the diagonal, rest zeros.

        Returns:
            Matrix:
        """
        return self.eye(*self.MATRIX_DIMS)

    def matrix_inverse(self):
        """
        Inverse of the matrix.

        Returns:
            Matrix:
        """
        return self.inv()

    # pylint: disable=no-member
    def symbolic(self, name, **kwargs):
        """
        Create with symbols.

        Args:
            name (str): Name prefix of the symbols
            **kwargs (dict): Forwarded to `sm.Symbol`

        Returns:
            Matrix:
        """
        rows, cols = self.MATRIX_DIMS  # pylint: disable=unpacking-non-sequence

        row_names = [str(r_i) for r_i in range(rows)]
        col_names = [str(c_i) for c_i in range(cols)]

        assert len(row_names) == rows
        assert len(col_names) == cols

        if cols == 1:
            symbols = []
            for r_i in range(rows):
                _name = "{}{}".format(name, row_names[r_i])
                symbols.append(sm.Symbol(_name, **kwargs))
        else:
            symbols = []
            for r_i in range(rows):
                col_symbols = []
                for c_i in range(cols):
                    _name = "{}{}_{}".format(name, row_names[r_i], col_names[c_i])
                    col_symbols.append(sm.Symbol(_name, **kwargs))
                symbols.append(col_symbols)

        return self.__class__(sm.Matrix(symbols))

    def simplify(self, *args, **kwargs):
        """
        Simplify this expression.

        This overrides the sympy implementation because that clobbers the class type.

        Returns:
            Matrix:
        """
        return self.__class__(sm.simplify(self, *args, **kwargs))

    def squared_norm(self):
        """
        Squared norm of a vector, equivalent to the dot product with itself.

        Returns:
            Scalar:
        """
        self._assert_is_vector()
        return self.dot(self)

    def norm(self, epsilon=0):
        """
        Norm of a vector (square root of magnitude).

        Args:
            epsilon (Scalar): Small number to avoid numerical error

        Returns:
            Scalar:
        """
        return sm.sqrt(self.squared_norm() + epsilon)

    def normalized(self, epsilon=0):  # pylint: disable=arguments-differ
        """
        Returns a unit vector in this direction (divide by norm).

        Args:
            epsilon (Scalar): Small number to avoid numerical error

        Returns:
            Matrix:
        """
        return self / self.norm(epsilon=epsilon)

    def __add__(self, right):
        """
        Add a scalar to a matrix.

        Args:
            right (Scalar):

        Returns:
            Matrix:
        """
        if StorageOps.scalar_like(right):
            return self.applyfunc(lambda x: x + right)

        return sm.Matrix.__add__(self, right)

    def __div__(self, right):
        """
        Divide a matrix by a scalar.

        Args:
            right: (Scalar):

        Returns:
            Matrix:
        """
        if StorageOps.scalar_like(right):
            return self.applyfunc(lambda x: x / right)

        return sm.Matrix.__div__(self, right)

    @staticmethod
    def are_parallel(a, b, epsilon):
        """
        Returns 1 if a and b are parallel within epsilon, and 0 otherwise.

        Args:
            a (Vector):
            b (Vector):
            epsilon (Scalar):

        Returns:
            Scalar: acts like a boolean 1/0
        """
        return (1 - sm.sign(a.cross(b).norm() - epsilon)) / 2

    def evalf(self, real=True):
        """
        Perform numerical evaluation of each element in the matrix.

        Args:
            real (bool): If True, assume no complex part.

        Returns:
            (self.__class__):
        """
        return self.__class__(
            [[self[i, j].evalf(real=real) for j in range(self.cols)] for i in range(self.rows)]
        )

    def to_numpy(self, scalar_type=np.float64):
        """
        Returns:
            np.ndarray:
        """
        return np.array(self.evalf().tolist(), dtype=scalar_type)

    @classmethod
    def column_stack(cls, *columns):
        """Take a sequence of 1-D vectors and stack them as columns to make a single 2-D Matrix.

        Args:
            columns (tuple(Matrix)): 1-D vectors

        Returns:
            Matrix:
        """
        if not columns:
            return cls()

        assert all([col.TANGENT_DIM == columns[0].TANGENT_DIM for col in columns])
        for col in columns:
            # assert that each column is a vector
            assert sum([dim > 1 for dim in col.shape]) <= 1
        cols_as_rows = [col.reshape(1, col.TANGENT_DIM) for col in columns]
        return cls([row for row in cols_as_rows]).T

    def _assert_is_vector(self):
        assert (self.shape[0] == 1) or (self.shape[1] == 1), "squared_norm() is only for vectors."


# -------------------------------------------------------------------------
# Typedefs and helpful constructors
# -------------------------------------------------------------------------


def vector_constructor(dim, args):
    """
    Construction helper for a vector with the given expected dimension.

    Args:
        dim: (int) Expected dimension
        args: (iterable) Args to forward

    Returns:
        (Matrix)
    """
    if len(args) == 0:
        return Matrix.zeros(rows=dim, cols=1)
    elif len(args) == dim:
        return Matrix(args)
    elif len(args) == 1 and isinstance(args[0], (list, tuple)) and len(args[0]) == dim:
        return Matrix(args[0])
    else:
        raise ArithmeticError(
            'Trying to construct vector of length {} with "{}".'.format(dim, args)
        )


# Vector constructor helper
Vector1 = V1 = lambda *args: vector_constructor(dim=1, args=args)
Vector2 = V2 = lambda *args: vector_constructor(dim=2, args=args)
Vector3 = V3 = lambda *args: vector_constructor(dim=3, args=args)
Vector4 = V4 = lambda *args: vector_constructor(dim=4, args=args)
Vector5 = V5 = lambda *args: vector_constructor(dim=5, args=args)
Vector6 = V6 = lambda *args: vector_constructor(dim=6, args=args)
Vector7 = V7 = lambda *args: vector_constructor(dim=7, args=args)
Vector8 = V8 = lambda *args: vector_constructor(dim=8, args=args)
Vector9 = V9 = lambda *args: vector_constructor(dim=9, args=args)

# Zero vector constructors
Z1 = Z11 = lambda: Matrix.zeros(rows=1, cols=1)
Z2 = Z22 = lambda: Matrix.zeros(rows=2, cols=1)
Z3 = Z33 = lambda: Matrix.zeros(rows=3, cols=1)
Z4 = Z44 = lambda: Matrix.zeros(rows=4, cols=1)
Z5 = Z55 = lambda: Matrix.zeros(rows=5, cols=1)
Z6 = Z66 = lambda: Matrix.zeros(rows=6, cols=1)
Z7 = Z77 = lambda: Matrix.zeros(rows=7, cols=1)
Z8 = Z88 = lambda: Matrix.zeros(rows=8, cols=1)
Z9 = Z99 = lambda: Matrix.zeros(rows=9, cols=1)

# Identity matrix short names
I1 = I11 = lambda: Matrix.eye(rows=1)
I2 = I22 = lambda: Matrix.eye(rows=2)
I3 = I33 = lambda: Matrix.eye(rows=3)
I4 = I44 = lambda: Matrix.eye(rows=4)
I5 = I55 = lambda: Matrix.eye(rows=5)
I6 = I66 = lambda: Matrix.eye(rows=6)
I7 = I77 = lambda: Matrix.eye(rows=7)
I8 = I88 = lambda: Matrix.eye(rows=8)
I9 = I99 = lambda: Matrix.eye(rows=9)
