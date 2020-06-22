import numpy as np

import symforce
from symforce import sympy as sm
from symforce import types as T
from symforce.ops import StorageOps
from symforce.python_util import classproperty

from .base import LieGroup


class Matrix(sm.Matrix, LieGroup):
    """
    Matrix type that inherits from the Sympy Matrix class. Care has been taken to allow this class
    to create fixed-size child classes like Matrix31. Anytime __new__ is called, the appropriate
    fixed size class is returned rather than the type of the arguments. The API is meant to parallel
    the way Eigen's C++ matrix classes work with dynamic and fixed sizes.

    References:

        https://docs.sympy.org/latest/tutorial/matrices.html
        https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html
        https://en.wikipedia.org/wiki/Vector_space

    It is also treated as lie group that represents the linear space of two dimensional matrices
    under the *addition* operation. This causes some confusion between the naming of methods, such
    as `.identity()` and `.inverse()`. The linear algebra equivalents are available at
    `.matrix_identity()` and `.matrix_inverse()`. Splitting the matrix type and the lie group ops
    is a possible action to make this better.
    """

    # Static dimensions of this type. (-1, -1) means there is no size information, like if
    # we are using geo.Matrix directly instead of geo.Matrix31.
    # Once a matrix is constructed it should be of a type where the .shape instance variable matches
    # this class variable as a strong internal consistency check.
    SHAPE = (-1, -1)

    @classproperty
    def STORAGE_DIM(cls):  # type: ignore
        return cls.SHAPE[0] * cls.SHAPE[1]

    TANGENT_DIM = STORAGE_DIM  # type: ignore

    def __new__(cls, *args, **kwargs):
        # type: (T.Any, T.Any) -> Matrix
        return cls._new(*args, **kwargs)

    @classmethod
    def _new(cls, *args, **kwargs):
        # type: (T.Any, T.Any) -> Matrix
        """
        Beast of a method for creating a Matrix. Handles a variety of construction use cases
        and *always* returns a fixed size child class of Matrix rather than Matrix itself. The
        available construction options depend on whether cls is a fixed size type or not.

        Generally modeled after the Eigen interface, but must also support internal use within
        sympy and symengine which we cannot change.

        Examples:
            1) Matrix32()  # Zero constructed Matrix32
            2) Matrix(sm.Matrix([[1, 2], [3, 4]]))  # Matrix22 with [1, 2, 3, 4] data
            3A) Matrix([[1, 2], [3, 4]])  # Matrix22 with [1, 2, 3, 4] data
            3B) Matrix22([1, 2, 3, 4])  # Matrix22 with [1, 2, 3, 4] data (must matched fixed shape)
            3C) Matrix([1, 2, 3, 4])  # Matrix41 with [1, 2, 3, 4] data - column vector assumed
            4) Matrix(4, 3)  # Zero constructed Matrix43
            5) Matrix(2, 2, [1, 2, 3, 4])  # Matrix22 with [1, 2, 3, 4] data (first two are shape)
            6) Matrix(2, 2, lambda row, col: row + col)  # Matrix22 with [0, 1, 1, 2] data
            7) Matrix22(1, 2, 3, 4)  # Matrix22 with [1, 2, 3, 4] data (must match fixed length)
        """

        # 1) Default construction allowed for fixed size.
        if len(args) == 0:
            assert cls._is_fixed_size(), "Cannot default construct non-fixed matrix."
            return cls.zero()

        # 2) Construct with another Matrix - this is easy
        elif len(args) == 1 and hasattr(args[0], "is_Matrix") and args[0].is_Matrix:
            rows, cols = args[0].shape
            flat_list = list(args[0])

        # 3) If there's one argument and it's an array, works for fixed or dynamic size.
        elif len(args) == 1 and isinstance(args[0], (T.Sequence, np.ndarray)):
            array = args[0]
            # 2D array, shape is known
            if isinstance(array[0], (T.Sequence, np.ndarray)):
                rows, cols = len(array), len(array[0])
                assert all(len(arr) == cols for arr in array), "Inconsistent columns: {}".format(
                    args
                )
                flat_list = [v for row in array for v in row]

            # 1D array - if fixed size this must match data length. If not, assume column vec.
            else:
                if cls._is_fixed_size():
                    assert len(array) == cls.STORAGE_DIM, "Gave args {} for {}".format(args, cls)
                    rows, cols = cls.SHAPE
                else:
                    rows, cols = len(array), 1
                flat_list = list(array)

        # 4) If there are two integer arguments and the type is not a 2-vector, treat that as
        # a zero constructor with the given shape. If it's a 2-vector, skip - we'll treat as values.
        elif (
            len(args) == 2
            and isinstance(args[0], int)
            and isinstance(args[1], int)
            and cls.SHAPE not in ((2, 1), (1, 2))
        ):
            rows = args[0]
            cols = args[1]
            flat_list = [0 for row in range(rows) for col in range(cols)]

        # 5) If there are two integer arguments and then a sequence, treat this as a shape and a
        # data list directly.
        elif len(args) == 3 and isinstance(args[-1], (np.ndarray, T.Sequence)):
            assert isinstance(args[0], int), args
            assert isinstance(args[1], int), args
            rows, cols = args[0], args[1]
            assert len(args[2]) == rows * cols, "Inconsistent args: {}".format(args)
            flat_list = list(args[2])

        # 6) Two integer arguments plus a callable to initialize values based on (row, col)
        # NOTE(hayk): sympy.Symbol is callable, hence the last check.
        elif len(args) == 3 and callable(args[-1]) and not hasattr(args[-1], "is_Symbol"):
            assert isinstance(args[0], int), args
            assert isinstance(args[1], int), args
            rows, cols = args[0], args[1]
            flat_list = [args[2](row, col) for row in range(rows) for col in range(cols)]

        # 7) If we have args equal to the fixed type, treat that as a convenience constructor like
        # Matrix31(1, 2, 3) which is the same as Matrix31(3, 1, [1, 2, 3]). Also works for
        # Matrix22([1, 2, 3, 4]).
        elif cls._is_fixed_size() and len(args) == cls.STORAGE_DIM:
            rows, cols = cls.SHAPE
            flat_list = list(args)

        # 8) No match, error out.
        else:
            raise AssertionError("Unknown {} constructor for: {}".format(cls, args))

        # Get the proper fixed size child class
        fixed_size_type = fixed_type_from_shape((rows, cols))

        if symforce.get_backend() == "sympy":
            # NOTE(hayk): Need to do this because sympy will recursively call _new otherwise.
            obj = object.__new__(fixed_size_type)
            obj.rows = rows
            obj.cols = cols
            obj._mat = flat_list
        else:
            obj = sm.Matrix.__new__(fixed_size_type, rows, cols, flat_list, **kwargs)

        return obj

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    @classmethod
    def from_storage(cls, vec):
        # type: (T.Sequence[T.Scalar]) -> Matrix
        assert cls._is_fixed_size(), "Type has no size info: {}".format(cls)
        return cls(vec)

    def to_storage(self):
        # type: () -> T.List[T.Scalar]
        return self.to_tangent()

    def __repr__(self):
        # type: () -> str
        return super(Matrix, self).__repr__()

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls):
        # type: () -> Matrix
        assert cls._is_fixed_size(), "Type has no size info: {}".format(cls)
        return cls.zeros(*cls.SHAPE)

    def compose(self, other):
        # type: (Matrix) -> Matrix
        return self + other

    def inverse(self):
        # type: () -> Matrix
        return -self

    # -------------------------------------------------------------------------
    # Lie group concept - see symforce.ops.lie_group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def from_tangent(cls, vec, epsilon=0):
        # type: (T.Sequence[T.Scalar], T.Scalar) -> Matrix
        assert cls._is_fixed_size(), "Type has no size info: {}".format(cls)
        if isinstance(vec, (list, tuple)):
            vec = cls(vec)
        return cls(vec.reshape(*cls.SHAPE).tolist())  # type: ignore

    def to_tangent(self, epsilon=0):
        # type: (T.Scalar) -> T.List[T.Scalar]
        return list(self.reshape(self.shape[0] * self.shape[1], 1))

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @classmethod
    def zero(cls):
        # type: () -> Matrix
        """
        Matrix of zeros.

        Returns:
            Matrix:
        """
        return cls.identity()

    @classmethod
    def zeros(cls, rows, cols):  # pylint: disable=signature-differs
        # type: (int, int) -> Matrix
        """
        Matrix of zeros.

        Args:
            rows (int):
            cols (int):

        Returns:
            Matrix:
        """
        return cls([[sm.S.Zero] * cols for _ in range(rows)])

    @classmethod
    def one(cls):
        # type: () -> Matrix
        """
        Matrix of ones.

        Returns:
            Matrix:
        """
        assert cls._is_fixed_size(), "Type has no size info: {}".format(cls)
        return cls.ones(*cls.SHAPE)

    @classmethod
    def ones(cls, rows, cols):  # pylint: disable=signature-differs
        # type: (int, int) -> Matrix
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
        # type: (T.List[T.Scalar]) -> Matrix
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
        # type: (int, int) -> Matrix
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

    @classmethod
    def matrix_identity(cls):
        # type: () -> Matrix
        """
        Identity matrix - ones on the diagonal, rest zeros.

        Returns:
            Matrix:
        """
        assert cls._is_fixed_size(), "Type has no size info: {}".format(cls)
        return cls.eye(*cls.SHAPE)

    def matrix_inverse(self):
        # type: () -> Matrix
        """
        Inverse of the matrix.

        Returns:
            Matrix:
        """
        return self.inv()

    @classmethod
    def symbolic(cls, name, **kwargs):
        # type: (str, T.Any) -> Matrix
        """
        Create with symbols.

        Args:
            name (str): Name prefix of the symbols
            **kwargs (dict): Forwarded to `sm.Symbol`

        Returns:
            Matrix:
        """
        assert cls._is_fixed_size(), "Type has no size info: {}".format(cls)
        rows, cols = cls.SHAPE  # pylint: disable=unpacking-non-sequence

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

        return cls(sm.Matrix(symbols))

    def simplify(self, *args, **kwargs):
        # type: (T.Any, T.Any) -> Matrix
        """
        Simplify this expression.

        This overrides the sympy implementation because that clobbers the class type.
        """
        return self.__class__(sm.simplify(self, *args, **kwargs))

    def squared_norm(self):
        # type: () -> T.Scalar
        """
        Squared norm of a vector, equivalent to the dot product with itself.
        """
        self._assert_is_vector()
        return self.dot(self)

    def norm(self, epsilon=0):
        # type: (T.Scalar) -> T.Scalar
        """
        Norm of a vector (square root of magnitude).
        """
        return sm.sqrt(self.squared_norm() + epsilon)

    def normalized(self, epsilon=0):  # pylint: disable=arguments-differ
        # type: (T.Scalar) -> Matrix
        """
        Returns a unit vector in this direction (divide by norm).
        """
        return self / self.norm(epsilon=epsilon)

    def __add__(self, right):
        # type: (T.Scalar) -> Matrix
        """
        Add a scalar to a matrix.
        """
        if StorageOps.scalar_like(right):
            return self.applyfunc(lambda x: x + right)

        return sm.Matrix.__add__(self, right)

    def __div__(self, right):
        # type: (T.Union[T.Scalar, Matrix]) -> Matrix
        """
        Divide a matrix by a scalar or a matrix (which takes the inverse).
        """
        if StorageOps.scalar_like(right):
            return self.applyfunc(lambda x: x / right)

        return self * right.inv()  # type: ignore

    __truediv__ = __div__

    @staticmethod
    def are_parallel(a, b, epsilon):
        # type: (Matrix, Matrix, T.Scalar) -> T.Scalar
        """
        Returns 1 if a and b are parallel within epsilon, and 0 otherwise.
        """
        return (1 - sm.sign(a.cross(b).norm() - epsilon)) / 2

    def evalf(self):
        # type: () -> Matrix
        """
        Perform numerical evaluation of each element in the matrix.
        """
        return self.__class__.from_storage([StorageOps.evalf(v) for v in self.to_storage()])

    def to_numpy(self, scalar_type=np.float64):
        # type: (type) -> np.ndarray
        """
        Convert to a numpy array.
        """
        return np.array(self.evalf().to_storage(), dtype=scalar_type).reshape(self.shape)

    @classmethod
    def column_stack(cls, *columns):
        # type: (Matrix) -> Matrix
        """Take a sequence of 1-D vectors and stack them as columns to make a single 2-D Matrix.

        Args:
            columns (tuple(Matrix)): 1-D vectors

        Returns:
            Matrix:
        """
        if not columns:
            return cls()

        for col in columns:
            # assert that each column is a vector
            assert col.shape == columns[0].shape
            assert sum([dim > 1 for dim in col.shape]) <= 1

        return cls([list(col) for col in columns]).T

    def _assert_is_vector(self):
        # type: () -> None
        assert (self.shape[0] == 1) or (self.shape[1] == 1), "Not a vector."

    def _assert_sanity(self):
        # type: () -> None
        assert self.shape == self.SHAPE, "Inconsistent Matrix!. shape={}, SHAPE={}".format(
            self.shape, self.SHAPE
        )

    def __hash__(self):
        # type: () -> int
        return LieGroup.__hash__(self)

    @classmethod
    def _is_fixed_size(cls):
        # type: () -> bool
        """
        Return True if this is a type with fixed dimensions set, ie Matrix31 instead of Matrix.
        """
        return cls.SHAPE[0] > 0 and cls.SHAPE[1] > 0


# -----------------------------------------------------------------------------
# Statically define fixed matrix types. We could dynamically generate in a
# loop but this is nice for IDE understanding and static analysis.
# -----------------------------------------------------------------------------

# TODO(hayk): It could be nice to put these in another file but there's a circular dependency..


class Matrix11(Matrix):
    SHAPE = (1, 1)


class Matrix21(Matrix):
    SHAPE = (2, 1)


class Matrix31(Matrix):
    SHAPE = (3, 1)


class Matrix41(Matrix):
    SHAPE = (4, 1)


class Matrix51(Matrix):
    SHAPE = (5, 1)


class Matrix61(Matrix):
    SHAPE = (6, 1)


class Matrix71(Matrix):
    SHAPE = (7, 1)


class Matrix81(Matrix):
    SHAPE = (8, 1)


class Matrix91(Matrix):
    SHAPE = (9, 1)


class Matrix12(Matrix):
    SHAPE = (1, 2)


class Matrix22(Matrix):
    SHAPE = (2, 2)


class Matrix32(Matrix):
    SHAPE = (3, 2)


class Matrix42(Matrix):
    SHAPE = (4, 2)


class Matrix52(Matrix):
    SHAPE = (5, 2)


class Matrix62(Matrix):
    SHAPE = (6, 2)


class Matrix13(Matrix):
    SHAPE = (1, 3)


class Matrix23(Matrix):
    SHAPE = (2, 3)


class Matrix33(Matrix):
    SHAPE = (3, 3)


class Matrix43(Matrix):
    SHAPE = (4, 3)


class Matrix53(Matrix):
    SHAPE = (5, 3)


class Matrix63(Matrix):
    SHAPE = (6, 3)


class Matrix14(Matrix):
    SHAPE = (1, 4)


class Matrix24(Matrix):
    SHAPE = (2, 4)


class Matrix34(Matrix):
    SHAPE = (3, 4)


class Matrix44(Matrix):
    SHAPE = (4, 4)


class Matrix54(Matrix):
    SHAPE = (5, 4)


class Matrix64(Matrix):
    SHAPE = (6, 4)


class Matrix15(Matrix):
    SHAPE = (1, 5)


class Matrix25(Matrix):
    SHAPE = (2, 5)


class Matrix35(Matrix):
    SHAPE = (3, 5)


class Matrix45(Matrix):
    SHAPE = (4, 5)


class Matrix55(Matrix):
    SHAPE = (5, 5)


class Matrix65(Matrix):
    SHAPE = (6, 5)


class Matrix16(Matrix):
    SHAPE = (1, 6)


class Matrix26(Matrix):
    SHAPE = (2, 6)


class Matrix36(Matrix):
    SHAPE = (3, 6)


class Matrix46(Matrix):
    SHAPE = (4, 6)


class Matrix56(Matrix):
    SHAPE = (5, 6)


class Matrix66(Matrix):
    SHAPE = (6, 6)


# Dictionary of shapes to static types.
DIMS_TO_FIXED_TYPE = {
    m.SHAPE: m
    for m in (
        Matrix11,
        Matrix12,
        Matrix13,
        Matrix14,
        Matrix15,
        Matrix16,
        Matrix21,
        Matrix22,
        Matrix23,
        Matrix24,
        Matrix25,
        Matrix26,
        Matrix31,
        Matrix32,
        Matrix33,
        Matrix34,
        Matrix35,
        Matrix36,
        Matrix41,
        Matrix42,
        Matrix43,
        Matrix44,
        Matrix45,
        Matrix46,
        Matrix51,
        Matrix52,
        Matrix53,
        Matrix54,
        Matrix55,
        Matrix56,
        Matrix61,
        Matrix62,
        Matrix63,
        Matrix64,
        Matrix65,
        Matrix66,
        Matrix71,
        Matrix81,
        Matrix91,
    )
}  # type: T.Dict[T.Tuple[int, int], type]


def fixed_type_from_shape(shape):
    # type: (T.Tuple[int, int]) -> type
    """
    Return a fixed size matrix type (like Matrix32) given a shape. Either use the statically
    defined ones or dynamically create a new one if not available.
    """
    if shape not in DIMS_TO_FIXED_TYPE:
        DIMS_TO_FIXED_TYPE[shape] = type(
            "Matrix{}_{}".format(shape[0], shape[1]), (Matrix,), {"SHAPE": shape}
        )

    return DIMS_TO_FIXED_TYPE[shape]


# Shorthand
M = Matrix

Vector1 = V1 = M11 = Matrix11
Vector2 = V2 = M21 = Matrix21
Vector3 = V3 = M31 = Matrix31
Vector4 = V4 = M41 = Matrix41
Vector5 = V5 = M51 = Matrix51
Vector6 = V6 = M61 = Matrix61
Vector7 = V7 = M71 = Matrix71
Vector8 = V8 = M81 = Matrix81
Vector9 = V9 = M91 = Matrix91

M12 = Matrix12
M22 = Matrix22
M32 = Matrix32
M42 = Matrix42
M52 = Matrix52
M62 = Matrix62
M13 = Matrix13
M23 = Matrix23
M33 = Matrix33
M43 = Matrix43
M53 = Matrix53
M63 = Matrix63
M14 = Matrix14
M24 = Matrix24
M34 = Matrix34
M44 = Matrix44
M54 = Matrix54
M64 = Matrix64
M15 = Matrix15
M25 = Matrix25
M35 = Matrix35
M45 = Matrix45
M55 = Matrix55
M65 = Matrix65
M16 = Matrix16
M26 = Matrix26
M36 = Matrix36
M46 = Matrix46
M56 = Matrix56
M66 = Matrix66


# Identity convenience names
I1 = I11 = M11.matrix_identity
I2 = I22 = M22.matrix_identity
I3 = I33 = M33.matrix_identity
I4 = I44 = M44.matrix_identity
I5 = I55 = M55.matrix_identity
I6 = I66 = M66.matrix_identity
