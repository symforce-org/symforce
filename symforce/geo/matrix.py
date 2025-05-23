# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import enum

import numpy as np

import symforce
import symforce.internal.symbolic as sf
from symforce import ops
from symforce import typing as _T  # We already have a Matrix.T which collides
from symforce import typing_util
from symforce.ops.interfaces import Storage

if _T.TYPE_CHECKING:
    import symengine


class Matrix(Storage):
    """
    Matrix type that wraps the SymPy Matrix class. Care has been taken to allow this class
    to create fixed-size child classes like :class:`Matrix31`. Anytime :meth:`__new__` is called,
    the appropriate fixed size class is returned rather than the type of the arguments. The API is
    meant to parallel the way Eigen's C++ matrix classes work with dynamic and fixed sizes, as well
    as internal use cases within SymPy and SymEngine.

    Examples::

        1) Matrix32()  # Zero constructed Matrix32
        2) Matrix(sm.Matrix([[1, 2], [3, 4]]))  # Matrix22 with [1, 2, 3, 4] data
        3A) Matrix([[1, 2], [3, 4]])  # Matrix22 with [1, 2, 3, 4] data
        3B) Matrix22([1, 2, 3, 4])  # Matrix22 with [1, 2, 3, 4] data (must matched fixed shape)
        3C) Matrix([1, 2, 3, 4])  # Matrix41 with [1, 2, 3, 4] data - column vector assumed
        4) Matrix(4, 3)  # Zero constructed Matrix43
        5) Matrix(2, 2, [1, 2, 3, 4])  # Matrix22 with [1, 2, 3, 4] data (first two are shape)
        6) Matrix(2, 2, lambda row, col: row + col)  # Matrix22 with [0, 1, 1, 2] data
        7) Matrix22(1, 2, 3, 4)  # Matrix22 with [1, 2, 3, 4] data (must match fixed length)

    References:
        https://docs.sympy.org/latest/tutorial/matrices.html
        https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html
        https://en.wikipedia.org/wiki/Vector_space

    Matrix does not implement the group or lie group concepts using instance/class methods directly,
    because we want it to represent the group R^{NxM}, not GL(n), which leads to the ``identity``
    and ``inverse`` methods being confusingly named.  For the group ops and lie group ops, use
    :class:`symforce.ops.group_ops.GroupOps` and :class:`symforce.ops.lie_group_ops.LieGroupOps`
    respectively, which use the implementation in
    :mod:`symforce.ops.impl.vector_class_lie_group_ops` of the R^{NxM} group under matrix addition.
    For the identity matrix and inverse matrix, see :meth:`Matrix.eye` and :meth:`Matrix.inv`
    respectively.
    """

    # Type that represents this or any subclasses
    MatrixT = _T.TypeVar("MatrixT", bound="Matrix")

    # Static dimensions of this type. (-1, -1) means there is no size information, like if
    # we are using sf.Matrix directly instead of sf.Matrix31.
    # Once a matrix is constructed it should be of a type where the .shape instance variable matches
    # this class variable as a strong internal consistency check.
    SHAPE = (-1, -1)

    def __new__(cls, *args: _T.Any, **kwargs: _T.Any) -> Matrix:  # noqa: PLR0912, PLR0915
        """
        Beast of a method for creating a Matrix. Handles a variety of construction use cases
        and *always* returns a fixed size child class of Matrix rather than Matrix itself. The
        available construction options depend on whether cls is a fixed size type or not.  See the
        Matrix docstring for a summary of the construction options.
        """

        # 1) Default construction allowed for fixed size.
        if len(args) == 0:
            if not cls._is_fixed_size():
                raise TypeError("Cannot default construct non-fixed matrix.")
            return cls.zero()

        # 2) Construct with another Matrix - this is easy
        elif len(args) == 1 and hasattr(args[0], "is_Matrix") and args[0].is_Matrix:
            rows, cols = args[0].shape
            if cls._is_fixed_size() and cls.SHAPE != (rows, cols):
                raise ValueError(
                    f"Inconsistent shape: expected shape {cls.SHAPE} but found shape {(rows, cols)}"
                )
            flat_list = list(args[0])

        # 3) If there's one argument and it's an array, works for fixed or dynamic size.
        elif len(args) == 1 and isinstance(args[0], (_T.Sequence, np.ndarray)):
            array = args[0]
            # 2D array, shape is known
            if len(array) > 0 and isinstance(array[0], (_T.Sequence, np.ndarray)):
                # 2D array of scalars
                if isinstance(array[0][0], Matrix):
                    raise TypeError("Use Matrix.block_matrix to construct using matrices")
                rows, cols = len(array), len(array[0])
                if cls._is_fixed_size() and (rows, cols) != cls.SHAPE:
                    raise ValueError(
                        f"{cls} has shape {cls.SHAPE} but arg has shape {(rows, cols)}"
                    )
                if not all(len(arr) == cols for arr in array):
                    raise ValueError(f"Inconsistent columns: {args}")
                flat_list = [v for row in array for v in row]

            # 1D array - if fixed size this must match data length. If not, assume column vec.
            else:
                if cls._is_fixed_size():
                    if len(array) != cls.storage_dim():
                        raise ValueError(
                            f"Expected {cls.storage_dim()} elements for {cls}, got {len(array)}"
                        )
                    rows, cols = cls.SHAPE
                elif len(array) == 0:
                    rows, cols = 0, 0
                else:
                    rows, cols = len(array), 1
                flat_list = list(array)

        # 4) If there are two arguments and this is not a fixed size matrix, treat it as a size
        # constructor with (rows, cols) arguments.
        # NOTE(hayk): I've had to override several routines on Matrix that in their symengine
        # versions construct a result with __class__(rows, cols), which for a fixed size type fails
        # here. We need it to fail because it's ambiguous in the case of sf.M21(10, 20) whether
        # the args are values or sizes. So I've overriden several operator methods to first convert
        # to an sm.Matrix, do the operation, then convert back.
        elif len(args) == 2 and cls.SHAPE == (-1, -1):
            rows, cols = args[0], args[1]
            if not isinstance(rows, int) or rows < 0:
                raise ValueError(f"rows must be a nonnegative integer, got {rows}")
            if not isinstance(cols, int) or cols < 0:
                raise ValueError(f"cols must be a nonnegative integer, got {cols}")
            flat_list = [0 for row in range(rows) for col in range(cols)]

        # 5) If there are two integer arguments and then a sequence, treat this as a shape and a
        # data list directly.
        elif len(args) == 3 and isinstance(args[-1], (np.ndarray, _T.Sequence)):
            if not isinstance(args[0], int) or args[0] < 0:
                raise ValueError(f"rows must be a nonnegative integer, got {args[0]}")
            if not isinstance(args[1], int) or args[1] < 0:
                raise ValueError(f"cols must be a nonnegative integer, got {args[1]}")
            rows, cols = args[0], args[1]
            if len(args[2]) != rows * cols:
                raise ValueError(f"Inconsistent args: {args}")
            flat_list = list(args[2])

        # 6) Two integer arguments plus a callable to initialize values based on (row, col)
        # NOTE(hayk): sympy.Symbol is callable, hence the last check.
        elif len(args) == 3 and callable(args[-1]) and not hasattr(args[-1], "is_Symbol"):
            if not isinstance(args[0], int) or args[0] < 0:
                raise ValueError(f"rows must be a nonnegative integer, got {args[0]}")
            if not isinstance(args[1], int) or args[1] < 0:
                raise ValueError(f"cols must be a nonnegative integer, got {args[1]}")
            rows, cols = args[0], args[1]
            flat_list = [args[2](row, col) for row in range(rows) for col in range(cols)]

        # 7) If we have args equal to the fixed type, treat that as a convenience constructor like
        # Matrix31(1, 2, 3) which is the same as Matrix31(3, 1, [1, 2, 3]). Also works for
        # Matrix22([1, 2, 3, 4]).
        elif cls._is_fixed_size() and len(args) == cls.storage_dim():
            rows, cols = cls.SHAPE
            flat_list = list(args)

        # 8) No match, error out.
        else:
            raise ValueError(f"Unknown {cls} constructor for: {args}")

        # Get the proper fixed size child class
        fixed_size_type = matrix_type_from_shape((rows, cols))

        # Build object
        instance = Storage.__new__(fixed_size_type)

        # Set the underlying sympy array
        instance.mat = sf.sympy.Matrix(rows, cols, flat_list, **kwargs)

        return instance

    def __init__(self, *args: _T.Any, **kwargs: _T.Any) -> None:
        if _T.TYPE_CHECKING:
            self.mat = sf.sympy.Matrix(*args, **kwargs)

        assert self.__class__.SHAPE == self.mat.shape, (
            f"Inconsistent Matrix: {self.__class__.SHAPE} != {self.mat.shape}"
        )

    @property
    def rows(self) -> int:
        return self.mat.rows

    @property
    def cols(self) -> int:
        return self.mat.cols

    @property
    def shape(self) -> _T.Tuple[int, int]:
        return self.mat.shape

    def __len__(self) -> int:
        return len(self.mat)

    @property
    def is_Matrix(self) -> bool:
        return True

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return self.mat.__repr__()

    @classmethod
    def storage_dim(cls) -> int:
        if not cls._is_fixed_size():
            raise TypeError(f"Type has no size info: {cls}")
        return cls.SHAPE[0] * cls.SHAPE[1]

    @classmethod
    def from_storage(
        cls: _T.Type[MatrixT], vec: _T.Union[_T.Sequence[_T.Scalar], Matrix]
    ) -> MatrixT:
        if not cls._is_fixed_size():
            raise TypeError(f"Type has no size info: {cls}")
        if isinstance(vec, Matrix):
            vec = list(vec)
        rows, cols = cls.SHAPE
        return _T.cast(Matrix.MatrixT, matrix_type_from_shape((cols, rows))(vec).transpose())

    def to_storage(self) -> _T.List[_T.Scalar]:
        return list(self.mat.transpose())

    @classmethod
    def tangent_dim(cls) -> int:
        return cls.storage_dim()

    @classmethod
    def from_tangent(
        cls: _T.Type[MatrixT], vec: _T.Sequence[_T.Scalar], epsilon: _T.Scalar = sf.epsilon()
    ) -> MatrixT:
        return cls.from_storage(vec)

    def to_tangent(self, epsilon: _T.Scalar = sf.epsilon()) -> _T.List[_T.Scalar]:
        return self.to_storage()

    def storage_D_tangent(self, epsilon: sf.Scalar = sf.epsilon()) -> Matrix:
        return Matrix.eye(self.storage_dim(), self.tangent_dim())

    def tangent_D_storage(self, epsilon: sf.Scalar = sf.epsilon()) -> Matrix:
        return Matrix.eye(self.tangent_dim(), self.storage_dim())

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @classmethod
    def zero(cls: _T.Type[MatrixT]) -> MatrixT:
        """
        Matrix of zeros.
        """
        if not cls._is_fixed_size():
            raise TypeError(f"Type has no size info: {cls}")
        return cls.zeros(*cls.SHAPE)

    @classmethod
    def zeros(cls: _T.Type[MatrixT], rows: int, cols: int) -> MatrixT:
        """
        Matrix of zeros.
        """
        if cls._is_fixed_size() and cls.SHAPE != (rows, cols):
            raise TypeError(f"Called zeros({rows=}, {cols=}) on matrix of shape {cls.SHAPE}")

        return cls([[sf.S.Zero] * cols for _ in range(rows)])

    @classmethod
    def one(cls: _T.Type[MatrixT]) -> MatrixT:
        """
        Matrix of ones.
        """
        if not cls._is_fixed_size():
            raise TypeError(f"Type has no size info: {cls}")
        return cls.ones(*cls.SHAPE)

    @classmethod
    def ones(cls: _T.Type[MatrixT], rows: int, cols: int) -> MatrixT:
        """
        Matrix of ones.
        """
        if cls._is_fixed_size() and cls.SHAPE != (rows, cols):
            raise TypeError(f"Called ones({rows=}, {cols=}) on matrix of shape {cls.SHAPE}")

        return cls([[sf.S.One] * cols for _ in range(rows)])

    @classmethod
    def diag(cls: _T.Type[MatrixT], diagonal: _T.Sequence[_T.Scalar]) -> MatrixT:
        """
        Construct a square matrix from the diagonal.
        """
        if cls._is_fixed_size():
            rows, cols = cls.SHAPE
            if rows != cols:
                raise TypeError(f"Cannot call .diag() on non-square shape {cls.SHAPE}")
            if rows != len(diagonal):
                raise ValueError(
                    f"Cannot call .diag() with a diagonal of length {len(diagonal)} on a matrix of shape {cls.SHAPE}"
                )

        mat = cls.zeros(len(diagonal), len(diagonal))
        for i, x in enumerate(diagonal):
            mat[i, i] = x
        return mat

    @classmethod
    def eye(
        cls: _T.Type[MatrixT], rows: _T.Optional[int] = None, cols: _T.Optional[int] = None
    ) -> MatrixT:
        """
        Construct an identity matrix

        If neither rows nor cols is provided, this must be called as a class method on a fixed-size
        class.

        If rows is provided, returns a square identity matrix of shape (rows x rows).

        If rows and cols are provided, returns a (rows x cols) matrix, with ones on the diagonal.
        """
        if rows is None and cols is None:
            if not cls._is_fixed_size():
                raise TypeError(
                    "Matrix.eye can only be called with no arguments on a fixed-size matrix type"
                )

            rows, cols = cls.SHAPE

        if rows is None:
            raise ValueError("If cols is not None, rows must not be None")

        orig_cols = cols
        if cols is None:
            cols = rows

        if cls._is_fixed_size() and cls.SHAPE != (rows, cols):
            raise TypeError(f"Called eye({rows=}, cols={orig_cols}) on matrix of shape {cls.SHAPE}")

        mat = cls.zeros(rows, cols)
        for i in range(min(rows, cols)):
            mat[i, i] = sf.S.One
        return mat

    def det(self) -> _T.Scalar:
        """
        Determinant of the matrix.
        """
        return self.mat.det()

    def inv(self: MatrixT, method: str = "LU") -> MatrixT:
        """
        Inverse of the matrix.
        """
        return self.__class__(self.mat.inv(method=method))

    @classmethod
    def symbolic(cls: _T.Type[MatrixT], name: str, **kwargs: _T.Any) -> MatrixT:
        """
        Create with symbols.

        Args:
            name (str): Name prefix of the symbols
            **kwargs (dict): Forwarded to `sf.Symbol`
        """
        if not cls._is_fixed_size():
            raise TypeError(f"Type has no size info: {cls}")
        rows, cols = cls.SHAPE

        row_names = [str(r_i) for r_i in range(rows)]
        col_names = [str(c_i) for c_i in range(cols)]

        if len(row_names) != rows:
            raise ValueError(f"Number of row names {len(row_names)} does not match rows {rows}")
        if len(col_names) != cols:
            raise ValueError(
                f"Number of column names {len(col_names)} does not match columns {cols}"
            )

        if cols == 1:
            if ops.StorageOps.use_latex_friendly_symbols():
                format_string = "{}_{}"
            else:
                format_string = "{}[{}]"

            symbols = []
            for r_i in range(rows):
                _name = format_string.format(name, row_names[r_i])
                symbols.append(sf.Symbol(_name, **kwargs))
        else:
            if ops.StorageOps.use_latex_friendly_symbols():
                format_string = "{}_{{{}, {}}}"
            else:
                format_string = "{}[{}, {}]"
            symbols = []
            for r_i in range(rows):
                for c_i in range(cols):
                    _name = format_string.format(name, row_names[r_i], col_names[c_i])
                    symbols.append(sf.Symbol(_name, **kwargs))

        return cls(sf.sympy.Matrix(rows, cols, symbols))

    def row_join(self, right: Matrix) -> Matrix:
        """
        Concatenates self with another matrix on the right
        """
        return Matrix(self.mat.row_join(right.mat))

    def col_join(self, bottom: Matrix) -> Matrix:
        """
        Concatenates self with another matrix below
        """
        return Matrix(self.mat.col_join(bottom.mat))

    @classmethod
    def block_matrix(cls, array: _T.Sequence[_T.Sequence[Matrix]]) -> Matrix:
        """
        Constructs a matrix from block elements.

        For example::

            [[Matrix22(...), Matrix23(...)], [Matrix11(...), Matrix14(...)]]

        constructs a :class:`Matrix35` with elements equal to given blocks
        """
        # Sum rows of matrices in the first column
        rows = sum(mat_row[0].shape[0] for mat_row in array)
        # Sum columns of matrices in the first row
        cols = sum(mat.shape[1] for mat in array[0])

        # Check for size consistency
        for mat_row in array:
            block_rows = mat_row[0].shape[0]
            block_cols = 0
            for mat in mat_row:
                if mat.shape[0] != block_rows:
                    raise ValueError(
                        "Inconsistent row number accross block: expected {}, got {}".format(
                            block_rows, mat.shape[0]
                        )
                    )
                block_cols += mat.shape[1]
            if block_cols != cols:
                raise ValueError(
                    "Inconsistent column number accross block: expected {}, got {}".format(
                        cols, block_cols
                    )
                )

        # Fill the new matrix data vector
        flat_list = []
        for mat_row in array:
            for row in range(mat_row[0].shape[0]):
                for mat in mat_row:
                    if mat.shape[1] == 1:
                        flat_list += [mat[row]]
                    else:
                        flat_list += list(mat[row, :])

        return Matrix(rows, cols, flat_list)

    def simplify(self, *args: _T.Any, **kwargs: _T.Any) -> Matrix:
        """
        Simplify this expression.

        This overrides the sympy implementation because that clobbers the class type.
        """
        return self.__class__(sf.simplify(self.mat, *args, **kwargs))

    def limit(self, *args: _T.Any, **kwargs: _T.Any) -> Matrix:
        """
        Take the limit at z = z0

        This overrides the sympy implementation because that clobbers the class type.
        """
        return self.from_flat_list([sf.limit(e, *args, **kwargs) for e in self.to_flat_list()])

    def jacobian(
        self, X: _T.Any, tangent_space: bool = True, epsilon: _T.Scalar = sf.epsilon()
    ) -> Matrix:
        """
        Compute the jacobian with respect to the tangent space of X if ``tangent_space = True``,
        otherwise returns the jacobian with respect to the storage elements of X.

        Note that the jacobian is always 2D, even if self or X are matrices - it will be M x N,
        where M is the size of self and N is the size of X
        """
        return ops.LieGroupOps.jacobian(self, X, tangent_space=tangent_space, epsilon=epsilon)

    def diff(self, *args: _T.Scalar) -> Matrix:
        """
        Differentiate w.r.t. a scalar.
        """
        return self.__class__(self.mat.diff(*args))

    @property
    def T(self) -> Matrix:
        """
        Matrix Transpose
        """
        return self.transpose()

    def transpose(self) -> Matrix:
        """
        Matrix Transpose
        """
        return Matrix(self.mat.transpose())

    def lower_triangle(self: MatrixT) -> MatrixT:
        """
        Returns the lower triangle (including diagonal) of self

        self must be square
        """
        rows, cols = self.shape
        if rows != cols:
            raise ValueError(
                f"Attempted to take lower triangle of non-square matrix (found shape {self.shape})"
            )

        lt = self.__class__()
        for k in range(rows):
            lt[k, : k + 1] = self[k, : k + 1]
        return lt

    class Triangle(enum.Enum):
        LOWER = "lower"
        UPPER = "upper"

    def symmetric_copy(self: MatrixT, upper_or_lower: Triangle) -> MatrixT:
        """
        Returns a symmetric copy of `self` by copying the lower or upper triangle to the opposite
        triangle.

        Args:
            upper_or_lower: The triangle to copy to the opposite triangle
        """
        if self.rows != self.cols:
            raise TypeError(f"Matrix must be square to make a symmetric copy, not {self.shape}")

        result = self[:, :]

        for i in range(self.rows):
            for j in range(i + 1, self.rows):
                if upper_or_lower == self.Triangle.LOWER:
                    result[i, j] = result[j, i]
                else:
                    result[j, i] = result[i, j]

        return result

    def reshape(self, rows: int, cols: int) -> Matrix:
        return Matrix(self.mat.reshape(rows, cols))

    def dot(self, other: Matrix) -> _T.Scalar:
        """
        Dot product, also known as inner product.

        Only supports mapping ``1 x n`` or ``n x 1`` Matrices to scalars. Note that both matrices
        must have the same shape.
        """
        if not (self.is_vector() and other.is_vector()):
            raise TypeError(
                f"Dot can only be called on vectors, got matrices of shapes {self.shape} and {other.shape}"
            )
        if self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1]:
            raise TypeError(
                f"Dot expects both vectors to be the same shape, got matrices of shapes {self.shape} and {other.shape}"
            )

        return self.mat.dot(other.mat)

    # NOTE(aaron): We could annotate this as (self, Vector3) -> Vector3.  However, many operations
    # on Matrix aren't shape-aware, e.g. *_join or matmul.  So it results in a lot of instances of
    # mypy getting mad about calling this on a Matrix instead of the Vector3 subclass.  So just
    # check the shape at runtime like we do for those other methods until mypy supports shapes
    # nicely
    def cross(self: MatrixT, other: MatrixT) -> Vector3:
        """
        Cross product.
        """
        if self.shape != (3, 1) or other.shape != (3, 1):
            raise TypeError(
                "Cross can only be called on shape (3, 1), got matrices of shapes {} and {}".format(
                    self.shape, other.shape
                )
            )

        return Vector3(self.mat.cross(other.mat))

    def squared_norm(self) -> _T.Scalar:
        """
        Squared norm of a vector, equivalent to the dot product with itself.
        """
        self._assert_is_vector()
        return self.dot(self)

    def norm(self, epsilon: _T.Scalar = sf.epsilon()) -> _T.Scalar:
        """
        Norm of a vector (square root of magnitude).
        """
        return sf.sqrt(self.squared_norm() + epsilon)

    def normalized(self: MatrixT, epsilon: _T.Scalar = sf.epsilon()) -> MatrixT:
        """
        Returns a unit vector in this direction (divide by norm).
        """
        return self / self.norm(epsilon=epsilon)

    def clamp_norm(
        self: MatrixT, max_norm: _T.Scalar, epsilon: _T.Scalar = sf.epsilon()
    ) -> MatrixT:
        """
        Clamp a vector to the given norm in a safe/differentiable way.

        Is **NOT** safe if max_norm can be negative, or if derivatives are needed w.r.t. max_norm and
        max_norm can be 0 or small enough that ``max_squared_norm / squared_norm`` is truncated to 0
        in the particular floating point type being used (e.g. all of these are true if ``max_norm``
        is optimized).

        Currently only L2 norm is supported
        """
        if self.shape[1] != 1:
            raise TypeError(
                f"clamp_norm can only be called on vectors, this matrix is shape {self.shape}"
            )

        squared_norm = self.squared_norm() + epsilon
        max_squared_norm = max_norm**2

        # This sqrt can be near 0, if max_norm can be exactly 0
        return self * sf.Min(1, sf.sqrt(max_squared_norm / squared_norm))

    def multiply_elementwise(self: MatrixT, rhs: MatrixT) -> MatrixT:
        """
        Do the elementwise multiplication between self and rhs, and return the result as a new
        :class:`Matrix`
        """
        if self.shape != rhs.shape:
            raise TypeError(f"Cannot multiply elementwise: shapes {self.shape} and {rhs.shape}")
        return self.__class__(self.mat.multiply_elementwise(rhs.mat))

    def applyfunc(self: MatrixT, func: _T.Callable) -> MatrixT:
        """
        Apply a unary operation to every scalar.
        """
        return self.__class__(self.mat.applyfunc(func))

    # Dummy __iter__ method for mypy
    # Matrix is Iterable because it implements __getitem__(int), but mypy only recognizes __iter__:
    # https://github.com/python/mypy/issues/2220
    if _T.TYPE_CHECKING:  # pragma: no cover

        def __iter__(self) -> _T.Iterator[_T.Any]:
            raise NotImplementedError()

    def __getitem__(self, item: _T.Any) -> _T.Any:
        """
        Get a scalar value or submatrix slice.

        Unlike sympy, for 1D matrices the submatrix slice is returned as a 1D matrix instead of as a
        list.
        """
        ret = self.mat.__getitem__(item)
        if isinstance(ret, sf.sympy.Matrix):
            return Matrix(ret)
        if isinstance(ret, list):
            if self.cols > 1:
                # Original matrix is a row vector, return a row vector
                return Matrix(1, len(ret), ret)
            # Original matrix is a column vector, return a column vector
            return Matrix(ret)
        return ret

    def __setitem__(
        self, key: _T.Any, value: _T.Union[_T.Scalar, Matrix, sf.sympy.MutableDenseMatrix]
    ) -> None:
        if isinstance(value, Matrix):
            value = value.mat
        ret = self.mat.__setitem__(key, value)
        if isinstance(ret, sf.sympy.Matrix):
            ret = self.__class__(ret)
        return ret

    def row(self, r: int) -> Matrix:
        """
        Extract a row of the matrix
        """
        return Matrix(self.mat.row(r))

    def col(self, c: int) -> Matrix:
        """
        Extract a column of the matrix
        """
        return Matrix(self.mat.col(c))

    def __neg__(self: MatrixT) -> MatrixT:
        """
        Negate matrix.
        """
        return self.__class__(-self.mat)

    def __add__(self: MatrixT, right: _T.Union[_T.Scalar, MatrixT]) -> MatrixT:
        """
        Add a scalar or matrix to this matrix.
        """
        if typing_util.scalar_like(right):
            return self.applyfunc(lambda x: x + right)
        elif isinstance(right, Matrix):
            return self.__class__(self.mat + right.mat)
        else:
            return self.__class__(self.mat + right)

    def __sub__(self: MatrixT, right: _T.Union[_T.Scalar, MatrixT]) -> MatrixT:
        """
        Subtract a scalar or matrix from this matrix.
        """
        if typing_util.scalar_like(right):
            return self.applyfunc(lambda x: x - right)
        elif isinstance(right, Matrix):
            return self.__class__(self.mat - right.mat)
        else:
            return self.__class__(self.mat - right)

    @_T.overload
    def __mul__(
        self, right: _T.Union[Matrix, sf.sympy.MutableDenseMatrix]
    ) -> Matrix:  # pragma: no cover
        pass

    @_T.overload
    def __mul__(self: MatrixT, right: _T.Scalar) -> MatrixT:  # pragma: no cover
        pass

    def __mul__(
        self, right: _T.Union[MatrixT, _T.Scalar, Matrix, sf.sympy.MutableDenseMatrix]
    ) -> _T.Union[MatrixT, Matrix]:
        """
        Multiply a matrix by a scalar or matrix
        """
        if typing_util.scalar_like(right):
            return self.applyfunc(lambda x: x * right)
        elif isinstance(right, Matrix):
            return Matrix(self.mat * right.mat)
        else:
            return Matrix(self.mat * right)

    @_T.overload
    def __rmul__(
        self, left: _T.Union[Matrix, sf.sympy.MutableDenseMatrix]
    ) -> Matrix:  # pragma: no cover
        pass

    @_T.overload
    def __rmul__(self: MatrixT, left: _T.Scalar) -> MatrixT:  # pragma: no cover
        pass

    def __rmul__(
        self, left: _T.Union[MatrixT, _T.Scalar, Matrix, sf.sympy.MutableDenseMatrix]
    ) -> _T.Union[MatrixT, Matrix]:
        """
        Left multiply a matrix by a scalar or matrix
        """
        if typing_util.scalar_like(left):
            return self.applyfunc(lambda x: left * x)
        elif isinstance(left, Matrix):
            return self.__class__(left.mat * self.mat)
        else:
            return self.__class__(left * self.mat)

    @_T.overload
    def __truediv__(
        self, right: _T.Union[Matrix, sf.sympy.MutableDenseMatrix]
    ) -> Matrix:  # pragma: no cover
        pass

    @_T.overload
    def __truediv__(self: MatrixT, right: _T.Scalar) -> MatrixT:  # pragma: no cover
        pass

    def __truediv__(
        self, right: _T.Union[MatrixT, _T.Scalar, Matrix, sf.sympy.MutableDenseMatrix]
    ) -> _T.Union[MatrixT, Matrix]:
        """
        Divide a matrix by a scalar or a matrix (which takes the inverse).
        """
        if typing_util.scalar_like(right):
            return self.applyfunc(lambda x: x / sf.S(right))
        elif isinstance(right, Matrix):
            return self * right.inv()
        else:
            return self.__class__(self.mat * _T.cast(sf.sympy.MutableDenseMatrix, right).inv())

    def _symengine_(self) -> symengine.Matrix:  # noqa: PLW3201
        symengine = symforce._find_symengine()  # noqa: SLF001
        return symengine.S(self.mat)

    def compute_AtA(self, lower_only: bool = False) -> Matrix:
        """
        Compute a symmetric product ``A.transpose() * A``

        Args:
            lower_only: If given, only fill the lower half and set upper to zero

        Returns:
            (Matrix(N, N)): Symmetric matrix ``AtA = self.transpose() * self``

        """
        AtA = self.T * self
        if lower_only:
            for i in range(self.cols):
                for j in range(i + 1, self.cols):
                    AtA[i, j] = 0

        return AtA

    def LU(
        self,
    ) -> _T.Union[_T.Tuple[Matrix, Matrix], _T.Tuple[Matrix, Matrix, _T.List[_T.Tuple[int, int]]]]:
        """
        LU matrix decomposition
        """
        if symforce.get_symbolic_api() == "sympy":
            L, U, perm = self.mat.LUdecomposition()
            return self.__class__(L), self.__class__(U), perm
        elif symforce.get_symbolic_api() == "symengine":
            L, U = self.mat.LU()  # type: ignore[attr-defined]
            return self.__class__(L), self.__class__(U)
        else:
            raise symforce.InvalidSymbolicApiError(symforce.get_symbolic_api())

    def LDL(self) -> _T.Tuple[Matrix, Matrix]:
        """
        LDL matrix decomposition (stable cholesky)
        """
        if symforce.get_symbolic_api() == "sympy":
            L, D = self.mat.LDLdecomposition()
        elif symforce.get_symbolic_api() == "symengine":
            L, D = self.mat.LDL()  # type: ignore[attr-defined]
        else:
            raise symforce.InvalidSymbolicApiError(symforce.get_symbolic_api())
        return self.__class__(L), self.__class__(D)

    def FFLU(self) -> _T.Tuple[Matrix, Matrix]:
        """
        Fraction-free LU matrix decomposition
        """
        if symforce.get_symbolic_api() == "sympy":
            raise NotImplementedError(
                "The FFLU decomposition does not exist on SymPy, use FFLDU instead"
            )
        elif symforce.get_symbolic_api() == "symengine":
            L, U = self.mat.FFLU()  # type: ignore[attr-defined]
            return self.__class__(L), self.__class__(U)
        else:
            raise symforce.InvalidSymbolicApiError(symforce.get_symbolic_api())

    def FFLDU(
        self,
    ) -> _T.Union[_T.Tuple[Matrix, Matrix, Matrix], _T.Tuple[Matrix, Matrix, Matrix, Matrix]]:
        """
        Fraction-free LDU matrix decomposition
        """
        if symforce.get_symbolic_api() == "sympy":
            P, L, D, U = self.mat.LUdecompositionFF()
            return self.__class__(P), self.__class__(L), self.__class__(D), self.__class__(U)
        elif symforce.get_symbolic_api() == "symengine":
            L, D, U = self.mat.FFLDU()  # type: ignore[attr-defined]
            return self.__class__(L), self.__class__(D), self.__class__(U)
        else:
            raise symforce.InvalidSymbolicApiError(symforce.get_symbolic_api())

    def solve(self, b: Matrix, method: str = "LU") -> Matrix:
        """
        Solve a linear system using the given method.
        """
        return self.__class__(self.mat.solve(b, method=method))

    @staticmethod
    def are_parallel(a: Vector3, b: Vector3, tolerance: _T.Scalar) -> _T.Scalar:
        """
        Returns 1 if a and b are parallel within tolerance, and 0 otherwise.
        """
        return (1 - sf.sign(a.cross(b).squared_norm() - tolerance**2)) / 2

    @staticmethod
    def skew_symmetric(a: Vector3) -> Matrix33:
        """
        Compute a skew-symmetric matrix of given a 3-vector.
        """
        return Matrix33([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])

    def evalf(self) -> Matrix:
        """
        Perform numerical evaluation of each element in the matrix.
        """
        return self.from_flat_list([ops.StorageOps.evalf(v) for v in self.to_flat_list()])

    def to_list(self) -> _T.List[_T.List[_T.Scalar]]:
        """
        Convert to a nested list
        """
        return self.mat.tolist()

    def to_flat_list(self) -> _T.List[_T.Scalar]:
        """
        Convert to a flattened list
        """
        return list(iter(self.mat))

    @classmethod
    def from_flat_list(cls, vec: _T.Sequence[_T.Scalar]) -> Matrix:
        if not cls._is_fixed_size():
            raise TypeError(f"Type has no size info: {cls}")
        return cls(vec)

    def to_numpy(self, scalar_type: type = np.float64) -> np.ndarray:
        """
        Convert to a numpy array.
        """
        return np.array(self.evalf().to_flat_list(), dtype=scalar_type).reshape(self.shape)

    @classmethod
    def column_stack(cls, *columns: Matrix) -> Matrix:
        """
        Take a sequence of 1-D vectors and stack them as columns to make a single 2-D Matrix.

        Args:
            columns: 1-D vectors
        """
        if not columns:
            return cls()

        for col in columns:
            # assert that each column is a vector
            if col.shape != columns[0].shape or sum(dim > 1 for dim in col.shape) > 1:
                raise TypeError(f"Column has shape {col.shape}, should be a vector (N, 1)")

        return cls([col.to_flat_list() for col in columns]).T

    def is_vector(self) -> bool:
        return (self.shape[0] == 1) or (self.shape[1] == 1)

    def _assert_is_vector(self) -> None:
        if not self.is_vector():
            raise TypeError(f"Not a vector, shape {self.shape}")

    def _assert_sanity(self) -> None:
        assert self.shape == self.SHAPE, "Inconsistent Matrix!. shape={}, SHAPE={}".format(
            self.shape, self.SHAPE
        )

    def __hash__(self) -> int:
        return Storage.__hash__(self)

    @classmethod
    def _is_fixed_size(cls) -> bool:
        """
        Return ``True`` if this is a type with fixed dimensions set, e.g. :class:`Matrix31` instead
        of :class:`Matrix`.
        """
        return cls.SHAPE[0] > -1 and cls.SHAPE[1] > -1

    def _ipython_display_(self) -> None:  # noqa: PLW3201
        """
        Display ``self.mat`` in IPython, with SymPy's pretty printing
        """
        display(self.mat)  # type: ignore[name-defined] # noqa: F821 # not defined outside of ipython

    @staticmethod
    def init_printing() -> None:
        """
        Initialize SymPy pretty printing

        ``_ipython_display_`` is sufficient in Jupyter, but this covers other locations
        """
        ip = None
        try:
            ip = get_ipython()  # type: ignore[name-defined] # only exists in ipython
        except NameError:
            pass

        if ip is not None:
            plaintext_formatter = ip.display_formatter.formatters["text/plain"]
            sympy_plaintext_formatter = plaintext_formatter.for_type(sf.sympy.Matrix)
            if sympy_plaintext_formatter is not None:
                plaintext_formatter.for_type(
                    Matrix, lambda arg, p, cycle: sympy_plaintext_formatter(arg.mat, p, cycle)
                )

            png_formatter = ip.display_formatter.formatters["image/png"]
            sympy_png_formatter = png_formatter.for_type(sf.sympy.Matrix)
            if sympy_png_formatter is not None:
                png_formatter.for_type(Matrix, lambda o: sympy_png_formatter(o.mat))

            latex_formatter = ip.display_formatter.formatters["text/latex"]
            sympy_latex_formatter = latex_formatter.for_type(sf.sympy.Matrix)
            if sympy_latex_formatter is not None:
                latex_formatter.for_type(Matrix, lambda o: sympy_latex_formatter(o.mat))


# -----------------------------------------------------------------------------
# Statically define fixed matrix types. We could dynamically generate in a
# loop but this is nice for IDE understanding and static analysis.
# -----------------------------------------------------------------------------

# TODO(hayk): It could be nice to put these in another file but there's a circular dependency..


class Matrix11(Matrix):
    SHAPE = (1, 1)


class Matrix21(Matrix):
    SHAPE = (2, 1)

    @staticmethod
    def unit_x() -> Vector2:
        """
        The unit vector [1, 0]
        """
        return Vector2(1, 0)

    @staticmethod
    def unit_y() -> Vector2:
        """
        The unit vector [0, 1]
        """
        return Vector2(0, 1)

    @property
    def x(self) -> sf.Scalar:
        """
        The entry self[0, 0]
        """
        return self[0, 0]

    @property
    def y(self) -> sf.Scalar:
        """
        The entry self[1, 0]
        """
        return self[1, 0]


class Matrix31(Matrix):
    SHAPE = (3, 1)

    @staticmethod
    def unit_x() -> Vector3:
        """
        The unit vector [1, 0, 0]
        """
        return Vector3(1, 0, 0)

    @staticmethod
    def unit_y() -> Vector3:
        """
        The unit vector [0, 1, 0]
        """
        return Vector3(0, 1, 0)

    @staticmethod
    def unit_z() -> Vector3:
        """
        The unit vector [0, 0, 1]
        """
        return Vector3(0, 0, 1)

    @property
    def x(self) -> sf.Scalar:
        """
        The entry self[0, 0]
        """
        return self[0, 0]

    @property
    def y(self) -> sf.Scalar:
        """
        The entry self[1, 0]
        """
        return self[1, 0]

    @property
    def z(self) -> sf.Scalar:
        """
        The entry self[2, 0]
        """
        return self[2, 0]


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


class Matrix72(Matrix):
    SHAPE = (7, 2)


class Matrix82(Matrix):
    SHAPE = (8, 2)


class Matrix92(Matrix):
    SHAPE = (9, 2)


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


class Matrix73(Matrix):
    SHAPE = (7, 3)


class Matrix83(Matrix):
    SHAPE = (8, 3)


class Matrix93(Matrix):
    SHAPE = (9, 3)


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


class Matrix74(Matrix):
    SHAPE = (7, 4)


class Matrix84(Matrix):
    SHAPE = (8, 4)


class Matrix94(Matrix):
    SHAPE = (9, 4)


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


class Matrix75(Matrix):
    SHAPE = (7, 5)


class Matrix85(Matrix):
    SHAPE = (8, 5)


class Matrix95(Matrix):
    SHAPE = (9, 5)


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


class Matrix76(Matrix):
    SHAPE = (7, 6)


class Matrix86(Matrix):
    SHAPE = (8, 6)


class Matrix96(Matrix):
    SHAPE = (9, 6)


class Matrix17(Matrix):
    SHAPE = (1, 7)


class Matrix27(Matrix):
    SHAPE = (2, 7)


class Matrix37(Matrix):
    SHAPE = (3, 7)


class Matrix47(Matrix):
    SHAPE = (4, 7)


class Matrix57(Matrix):
    SHAPE = (5, 7)


class Matrix67(Matrix):
    SHAPE = (6, 7)


class Matrix77(Matrix):
    SHAPE = (7, 7)


class Matrix87(Matrix):
    SHAPE = (8, 7)


class Matrix97(Matrix):
    SHAPE = (9, 7)


class Matrix18(Matrix):
    SHAPE = (1, 8)


class Matrix28(Matrix):
    SHAPE = (2, 8)


class Matrix38(Matrix):
    SHAPE = (3, 8)


class Matrix48(Matrix):
    SHAPE = (4, 8)


class Matrix58(Matrix):
    SHAPE = (5, 8)


class Matrix68(Matrix):
    SHAPE = (6, 8)


class Matrix78(Matrix):
    SHAPE = (7, 8)


class Matrix88(Matrix):
    SHAPE = (8, 8)


class Matrix98(Matrix):
    SHAPE = (9, 8)


class Matrix19(Matrix):
    SHAPE = (1, 9)


class Matrix29(Matrix):
    SHAPE = (2, 9)


class Matrix39(Matrix):
    SHAPE = (3, 9)


class Matrix49(Matrix):
    SHAPE = (4, 9)


class Matrix59(Matrix):
    SHAPE = (5, 9)


class Matrix69(Matrix):
    SHAPE = (6, 9)


class Matrix79(Matrix):
    SHAPE = (7, 9)


class Matrix89(Matrix):
    SHAPE = (8, 9)


class Matrix99(Matrix):
    SHAPE = (9, 9)


# Dictionary of shapes to static types.
DIMS_TO_FIXED_TYPE: _T.Dict[_T.Tuple[int, int], type] = {}
for rows in range(1, 10):
    for cols in range(1, 10):
        m = vars()[f"Matrix{rows}{cols}"]
        DIMS_TO_FIXED_TYPE[m.SHAPE] = m


def matrix_type_from_shape(shape: _T.Tuple[int, int]) -> _T.Type[Matrix]:
    """
    Return a fixed size matrix type (like :class:`Matrix32`) given a shape

    Either uses the statically defined ones or dynamically creates a new one if not available.
    """
    if shape not in DIMS_TO_FIXED_TYPE:
        DIMS_TO_FIXED_TYPE[shape] = type(
            "Matrix{}_{}".format(shape[0], shape[1]), (Matrix,), {"SHAPE": shape}
        )

    return DIMS_TO_FIXED_TYPE[shape]


# Shorthand
M = Matrix

Vector1 = Matrix11
Vector2 = Matrix21
Vector3 = Matrix31
Vector4 = Matrix41
Vector5 = Matrix51
Vector6 = Matrix61
Vector7 = Matrix71
Vector8 = Matrix81
Vector9 = Matrix91

V1 = Vector1
V2 = Vector2
V3 = Vector3
V4 = Vector4
V5 = Vector5
V6 = Vector6
V7 = Vector7
V8 = Vector8
V9 = Vector9

M11 = Matrix11
M21 = Matrix21
M31 = Matrix31
M41 = Matrix41
M51 = Matrix51
M61 = Matrix61
M71 = Matrix71
M81 = Matrix81
M91 = Matrix91
M12 = Matrix12
M22 = Matrix22
M32 = Matrix32
M42 = Matrix42
M52 = Matrix52
M62 = Matrix62
M72 = Matrix72
M82 = Matrix82
M92 = Matrix92
M13 = Matrix13
M23 = Matrix23
M33 = Matrix33
M43 = Matrix43
M53 = Matrix53
M63 = Matrix63
M73 = Matrix73
M83 = Matrix83
M93 = Matrix93
M14 = Matrix14
M24 = Matrix24
M34 = Matrix34
M44 = Matrix44
M54 = Matrix54
M64 = Matrix64
M74 = Matrix74
M84 = Matrix84
M94 = Matrix94
M15 = Matrix15
M25 = Matrix25
M35 = Matrix35
M45 = Matrix45
M55 = Matrix55
M65 = Matrix65
M75 = Matrix75
M85 = Matrix85
M95 = Matrix95
M16 = Matrix16
M26 = Matrix26
M36 = Matrix36
M46 = Matrix46
M56 = Matrix56
M66 = Matrix66
M76 = Matrix76
M86 = Matrix86
M96 = Matrix96
M17 = Matrix17
M27 = Matrix27
M37 = Matrix37
M47 = Matrix47
M57 = Matrix57
M67 = Matrix67
M77 = Matrix77
M87 = Matrix87
M97 = Matrix97
M18 = Matrix18
M28 = Matrix28
M38 = Matrix38
M48 = Matrix48
M58 = Matrix58
M68 = Matrix68
M78 = Matrix78
M88 = Matrix88
M98 = Matrix98
M19 = Matrix19
M29 = Matrix29
M39 = Matrix39
M49 = Matrix49
M59 = Matrix59
M69 = Matrix69
M79 = Matrix79
M89 = Matrix89
M99 = Matrix99


# Identity convenience names
I1 = I11 = M11.eye
I2 = I22 = M22.eye
I3 = I33 = M33.eye
I4 = I44 = M44.eye
I5 = I55 = M55.eye
I6 = I66 = M66.eye
I7 = I77 = M77.eye
I8 = I88 = M88.eye
I9 = I99 = M99.eye


# Register printing for ipython
Matrix.init_printing()


# Register ops
from symforce.ops.impl.vector_class_lie_group_ops import VectorClassLieGroupOps

ops.LieGroupOps.register(Matrix, VectorClassLieGroupOps)
