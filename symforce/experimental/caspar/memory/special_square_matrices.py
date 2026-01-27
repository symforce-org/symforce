# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

from abc import abstractmethod

import symforce.symbolic as sf
from symforce import typing as T
from symforce.ops.interfaces import Storage

SpecialSquareMatT = T.TypeVar("SpecialSquareMatT", bound="_SpecialSquareMatrix")


class _SpecialSquareMatrix(Storage):
    """Special square matrix"""

    storage: T.List[sf.Scalar]

    SHAPE: T.ClassVar[int] = -1

    @classmethod
    def get_class(cls, shape: int) -> T.Type[_SpecialSquareMatrix]:
        return type(f"{cls.__name__}{shape}", (cls,), {"SHAPE": shape})

    def __new__(cls: T.Type[SpecialSquareMatT], matrix: sf.Matrix) -> SpecialSquareMatT:
        if cls.SHAPE == -1:
            sizedT = type(f"{cls.__name__}{matrix.shape[0]}", (cls,), {"SHAPE": matrix.shape[0]})
            return T.cast(SpecialSquareMatT, super().__new__(sizedT))
        return super().__new__(cls)

    @classmethod
    def check_size(cls, mat: sf.Matrix) -> None:
        if (target := (cls.SHAPE, cls.SHAPE)) != mat.SHAPE:
            raise ValueError(f"Expected matrix of dim {target}, got shape {mat.SHAPE}")

    def __repr__(self) -> str:
        return self.mat().__repr__()

    def to_storage(self) -> T.List[T.Scalar]:
        """
        Flat list representation of the underlying storage, length of :meth:`storage_dim`.
        This is used purely for plumbing, it is NOT like a tangent space.
        """
        return self.storage.copy()

    @classmethod
    def from_storage(
        cls: T.Type[SpecialSquareMatT], elements: T.Sequence[T.Scalar]
    ) -> SpecialSquareMatT:
        """
        Construct from a flat list representation. Opposite of :meth:`to_storage`.
        """
        if cls.SHAPE == -1:
            raise ValueError("SHAPE not set")
        if len(elements) != cls.storage_dim():
            raise ValueError(
                f"Expected {cls.storage_dim()} elements for {cls}, got {len(elements)}"
            )
        instance = object.__new__(cls)
        instance.storage = list(elements)
        return instance

    @abstractmethod
    def mat(self) -> sf.Matrix:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def storage_dim(cls) -> int:
        raise NotImplementedError


class LowerTriangularMatrix(_SpecialSquareMatrix):
    """Lower triangular matrix, including the diagonal"""

    def __init__(self, mat: sf.Matrix):
        self.check_size(mat)
        self.storage = []
        for i in range(self.SHAPE):
            for j in range(i + 1):
                if i != j and mat[j, i] != 0:
                    raise ValueError("Matrix is not lower triangular")
                self.storage.append(mat[i, j])
        assert self.mat() == mat

    @classmethod
    def lower_storage_indices(cls) -> tuple[int, ...]:
        out = []
        for i in range(cls.SHAPE):
            for j in range(i + 1):
                out.append(i + j * cls.SHAPE)
        return tuple(out)

    def mat(self) -> sf.Matrix:
        mat = sf.Matrix(self.SHAPE, self.SHAPE)
        offset = 0
        for i in range(self.SHAPE):
            for j in range(i + 1):
                mat[i, j] = self.storage[offset]
                offset += 1
        return mat

    @classmethod
    def storage_dim(cls) -> int:
        return cls.SHAPE * (cls.SHAPE + 1) // 2


class UpperTriangularMatrix(_SpecialSquareMatrix):
    """Upper triangular matrix, including the diagonal"""

    def __init__(self, mat: sf.Matrix):
        self.check_size(mat)
        self.storage = []
        for i in range(self.SHAPE):
            for j in range(i, self.SHAPE):
                if i != j and mat[j, i] != 0:
                    raise ValueError("Matrix is not upper triangular")
                self.storage.append(mat[i, j])
        assert self.mat() == mat

    @classmethod
    def upper_storage_indices(cls) -> tuple[int, ...]:
        out = []
        for i in range(cls.SHAPE):
            for j in range(i, cls.SHAPE):
                out.append(i + j * cls.SHAPE)
        return tuple(out)

    def mat(self) -> sf.Matrix:
        mat = sf.Matrix(self.SHAPE, self.SHAPE)
        offset = 0
        for i in range(self.SHAPE):
            for j in range(i, self.SHAPE):
                mat[i, j] = self.storage[offset]
                offset += 1
        return mat

    @classmethod
    def storage_dim(cls) -> int:
        return cls.SHAPE * (cls.SHAPE + 1) // 2


class SymmetricMatrix(UpperTriangularMatrix):
    """Symmetrical matrix"""

    def __init__(self, mat: sf.Matrix):
        self.check_size(mat)
        self.storage = []
        for i in range(self.SHAPE):
            for j in range(i, self.SHAPE):
                if i != j and mat[i, j] != mat[j, i]:
                    raise ValueError("Matrix is not symmetric")
                self.storage.append(mat[i, j])
        assert self.mat() == mat

    def mat(self) -> sf.Matrix:
        mat = super().mat()
        for i in range(self.SHAPE):
            for j in range(i):
                mat[i, j] = mat[j, i]
        return mat


class LMat11(LowerTriangularMatrix):
    SHAPE = 1


class LMat22(LowerTriangularMatrix):
    SHAPE = 2


class LMat33(LowerTriangularMatrix):
    SHAPE = 3


class LMat44(LowerTriangularMatrix):
    SHAPE = 4


class LMat55(LowerTriangularMatrix):
    SHAPE = 5


class LMat66(LowerTriangularMatrix):
    SHAPE = 6


class LMat77(LowerTriangularMatrix):
    SHAPE = 7


class LMat88(LowerTriangularMatrix):
    SHAPE = 8


class LMat99(LowerTriangularMatrix):
    SHAPE = 9


class UMat11(UpperTriangularMatrix):
    SHAPE = 1


class UMat22(UpperTriangularMatrix):
    SHAPE = 2


class UMat33(UpperTriangularMatrix):
    SHAPE = 3


class UMat44(UpperTriangularMatrix):
    SHAPE = 4


class UMat55(UpperTriangularMatrix):
    SHAPE = 5


class UMat66(UpperTriangularMatrix):
    SHAPE = 6


class UMat77(UpperTriangularMatrix):
    SHAPE = 7


class UMat88(UpperTriangularMatrix):
    SHAPE = 8


class UMat99(UpperTriangularMatrix):
    SHAPE = 9


class SMat11(SymmetricMatrix):
    SHAPE = 1


class SMat22(SymmetricMatrix):
    SHAPE = 2


class SMat33(SymmetricMatrix):
    SHAPE = 3


class SMat44(SymmetricMatrix):
    SHAPE = 4


class SMat55(SymmetricMatrix):
    SHAPE = 5


class SMat66(SymmetricMatrix):
    SHAPE = 6


class SMat77(SymmetricMatrix):
    SHAPE = 7


class SMat88(SymmetricMatrix):
    SHAPE = 8


class SMat99(SymmetricMatrix):
    SHAPE = 9
