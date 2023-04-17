# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from __future__ import annotations

import sympy

from symforce import typing as T


class DataBuffer(sympy.MatrixSymbol):
    """
    Custom class to make sympy's MatrixSymbol consistent with symengine, where we have a custom
    1-D Databuffer. We want to force Databuffers to be 1-D since otherwise CSE will (rightfully)
    treat each index as a separate expression.
    """

    # HACK(harrison): needed to get around the flast that DataBuffer needs to be called from
    # initialization.py
    __sympy_module__: T.Any = None

    def __new__(
        cls, name: str, n: T.Optional[T.Scalar] = None, m: T.Optional[T.Scalar] = None
    ) -> DataBuffer:
        if n is None:
            n = DataBuffer.__sympy_module__.Symbol(name + "_dim")
        if m is not None:
            assert m == 1, "DataBuffer is 1-D only!"
        instance = super(DataBuffer, cls).__new__(cls, name, n, DataBuffer.__sympy_module__.S(1))
        return instance

    def __getitem__(self, key: T.Any) -> sympy.matrices.expressions.matexpr.MatrixElement:
        return super().__getitem__((key, DataBuffer.__sympy_module__.S(0)))
