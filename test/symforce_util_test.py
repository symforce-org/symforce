# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_symbol()

import importlib
import unittest
from dataclasses import dataclass

import numpy as np

import sym
import symforce.symbolic as sf
from symforce import typing as T
from symforce import util
from symforce.test_util import TestCase


class SymforceUtilTest(TestCase):
    def test_symbolic_eval(self) -> None:
        def f(x: T.Scalar, y: sf.V1, z: sf.V2, w: sf.M22, r: sf.Rot3) -> T.Scalar:
            return (x, y, z, w, r)

        x, y, z, w, r = util.symbolic_eval(f)
        self.assertIsInstance(x, sf.Symbol)
        self.assertIsInstance(y, sf.V1)
        self.assertIsInstance(z, sf.V2)
        self.assertIsInstance(w, sf.M22)
        self.assertIsInstance(r, sf.Rot3)

    def test_lambdify(self) -> None:
        def f(x: T.Scalar, y: sf.V1, z: sf.V2, w: sf.M22, r: sf.Rot3) -> T.Scalar:
            return (x, y, z, w, r)

        numeric_f = util.lambdify(f)
        x, y, z, w, r = numeric_f(0.0, np.zeros((1,)), np.zeros((2,)), np.zeros((2, 2)), sym.Rot3())
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(z, np.ndarray)
        self.assertIsInstance(w, np.ndarray)
        self.assertIsInstance(r, sym.Rot3)

    def test_lambdify_lcmtypes(self) -> None:
        @dataclass
        class TestType:
            x: sf.Scalar
            y: sf.V2

        @dataclass
        class TestTypeContainer:
            inner: TestType

        def f(a: sf.Scalar, b: TestTypeContainer) -> TestType:
            return TestType(x=a + b.inner.x, y=b.inner.y)

        # Test that lambdify works with lcmtypes generated from Values/dataclasses
        numeric_f = util.lambdify(f)

        # Result can be invoked with any type with the same structure as TestTypeContainer.
        # Technically the expected argument type is the generated lcmtype
        result = numeric_f(1.0, TestTypeContainer(inner=TestType(x=2.0, y=np.array([3.0, 4.0]))))  # type: ignore[arg-type]

        # Return types of python functions (including lambdified functions) are currently the
        # storage of the result for lcmtypes, which is real dumb
        # self.assertIsInstance(result, TestType)
        self.assertEqual(result, [3.0, 3.0, 4.0])

    def test_lambdify_expr(self) -> None:
        x, y, z = sf.symbols("x y z")
        R = sf.Rot3.symbolic("R")
        expr0 = R * sf.V3(x, y, z)
        expr1 = x + y

        f0 = util.lambdify([x, y, z, R], expr0)
        f1 = util.lambdify([x, y, z, R], [expr0])
        f2 = util.lambdify([x, y, z, R], [expr0, expr1])

        result0 = f0(1.0, 2.0, 3.0, sym.Rot3.identity())
        result1 = f1(1.0, 2.0, 3.0, sym.Rot3.identity())
        result2, result3 = f2(1.0, 2.0, 3.0, sym.Rot3.identity())

        self.assertEqual(result0, np.array([1.0, 2.0, 3.0]))
        self.assertEqual(result1, np.array([1.0, 2.0, 3.0]))
        self.assertEqual(result2, np.array([1.0, 2.0, 3.0]))
        self.assertEqual(result3, 3.0)

    @unittest.skipIf(importlib.util.find_spec("numba") is None, "Requires numba")
    def test_numbify(self) -> None:
        import numba

        def f(x: T.Scalar, y: sf.V1, z: sf.V2, w: sf.M22) -> T.Scalar:
            return (x, y, z, w)

        numeric_f = util.numbify(f)
        x, y, z, w = numeric_f(0.0, np.zeros((1,)), np.zeros((2,)), np.zeros((2, 2)))
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(z, np.ndarray)
        self.assertIsInstance(w, np.ndarray)

        def f_bad(x: T.Scalar, y: sf.V1, z: sf.V2, w: sf.M22, r: sf.Rot3) -> T.Scalar:
            return (x, y, z, w, r)

        numeric_f = util.numbify(f_bad)
        with self.assertRaises(numba.core.errors.TypingError):
            numeric_f(0.0, np.zeros((1,)), np.zeros((2,)), np.zeros((2, 2)), sym.Rot3())


if __name__ == "__main__":
    SymforceUtilTest.main()
