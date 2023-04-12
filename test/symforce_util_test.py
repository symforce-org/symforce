# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_symbol()

import importlib
import unittest

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
