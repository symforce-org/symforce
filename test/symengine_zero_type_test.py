# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Zero-multiplication type semantics for symengine.

Integer-zero short-circuits to the Zero singleton; RealDouble(0.0) does not
short-circuit (float 0.0 * x isn't always 0.0 for NaN/Inf). These tests pin
down both behaviors so a future symengine upgrade can't silently change them.
"""

import symforce

symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from symforce import codegen
from symforce.test_util import TestCase
from symforce.test_util import symengine_only


class SymEngineZeroTypeTest(TestCase):
    """Zero-multiplication result types and C++ codegen output."""

    @symengine_only
    def test_integer_zero_times_symbol_is_zero_singleton(self) -> None:
        """0 * Symbol collapses to the Zero singleton."""
        x = sf.Symbol("x")
        result = 0 * x
        self.assertEqual(type(result).__name__, "Zero")
        self.assertEqual(repr(result), "0")

    @symengine_only
    def test_integer_zero_times_realdouble_is_zero_singleton(self) -> None:
        """Integer(0) * RealDouble(3.14) collapses to the Zero singleton (integer-zero short-circuits)."""
        result = sf.S(0) * sf.Float(3.14)
        self.assertEqual(type(result).__name__, "Zero")
        self.assertEqual(repr(result), "0")

    @symengine_only
    def test_realdouble_zero_times_integer_stays_realdouble(self) -> None:
        """RealDouble(0.0) * Integer(5) stays as RealDouble(0.0) (float zero does not short-circuit)."""
        result = sf.Float(0.0) * sf.S(5)
        self.assertEqual(type(result).__name__, "RealDouble")
        self.assertEqual(repr(result), "0.0")

    @symengine_only
    def test_zero_multiplication_cpp_codegen(self) -> None:
        """C++ printer renders integer zeros as '0' and RealDouble zero as 'Scalar(0.0)'."""
        x = sf.Symbol("x")
        printer = codegen.CppConfig().printer()

        self.assertEqual(printer.doprint(0 * x), "0")
        self.assertEqual(printer.doprint(sf.S(0) * sf.Float(3.14)), "0")
        self.assertEqual(printer.doprint(sf.Float(0.0) * sf.S(5)), "Scalar(0.0)")


if __name__ == "__main__":
    TestCase.main()
