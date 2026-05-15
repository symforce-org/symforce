# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Smoke tests for SymForce-specific patches to symengine / symenginepy.

Each test validates one patch from the set carried on top of the upstream
symengine and symenginepy repositories.  The intent is to run this file on
every symengine version upgrade so that patch regressions are caught early.
"""

import typing as T

import symforce.symbolic as sf
from symforce.test_util import TestCase
from symforce.test_util import symengine_only


class SymEnginePatchesTest(TestCase):
    """Tests for each SymForce-specific symengine/symenginepy patch."""

    # ------------------------------------------------------------------
    # Patch: sign_no_zero / copysign_no_zero
    # ------------------------------------------------------------------

    @symengine_only
    def test_sign_no_zero_derivative(self) -> None:
        """SignNoZero(x) should have derivative 0 with respect to x."""
        x = sf.Symbol("x")
        expr = sf.sign_no_zero(x)
        deriv = sf.diff(expr, x)
        self.assertEqual(deriv, 0)

    @symengine_only
    def test_sign_no_zero_eval(self) -> None:
        """SignNoZero evaluates to -1 for negatives, +1 for positives, +1 for zero."""
        self.assertEqual(float(sf.sign_no_zero(sf.S(-3))), -1.0)
        self.assertEqual(float(sf.sign_no_zero(sf.S(7))), 1.0)
        # Key patch behavior: sign of zero is +1, not 0
        self.assertEqual(float(sf.sign_no_zero(sf.S(0))), 1.0)

    # ------------------------------------------------------------------
    # Patch: copysign_no_zero
    # ------------------------------------------------------------------

    @symengine_only
    def test_copysign_no_zero_basic(self) -> None:
        """CopysignNoZero(a, b) returns |a| * sign_no_zero(b)."""
        self.assertEqual(float(sf.copysign_no_zero(sf.S(5), sf.S(-2))), -5.0)
        self.assertEqual(float(sf.copysign_no_zero(sf.S(-5), sf.S(2))), 5.0)
        self.assertEqual(float(sf.copysign_no_zero(sf.S(-5), sf.S(-2))), -5.0)
        # When y is zero, returns positive x
        self.assertEqual(float(sf.copysign_no_zero(sf.S(5), sf.S(0))), 5.0)
        self.assertEqual(float(sf.copysign_no_zero(sf.S(-5), sf.S(0))), 5.0)

    @symengine_only
    def test_copysign_no_zero_derivative(self) -> None:
        """CopysignNoZero derivative w.r.t. second argument should be 0."""
        a, b = sf.symbols("a b")
        expr = sf.copysign_no_zero(a, b)
        deriv_b = sf.diff(expr, b)
        self.assertEqual(deriv_b, 0)

    # ------------------------------------------------------------------
    # Patch: 0-size matrix
    # ------------------------------------------------------------------

    @symengine_only
    def test_zero_size_matrix_iteration(self) -> None:
        """Matrix(0, 0) should be iterable (yielding no elements) without error."""
        m = sf.sympy.Matrix(0, 0, [])
        elements = list(m)  # type: ignore[call-overload]
        self.assertEqual(elements, [])

    @symengine_only
    def test_zero_size_matrix_str(self) -> None:
        """str() on a 0x0 matrix should not crash."""
        m = sf.sympy.Matrix(0, 0, [])
        result = str(m)
        self.assertIsInstance(result, str)

    # ------------------------------------------------------------------
    # Patch: DataBufferElement CSE
    # ------------------------------------------------------------------

    @symengine_only
    def test_databuffer_cse(self) -> None:
        """CSE should work on expressions containing DataBufferElement references."""
        buf = sf.DataBuffer("buf")
        x = sf.Symbol("x")
        # Build two expressions that share a common DataBufferElement sub-expression
        common = buf[x]
        e1 = common + x
        e2 = common * x
        replacements, reduced = sf.cse([e1, e2])
        # The DataBufferElement access should be factored out as a common sub-expression
        # or at minimum the call should not crash.
        self.assertIsInstance(replacements, list)
        self.assertIsInstance(reduced, list)
        self.assertEqual(len(reduced), 2)

    # ------------------------------------------------------------------
    # Patch: CSE custom symbol generator
    # ------------------------------------------------------------------

    @symengine_only
    def test_cse_custom_symbols(self) -> None:
        """cse() with a custom symbol generator should use those symbols as temporaries."""
        x, y = sf.symbols("x y")
        expr = (x + y) ** 2 + sf.sqrt(x + y)

        def test_symbols() -> T.Iterator[sf.Symbol]:
            idx = 0
            while True:
                yield sf.Symbol(f"test{idx}")
                idx += 1

        replacements, reduced = sf.cse([expr], symbols=test_symbols())
        self.assertEqual(len(replacements), 1)
        self.assertEqual(replacements[0][0], sf.Symbol("test0"))
        self.assertEqual(replacements[0][1], x + y)
        self.assertEqual(reduced, [sf.Symbol("test0") ** 2 + sf.sqrt(sf.Symbol("test0"))])

    # ------------------------------------------------------------------
    # Patch: count_ops improvements
    # ------------------------------------------------------------------

    @symengine_only
    def test_count_ops_inverse(self) -> None:
        """count_ops should count x**(-1) (division) as one operation."""
        x = sf.Symbol("x")
        self.assertEqual(sf.count_ops(x ** (-1)), 1)

    @symengine_only
    def test_count_ops_negation(self) -> None:
        """count_ops should count -x as one operation."""
        x = sf.Symbol("x")
        self.assertEqual(sf.count_ops(-x), 1)

    # ------------------------------------------------------------------
    # Patch: Hot-reload guard
    # ------------------------------------------------------------------

    @symengine_only
    def test_hot_reload_guard(self) -> None:
        """Importing symengine a second time should not crash."""
        import importlib

        import symengine

        # Force a re-execution of the module init code
        importlib.reload(symengine)
        # If we get here, the hot-reload guard worked
        self.assertTrue(hasattr(symengine, "Symbol"))

    # ------------------------------------------------------------------
    # Patch: Ceiling infinite recursion guard (ceiling half of b14b7e07f9d6;
    # floor half upstreamed in v0.10.1)
    # ------------------------------------------------------------------

    @symengine_only
    def test_ceiling_no_infinite_recursion(self) -> None:
        """
        ceiling() on an Add with zero integer coefficient must terminate.

        Without the is_zero() guard in Ceiling::create, the recursive branch
        wraps the same Add back up without shrinking it, causing a hang.
        """
        import signal

        def handler(signum: int, frame: T.Any) -> None:
            raise TimeoutError("ceiling() did not terminate within 5s")

        x = sf.Symbol("x")
        expr = sf.S(0) + x  # Add with coef=0
        old = signal.signal(signal.SIGALRM, handler)
        try:
            signal.alarm(5)
            result = sf.ceiling(expr)
            signal.alarm(0)
            self.assertIsNotNone(result)
        finally:
            signal.signal(signal.SIGALRM, old)

    # ------------------------------------------------------------------
    # Patch: Matrix __setitem__ with np.float64 (b2907c41e52e)
    # ------------------------------------------------------------------

    @symengine_only
    def test_matrix_setitem_np_float64(self) -> None:
        """
        Matrix.__setitem__ must accept np.float64 scalar values.

        np.float64 raises IndexError (not TypeError) when indexed; the patch
        adds IndexError to the exception tuple so scalar assignment works.
        """
        import numpy as np

        m = sf.sympy.Matrix(2, 2, [0, 0, 0, 0])
        m[0, 0] = np.float64(1.5)  # would raise IndexError without the patch
        self.assertAlmostEqual(float(m[0, 0]), 1.5)
        m[1, 1] = np.float64(-2.25)
        self.assertAlmostEqual(float(m[1, 1]), -2.25)

    # ------------------------------------------------------------------
    # Patch: DataBuffer not iterable (182b503946f5)
    # ------------------------------------------------------------------

    @symengine_only
    def test_databuffer_not_iterable(self) -> None:
        """
        iter(DataBuffer) must raise TypeError.

        DataBuffer has undefined size; without the explicit __iter__ guard,
        iteration would silently loop forever via __getitem__[0], [1], [2]...
        """
        buf = sf.DataBuffer("buf")
        with self.assertRaises(TypeError):
            list(buf)
        with self.assertRaises(TypeError):
            iter(buf)

    # ------------------------------------------------------------------
    # Patch: __EVAL_ON_SYMPY__ module attribute (7484fc4cd277 canary)
    # ------------------------------------------------------------------

    @symengine_only
    def test_eval_on_sympy_flag_exists(self) -> None:
        """
        symengine_wrapper module must expose __EVAL_ON_SYMPY__ attribute.

        The set_eval_on_sympify() API in symforce sets this attribute; it
        must exist for the API to work.
        """
        import symengine.lib.symengine_wrapper as wrapper

        self.assertTrue(hasattr(wrapper, "__EVAL_ON_SYMPY__"))


if __name__ == "__main__":
    TestCase.main()
