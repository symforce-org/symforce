# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import sympy
from sympy.printing.pycode import PythonCodePrinter as _PythonCodePrinter


class PythonCodePrinter(_PythonCodePrinter):
    """
    Symforce customized code printer for Python. Modifies the Sympy printing
    behavior for codegen compatibility and efficiency.
    """

    def _print_Rational(self, expr: sympy.Rational) -> str:
        """
        Customizations:
            * Decimal points for Python2 support, doesn't exist in some sympy versions.
        """
        return f"{expr.p}./{expr.q}."

    def _print_Max(self, expr: sympy.Max) -> str:
        """
        Max is not supported by default, so we add a version here.
        """
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        else:
            return "max({})".format(", ".join([self._print(arg) for arg in expr.args]))

    def _print_Min(self, expr: sympy.Min) -> str:
        """
        Min is not supported by default, so we add a version here.
        """
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        else:
            return "min({})".format(", ".join([self._print(arg) for arg in expr.args]))

    # NOTE(brad): We type ignore the signature because mypy complains that it
    # does not match that of the sympy base class CodePrinter. This is because the base class
    # defines _print_Heaviside with: _print_Heaviside = None (see
    # https://github.com/sympy/sympy/blob/95f0228c033d27731f8707cdbb5bb672e500847d/sympy/printing/codeprinter.py#L446
    # ).
    # Despite this, our signature here matches the signatures of the sympy defined subclasses
    # of CodePrinter. I don't know of any other way to resolve this issue other than to
    # to type ignore.
    def _print_Heaviside(self, expr: "sympy.Heaviside") -> str:  # type: ignore[override]
        """
        Heaviside is not supported by default, so we add a version here.
        """
        return f"(0.0 if ({self._print(expr.args[0])}) < 0 else 1.0)"

    def _print_MatrixElement(self, expr: sympy.matrices.expressions.matexpr.MatrixElement) -> str:
        """
        default printer doesn't cast to int
        """
        return "{}[int({})]".format(
            expr.parent, self._print(expr.j + expr.i * expr.parent.shape[1])
        )
