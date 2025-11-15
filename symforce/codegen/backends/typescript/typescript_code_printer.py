# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import sympy
from sympy.printing.jscode import JavascriptCodePrinter

import symforce.internal.symbolic as sf


class TypeScriptCodePrinter(JavascriptCodePrinter):
    """
    SymForce code printer for TypeScript. Converts SymPy expressions to TypeScript code.
    """

    printmethod = "_typescript"
    language = "TypeScript"

    def _print_sign(self, expr: sympy.sign) -> str:
        return f"Math.sign({self._print(expr.args[0])})"

    def _print_SignNoZero(self, expr: sf.SymPySignNoZero) -> str:
        arg = self._print(expr.args[0])
        return f"(({arg}) >= 0.0 ? 1.0 : -1.0)"

    def _print_CopysignNoZero(self, expr: sf.SymPyCopysignNoZero) -> str:
        return f"(({self._print(expr.args[1])}) >= 0.0 ? Math.abs({self._print(expr.args[0])}) : -Math.abs({self._print(expr.args[0])}))"

    # NOTE(brad): We type ignore the signature because mypy complains that it
    # does not match that of the sympy base class CodePrinter. This is because the base class
    # defines _print_Heaviside with: _print_Heaviside = None (see
    # https://github.com/sympy/sympy/blob/95f0228c033d27731f8707cdbb5bb672e500847d/sympy/printing/codeprinter.py#L446
    # ).
    # Despite this, our signature here matches the signatures of the sympy defined subclasses
    # of CodePrinter. I don't know of any other way to resolve this issue other than to
    # to type ignore.
    def _print_Heaviside(self, expr: "sympy.Heaviside") -> str:  # type: ignore[override]
        return f"(({self._print(expr.args[0])}) < 0.0 ? 0.0 : 1.0)"

    def _print_MatrixElement(self, expr: sympy.matrices.expressions.matexpr.MatrixElement) -> str:
        """
        Print matrix element access
        """
        return "{}[Math.trunc({})]".format(
            expr.parent, self._print(expr.j + expr.i * expr.parent.shape[1])
        )
