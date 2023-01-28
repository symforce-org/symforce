# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from enum import Enum

import sympy
from sympy.codegen.ast import float32
from sympy.codegen.ast import float64
from sympy.codegen.ast import real
from sympy.printing.c import C11CodePrinter

from symforce import typing as T


class ScalarType(Enum):
    FLOAT = float32
    DOUBLE = float64


class CudaCodePrinter(C11CodePrinter):
    """
    SymForce code printer for CUDA. Based on the SymPy C printer.
    """

    def __init__(
        self,
        scalar_type: ScalarType,
        settings: T.Optional[T.Dict[str, T.Any]] = None,
        override_methods: T.Optional[T.Dict[sympy.Function, str]] = None,
    ) -> None:
        super().__init__(dict(settings or {}, type_aliases={real: scalar_type.value}))

        self.override_methods = override_methods or {}
        for (expr, name) in self.override_methods.items():
            self._set_override_methods(expr, name)

    def _set_override_methods(self, expr: sympy.Function, name: str) -> None:
        method_name = f"_print_{str(expr)}"

        def _print_expr(expr: sympy.Expr) -> str:
            expr_string = ", ".join(map(self._print, expr.args))
            return f"{name}({expr_string})"

        setattr(self, method_name, _print_expr)

    def _print_ImaginaryUnit(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(
            "You tried to print an expression that contains the imaginary unit `i`.  SymForce does "
            "not support complex numbers in CUDA"
        )

    # NOTE(brad): We type ignore the signature because mypy complains that it
    # does not match that of the sympy base class CodePrinter. This is because the base class
    # defines _print_Heaviside with: _print_Heaviside = None (see
    # https://github.com/sympy/sympy/blob/95f0228c033d27731f8707cdbb5bb672e500847d/sympy/printing/codeprinter.py#L446
    # ).
    # Despite this, our signature here matches the signatures of the sympy defined subclasses
    # of CodePrinter. I don't know of any other way to resolve this issue other than to
    # to type ignore.
    def _print_Heaviside(self, expr: sympy.Heaviside) -> str:  # type: ignore[override]
        """
        Heaviside is not supported by default in C++, so we add a version here.
        """
        return "{0}*(((({1}) >= 0) - (({1}) < 0)) + 1)".format(
            self._print_Float(sympy.S(0.5)), self._print(expr.args[0])
        )

    def _print_MatrixElement(self, expr: sympy.matrices.expressions.matexpr.MatrixElement) -> str:
        """
        default printer doesn't cast to int
        """
        return "{}[static_cast<size_t>({})]".format(
            expr.parent, self._print(expr.j + expr.i * expr.parent.shape[1])
        )
