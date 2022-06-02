from sympy.printing.jscode import JavascriptCodePrinter as SympyJsCodePrinter

from symforce import typing as T

# Everything in this file is SymPy, not SymEngine (even when SymForce is on the SymEngine backend)
import sympy


class JavascriptCodePrinter(SympyJsCodePrinter):
    """
    Symforce customized code printer for Javascript. Modifies the Sympy printing
    behavior for codegen compatibility and efficiency.
    """

    def __init__(self, settings: T.Dict[str, T.Any] = None) -> None:
        settings = dict(settings or {},)
        super().__init__(settings)
