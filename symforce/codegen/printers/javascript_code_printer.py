# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from sympy.printing.jscode import JavascriptCodePrinter as SympyJsCodePrinter

from symforce import typing as T


class JavascriptCodePrinter(SympyJsCodePrinter):
    """
    Symforce customized code printer for Javascript. Modifies the Sympy printing
    behavior for codegen compatibility and efficiency.
    """

    def __init__(self, settings: T.Dict[str, T.Any] = None) -> None:
        settings = dict(settings or {},)
        super().__init__(settings)
