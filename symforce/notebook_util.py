# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Helpers for interactive use in a Jupyter notebook with an IPython kernel.
"""

import IPython
import sympy as sympy_py

sympy_py.init_printing()

import symforce
import symforce.symbolic as sf
from symforce import typing as T

if symforce.get_symbolic_api() == "symengine":
    sf.sympy.init_printing()


def display(*args: T.Any) -> None:
    """
    Display the given expressions in latex, or print if not an expression.
    """
    # TODO(aaron): This should all be unnecessary on new symengine.  The problem is that our version
    # of symengine does not define `DenseMatrixBase._repr_latex_`, so we need to convert symengine
    # matrices to sympy

    if symforce.get_symbolic_api() == "sympy":
        IPython.display.display(*args)
        return

    converted_args = []
    for arg in args:
        if isinstance(arg, sf.Matrix):
            converted_args.append(arg.mat)
        else:
            converted_args.append(arg)

    try:
        IPython.display.display(
            *[sympy_py.S(converted_arg, strict=True) for converted_arg in converted_args]
        )
    except (sympy_py.SympifyError, AttributeError, TypeError):
        IPython.display.display(*args)


def display_code(code: str, language: T.Optional[str] = None) -> None:
    """
    Display code with syntax highlighting.

    Args:
        code: Source code
        language: {python, c++, anything supported by pygments}
    """
    IPython.display.display(IPython.display.Code(code, language=language))


def display_code_file(path: T.Openable, language: str) -> None:
    """
    Display code from a file path with syntax highlighting.

    Args:
        path: Path to source file
        language: {python, c++, anything supported by pygments}
    """
    with open(path) as f:
        code = f.read()

    display_code(code, language)


def print_expression_tree(expr: sf.Expr, assumptions: bool = False) -> None:
    """
    Print a SymPy expression tree, ignoring node attributes

    Args:
        expr: The expression to print
        assumptions: Whether to include assumption information for nodes.  See
            ``sympy.printing.tree`` for more information.
    """
    from sympy.printing.tree import tree

    unfiltered_tree = tree(expr, assumptions=assumptions).split("\n")
    filtered_tree = "\n".join(v for i, v in enumerate(unfiltered_tree) if "+-" in v or i == 0)
    print(filtered_tree)
