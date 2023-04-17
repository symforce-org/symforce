# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Helpers for interactive use in a Jupyter notebook with an IPython kernel.
"""
import warnings

import IPython
import matplotlib
import pygments

# NOTE(aaron): This is currently nice-to-have, otherwise every time we display something LaTeX we
# get this warning.  It's fixed in IPython master (https://github.com/ipython/ipython/pull/12889),
# so once that fix is in a release this can be removed
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

import sympy as sympy_py

sympy_py.init_printing()

import symforce
import symforce.symbolic as sf
from symforce import typing as T


def display(*args: T.Any) -> None:
    """
    Display the given expressions in latex, or print if not an expression.
    """
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
        IPython.display.display(sympy_py.S(*converted_args, strict=True))
    except (sympy_py.SympifyError, AttributeError, TypeError):
        IPython.display.display(*args)


def display_code(code: str, language: str) -> None:
    """
    Display code with syntax highlighting.

    Args:
        code (str): Source code
        language (str): {python, c++, anything supported by pygments}
    """
    # types-pygments doesn't have the type for this
    lexer = T.cast(T.Any, pygments).lexers.get_lexer_by_name(language)

    # And sometimes not this either
    formatter = T.cast(T.Any, pygments).formatters.HtmlFormatter(  # pylint: disable=no-member
        noclasses=True
    )
    html = pygments.highlight(code, lexer, formatter)

    IPython.display.display(IPython.display.HTML(html))


def display_code_file(path: T.Openable, language: str) -> None:
    """
    Display code from a file path with syntax highlighting.

    Args:
        path (T.Openable): Path to source file
        language (str): {python, c++, anything supported by pygments}
    """
    with open(path) as f:
        code = f.read()

    display_code(code, language)


def print_expression_tree(expr: sf.Expr) -> None:
    """
    Print a SymPy expression tree, ignoring node attributes
    """
    from sympy.printing.tree import tree

    unfiltered_tree = tree(expr).split("\n")
    filtered_tree = "\n".join(v for i, v in enumerate(unfiltered_tree) if "+-" in v or i == 0)
    print(filtered_tree)
