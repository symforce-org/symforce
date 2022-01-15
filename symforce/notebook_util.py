"""
Helpers for interactive use in a Jupyter notebook with an IPython kernel.
"""
import IPython
import matplotlib
import pygments
import warnings

# NOTE(aaron): This is currently nice-to-have, otherwise every time we display something LaTeX we
# get this warning.  It's fixed in IPython master (https://github.com/ipython/ipython/pull/12889),
# so once that fix is in a release this can be removed
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

import sympy as sympy_py

sympy_py.init_printing()

import symforce
from symforce import sympy as sm
from symforce import typing as T


def display(*args: T.List) -> None:
    """
    Display the given expressions in latex, or print if not an expression.
    """
    if symforce.get_backend() == "sympy":
        IPython.display.display(*args)
        return

    try:
        IPython.display.display(sympy_py.S(*args, strict=True))
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


def display_code_file(path: str, language: str) -> None:
    """
    Display code from a file path with syntax highlighting.

    Args:
        path (str): Path to source file
        language (str): {python, c++, anything supported by pygments}
    """
    with open(path) as f:
        code = f.read()

    display_code(code, language)


def print_expression_tree(expr: sm.Expr) -> None:
    """
    Print a SymPy expression tree, ignoring node attributes
    """
    from sympy.printing.tree import tree

    unfiltered_tree = tree(expr).split("\n")
    filtered_tree = "\n".join(v for i, v in enumerate(unfiltered_tree) if "+-" in v or i == 0)
    print(filtered_tree)
