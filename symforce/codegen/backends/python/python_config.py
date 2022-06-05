# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from symforce import typing as T
from symforce.codegen.codegen_config import CodegenConfig


CURRENT_DIR = Path(__file__).parent


@dataclass
class PythonConfig(CodegenConfig):
    """
    Code generation config for the Python backend.

    Args:
        doc_comment_line_prefix: Prefix applied to each line in a docstring
        line_length: Maximum allowed line length in docstrings; used for formatting docstrings.
        use_eigen_types: Use eigen_lcm types for vectors instead of lists
        autoformat: Run a code formatter on the generated code
        cse_optimizations: Optimizations argument to pass to sm.cse
        use_numba: Add the `@numba.njit` decorator to generated functions.  This will greatly
                   speed up functions by compiling them to machine code, but has large overhead
                   on the first call and some overhead on subsequent calls, so it should not be
                   used for small functions or functions that are only called a handfull of
                   times.
        matrix_is_1D: geo.Matrix symbols get formatted as a 1D array
    """

    doc_comment_line_prefix: str = ""
    line_length: int = 100
    use_eigen_types: bool = True
    use_numba: bool = False
    matrix_is_1d: bool = True

    @classmethod
    def backend_name(cls) -> str:
        return "python"

    @classmethod
    def template_dir(cls) -> Path:
        return CURRENT_DIR / "templates"

    def templates_to_render(self, generated_file_name: str) -> T.List[T.Tuple[str, str]]:
        return [
            ("function/FUNCTION.py.jinja", f"{generated_file_name}.py"),
            ("function/__init__.py.jinja", "__init__.py"),
        ]

    def printer(self) -> "sympy.CodePrinter":
        from symforce.codegen.backends.python import python_code_printer

        return python_code_printer.PythonCodePrinter()
