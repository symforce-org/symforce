# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path

from sympy.printing.codeprinter import CodePrinter

from symforce import typing as T
from symforce.codegen.backends.python import python_code_printer
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
        custom_preamble: An optional string to be prepended on the front of the rendered template
        cse_optimizations: Optimizations argument to pass to sf.cse
        zero_epsilon_behavior: What should codegen do if a default epsilon is not set?
        use_numba: Add the `@numba.njit` decorator to generated functions.  This will greatly
                   speed up functions by compiling them to machine code, but has large overhead
                   on the first call and some overhead on subsequent calls, so it should not be
                   used for small functions or functions that are only called a handfull of
                   times.
        reshape_vectors: Allow rank 1 ndarrays to be passed in for row and column vectors by
                         automatically reshaping the input.
        return_2d_vectors: Return all matrices as 2d ndarrays if True. If False and a matrix has
                           either only 1 row or only 1 column, return as a 1d ndarray.
    """

    doc_comment_line_prefix: str = ""
    line_length: int = 100
    use_eigen_types: bool = True
    use_numba: bool = False
    reshape_vectors: bool = True
    return_2d_vectors: bool = False

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

    def printer(self) -> CodePrinter:
        return python_code_printer.PythonCodePrinter()

    def format_matrix_accessor(self, key: str, i: int, j: int, *, shape: T.Tuple[int, int]) -> str:
        PythonConfig._assert_indices_in_bounds(i, j, shape)
        return f"{key}[{i}, {j}]"

    @staticmethod
    def format_eigen_lcm_accessor(key: str, i: int) -> str:
        """
        Format accessor for eigen_lcm types.
        """
        return f"{key}.data[{i}]"
