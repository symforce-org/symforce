# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path

from sympy.printing.codeprinter import CodePrinter

from symforce import typing as T
from symforce.codegen.backends.typescript import typescript_code_printer
from symforce.codegen.codegen_config import CodegenConfig

CURRENT_DIR = Path(__file__).parent


@dataclass
class TypeScriptConfig(CodegenConfig):
    """
    Code generation config for the TypeScript backend.

    Args:
        doc_comment_line_prefix: Prefix applied to each line in a docstring
        line_length: Maximum allowed line length in docstrings; used for formatting docstrings.
        use_eigen_types: Use eigen_lcm types for vectors instead of arrays (not supported in TypeScript)
        autoformat: Run a code formatter on the generated code
        custom_preamble: An optional string to be prepended on the front of the rendered template
        cse_optimizations: Optimizations argument to pass to :func:`sf.cse <symforce.symbolic.cse>`
        zero_epsilon_behavior: What should codegen do if a default epsilon is not set?
        normalize_results: Should function outputs be explicitly projected onto the manifold before
                           returning?
    """

    doc_comment_line_prefix: str = "//"
    line_length: int = 100
    use_eigen_types: bool = False  # Not supported in TypeScript

    @classmethod
    def backend_name(cls) -> str:
        return "typescript"

    @classmethod
    def template_dir(cls) -> Path:
        return CURRENT_DIR / "templates"

    @staticmethod
    def templates_to_render(generated_file_name: str) -> T.List[T.Tuple[str, str]]:
        return [("function/FUNCTION.ts.jinja", f"{generated_file_name}.ts")]

    @staticmethod
    def printer() -> CodePrinter:
        return typescript_code_printer.TypeScriptCodePrinter()

    @staticmethod
    def format_matrix_accessor(key: str, i: int, j: int, *, shape: T.Tuple[int, int]) -> str:
        """
        Format accessor for matrix types.

        Assumes matrices are stored as flat arrays in row-major order.
        """
        TypeScriptConfig._assert_indices_in_bounds(i, j, shape)
        if shape[1] == 1:
            # Column vector
            return f"{key}[{i}]"
        if shape[0] == 1:
            # Row vector
            return f"{key}[{j}]"
        return f"{key}[{i}][{j}]"

    @staticmethod
    def format_eigen_lcm_accessor(key: str, i: int) -> str:
        """
        Format accessor for eigen_lcm types.
        """
        raise NotImplementedError("TypeScript does not support eigen_lcm")
