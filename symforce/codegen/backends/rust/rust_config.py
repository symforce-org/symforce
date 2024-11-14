# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path

import sympy
from sympy.printing.codeprinter import CodePrinter

from symforce import typing as T
from symforce.codegen.backends.rust import rust_code_printer
from symforce.codegen.codegen_config import CodegenConfig

CURRENT_DIR = Path(__file__).parent


@dataclass
class RustConfig(CodegenConfig):
    """
    Code generation config for the Rust backend.

    Args:
        doc_comment_line_prefix: Prefix applied to each line in a docstring
        line_length: Maximum allowed line length in docstrings; used for formatting docstrings.
        scala_type: The scalar type to use (float or double)
        use_eigen_types: Use eigen_lcm types for vectors instead of lists
        render_template_config: Configuration for template rendering, see RenderTemplateConfig for
                                more information
        cse_optimizations: Optimizations argument to pass to :func:`sf.cse <symforce.symbolic.cse>`
        zero_epsilon_behavior: What should codegen do if a default epsilon is not set?
        normalize_results: Should function outputs be explicitly projected onto the manifold before
                           returning?
    """

    doc_comment_line_prefix: str = "///"
    line_length: int = 100
    scalar_type: rust_code_printer.ScalarType = rust_code_printer.ScalarType.DOUBLE
    use_eigen_types: bool = False

    @classmethod
    def backend_name(cls) -> str:
        return "rust"

    @classmethod
    def template_dir(cls) -> Path:
        return CURRENT_DIR / "templates"

    def templates_to_render(self, generated_file_name: str) -> T.List[T.Tuple[str, str]]:
        return [("function/FUNCTION.rs.jinja", f"{generated_file_name}.rs")]

    def printer(self) -> CodePrinter:
        kwargs: T.Mapping[str, T.Any] = {}
        return rust_code_printer.RustCodePrinter(scalar_type=self.scalar_type, **kwargs)

    def format_matrix_accessor(self, key: str, i: int, j: int, *, shape: T.Tuple[int, int]) -> str:
        """
        Format accessor for matrix types.

        Assumes matrices are row-major.
        """
        RustConfig._assert_indices_in_bounds(i, j, shape)
        if shape[1] == 1:
            return f"{key}[{i}]"
        if shape[0] == 1:
            return f"{key}[{j}]"
        return f"{key}[({i}, {j})]"

    @staticmethod
    def format_eigen_lcm_accessor(key: str, i: int) -> str:
        """
        Format accessor for eigen_lcm types.
        """
        raise NotImplementedError("Rust does not support eigen_lcm")
