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
class JavascriptConfig(CodegenConfig):
    """
    Code generation config for the javascript backend.

    Args:
        doc_comment_line_prefix: Prefix applied to each line in a docstring
        line_length: Maximum allowed line length in docstrings; used for formatting docstrings.
        use_eigen_types: Use eigen_lcm types for vectors instead of lists
        autoformat: Run a code formatter on the generated code
        matrix_is_1D: geo.Matrix symbols get formatted as a 1D array
    """

    doc_comment_line_prefix: str = " * "
    line_length: int = 100
    use_eigen_types: bool = True
    # NOTE(hayk): Add JS autoformatter
    autoformat: bool = False
    matrix_is_1d: bool = True

    @classmethod
    def backend_name(cls) -> str:
        return "javascript"

    @classmethod
    def template_dir(cls) -> Path:
        return CURRENT_DIR / "templates"

    def templates_to_render(self, generated_file_name: str) -> T.List[T.Tuple[str, str]]:
        return [
            ("function/FUNCTION.js.jinja", f"{generated_file_name}.js"),
        ]

    def printer(self) -> "sm.CodePrinter":
        from symforce.codegen.printers import javascript_code_printer

        return javascript_code_printer.JavascriptCodePrinter()
