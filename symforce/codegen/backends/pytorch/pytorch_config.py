# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path

from sympy.printing.codeprinter import CodePrinter

from symforce import typing as T
from symforce.codegen.backends.pytorch import pytorch_code_printer
from symforce.codegen.codegen_config import CodegenConfig

CURRENT_DIR = Path(__file__).parent


@dataclass
class PyTorchConfig(CodegenConfig):
    """
    Code generation config for the PyTorch backend.

    Args:
        doc_comment_line_prefix: Prefix applied to each line in a docstring
        line_length: Maximum allowed line length in docstrings; used for formatting docstrings.
        use_eigen_types: Use eigen_lcm types for vectors instead of lists
        autoformat: Run a code formatter on the generated code
        custom_preamble: An optional string to be prepended on the front of the rendered template
        cse_optimizations: Optimizations argument to pass to sf.cse
        zero_epsilon_behavior: What should codegen do if a default epsilon is not set?
    """

    doc_comment_line_prefix: str = ""
    line_length: int = 100
    use_eigen_types: bool = False

    @classmethod
    def backend_name(cls) -> str:
        return "pytorch"

    @classmethod
    def template_dir(cls) -> Path:
        return CURRENT_DIR / "templates"

    def templates_to_render(self, generated_file_name: str) -> T.List[T.Tuple[str, str]]:
        return [
            ("function/FUNCTION.py.jinja", f"{generated_file_name}.py"),
            ("function/__init__.py.jinja", "__init__.py"),
        ]

    def printer(self) -> CodePrinter:
        return pytorch_code_printer.PyTorchCodePrinter()

    def format_matrix_accessor(self, key: str, i: int, j: int, *, shape: T.Tuple[int, int]) -> str:
        PyTorchConfig._assert_indices_in_bounds(i, j, shape)
        if (shape[0] == 1) ^ (shape[1] == 1):
            return f"{key}[..., {max(i, j)}]"
        elif shape[0] == 1 and shape[1] == 1:
            return key
        else:
            return f"{key}[..., {i}, {j}]"

    @staticmethod
    def format_eigen_lcm_accessor(key: str, i: int) -> str:
        """
        Format accessor for eigen_lcm types.
        """
        raise NotImplementedError("Can't pass eigen_lcm types to PyTorch functions")
