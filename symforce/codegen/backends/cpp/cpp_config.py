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
class CppConfig(CodegenConfig):
    """
    Code generation config for the C++ backend.

    Args:
        doc_comment_line_prefix: Prefix applied to each line in a docstring
        line_length: Maximum allowed line length in docstrings; used for formatting docstrings.
        use_eigen_types: Use eigen_lcm types for vectors instead of lists
        autoformat: Run a code formatter on the generated code
        cse_optimizations: Optimizations argument to pass to sm.cse
        support_complex: Generate code that can work with std::complex or with regular float types
        force_no_inline: Mark generated functions as `__attribute__((noinline))`
        zero_initialization_sparsity_threshold: Threshold between 0 and 1 for the sparsity below
                                                which we'll initialize an output matrix to 0, so we
                                                don't have to generate a line to set each zero
                                                element to 0 individually
        explicit_template_instantiation_types: Explicity instantiates templated functions in a `.cc`
            file for each given type. This allows the generated function to be compiled in its own
            translation unit. Useful for large functions which take a long time to compile.
    """

    doc_comment_line_prefix: str = " * "
    line_length: int = 100
    use_eigen_types: bool = True
    support_complex: bool = False
    force_no_inline: bool = False
    zero_initialization_sparsity_threshold: float = 0.5
    explicit_template_instantiation_types: T.Optional[T.Sequence[str]] = None

    def printer(self) -> "sm.CodePrinter":
        # NOTE(hayk): Is there any benefit to this being lazy?
        from symforce.codegen.backends.cpp import cpp_code_printer

        if self.support_complex:
            return cpp_code_printer.ComplexCppCodePrinter()
        else:
            return cpp_code_printer.CppCodePrinter()

    @classmethod
    def template_dir(cls) -> Path:
        return CURRENT_DIR / "templates"

    @staticmethod
    def format_data_accessor(prefix: str, index: int) -> str:
        return f"{prefix}.Data()[{index}]"
