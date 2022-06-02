# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass

from symforce import typing as T


@dataclass
class CodegenConfig:
    """
    Base class for language-specific arguments for code generation

    Args:
        doc_comment_line_prefix: Prefix applied to each line in a docstring, e.g. " * " for C++
                                 block-style docstrings
        line_length: Maximum allowed line length in docstrings; used for formatting docstrings.
        use_eigen_types: Use eigen_lcm types for vectors instead of lists
        autoformat: Run a code formatter on the generated code
        cse_optimizations: Optimizations argument to pass to sm.cse
        matrix_is_1d: Whether geo.Matrix symbols get formatted as 1D
    """

    doc_comment_line_prefix: str
    line_length: int
    use_eigen_types: bool
    autoformat: bool = True
    cse_optimizations: T.Optional[
        T.Union[T.Literal["basic"], T.Sequence[T.Tuple[T.Callable, T.Callable]]]
    ] = None
    # TODO(hayk): Remove this parameter (by making everything 2D?)
    matrix_is_1d: bool = False

    def printer(self) -> "sm.CodePrinter":
        """
        Return the code printer to use for this language.
        """
        raise NotImplementedError()

    # TODO(hayk): Move this into code printer.
    @staticmethod
    def format_data_accessor(prefix: str, index: int) -> str:
        """
        Format data for accessing a data array in code.
        """
        return f"{prefix}.data[{index}]"


@dataclass
class CppConfig(CodegenConfig):
    """
    C++ Codegen configuration

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
        from symforce.codegen.printers import cpp_code_printer
        if self.support_complex:
            return cpp_code_printer.ComplexCppCodePrinter()
        else:
            return cpp_code_printer.CppCodePrinter()

    @staticmethod
    def format_data_accessor(prefix: str, index: int) -> str:
        return f"{prefix}.Data()[{index}]"

@dataclass
class PythonConfig(CodegenConfig):
    """
    Python Codegen configuration

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

    def printer(self) -> "sm.CodePrinter":
        from symforce.codegen.printers import python_code_printer
        return python_code_printer.PythonCodePrinter()

