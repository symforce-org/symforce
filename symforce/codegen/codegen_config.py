# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

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
    """

    doc_comment_line_prefix: str
    line_length: int
    use_eigen_types: bool
    autoformat: bool = True
    cse_optimizations: T.Optional[
        T.Union[T.Literal["basic"], T.Sequence[T.Tuple[T.Callable, T.Callable]]]
    ] = None


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
        use_explicit_template_instantiation: Explicity instantiate templated functions in a `.cc`
            file so that generated function can be compiled in its own translation unit. Useful
            for large functions which take a long time to compile.
    """

    doc_comment_line_prefix: str = " * "
    line_length: int = 100
    use_eigen_types: bool = True
    support_complex: bool = False
    force_no_inline: bool = False
    zero_initialization_sparsity_threshold: float = 0.5
    use_explicit_template_instantiation: bool = False


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
    """

    doc_comment_line_prefix: str = ""
    line_length: int = 100
    use_eigen_types: bool = True
    use_numba: bool = False
