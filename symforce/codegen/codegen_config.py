from dataclasses import dataclass
from enum import Enum


@dataclass
class CodegenConfig:
    """
    Base class for language-specific arguments for code generation

    Args:
        doc_comment_line_prefix: Prefix applied to each line in a docstring, e.g. " * " for C++
                                 block-style docstrings
        line_length: Maximum allowed line length in docstrings; used for formatting docstrings.
    """

    doc_comment_line_prefix: str
    line_length: int


@dataclass
class CppConfig(CodegenConfig):
    """
    C++ Codegen configuration

    Args:
        doc_comment_line_prefix: Prefix applied to each line in a docstring
        line_length: Maximum allowed line length in docstrings; used for formatting docstrings.
    """

    doc_comment_line_prefix: str = " * "
    line_length: int = 100


@dataclass
class PythonConfig(CodegenConfig):
    """
    Python Codegen configuration

    Args:
        standard: Version of the Python language, either both 2 and 3 or just 3
        doc_comment_line_prefix: Prefix applied to each line in a docstring
        line_length: Maximum allowed line length in docstrings; used for formatting docstrings.
        use_numba: Add the `@numba.njit` decorator to generated functions.  This will greatly
                   speed up functions by compiling them to machine code, but has large overhead
                   on the first call and some overhead on subsequent calls, so it should not be
                   used for small functions or functions that are only called a handfull of
                   times.
    """

    class PythonVersion(Enum):
        PYTHON2AND3 = "python2"
        PYTHON3 = "python3"

    standard: PythonVersion = PythonVersion.PYTHON2AND3
    doc_comment_line_prefix: str = ""
    line_length: int = 100
    use_numba: bool = False
