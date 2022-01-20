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
        use_eigen_types: Use eigen_lcm types for vectors instead of lists
    """

    doc_comment_line_prefix: str
    line_length: int
    use_eigen_types: bool


@dataclass
class CppConfig(CodegenConfig):
    """
    C++ Codegen configuration

    Args:
        doc_comment_line_prefix: Prefix applied to each line in a docstring
        line_length: Maximum allowed line length in docstrings; used for formatting docstrings.
        use_eigen_types: Use eigen_lcm types for vectors instead of lists
        support_complex: Generate code that can work with std::complex or with regular float types
    """

    doc_comment_line_prefix: str = " * "
    line_length: int = 100
    use_eigen_types: bool = True
    support_complex: bool = False


@dataclass
class PythonConfig(CodegenConfig):
    """
    Python Codegen configuration

    Args:
        standard: Version of the Python language, either both 2 and 3 or just 3
        doc_comment_line_prefix: Prefix applied to each line in a docstring
        line_length: Maximum allowed line length in docstrings; used for formatting docstrings.
        use_eigen_types: Use eigen_lcm types for vectors instead of lists
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
    use_eigen_types: bool = True
    use_numba: bool = False
