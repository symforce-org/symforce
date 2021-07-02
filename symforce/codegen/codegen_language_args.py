from dataclasses import dataclass

from symforce.codegen import codegen_util


@dataclass
class LanguageArgs:
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
class CppLanguageArgs(LanguageArgs):
    """
    C++-specific arguments for code generation

    Args:
        doc_comment_line_prefix: Prefix applied to each line in a docstring
        line_length: Maximum allowed line length in docstrings; used for formatting docstrings.
    """

    doc_comment_line_prefix: str = " * "
    line_length: int = 100


@dataclass
class PythonLanguageArgs(LanguageArgs):
    """
    Python-specific arguments for code generation

    Args:
        doc_comment_line_prefix: Prefix applied to each line in a docstring
        line_length: Maximum allowed line length in docstrings; used for formatting docstrings.
        use_numba: Add the `@numba.njit` decorator to generated functions.  This will greatly
                   speed up functions by compiling them to machine code, but has large overhead
                   on the first call and some overhead on subsequent calls, so it should not be
                   used for small functions or functions that are only called a handfull of
                   times.
    """

    doc_comment_line_prefix: str = ""
    line_length: int = 100
    use_numba: bool = False


_registry = {
    codegen_util.CodegenMode.CPP: CppLanguageArgs,
    codegen_util.CodegenMode.PYTHON2: PythonLanguageArgs,
    codegen_util.CodegenMode.PYTHON3: PythonLanguageArgs,
}


def for_codegen_mode(mode: codegen_util.CodegenMode) -> LanguageArgs:
    """
    Gets the LanguageArgs subclass for the given CodegenMode
    """
    return _registry[mode]()
