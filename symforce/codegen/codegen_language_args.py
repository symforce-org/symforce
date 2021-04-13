from symforce import types as T
from symforce.codegen import codegen_util


class LanguageArgs:
    """
    Base class for language-specific arguments for code generation, passed directly to the templates
    """

    pass


class CppLanguageArgs(LanguageArgs):
    """
    C++-specific arguments for code generation, passed directly to the templates
    """

    pass


class PythonLanguageArgs(LanguageArgs):
    """
    Python-specific arguments for code generation, passed directly to the templates

    Args:
        use_numba: Add the `@numba.njit` decorator to generated functions.  This will greatly
                   speed up functions by compiling them to machine code, but has large overhead
                   on the first call and some overhead on subsequent calls, so it should not be
                   used for small functions or functions that are only called a handfull of
                   times.
    """

    def __init__(self, use_numba: bool = False) -> None:
        self.use_numba = use_numba


_registry = {
    codegen_util.CodegenMode.CPP: CppLanguageArgs,
    codegen_util.CodegenMode.PYTHON2: PythonLanguageArgs,
    codegen_util.CodegenMode.PYTHON3: PythonLanguageArgs,
}


def for_codegen_mode(mode: codegen_util.CodegenMode) -> T.Type[LanguageArgs]:
    """
    Gets the LanguageArgs subclass for the given CodegenMode
    """
    return _registry[mode]
