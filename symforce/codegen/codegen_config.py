# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from sympy.printing.codeprinter import CodePrinter

from symforce import typing as T

CURRENT_DIR = Path(__file__).parent


@dataclass
class CodegenConfig:
    """
    Base class for backend-specific arguments for code generation.

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

    @classmethod
    @abstractmethod
    def backend_name(cls) -> str:
        """
        String name for the backend. This should match the directory name in codegen/backends
        and will be used to namespace by backend in generated code.
        """
        pass

    @classmethod
    @abstractmethod
    def template_dir(cls) -> Path:
        """
        Directory for jinja templates.
        """
        pass

    @abstractmethod
    def templates_to_render(self, generated_file_name: str) -> T.List[T.Tuple[str, str]]:
        """
        Given a single symbolic function's filename, provide one or more Jinja templates to
        render and the relative output paths where they should go.
        """
        pass

    @abstractmethod
    def printer(self) -> CodePrinter:
        """
        Return an instance of the code printer to use for this language.
        """
        pass

    # TODO(hayk): Move this into code printer.
    @staticmethod
    def format_data_accessor(prefix: str, index: int) -> str:
        """
        Format data for accessing a data array in code.
        """
        return f"{prefix}.data[{index}]"
