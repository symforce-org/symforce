# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

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
    """

    doc_comment_line_prefix: str
    line_length: int
    use_eigen_types: bool
    autoformat: bool = True
    cse_optimizations: T.Optional[
        T.Union[T.Literal["basic"], T.Sequence[T.Tuple[T.Callable, T.Callable]]]
    ] = None

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
    def printer(self) -> "sympy.CodePrinter":
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

    @staticmethod
    @abstractmethod
    def format_matrix_accessor(key: str, i: int, j: int = None) -> str:
        """
        Format accessor for 2D matrices. If j is None, it is a 1D vector type, which for some
        languages is accessed with 2D indices and in some with 1D.
        """
        pass
