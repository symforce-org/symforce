# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path

from sympy.printing.codeprinter import CodePrinter

from symforce import typing as T

CURRENT_DIR = Path(__file__).parent


class ZeroEpsilonBehavior(Enum):
    """
    Options for what to do when attempting to generate code with the default epsilon set to 0
    """

    FAIL = 0
    WARN = 1
    ALLOW = 2


# Default for new codegen configs - this lets you modify the default for all configs, e.g. for all
# codegen tests
DEFAULT_ZERO_EPSILON_BEHAVIOR = ZeroEpsilonBehavior.WARN


# TODO(hayk): Address this type ignore, which comes from having abstract methods on a dataclass.
@dataclass  # type: ignore
class CodegenConfig:
    """
    Base class for backend-specific arguments for code generation.

    Args:
        doc_comment_line_prefix: Prefix applied to each line in a docstring, e.g. " * " for C++
                                 block-style docstrings
        line_length: Maximum allowed line length in docstrings; used for formatting docstrings.
        use_eigen_types: Use eigen_lcm types for vectors instead of lists
        autoformat: Run a code formatter on the generated code
        cse_optimizations: Optimizations argument to pass to sf.cse
        zero_epsilon_behavior: What should codegen do if a default epsilon is not set?
    """

    doc_comment_line_prefix: str
    line_length: int
    use_eigen_types: bool
    autoformat: bool = True
    cse_optimizations: T.Optional[
        T.Union[T.Literal["basic"], T.Sequence[T.Tuple[T.Callable, T.Callable]]]
    ] = None
    zero_epsilon_behavior: ZeroEpsilonBehavior = field(
        default_factory=lambda: DEFAULT_ZERO_EPSILON_BEHAVIOR
    )

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

    @staticmethod
    def _assert_indices_in_bounds(row: int, col: int, shape: T.Tuple[int, int]) -> None:
        if row < 0 or shape[0] <= row:
            raise IndexError(f"Row index {row} is out of bounds (size {shape[0]})")
        if col < 0 or shape[1] <= col:
            raise IndexError(f"Column index {col} is out of bounds (size {shape[1]})")

    @abstractmethod
    def format_matrix_accessor(self, key: str, i: int, j: int, *, shape: T.Tuple[int, int]) -> str:
        """
        Format accessor for 2D matrices.
        Raises an index exception if either of the following is false:
            0 <= i < shape[0]
            0 <= j < shape[1]
        """
        pass

    @staticmethod
    @abstractmethod
    def format_eigen_lcm_accessor(key: str, i: int) -> str:
        """
        Format accessor for eigen_lcm types.
        """
        pass
