# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import sympy
from sympy.printing.codeprinter import CodePrinter

from symforce import typing as T
from symforce.codegen.backends.rust import rust_code_printer
from symforce.codegen.codegen_config import CodegenConfig

CURRENT_DIR = Path(__file__).parent


@dataclass
class RustConfig(CodegenConfig):
    """
    Code generation config for the Rust backend.

    Args:
        doc_comment_line_prefix: Prefix applied to each line in a docstring
        line_length: Maximum allowed line length in docstrings; used for formatting docstrings.
        use_eigen_types: Use eigen_lcm types for vectors instead of lists
        autoformat: Run a code formatter on the generated code
        custom_preamble: An optional string to be prepended on the front of the rendered template
        cse_optimizations: Optimizations argument to pass to :func:`sf.cse <symforce.symbolic.cse>`
        zero_epsilon_behavior: What should codegen do if a default epsilon is not set?
        normalize_results: Should function outputs be explicitly projected onto the manifold before
                           returning?
        override_methods: Add special function overrides in dictionary with symforce function keys
            (e.g. ``sympy.sin``) and a string for the new method (e.g. ``"fast_math::sin_lut"``),
            note that this bypasses the default namespace (so std:: won't be added in front
            automatically). Note that the keys here need to be sympy keys, not symengine (e.g.
            ``sympy.sin`` NOT ``sf.sin`` with the symengine backend). SymEngine to SymPy conversion
            does not work for ``Function`` types. Note that this function works in the code printer,
            and should only be used for replacing functions that compute the same thing but in a
            different way, e.g. replacing ``sin`` with ``my_lib::sin``. It should `not` be used for
            substituting a function with a different function, which will break derivatives and
            certain simplifications, e.g. you should not use this to replace ``sin`` with ``cos``
            or ``sin`` with ``my_lib::cos``.
        extra_imports: Add extra imports to the file if you use custom overrides for some functions
            (e.g. add fast_math.h). Note that these are only added on a call to
            :meth:`generate_function <symforce.codegen.codegen.Codegen.generate_function>`, i.e.
            you can't define custom functions in e.g. the geo package using this
        scalar_type: The scalar type to use (float or double)
        inline: Whether to generate inline functions (in the header) or a separate ``.cu`` file
                containing the function definition
    """

    doc_comment_line_prefix: str = "///"
    line_length: int = 100
    override_methods: T.Optional[T.Dict[sympy.Function, str]] = None
    extra_imports: T.Optional[T.List[str]] = None
    scalar_type: rust_code_printer.ScalarType = rust_code_printer.ScalarType.FLOAT
    use_eigen_types: bool = False

    @classmethod
    def backend_name(cls) -> str:
        return "rust"

    @classmethod
    def template_dir(cls) -> Path:
        return CURRENT_DIR / "templates"

    def templates_to_render(self, generated_file_name: str) -> T.List[T.Tuple[str, str]]:
        return [("function/FUNCTION.rs.jinja", f"{generated_file_name}.rs")]

    def printer(self) -> CodePrinter:
        kwargs: T.Mapping[str, T.Any] = {"override_methods": self.override_methods}
        return rust_code_printer.RustCodePrinter(scalar_type=self.scalar_type, **kwargs)

    def format_matrix_accessor(self, key: str, i: int, j: int, *, shape: T.Tuple[int, int]) -> str:
        """
        Format accessor for matrix types.

        Assumes matrices are row-major.
        """
        RustConfig._assert_indices_in_bounds(i, j, shape)
        if shape[1] == 1:
            return f"{key}[{i}]"
        if shape[0] == 1:
            return f"{key}[{j}]"
        return f"{key}[({i}, {j})]"

    @staticmethod
    def format_eigen_lcm_accessor(key: str, i: int) -> str:
        """
        Format accessor for eigen_lcm types.
        """
        raise NotImplementedError("Rust does not support eigen_lcm")
