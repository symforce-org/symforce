# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path

import sympy
from sympy.printing.codeprinter import CodePrinter

from symforce import typing as T
from symforce.codegen.backends.cpp import cpp_code_printer
from symforce.codegen.codegen_config import CodegenConfig

CURRENT_DIR = Path(__file__).parent


@dataclass
class CppConfig(CodegenConfig):
    """
    Code generation config for the C++ backend.

    Args:
        doc_comment_line_prefix: Prefix applied to each line in a docstring
        line_length: Maximum allowed line length in docstrings; used for formatting docstrings.
        use_eigen_types: Use eigen_lcm types for vectors instead of lists
        autoformat: Run a code formatter on the generated code
        cse_optimizations: Optimizations argument to pass to sf.cse
        zero_epsilon_behavior: What should codegen do if a default epsilon is not set?
        support_complex: Generate code that can work with std::complex or with regular float types
        force_no_inline: Mark generated functions as `__attribute__((noinline))`
        zero_initialization_sparsity_threshold: Threshold between 0 and 1 for the sparsity below
                                                which we'll initialize an output matrix to 0, so we
                                                don't have to generate a line to set each zero
                                                element to 0 individually
        explicit_template_instantiation_types: Explicity instantiates templated functions in a `.cc`
            file for each given type. This allows the generated function to be compiled in its own
            translation unit. Useful for large functions which take a long time to compile
        override_methods: Add special function overrides in dictionary with symforce function keys
            (e.g. sf.sin) and a string for the new method (e.g. fast_math::sin_lut), note that this bypasses
            the default namespace (so std:: won't be added in front automatically). Note that the keys here
            need to be sympy keys, not symengine (i.e sympy.sin NOT sf.sin with symengine backend). Symengine to
            sympy conversion does not work for Function types. Note that this function works in the code printer,
            and should only be used for replacing functions that compute the same thing but in a different way,
            e.g. replacing `sin` with `my_lib::sin`. It should _not_ be used for substituting a function
            with a different function, which will break derivatives and certain simplifications,
            e.g. you should not use this to replace `sin` with `cos` or `sin` with `my_lib::cos`
        extra_imports: Add extra imports to the file if you use custom overrides for some functions
            (i.e. add fast_math.h). Note that these are only added on a call to `generate_function`, i.e.
            you can't define custom functions in e.g. the geo package using this
    """

    doc_comment_line_prefix: str = " * "
    line_length: int = 100
    use_eigen_types: bool = True
    support_complex: bool = False
    force_no_inline: bool = False
    zero_initialization_sparsity_threshold: float = 0.5
    explicit_template_instantiation_types: T.Optional[T.Sequence[str]] = None
    override_methods: T.Optional[T.Dict[sympy.Function, str]] = None
    extra_imports: T.Optional[T.List[str]] = None

    @classmethod
    def backend_name(cls) -> str:
        return "cpp"

    @classmethod
    def template_dir(cls) -> Path:
        return CURRENT_DIR / "templates"

    def templates_to_render(self, generated_file_name: str) -> T.List[T.Tuple[str, str]]:
        # Generate code into a header (since the code is templated)
        templates = [("function/FUNCTION.h.jinja", f"{generated_file_name}.h")]

        # Generate a cc file only if we need explicit instantiation.
        if self.explicit_template_instantiation_types is not None:
            templates.append(("function/FUNCTION.cc.jinja", f"{generated_file_name}.cc"))

        return templates

    def printer(self) -> CodePrinter:
        kwargs: T.Mapping[str, T.Any] = {"override_methods": self.override_methods}

        if self.support_complex:
            return cpp_code_printer.ComplexCppCodePrinter(**kwargs)

        return cpp_code_printer.CppCodePrinter(**kwargs)

    @staticmethod
    def format_data_accessor(prefix: str, index: int) -> str:
        return f"{prefix}.Data()[{index}]"

    def format_matrix_accessor(self, key: str, i: int, j: int, *, shape: T.Tuple[int, int]) -> str:
        CppConfig._assert_indices_in_bounds(i, j, shape)
        return f"{key}({i}, {j})"

    @staticmethod
    def format_eigen_lcm_accessor(key: str, i: int) -> str:
        """
        Format accessor for eigen_lcm types.
        """
        return f"{key}.data()[{i}]"
