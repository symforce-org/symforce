# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import os

import symforce

from symforce import codegen
import symforce.symbolic as sf
from symforce.test_util import TestCase, sympy_only

SYMFORCE_DIR = os.path.dirname(os.path.dirname(__file__))
TEST_DATA_DIR = os.path.join(
    SYMFORCE_DIR, "test", "symforce_function_codegen_test_data", symforce.get_symbolic_api()
)


class SymforceCppCodePrinterTest(TestCase):
    """
    Test CppCodePrinter
    """

    def test_max_min(self) -> None:
        printer = codegen.CppConfig().printer()

        a = sf.Symbol("a")
        b = sf.Symbol("b")

        expr: sf.Expr = sf.Max(a ** 2, b ** 2)
        self.assertEqual(
            printer.doprint(expr),
            "std::max<Scalar>(std::pow(a, Scalar(2)), std::pow(b, Scalar(2)))",
        )

        expr = sf.Min(a ** 2, b ** 2)
        self.assertEqual(
            printer.doprint(expr),
            "std::min<Scalar>(std::pow(a, Scalar(2)), std::pow(b, Scalar(2)))",
        )

    @sympy_only
    def test_heaviside(self) -> None:
        output_dir = self.make_output_dir("sf_heaviside_test_")

        def heaviside(x: sf.Symbol) -> sf.Expr:
            return sf.sympy.functions.special.delta_functions.Heaviside(x)

        heaviside_codegen = codegen.Codegen.function(func=heaviside, config=codegen.CppConfig())
        heaviside_codegen_data = heaviside_codegen.generate_function(
            output_dir=output_dir, namespace="cpp_code_printer_test"
        )

        # Compare to expected
        expected_code_file = os.path.join(TEST_DATA_DIR, "heaviside.h")
        output_function = heaviside_codegen_data.function_dir / "heaviside.h"
        self.compare_or_update_file(expected_code_file, output_function)

    def test_custom_function_replacement(self) -> None:
        output_dir = self.make_output_dir("sf_custom_function_replacement_test_")

        # Simple test expression with one function that should be overwritten
        def test_expression(x: sf.Symbol) -> sf.Expr:
            return sf.sin(x) + sf.cos(x)

        codegen_config = codegen.CppConfig(
            override_methods={sf.sympy.sin: "fast_math::sin"},
            extra_imports=["custom_function_replacement_header.h"],
        )
        codegen_function = codegen.Codegen.function(func=test_expression, config=codegen_config)

        codegen_data = codegen_function.generate_function(
            output_dir=output_dir, namespace="cpp_code_printer_test"
        )

        # Compare
        expected = os.path.join(TEST_DATA_DIR, "custom_function_replacement.h")
        output = codegen_data.function_dir / "test_expression.h"
        self.compare_or_update_file(expected, output)


if __name__ == "__main__":
    TestCase.main()
