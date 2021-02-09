import logging
import os

import symforce

from symforce import logger
from symforce import python_util
from symforce import types as T
from symforce.test_util import TestCase, requires_sympy
from symforce.values import Values

from symforce.codegen import codegen_util, Codegen, CodegenMode
from symforce import sympy as sm

SYMFORCE_DIR = os.path.dirname(os.path.dirname(__file__))
TEST_DATA_DIR = os.path.join(
    SYMFORCE_DIR, "test", "symforce_function_codegen_test_data", symforce.get_backend()
)


class SymforceCppCodePrinterTest(TestCase):
    """
    Test CppCodePrinter
    """

    def test_max_min(self) -> None:
        printer = codegen_util.get_code_printer(CodegenMode.CPP)

        a = sm.Symbol("a")
        b = sm.Symbol("b")

        expr = sm.Max(a ** 2, b ** 2)
        self.assertEqual(printer.doprint(expr), "std::max<Scalar>((a * a), (b * b))")

        expr = sm.Min(a ** 2, b ** 2)
        self.assertEqual(printer.doprint(expr), "std::min<Scalar>((a * a), (b * b))")

    @requires_sympy
    def test_heaviside(self) -> None:
        def f(x: sm.Symbol) -> sm.Symbol:
            return sm.functions.special.delta_functions.Heaviside(x)

        heaviside_codegen = Codegen.function(name="Heaviside", func=f, mode=CodegenMode.CPP)
        heaviside_codegen_data = heaviside_codegen.generate_function(
            namespace="cpp_code_printer_test"
        )

        # Compare to expected
        expected_code_file = os.path.join(TEST_DATA_DIR, "heaviside.h")
        output_function = os.path.join(heaviside_codegen_data["cpp_function_dir"], "heaviside.h")
        self.compare_or_update_file(expected_code_file, output_function)

        # Clean up
        if logger.level != logging.DEBUG:
            python_util.remove_if_exists(heaviside_codegen_data["output_dir"])


if __name__ == "__main__":
    TestCase.main()
