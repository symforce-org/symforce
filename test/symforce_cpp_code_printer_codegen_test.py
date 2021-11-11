import os

import symforce

from symforce import typing as T
from symforce.test_util import TestCase, sympy_only
from symforce.values import Values

from symforce import codegen
from symforce.codegen import codegen_util
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
        printer = codegen_util.get_code_printer(codegen.CppConfig())

        a = sm.Symbol("a")
        b = sm.Symbol("b")

        expr = sm.Max(a ** 2, b ** 2)
        self.assertEqual(
            printer.doprint(expr),
            "std::max<Scalar>(std::pow(a, Scalar(2)), std::pow(b, Scalar(2)))",
        )

        expr = sm.Min(a ** 2, b ** 2)
        self.assertEqual(
            printer.doprint(expr),
            "std::min<Scalar>(std::pow(a, Scalar(2)), std::pow(b, Scalar(2)))",
        )

    @sympy_only
    def test_heaviside(self) -> None:
        output_dir = self.make_output_dir("sf_heaviside_test_")

        def heaviside(x: sm.Symbol) -> sm.Symbol:
            return sm.functions.special.delta_functions.Heaviside(x)

        heaviside_codegen = codegen.Codegen.function(func=heaviside, config=codegen.CppConfig())
        heaviside_codegen_data = heaviside_codegen.generate_function(
            output_dir=output_dir, namespace="cpp_code_printer_test",
        )

        # Compare to expected
        expected_code_file = os.path.join(TEST_DATA_DIR, "heaviside.h")
        output_function = os.path.join(heaviside_codegen_data["cpp_function_dir"], "heaviside.h")
        self.compare_or_update_file(expected_code_file, output_function)


if __name__ == "__main__":
    TestCase.main()
