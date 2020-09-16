from symforce import logger
from symforce import python_util
from symforce import types as T
from symforce.test_util import TestCase

from symforce.codegen import codegen_util, CodegenMode
from symforce import sympy as sm


class SymforceCppCodePrinterTest(TestCase):
    """
    Test CppCodePrinter
    """

    def test_max_min(self):
        # type: () -> None
        printer = codegen_util.get_code_printer(CodegenMode.CPP)

        a = sm.Symbol("a")
        b = sm.Symbol("b")

        expr = sm.Max(a ** 2, b ** 2)
        self.assertEqual(printer.doprint(expr), "std::max<Scalar>((a * a), (b * b))")

        expr = sm.Min(a ** 2, b ** 2)
        self.assertEqual(printer.doprint(expr), "std::min<Scalar>((a * a), (b * b))")


if __name__ == "__main__":
    TestCase.main()
