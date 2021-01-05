from sympy.printing.pycode import NumPyPrinter

from symforce import sympy as sm


class PythonCodePrinter(NumPyPrinter):
    """
    Symforce customized code printer for Python. Modifies the Sympy printing
    behavior for codegen compatibility and efficiency.
    """

    def _print_Rational(self, expr: sm.Rational) -> str:
        """
        Customizations:
            * Decimal points for Python2 support, doesn't exist in some sympy versions.
        """
        if self.standard == "python2":
            return f"{expr.p}./{expr.q}."
        return f"{expr.p}/{expr.q}"

    def _print_Heaviside(self, expr: "sm.Heaviside") -> str:
        """
        Heaviside is not supported by default, so we add a version here.
        """
        return "numpy.heaviside({}, 0.5)".format(expr.args[0])
