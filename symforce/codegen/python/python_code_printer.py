from symforce import sympy as sm

from sympy.printing.pycode import NumPyPrinter


class PythonCodePrinter(NumPyPrinter):
    """
    Symforce customized code printer for Python. Modifies the Sympy printing
    behavior for codegen compatibility and efficiency.
    """

    def _print_Rational(self, expr):
        # type: (sm.Rational) -> str
        """
        Customizations:
            * Decimal points for Python2 support, doesn't exist in some sympy versions.
        """
        if self.standard == "python2":
            return "{}./{}.".format(expr.p, expr.q)
        return "{}/{}".format(expr.p, expr.q)
