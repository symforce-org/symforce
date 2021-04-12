from sympy.printing.pycode import PythonCodePrinter as _PythonCodePrinter

from symforce import sympy as sm


class PythonCodePrinter(_PythonCodePrinter):
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

    def _print_Max(self, expr: sm.Max) -> str:
        """
        Max is not supported by default, so we add a version here.
        """
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        else:
            return "max({})".format(", ".join([self._print(arg) for arg in expr.args]))

    def _print_Min(self, expr: sm.Min) -> str:
        """
        Min is not supported by default, so we add a version here.
        """
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        else:
            return "min({})".format(", ".join([self._print(arg) for arg in expr.args]))

    def _print_Heaviside(self, expr: "sm.Heaviside") -> str:
        """
        Heaviside is not supported by default, so we add a version here.
        """
        return f"(0.0 if ({self._print(expr.args[0])}) < 0 else 1.0)"
