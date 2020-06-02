from symforce import sympy as sm

from sympy.printing.cxxcode import CXX11CodePrinter


class CppCodePrinter(CXX11CodePrinter):
    """
    Symforce customized code printer for C++. Modifies the Sympy printing
    behavior for codegen compatibility and efficiency.
    """

    def _print_Pow(self, expr):
        # type: (sm.Pow) -> str
        """
        Customizations:
            * Convert small powers into multiplies, divides, and square roots.
        """
        base_str = self._print(expr.base)

        if expr.exp == -1:
            return "1.0 / ({})".format(base_str)
        elif expr.exp == 2:
            return "({} * {})".format(base_str, base_str)
        elif expr.exp == 3:
            return "({} * {} * {})".format(base_str, base_str, base_str)
        elif expr.exp == sm.S.One / 2:
            return "{}sqrt({})".format(self._ns, base_str)
        elif expr.exp == sm.S(3) / 2:
            return "({} * {}sqrt({}))".format(base_str, self._ns, base_str)
        else:
            return "{}pow({}, {})".format(self._ns, base_str, self._print(expr.exp))

    def _print_Max(self, expr):
        # type: (sm.Max) -> str
        """
        Customizations:
            * Emit template type to avoid deduction errors.
        """
        from sympy import Max

        if len(expr.args) == 1:
            return self._print(expr.args[0])

        if len(expr.args) == 2:
            rhs = self._print(expr.args[1])
        else:
            rhs = self._print(Max(*expr.args[1:]))

        return "{}max<Scalar>({}, {})".format(self._ns, expr.args[0], rhs)

    def _print_Min(self, expr):
        # type: (sm.Min) -> str
        """
        Customizations:
            * Emit template type to avoid deduction errors.
        """
        from sympy import Min

        if len(expr.args) == 1:
            return self._print(expr.args[0])

        if len(expr.args) == 2:
            rhs = self._print(expr.args[1])
        else:
            rhs = self._print(Min(*expr.args[1:]))

        return "{}min<Scalar>({}, {})".format(self._ns, expr.args[0], rhs)
