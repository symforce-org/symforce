from sympy.printing.cxx import CXX11CodePrinter

from symforce import sympy as sm


class CppCodePrinter(CXX11CodePrinter):
    """
    Symforce customized code printer for C++. Modifies the Sympy printing
    behavior for codegen compatibility and efficiency.
    """

    def _print_Rational(self, expr: sm.Rational) -> str:
        """
        Customizations:
            * Cast all literals to Scalar at compile time instead of using a suffix at codegen time
        """
        return f"Scalar({expr.p})/Scalar({expr.q})"

    def _print_Float(self, flt: sm.Float) -> str:
        """
        Customizations:
            * Cast all literals to Scalar at compile time instead of using a suffix at codegen time
        """
        return f"Scalar({super()._print_Float(flt)})"

    def _print_Pow(self, expr: sm.Pow) -> str:
        """
        Customizations:
            * Convert small powers into multiplies, divides, and square roots.
        """
        base_str = self._print(expr.base)

        # We don't special-case 2, because std::pow(x, 2) compiles to x * x under all circumstances
        # we tested (floats or doubles, fast-math or not)
        if expr.exp == -1:
            return f"1.0 / ({base_str})"
        elif expr.exp == 3:
            return f"[&]() {{ const Scalar base = {base_str}; return base * base * base; }}()"
        elif expr.exp == sm.S.One / 2:
            return f"{self._ns}sqrt({base_str})"
        elif expr.exp == sm.S(3) / 2:
            return f"({base_str} * {self._ns}sqrt({base_str}))"
        else:
            return "{}pow<Scalar>({}, {})".format(self._ns, base_str, self._print(expr.exp))

    def _print_Max(self, expr: sm.Max) -> str:
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

        return "{}max<Scalar>({}, {})".format(self._ns, self._print(expr.args[0]), rhs)

    def _print_Min(self, expr: sm.Min) -> str:
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

        return "{}min<Scalar>({}, {})".format(self._ns, self._print(expr.args[0]), rhs)

    def _print_Heaviside(self, expr: "sm.Heaviside") -> str:
        """
        Heaviside is not supported by default in C++, so we add a version here.
        """
        return "0.5*(((({0}) >= 0) - (({0}) < 0)) + 1)".format(self._print(expr.args[0]))
