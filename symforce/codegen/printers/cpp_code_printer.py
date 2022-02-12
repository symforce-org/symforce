# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from sympy.printing.c import get_math_macros
from sympy.printing.cxx import CXX11CodePrinter

from symforce import typing as T

# Everything in this file is SymPy, not SymEngine (even when SymForce is on the SymEngine backend)
import sympy


class CppCodePrinter(CXX11CodePrinter):
    """
    Symforce customized code printer for C++. Modifies the Sympy printing
    behavior for codegen compatibility and efficiency.
    """

    def __init__(self, settings: T.Dict[str, T.Any] = None) -> None:
        settings = dict(
            settings or {},
            math_macros={key: f"Scalar({macro})" for key, macro in get_math_macros().items()},
        )
        super().__init__(settings)

    def _print_Rational(self, expr: sympy.Rational) -> str:
        """
        Customizations:
            * Cast all literals to Scalar at compile time instead of using a suffix at codegen time
        """
        return f"Scalar({expr.p})/Scalar({expr.q})"

    def _print_Float(self, flt: sympy.Float) -> str:
        """
        Customizations:
            * Cast all literals to Scalar at compile time instead of using a suffix at codegen time
        """
        return f"Scalar({super()._print_Float(flt)})"

    def _print_ImaginaryUnit(self, expr: sympy.Expr) -> str:
        """
        Don't use this method - if you need complex number support, use the ComplexCppCodePrinter
        subclass
        """
        raise NotImplementedError(
            "You tried to print an expression that contains the imaginary unit `i`.  If this is"
            "intentional and you need complex number support, set `support_complex=True` in the"
            "CppConfig"
        )

    def _print_Pow(self, expr: sympy.Pow) -> str:
        """
        Customizations:
            * Convert small powers into multiplies, divides, and square roots.
        """
        # std::pow(float, integral_type), std::pow(integral_type, float), and
        # std::sqrt(integral_type) will convert all arguments to double; so, we have to cast
        # arguments to Scalar first.  We can't just cast them if they're sympy.Integer, because they
        # may be larger expressions that evaluate to integers, so we can't tell whether the type of
        # the C++ expression is an integral type from the type of the top-level symbolic expression
        # (I think...).  The only case where we know something is already of type Scalar is if it's
        # a Symbol
        # https://en.cppreference.com/w/cpp/numeric/math/pow
        # https://en.cppreference.com/w/cpp/numeric/math/sqrt
        raw_base_str = self._print(expr.base)
        if isinstance(expr.base, sympy.Symbol):
            base_str = raw_base_str
        else:
            base_str = f"Scalar({raw_base_str})"

        raw_exp_str = self._print(expr.exp)
        if isinstance(expr.exp, sympy.Symbol):
            exp_str = raw_exp_str
        else:
            exp_str = f"Scalar({raw_exp_str})"

        # We don't special-case 2, because std::pow(x, 2) compiles to x * x under all circumstances
        # we tested (floats or doubles, fast-math or not)

        if expr.exp == -1:
            return f"{self._print_Float(sympy.S(1.0))} / ({raw_base_str})"
        elif expr.exp == 3:
            return f"[&]() {{ const Scalar base = {raw_base_str}; return base * base * base; }}()"
        elif expr.exp == sympy.S.One / 2:
            return f"{self._ns}sqrt({base_str})"
        elif expr.exp == sympy.S(3) / 2:
            return f"({raw_base_str} * {self._ns}sqrt({base_str}))"
        else:
            return f"{self._ns}pow({base_str}, {exp_str})"

    def _print_Max(self, expr: sympy.Max) -> str:
        """
        Customizations:
            * Emit template type to avoid deduction errors.
        """
        if len(expr.args) == 1:
            return self._print(expr.args[0])

        if len(expr.args) == 2:
            rhs = self._print(expr.args[1])
        else:
            rhs = self._print(sympy.Max(*expr.args[1:]))

        return "{}max<Scalar>({}, {})".format(self._ns, self._print(expr.args[0]), rhs)

    def _print_Min(self, expr: sympy.Min) -> str:
        """
        Customizations:
            * Emit template type to avoid deduction errors.
        """
        if len(expr.args) == 1:
            return self._print(expr.args[0])

        if len(expr.args) == 2:
            rhs = self._print(expr.args[1])
        else:
            rhs = self._print(sympy.Min(*expr.args[1:]))

        return "{}min<Scalar>({}, {})".format(self._ns, self._print(expr.args[0]), rhs)

    def _print_Heaviside(self, expr: sympy.Heaviside) -> str:
        """
        Heaviside is not supported by default in C++, so we add a version here.
        """
        return "{0}*(((({1}) >= 0) - (({1}) < 0)) + 1)".format(
            self._print_Float(sympy.S(0.5)), self._print(expr.args[0])
        )


class ComplexCppCodePrinter(CppCodePrinter):  # pylint: disable=too-many-ancestors
    def _print_Integer(self, expr: sympy.Integer) -> str:
        """
        Customizations:
            * Cast all integers to Scalar, since binary ops between integers and complex aren't
              defined
        """
        return f"Scalar({expr.p})"

    def _print_ImaginaryUnit(self, expr: sympy.Expr) -> str:
        """
        Customizations:
            * Print 1i instead of I
            * Cast to Scalar, since the literal is of type std::complex<double>
        """
        return "Scalar(1i)"
