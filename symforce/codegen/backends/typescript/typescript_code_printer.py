# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from sympy.printing.codeprinter import CodePrinter

# Everything in this file is SymPy, not SymEngine (even when SymForce is on the SymEngine backend)
import sympy

from sympy.core import S


_kw = {
    "break",
    "as",
    "any",
    "switch",
    "case",
    "if",
    "throw",
    "else",
    "var",
    "number",
    "string",
    "get",
    "module",
    "type",
    "instanceof",
    "typeof",
    "public",
    "private",
    "enum",
    "export",
    "finally",
    "for",
    "while",
    "void",
    "null",
    "super",
    "this",
    "new",
    "in",
    "return",
    "true",
    "false",
    "any",
    "extends",
    "static",
    "let",
    "package",
    "implements",
    "interface",
    "function",
    "new",
    "try",
    "yield",
    "const",
    "continue",
    "do",
    "catch",
}

_known_functions = {
    "Abs": "Math.abs",
    "acos": "Math.acos",
    "acosh": "Math.acosh",
    "asin": "Math.asin",
    "asinh": "Math.asinh",
    "atan": "Math.atan",
    "atan2": "Math.atan2",
    "atanh": "Math.atanh",
    "ceiling": "Math.ceil",
    "cos": "Math.cos",
    "cosh": "Math.cosh",
    "erf": "Math.erf",
    "erfc": "Math.erfc",
    "exp": "Math.exp",
    "expm1": "Math.expm1",
    "factorial": "Math.factorial",
    "floor": "Math.floor",
    "gamma": "Math.gamma",
    "hypot": "Math.hypot",
    "loggamma": "Math.lgamma",
    "log": "Math.log",
    "ln": "Math.log",
    "log10": "Math.log10",
    "log1p": "Math.log1p",
    "log2": "Math.log2",
    "sin": "Math.sin",
    "sinh": "Math.sinh",
    "Sqrt": "Math.sqrt",
    "tan": "Math.tan",
    "tanh": "Math.tanh",
}

_known_constants = {
    "Exp1": "E",
    "Pi": "PI",
    "E": "E",
    "Infinity": "Infinity",
    "NaN": "NaN",
    "ComplexInfinity": "NaN",
}


class AbstractTypescriptCodePrinter(CodePrinter):
    printmethod = "_typescriptcode"
    language = "Typescript"
    reserved_words = _kw
    modules = None  # initialized to a set in __init__
    tab = "    "
    _kf = _known_functions
    _kc = _known_constants
    _operators = {"and": "&&", "or": "||", "not": "!"}
    _default_settings = dict(
        CodePrinter._default_settings,
        user_functions={},
        precision=17,
        inline=True,
        fully_qualified_modules=True,
        contract=False,
        standard="typescrispt3",
    )

    def __init__(self, settings=None):
        super().__init__(settings)

        # Typescript standard handler
        std = self._settings["standard"]
        if std is None:
            import sys

            std = "typescript{}".format(sys.version_info.major)
        if std != "typescript3":
            raise ValueError("Only Typescript 3 is supported.")
        self.standard = std

        self.module_imports = defaultdict(set)

        # Known functions and constants handler
        self.known_functions = dict(
            self._kf, **(settings or {}).get("user_functions", {})
        )
        self.known_constants = dict(
            self._kc, **(settings or {}).get("user_constants", {})
        )

    def _declare_number_const(self, name, value):
        return "%s = %s" % (name, value)

    def _module_format(self, fqn, register=True):
        parts = fqn.split(".")
        if register and len(parts) > 1:
            self.module_imports[".".join(parts[:-1])].add(parts[-1])

        if self._settings["fully_qualified_modules"]:
            return fqn
        else:
            return fqn.split("(")[0].split("[")[0].split(".")[-1]

    def _format_code(self, lines):
        return lines

    def _get_statement(self, codestring):
        return "{}".format(codestring)

    def _get_comment(self, text):
        return "  # {}".format(text)

    def _expand_fold_binary_op(self, op, args):
        """
        This method expands a fold on binary operations.

        ``functools.reduce`` is an example of a folded operation.

        For example, the expression

        `A + B + C + D`

        is folded into

        `((A + B) + C) + D`
        """
        if len(args) == 1:
            return self._print(args[0])
        else:
            return "%s(%s, %s)" % (
                self._module_format(op),
                self._expand_fold_binary_op(op, args[:-1]),
                self._print(args[-1]),
            )

    def _expand_reduce_binary_op(self, op, args):
        """
        This method expands a reductin on binary operations.

        Notice: this is NOT the same as ``functools.reduce``.

        For example, the expression

        `A + B + C + D`

        is reduced into:

        `(A + B) + (C + D)`
        """
        if len(args) == 1:
            return self._print(args[0])
        else:
            N = len(args)
            Nhalf = N // 2
            return "%s(%s, %s)" % (
                self._module_format(op),
                self._expand_reduce_binary_op(args[:Nhalf]),
                self._expand_reduce_binary_op(args[Nhalf:]),
            )

    def _get_einsum_string(self, subranks, contraction_indices):
        letters = self._get_letter_generator_for_einsum()
        contraction_string = ""
        counter = 0
        d = {j: min(i) for i in contraction_indices for j in i}
        indices = []
        for rank_arg in subranks:
            lindices = []
            for i in range(rank_arg):
                if counter in d:
                    lindices.append(d[counter])
                else:
                    lindices.append(counter)
                counter += 1
            indices.append(lindices)
        mapping = {}
        letters_free = []
        letters_dum = []
        for i in indices:
            for j in i:
                if j not in mapping:
                    l = next(letters)
                    mapping[j] = l
                else:
                    l = mapping[j]
                contraction_string += l
                if j in d:
                    if l not in letters_dum:
                        letters_dum.append(l)
                else:
                    letters_free.append(l)
            contraction_string += ","
        contraction_string = contraction_string[:-1]
        return contraction_string, letters_free, letters_dum

    def _print_NaN(self, expr):
        return "float('nan')"

    def _print_Infinity(self, expr):
        return "float('inf')"

    def _print_NegativeInfinity(self, expr):
        return "float('-inf')"

    def _print_ComplexInfinity(self, expr):
        return self._print_NaN(expr)

    def _print_Mod(self, expr):
        PREC = precedence(expr)
        return "{} % {}".format(*map(lambda x: self.parenthesize(x, PREC), expr.args))

    def _print_Piecewise(self, expr):
        result = []
        i = 0
        for arg in expr.args:
            e = arg.expr
            c = arg.cond
            if i == 0:
                result.append("(")
            result.append("(")
            result.append(self._print(e))
            result.append(")")
            result.append(" if ")
            result.append(self._print(c))
            result.append(" else ")
            i += 1
        result = result[:-1]
        if result[-1] == "True":
            result = result[:-2]
            result.append(")")
        else:
            result.append(" else None)")
        return "".join(result)

    def _print_Relational(self, expr):
        "Relational printer for Equality and Unequality"
        op = {
            "==": "equal",
            "!=": "not_equal",
            "<": "less",
            "<=": "less_equal",
            ">": "greater",
            ">=": "greater_equal",
        }
        if expr.rel_op in op:
            lhs = self._print(expr.lhs)
            rhs = self._print(expr.rhs)
            return "({lhs} {op} {rhs})".format(op=expr.rel_op, lhs=lhs, rhs=rhs)
        return super()._print_Relational(expr)

    def _print_ITE(self, expr):
        from sympy.functions.elementary.piecewise import Piecewise

        return self._print(expr.rewrite(Piecewise))

    def _print_Sum(self, expr):
        loops = (
            "for {i} in range({a}, {b}+1)".format(
                i=self._print(i), a=self._print(a), b=self._print(b)
            )
            for i, a, b in expr.limits
        )
        return "(builtins.sum({function} {loops}))".format(
            function=self._print(expr.function), loops=" ".join(loops)
        )

    def _print_ImaginaryUnit(self, expr):
        return "1j"

    def _print_KroneckerDelta(self, expr):
        a, b = expr.args

        return "(1 if {a} == {b} else 0)".format(a=self._print(a), b=self._print(b))

    def _print_MatrixBase(self, expr):
        name = expr.__class__.__name__
        func = self.known_functions.get(name, name)
        return "%s(%s)" % (func, self._print(expr.tolist()))

    _print_SparseRepMatrix = (
        _print_MutableSparseMatrix
    ) = (
        _print_ImmutableSparseMatrix
    ) = (
        _print_Matrix
    ) = (
        _print_DenseMatrix
    ) = (
        _print_MutableDenseMatrix
    ) = (
        _print_ImmutableMatrix
    ) = _print_ImmutableDenseMatrix = lambda self, expr: self._print_MatrixBase(expr)

    def _indent_codestring(self, codestring):
        return "\n".join([self.tab + line for line in codestring.split("\n")])

    def _print_FunctionDefinition(self, fd):
        body = "\n".join(map(lambda arg: self._print(arg), fd.body))
        return "def {name}({parameters}):\n{body}".format(
            name=self._print(fd.name),
            parameters=", ".join([self._print(var.symbol) for var in fd.parameters]),
            body=self._indent_codestring(body),
        )

    def _print_While(self, whl):
        body = "\n".join(map(lambda arg: self._print(arg), whl.body))
        return "while({cond})\{\n{body}\n\}".format(
            cond=self._print(whl.condition), body=self._indent_codestring(body)
        )

    def _print_Declaration(self, decl):
        return "%s = %s;" % (
            self._print(decl.variable.symbol),
            self._print(decl.variable.value),
        )

    def _print_Return(self, ret):
        (arg,) = ret.args
        return "return %s;" % self._print(arg)

    def _print_Print(self, prnt):
        print_args = ", ".join(map(lambda arg: self._print(arg), prnt.print_args))
        if prnt.format_string != None:  # Must be '!= None', cannot be 'is not None'
            print_args = "{} % ({})".format(self._print(prnt.format_string), print_args)
        if prnt.file != None:  # Must be '!= None', cannot be 'is not None'
            print_args += ", file=%s" % self._print(prnt.file)

        return "console.log(%s);" % print_args

    def _print_Stream(self, strm):
        if str(strm.name) == "stdout":
            return "console.log"
        elif str(strm.name) == "stderr":
            return "console.error"
        else:
            return self._print(strm.name)

    def _print_NoneToken(self, arg):
        return "undefined"

    def _hprint_Pow(self, expr, rational=False, sqrt="Math.sqrt"):
        """Printing helper function for ``Pow``

        Notes
        =====

        This only preprocesses the ``sqrt`` as math formatter

        Examples
        ========

        >>> from sympy import sqrt
        >>> from sympy.printing.pycode import TypescriptCodePrinter
        >>> from sympy.abc import x

        Typescript code printer automatically looks up ``Math.sqrt``.

        >>> printer = TypescriptCodePrinter()
        >>> printer._hprint_Pow(sqrt(x), rational=True)
        'x**(1/2)'
        >>> printer._hprint_Pow(sqrt(x), rational=False)
        'Math.sqrt(x)'
        >>> printer._hprint_Pow(1/sqrt(x), rational=True)
        'x**(-1/2)'
        >>> printer._hprint_Pow(1/sqrt(x), rational=False)
        '1/Math.sqrt(x)'

        Using sqrt from numpy or mpmath

        >>> printer._hprint_Pow(sqrt(x), sqrt='numpy.sqrt')
        'numpy.sqrt(x)'

        See Also
        ========

        sympy.printing.str.StrPrinter._print_Pow
        """

        if expr.exp == S.Half and not rational:
            func = self._module_format(sqrt)
            arg = self._print(expr.base)
            return "{func}({arg})".format(func=func, arg=arg)

        if expr.is_commutative:
            if -expr.exp is S.Half and not rational:
                func = self._module_format(sqrt)
                num = self._print(S.One)
                arg = self._print(expr.base)
                return "{num}/{func}({arg})".format(num=num, func=func, arg=arg)

        return "({})**({})".format(expr.base, expr.exp)


class _TypescriptCodePrinter(AbstractTypescriptCodePrinter):
    def _print_sign(self, expr):
        return "Math.sign({})".format(self._print(expr.args[0]))

    def _print_Not(self, expr):
        return "(!({}))".format(self._print(expr.args[0]))

    def _print_Indexed(self, expr):
        base = expr.args[0]
        index = expr.args[1:]
        return "{}{}".format(
            str(base), "".join(["[{}]".format(self._print(ind)) for ind in index])
        )

    def _print_Pow(self, expr, rational=False):
        return self._hprint_Pow(expr, rational=rational)

    def _print_Rational(self, expr):
        return "{}/{}".format(expr.p, expr.q)

    def _print_Half(self, expr):
        return self._print_Rational(expr)

    def _print_frac(self, expr):
        from sympy.core.mod import Mod

        return self._print_Mod(Mod(expr.args[0], 1))

    def _print_Symbol(self, expr):

        name = super()._print_Symbol(expr)

        if name in self.reserved_words:
            if self._settings["error_on_reserved"]:
                msg = (
                    'This expression includes the symbol "{}" which is a '
                    "reserved keyword in this language."
                )
                raise ValueError(msg.format(name))
            return name + self._settings["reserved_word_suffix"]
        elif "{" in name:  # Remove curly braces from subscripted variables
            return name.replace("{", "").replace("}", "")
        else:
            return name

    _print_lowergamma = CodePrinter._print_not_supported
    _print_uppergamma = CodePrinter._print_not_supported
    _print_fresnelc = CodePrinter._print_not_supported
    _print_fresnels = CodePrinter._print_not_supported


class TypescriptCodePrinter(_TypescriptCodePrinter):
    """
    Symforce customized code printer for Typescript. Modifies the Sympy printing
    behavior for codegen compatibility and efficiency.
    """

    def _print_Rational(self, expr: sympy.Rational) -> str:
        """
        Customizations:
            * Decimal points for Typescript2 support, doesn't exist in some sympy versions.
        """
        return f"{expr.p}/{expr.q}."

    def _print_Max(self, expr: sympy.Max) -> str:
        """
        Max is not supported by default, so we add a version here.
        """
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        else:
            return "Math.max({})".format(
                ", ".join([self._print(arg) for arg in expr.args])
            )

    def _print_Min(self, expr: sympy.Min) -> str:
        """
        Min is not supported by default, so we add a version here.
        """
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        else:
            return "Math.min({})".format(
                ", ".join([self._print(arg) for arg in expr.args])
            )

    # NOTE(vincent): We type ignore the signature because mypy complains that it
    # does not match that of the sympy base class CodePrinter. This is because the base class
    # defines _print_Heaviside with: _print_Heaviside = None (see
    # https://github.com/sympy/sympy/blob/95f0228c033d27731f8707cdbb5bb672e500847d/sympy/printing/codeprinter.py#L446
    # ).
    # Despite this, our signature here matches the signatures of the sympy defined subclasses
    # of CodePrinter. I don't know of any other way to resolve this issue other than to
    # to type ignore.
    def _print_Heaviside(self, expr: "sympy.Heaviside") -> str:  # type: ignore[override]
        """
        Heaviside is not supported by default, so we add a version here.
        """
        return f"(({self._print(expr.args[0])}) < 0 ? 0.0 : 1.0)"

    def _print_MatrixElement(
        self, expr: sympy.matrices.expressions.matexpr.MatrixElement
    ) -> str:
        """
        default printer doesn't cast to int
        """
        return "{}[parseInt({})]".format(
            expr.parent, self._print(expr.j + expr.i * expr.parent.shape[1])
        )
