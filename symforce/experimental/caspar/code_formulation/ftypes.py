# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

# ruff: noqa: PLR6301

from __future__ import annotations

from abc import abstractmethod

from symengine.lib import symengine_wrapper

import symforce.symbolic as sf
from symforce import typing as T


class Func:
    data: T.Any
    custom_code: str

    n_outs: T.ClassVar[int] = 1
    is_unique: T.ClassVar[bool] = False
    is_accumulator: T.ClassVar[bool] = False
    is_fmaprod: T.ClassVar[bool] = False

    is_start_accumulator: T.ClassVar[bool] = False
    is_contribute: T.ClassVar[bool] = False
    is_finish_accumulator: T.ClassVar[bool] = False

    is_contrib_marker: T.ClassVar[bool] = False

    def __init__(self, data: T.Any = None, custom_code: str = "") -> None:
        self.data = data
        self.custom_code = custom_code

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, self.__class__):
            return NotImplemented
        return (
            self.__class__ == value.__class__
            and self.data == value.data
            and self.custom_code == value.custom_code
        )

    def __hash__(self) -> int:
        return hash((self.__class__, self.data, self.custom_code))

    @abstractmethod
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self.data}>"


class Finalize(Func):
    n_outs = 0


class Accumulator(Func):
    """
    Parent class of accumulators.

    Here, an accumulator is an associative and commutative function with more than two arguments.
    Examples are Sum, Prod, Min, Max.

    Accumulators make it possible to generate code that is more efficient than the naive approach,
    since at most two of the arguments need to be live at the same time.
    a+b+c+d -> r0 = a+b; r0 = r0+c; r0 = r0+d;
    """

    n_outs = 1
    is_accumulator = True


class ContributeMarker(Func):
    n_outs = 0

    is_unique = True
    is_contrib_marker = True

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return ""


class FmaProd2Marker(Func):
    n_outs = 0
    is_unique = True
    is_contrib_marker = True
    is_fmaprod = True

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return ""


class FmaProdMarker(Func):
    n_outs = 0
    is_unique = True
    is_contrib_marker = True
    is_fmaprod = True

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return ""


class StartAccumulate(Func):
    data: Func
    is_unique = True
    is_start_accumulator = True

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return self.data.assign_code(outs, args)


class StartFma(Func):
    is_unique = True
    is_start_accumulator = True

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        if len(args) == 2:
            return f"{outs[0]} = {args[0]} + {args[1]};"
        elif len(args) == 3:
            return f"{outs[0]} = fmaf({args[1]}, {args[2]}, {args[0]});"
        elif len(args) == 4:
            return f"{outs[0]} = fmaf({args[2]}, {args[3]}, {args[0]} * {args[1]});"
        raise NotImplementedError


class Contribute(Func):
    """
    Helper function that is a contribution to an accumulator.

    e.g. a + b + c -> Sum(Contribute(a), Contribute(b), Contribute(c))
    """

    n_outs = 0
    data: Func
    is_unique = True
    is_contribute = True

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return self.data.assign_code(outs, args)


class ContributeFmaProd(Func):
    """
    Helper function that is a contribution to an accumulator.

    e.g. a + b + c -> Sum(Contribute(a), Contribute(b), Contribute(c))
    """

    n_outs = 0
    is_unique = True
    is_contribute = True

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return ""


class ContributeFmaProd2(Func):
    """
    Helper function that is a contribution to an accumulator.

    e.g. a + b + c -> Sum(Contribute(a), Contribute(b), Contribute(c))
    """

    n_outs = 0
    is_unique = True
    is_fmaprod = True
    is_contribute = True

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        assert len(args) == 3
        return f"{outs[0]} =fmaf({args[1]}, {args[2]}, {args[0]});"


class ContributeToFmaProd(Func):
    """
    Helper function that is a contribution to an accumulator.

    e.g. a + b + c -> Sum(Contribute(a), Contribute(b), Contribute(c))
    """

    n_outs = 0
    is_unique = True
    is_fmaprod = True
    is_contribute = True

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        if len(args) == 2:
            return f"{outs[0]} = {args[0]} * {args[1]};"
        elif len(args) == 3:
            return f"{outs[0]} = fmaf({args[1]}, {args[2]}, {args[0]});"
        raise NotImplementedError


class FinishAccumulate(Func):
    is_finish_accumulator = True

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        assert len(args) == 1 and len(outs) == 1 and args[0] == outs[0]
        return ""


class FinishFmaProd(Func):
    is_fmaprod = True
    is_finish_accumulator = True

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        assert len(args) == 1 and len(outs) == 1 and args[0] == outs[0]
        return ""


class FreeMarker(Func):
    n_outs = 0
    is_unique = True

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return ""


class Write(Func):
    """
    Writes data to global memory using vectorized stores.

    The method for writing is defined directly in the class and corresponds to the different
    accessor types like Sequential, Shared, Sum, etc...
    """

    n_outs = 0
    data: str

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{self.custom_code.format(**{f'arg_{i}': f'{a}' for i, a in enumerate(args)})}"


class Write2(Write):
    n_outs = 0


class Write3(Write):
    n_outs = 0


class Write4(Write):
    n_outs = 0


class Write6(Write):
    n_outs = 0


class Write8(Write):
    n_outs = 0


class Read(Func):
    """
    The read classes reads data from global memory using vectorized loads.

    The method for writing is defined directly in the class and corresponds to the different
    accessor types like Sequential, Shared, Indexed, etc...
    """

    data: str

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{self.custom_code.format(**{f'arg_{i}': f'{a}' for i, a in enumerate(outs)})}"


class Read2(Read):
    n_outs = 2


class Read3(Read):
    n_outs = 3


class Read4(Read):
    n_outs = 4


class Read6(Read):
    n_outs = 6


class Read8(Read):
    n_outs = 8


class Store(Func):
    data: float

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = {self.data:.8e}f;"


class Sum(Accumulator):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = {' + '.join(args)};"


class Prod(Accumulator):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = {' * '.join(args)};"


class Barrier(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        assert len(args) == 1 and len(outs) == 1 and args[0] == outs[0]
        return ""


class Fma3(Func):
    """
    a * b + c.

    This class is not necessary, but speeds up the code generation.
    Fma(a,b,c) is simpler than Fma(FmaProd(a, b), Contribute(c))
    """

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = fmaf({args[0]}, {args[1]}, {args[2]});"


# ARITHMETIC FUNCTIONS
class Minus(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = {args[0]} - {args[1]};"


class Neg(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = -{args[0]};"


class Abs(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = fabsf({args[0]});"


class Sign(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = copysignf(1.0f, {args[0]});"


class CopysignNoZero(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = copysignf({args[0]}, {args[1]});"


class Div(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = {args[0]}/{args[1]};"


# TRIGONOMETRIC FUNCTIONS
class Cos(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = cosf({args[0]});"


class Sin(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = sinf({args[0]});"


class Tan(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = tanf({args[0]});"


class ACos(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = acosf({args[0]});"


class ASin(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = asinf({args[0]});"


class ATan(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = atanf({args[0]});"


class ATan2(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = atan2f({args[0]}, {args[1]});"


class SinCos(Func):
    n_outs = 2

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"sincosf({args[0]}, &{outs[0]}, &{outs[1]});"


# NORMS
class Norm(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = norm{len(args)}df({', '.join(args)});"


class RNorm(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = rnorm{len(args)}df({', '.join(args)});"


class Ldexp(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = ldexpf({args[0]}, {self.data});"


# EXPONENTS
class Exponent(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        raise NotImplementedError


class Pow(Exponent):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = powf({args[0]}, {args[1]});"


class Square(Exponent):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = {args[0]} * {args[0]};"


class Rcp(Exponent):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = __frcp_rn({args[0]});"


class Sqrt(Exponent):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = sqrtf({args[0]});"


class RSqrt(Exponent):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = rsqrtf({args[0]});"


class Cbrt(Exponent):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        raise NotImplementedError


class RCbrt(Exponent):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        raise NotImplementedError


# MIN MAX
class Min(Accumulator):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = fminf({args[0]}, {args[1]});"


class Max(Accumulator):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = fmaxf({args[0]}, {args[1]});"


# MISCELLANEOUS
class Ternary(Func):
    """
    Ternary operator.

    The comparison type is stored in self.data.
    """

    compmap = {
        symengine_wrapper.LessThan: "<=",
        symengine_wrapper.StrictLessThan: "<",
        symengine_wrapper.GreaterThan: ">=",
        symengine_wrapper.StrictGreaterThan: ">",
    }

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = {args[0]} {self.data} {args[1]} ? {args[2]} : {args[3]};"


READ_FUNCS: dict[int, T.Type[Func]] = {1: Read, 2: Read2, 3: Read3, 4: Read4, 6: Read6, 8: Read8}

WRITE_FUNCS = {1: Write, 2: Write2, 3: Write3, 4: Write4, 6: Write6, 8: Write8}
EXPR_TO_FUNC: dict[T.Type[sf.Basic], T.Type[Func]] = {
    sf.Symbol: Read,
    symengine_wrapper.Symbol: Read,
    sf.Add: Sum,
    sf.Mul: Prod,
    sf.Pow: Pow,
    symengine_wrapper.cos: Cos,
    symengine_wrapper.sin: Sin,
    symengine_wrapper.tan: Tan,
    symengine_wrapper.acos: ACos,
    symengine_wrapper.asin: ASin,
    symengine_wrapper.atan: ATan,
    symengine_wrapper.atan2: ATan2,
    symengine_wrapper.sign: Sign,
    symengine_wrapper.CopysignNoZero: CopysignNoZero,
    symengine_wrapper.SignNoZero: Sign,
    sf.Min: Min,
    sf.Max: Max,
    sf.Abs: Abs,
}
