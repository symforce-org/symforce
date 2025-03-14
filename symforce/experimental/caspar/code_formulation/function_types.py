# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

# ruff: noqa: PLR6301

from __future__ import annotations

import hashlib
from abc import abstractmethod

from symengine.lib import symengine_wrapper

import symforce.symbolic as sf
from symforce import typing as T

if T.TYPE_CHECKING:
    from .compute_graph_sorting_helpers import FuncOrderingData
    from .compute_graph_sorting_helpers import VarOrderingData


class Var:
    """
    A class representing a variable in the expression tree.
    The variable is defined by the function it belongs to and the index of the output.
    """

    func: Func
    idx: int

    vod: VarOrderingData  # This name is short as it is often nested
    _str: str | None = None

    def __init__(self, func: Func, idx: int) -> None:
        self.func = func
        self.idx = idx
        if not isinstance(func, Func):
            raise ValueError(f"Expected Func, got {type(func)}")
        assert idx < func.n_outs
        self._hash = hash((func, idx))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Var) and self.func == other.func and self.idx == other.idx

    def __repr__(self) -> str:
        if self._str is None:
            return repr(self.func) + (f"[{self.idx}]" if self.idx else "")
        return self._str


class Func:
    """
    A parent class representing a function in the expression tree.
    The function is defined by its arguments and the data it holds.
    """

    args: tuple[Var, ...]
    data: T.Any
    unique_id = 0  # used to create duplicate instances with unique hashes

    outs: list[Var]

    n_outs = 1
    fod: FuncOrderingData  # This name is short as it is often nested
    _hash: int

    def __init__(
        self,
        *args: Var,
        data: T.Any = None,
        unique_id: int = 0,
        custom_code: str = "",
    ) -> None:
        self.args = args
        self.data = data
        self.unique_id = unique_id
        self.custom_code = custom_code
        self.outs = [Var(self, i) for i in range(self.n_outs)]

        assert isinstance(data, (float, int, str, tuple)) or data is None
        assert all([isinstance(arg, Var) for arg in self.args])

    @property
    def n_args(self) -> int:
        return len(self.args)

    def rebind(self, other: Func, idx_self: int = 0, idx_other: int = 0) -> None:
        """
        Rebinds the output of another function to the output of this function.

        The other function is no longer valid.
        """
        assert isinstance(other, Func)
        if other is self:
            raise ValueError("Cannot rebind to self")

        self[idx_self].func = None  # type: ignore[assignment]
        self.outs[idx_self] = other[idx_other]
        other.outs[idx_other] = None  # type: ignore[call-overload]
        self[idx_self].func = self
        self[idx_self].idx = idx_self
        self[idx_self]._hash = hash((self, idx_self))  # noqa: SLF001

    def update_args(self, *args: Var) -> None:
        """
        Used when fusing multiple instances of the same function.
        """
        assert args == self.args
        assert hash(args) == hash(self.args)
        self.args = args

    @abstractmethod
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Var:
        return self.outs[idx]

    def __repr__(self) -> str:
        rep = f"{self.__class__.__name__}({','.join(map(str, self.args))})"
        if len(rep) > 50:
            return f"{self.__class__.__name__}(...[{len(self.args)}])"
        return rep

    def __hash__(self) -> int:
        if self.is_accumulator():
            args = tuple(sorted(self.args, key=hash))
        else:
            args = self.args

        # we want hashes to be deterministic and hash(str) is not
        _data_hash = int(
            hashlib.md5((str(self.data) + self.__class__.__name__).encode()).hexdigest(), 16
        )

        return hash((_data_hash, args, self.unique_id))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.args == other.args
            and self.data == other.data
            and self.unique_id == other.unique_id
        )

    def is_write(self) -> bool:
        return isinstance(self, Write)

    def is_load(self) -> bool:
        return isinstance(self, Read)

    def is_store(self) -> bool:
        return isinstance(self, Store)

    def is_sum(self) -> bool:
        return isinstance(self, Sum)

    def is_minus(self) -> bool:
        return isinstance(self, Minus)

    def is_prod(self) -> bool:
        return isinstance(self, Prod)

    def is_div(self) -> bool:
        return isinstance(self, Div)

    def is_sincos(self) -> bool:
        return isinstance(self, SinCos)

    def is_cos(self) -> bool:
        return isinstance(self, Cos)

    def is_sin(self) -> bool:
        return isinstance(self, Sin)

    def is_norm(self) -> bool:
        return isinstance(self, Norm)

    def is_rnorm(self) -> bool:
        return isinstance(self, RNorm)

    def is_pow(self) -> bool:
        return isinstance(self, Pow)

    def is_square(self) -> bool:
        return isinstance(self, Square)

    def is_rcp(self) -> bool:
        return isinstance(self, Rcp)

    def is_sqrt(self) -> bool:
        return isinstance(self, Sqrt)

    def is_rsqrt(self) -> bool:
        return isinstance(self, RSqrt)

    def is_cbrt(self) -> bool:
        return isinstance(self, Cbrt)

    def is_rcbrt(self) -> bool:
        return isinstance(self, RCbrt)

    def is_anypow(self) -> bool:
        return isinstance(self, Exponent)

    def is_neg(self) -> bool:
        return isinstance(self, Neg)

    def is_fma_none(self) -> bool:
        return isinstance(self, FmaNone)

    def is_fma_any(self) -> bool:
        return isinstance(self, Fma)

    def is_fma(self) -> bool:
        return self.is_fma_any() and not self.is_fma_none()

    def is_fmaprod(self) -> bool:
        return isinstance(self, FmaProd)

    def is_accumulator(self) -> bool:
        return self.is_fma_any() or (isinstance(self, Accumulator) and self.n_args > 2)

    def is_contrib(self) -> bool:
        return isinstance(self, Contribute)

    def is_ternary(self) -> bool:
        return isinstance(self, Ternary)


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


class Contribute(Func):
    """
    Helper function that is a contribution to an accumulator.

    e.g. a + b + c -> Sum(Contribute(a), Contribute(b), Contribute(c))
    """

    data: Func

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return self.fod.contrib_parent_acc.assign_code(outs, args)


class Write(Func):
    """
    Writes data to global memory using vectorized stores.

    The method for writing is defined directly in the class and corresponds to the different
    accessor types like Sequential, Shared, Sum, etc...
    """

    n_outs = 0
    data: str

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{self.custom_code.format(**{k: f'{a}' for k, a in zip('xyzw', args)})}"


class Write2(Write):
    n_outs = 0


class Write3(Write):
    n_outs = 0


class Write4(Write):
    n_outs = 0


class Read(Func):
    """
    The read classes reads data from global memory using vectorized loads.

    The method for writing is defined directly in the class and corresponds to the different
    accessor types like Sequential, Shared, Indexed, etc...
    """

    data: str

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{self.custom_code.format(**{k: f'{a}' for k, a in zip('xyzw', outs)})}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({','.join(map(str, self.data))})"


class Read2(Read):
    n_outs = 2


class Read3(Read):
    n_outs = 3


class Read4(Read):
    n_outs = 4


class Store(Func):
    data: float

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = {self.data:.8e}f;"

    def __repr__(self) -> str:
        return f"{self.data:.2e}"


class Sum(Accumulator):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = {' + '.join(args)};"


class Prod(Accumulator):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = {' * '.join(args)};"


# FUSED MULTIPLY-ADD
class Fma(Sum):
    """
    A Sum with at least one product term and at least one other term.

    The first add operation will never be between two product terms.

    This class is responsible for much of the complexity in the expression ordering as
    it generates a lot of dependencies between variables. This is necessary to ensure fma operations
    are used where possible, as a non-product term has to be calculated first.

    e.g. a*b*c + d*e + f + g -> Fma(FmaProd(Contribute(a), Contribute(b), Contribute(c)),
                                    FmaProd(d, e),
                                    Contribute(f),
                                    Contribute(g))

    The above equation can be solved in many ways, but f or g have to be live before
    d*e is calculated and before the last contribution is added to a*b*c.
    """


class FmaNone(Fma):
    """
    Fma with only product terms.

    This means we 'lose' one fma operation and don't have to wait for a non-product term
    to be calculated first.
    """


class FmaProd(Prod):
    """
    A product that is part of an Fma operation.

    This class can be seen as both an accumulator (a product) and a contribution (to Fma / FmaNone).
    """

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        if len(args) == 2:
            return Prod.assign_code(self, outs, args[:2])
        elif len(args) == 3:
            return f"{outs[0]} = fmaf({args[0]}, {args[1]}, {args[2]});"
        raise NotImplementedError


class Fma3(Func):
    """
    a * b + c.

    This class is not necessary, but speeds up the code generation.
    Fma(a,b,c) is simpler than Fma(FmaProd(a, b), Contribute(c))
    """

    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = fma({args[0]}, {args[1]}, {args[2]});"


# ARITHMETIC FUNCTIONS
class Minus(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = {args[0]} - {args[1]};"


class Neg(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = -{args[0]};"


class Abs(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = abs({args[0]});"


class Sign(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = copysignf(1.0f, {args[0]});"


class Div(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = {args[0]}/{args[1]};"


# TRIGONOMETRIC FUNCTIONS
class Cos(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = cos({args[0]});"


class Sin(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = sin({args[0]});"


class Tan(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = tan({args[0]});"


class ACos(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = acos({args[0]});"


class ASin(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = asin({args[0]});"


class ATan(Func):
    def assign_code(self, outs: list[str], args: list[str]) -> str:
        return f"{outs[0]} = atan({args[0]});"


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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({','.join(map(str, self.args))}, {self.data})"


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
        return f"{outs[0]} = 1.f / {args[0]};"


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


READ_FUNCS: list[T.Type[Func]] = [Read, Read2, Read3, Read4]
WRITE_FUNCS = [Write, Write2, Write3, Write4]
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
    symengine_wrapper.sign: Sign,
    sf.Min: Min,
    sf.Max: Max,
    sf.Abs: Abs,
}
