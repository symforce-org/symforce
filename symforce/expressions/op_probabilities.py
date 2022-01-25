from dataclasses import dataclass

from symforce import sympy as sm
from symforce import typing as T


@dataclass
class OpProbability:
    """
    Represents a logical symbolic operation and the probability
    of it occurring within regular use, for the purpose of generating and
    profiling random expressions containing these operations.

    Probabilities are relative across ops with the same arity, and do not
    have to sum to one.
    """

    name: str
    func: T.Callable
    prob: float

    @property
    def arity(self) -> int:
        return self.func.__code__.co_argcount


# Some reasonable defaults for generating expressions

DEFAULT_UNARY_OPS = (
    OpProbability("neg", lambda x: -x, 3),
    OpProbability("abs", sm.Abs, 3),
    OpProbability("sign", sm.sign_no_zero, 3),
    OpProbability("sqrt", lambda x: sm.sqrt(sm.Abs(x)), 2),
    OpProbability("exp", sm.exp, 0.1),
    OpProbability("log", lambda x: sm.log(sm.Abs(x)), 0.1),
    OpProbability("sin", sm.sin, 0.5),
    OpProbability("cos", sm.cos, 0.5),
    OpProbability("tan", sm.tan, 0.3),
    OpProbability("pow2", lambda x: x ** 2, 3),
    OpProbability("pow3", lambda x: x ** 3, 1),
    OpProbability("asin", sm.asin_safe, 0.2),
    OpProbability("acos", sm.acos_safe, 0.2),
    OpProbability("atan", sm.atan, 0.1),
)

DEFAULT_BINARY_OPS = (
    OpProbability("add", lambda x, y: x + y, 4),
    OpProbability("sub", lambda x, y: x - y, 2),
    OpProbability("mul", lambda x, y: x * y, 5),
    OpProbability("div", lambda x, y: x / y, 1),
    OpProbability("pow", lambda x, y: x ** y, 0.5),
    OpProbability("atan2", sm.atan2, 0.2),
)

DEFAULT_INTEGER_LEAVES = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 9)
DEFAULT_SYMBOL_LEAVES = sm.symbols("x:10")
DEFAULT_LEAVES = DEFAULT_INTEGER_LEAVES + DEFAULT_SYMBOL_LEAVES
