# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import typing as T
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from itertools import product

from . import function_types
from .function_types import Func
from .function_types import Var


@lru_cache
def get_pow_map(
    cost_map: dict[Fraction, float] | None = None, powf_cost: float = 6.0
) -> dict[Fraction, tuple | Fraction]:
    """
    Finds the optimal way to express an rational exponential
    given a mapping between available power intrinsics (as fractions) and their costs,
    and the cost of a general power operation.

    Operations can either be nested (e.g. x^(1/4) = sqrt(sqrt(x)))
    or multiplied (e.g. x^(5/6) = sqrt(x)*cbrt(x)).
    The first element in each value of the output is a boolean indicating if the operations are
    multiplied, the remaining two elements are the exponents of the operations.
    """
    if cost_map is None:
        cost_map = {
            Fraction(-1, 1): 2.0,
            Fraction(2, 1): 1.0,
            Fraction(1, 2): 2.0,
            Fraction(1, 3): 2.0,
            Fraction(-1, 3): 2.0,
            Fraction(-1, 2): 2.0,
        }
    costs = {**cost_map, **{Fraction(1, 1): 0}}
    exp_maps: dict[Fraction, Fraction | tuple[bool, Fraction, Fraction]] = {k: k for k in costs}
    anynew = True

    def add_if_new(e0: Fraction, cost0: float, e1: Fraction, cost1: float, multiply: bool) -> bool:
        exp_new = e0 + e1 if multiply else e0 * e1
        cost = cost0 + cost1 + 1 if multiply else cost0 + cost1
        if cost < costs.get(exp_new, powf_cost):
            costs[exp_new] = cost
            exp_maps[exp_new] = (multiply, e0, e1)
            return True
        return False

    while anynew:
        anynew = False
        for (e0, cost0), (e1, cost1) in list(product(costs.items(), costs.items())):
            anynew |= add_if_new(e0, cost0, e1, cost1, True)
        for (e0, cost0), (e1, cost1) in list(product(cost_map.items(), costs.items())):
            anynew |= add_if_new(e0, cost0, e1, cost1, False)
    return exp_maps


EXPONENTIAL_MAP = get_pow_map()
OPERATION_MAP = {
    Fraction(-1, 1): function_types.Rcp,
    Fraction(2, 1): function_types.Square,
    Fraction(1, 2): function_types.Sqrt,
    Fraction(1, 3): function_types.Cbrt,
    Fraction(-1, 3): function_types.RCbrt,
    Fraction(-1, 2): function_types.RSqrt,
}


def fix_pow(func: Func) -> Func:
    """
    Fixes the power function using the generated maps above.
    """

    def inner(base: Var, exponent: Fraction) -> Var:
        exp = EXPONENTIAL_MAP.get(exponent, None)
        if exp is None:
            return function_types.Pow(base, function_types.Store(data=float(exponent))[0])[0]

        if isinstance(exp, Fraction):
            if exp == 1:
                return base
            else:
                return OPERATION_MAP[exp](base)[0]

        mul, e0, e1 = exp
        if mul:
            v0 = inner(base, e0)
            v1 = inner(base, e1)
            return function_types.Prod(v0, v1)[0]
        else:
            v1 = inner(base, e1)
            return inner(v1, e0)

    if not func.args[1].func.is_store():
        return func
    exponent = Fraction(float(func.args[1].func.data)).limit_denominator()
    return inner(func.args[0], exponent).func


def make_accumulators_shared(funcs: list[Func], ftype: T.Type[Func]) -> list[Func]:
    """
    Search for the best common subexpressions from a colloction of
    accumulators on the same type (e.g. Sums or Prods).

    The function finds the pair of variables that are used the most across all accumulators,
    extracts them from the functions and repeats until no more pairs of variables are shared.
    """
    # TODO(Emil Martens): speedup this function by using dynamic updates (don't regenerate all
    # pairs)

    pair_to_uses: dict[tuple[Var, Var], set[Func]] = {}
    for func0, func1 in combinations(funcs, 2):
        for pair in combinations(set(func0.args) & set(func1.args), 2):
            pair_to_uses.setdefault(pair, set()).update([func0, func1])

    siblings = set(frozenset(v) for v in pair_to_uses.values())

    max_len = max(map(len, siblings), default=0)
    if max_len == 0:
        return funcs

    candidate_sets = [v for v in siblings if len(v) == max_len]
    best_shared: set[Var] = set()
    best_fset: frozenset[Func] = frozenset()
    for fset in candidate_sets:
        it = iter(fset)
        shared = set(next(it).args)
        for func in it:
            shared &= set(func.args)
        if len(shared) > len(best_shared):
            best_shared = shared
            best_fset = fset
    out = []
    shared_var = ftype(*best_shared)[0]
    for func in funcs:
        if func in best_fset:
            func = ftype(*(set(func.args) - best_shared), shared_var)
        out.append(func)
    return make_accumulators_shared(out, ftype)
