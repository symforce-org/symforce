# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from itertools import product

from . import ftypes
from .dabseg import Call
from .dabseg import CallId
from .dabseg import Dabseg
from .dabseg import Val


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
    multiplied (True) or nested (False).
    The remaining two elements are the exponents of the operations.

    TODO: Take cse into account.
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

    def add_if_new(e0: Fraction, cost0: float, e1: Fraction, cost1: float, multiply: bool) -> bool:
        exp_new = e0 + e1 if multiply else e0 * e1
        cost = cost0 + cost1 + 1 if multiply else cost0 + cost1
        if is_new := cost < costs.get(exp_new, powf_cost):
            costs[exp_new] = cost
            exp_maps[exp_new] = (multiply, e0, e1)
        return is_new

    anynew = True
    while anynew:
        anynew = False
        keys = list(costs)
        for key0, key1 in combinations(keys, 2):
            anynew |= add_if_new(key0, costs[key0], key1, costs[key1], True)
        for key0, key1 in product(cost_map, keys):
            anynew |= add_if_new(key0, costs[key0], key1, costs[key1], False)
    return {k: exp_maps[k] for k in sorted(exp_maps)}


EXPONENTIAL_MAP = get_pow_map()
OPERATION_MAP = {
    Fraction(-1, 1): ftypes.Rcp,
    Fraction(2, 1): ftypes.Square,
    Fraction(1, 2): ftypes.Sqrt,
    Fraction(1, 3): ftypes.Cbrt,
    Fraction(-1, 3): ftypes.RCbrt,
    Fraction(-1, 2): ftypes.RSqrt,
}


def fix_pow(dabseg: Dabseg, call: Call) -> Call:
    """
    Fixes the power function using the generated maps above.
    """
    if not call.is_a(ftypes.Pow) or not call.args[1].call.is_a(ftypes.Store):
        return call

    def inner(base: Val, exponent: Fraction) -> Val:
        exp = EXPONENTIAL_MAP.get(exponent, None)
        if exp is None:
            exp_val = dabseg.add_call(ftypes.Store(data=float(exponent)))[0]
            return dabseg.add_call(ftypes.Pow(), (base, exp_val))[0]

        if isinstance(exp, Fraction):
            if exp == 1:
                return base
            else:
                return dabseg.add_call(OPERATION_MAP[exp](), (base,))[0]
        mul, e0, e1 = exp
        if mul:
            v0 = inner(base, e0)
            v1 = inner(base, e1)
            return dabseg.add_call(ftypes.Prod(), (v0, v1))[0]
        else:
            v1 = inner(base, e1)
            return inner(v1, e0)

    exponent = Fraction(float(call.args[1].call.func.data)).limit_denominator()
    return dabseg.rebind(inner(call.args[0], exponent).call, call)


def unpack_square(dabseg: Dabseg, call: Call) -> Call:
    if not call.is_a(ftypes.Square):
        return call
    new_call = dabseg.add_call(ftypes.Prod(), (call.args[0], call.args[0]), fix_accumulator=False)
    return dabseg.rebind(new_call, call)


def split_accumulators(dabseg: Dabseg) -> None:  # noqa: PLR0912, PLR0915
    """
    Used to split accumulators into single argument contributors.
    To facilitate instruction ordering additional marker calls are generated to limit sorting.
    """
    relations_map: dict[CallId, list[CallId]] = {}
    for call in dabseg.call_iter():
        if call.is_accumulator:
            relations_map.setdefault(call.id, [])
        for arg in call.args:
            if arg.call.is_accumulator:
                relations_map.setdefault(arg.call.id, []).append(call.id)

    def is_fma_prod(call: Call) -> bool:
        return (
            call.is_a(ftypes.Prod)
            and len(rel := relations_map[call.id]) == 1
            and dabseg.call(rel[0]).is_a(ftypes.Sum)
        )

    for call_id, relations in relations_map.items():
        call = dabseg.call(call_id)
        if (is_fma_prod(call) and call.n_args == 2) or not relations:
            continue

        # ReadyMarker
        markers: list[Call] = []
        non_fma_prod_markers: list[Call] = []
        is_fma = False
        fma_prod_markers: list[Call] = []
        for arg in call.args:
            acall = arg.call
            if is_fma_prod(acall) and acall.n_args == 2:
                is_fma = True
                marker = dabseg.add_call(ftypes.FmaProd2Marker(), acall.args)
            elif acall.is_a(ftypes.FinishFmaProd):
                is_fma = True
                marker = dabseg.add_call(ftypes.FmaProdMarker(), (arg,))
                fma_prod_markers.append(acall)
            else:
                marker = dabseg.add_call(ftypes.ContributeMarker(), (arg,))
                non_fma_prod_markers.append(marker)
            markers.append(marker)

        # Initialize
        if is_fma:
            assert call.is_a(ftypes.Sum)
            init = dabseg.add_call(ftypes.StartFma(), depends=tuple(markers))
            for contrib in (c for keep in fma_prod_markers for c in keep.depends):
                free = dabseg.add_call(ftypes.FreeMarker(), contrib.args, depends=(init,))
                contrib.add_relation(free, "free")
                contrib.add_relation(init, "fma")
        else:
            init = dabseg.add_call(ftypes.StartAccumulate(call.func), depends=tuple(markers))
        for marker in markers:
            marker.add_relation(init, "starter")

        # Contribute / FreeMarker
        contribs: list[Call] = []
        for arg, marker in zip(call.args, markers):
            acall = arg.call
            if is_fma_prod(acall) and acall.n_args == 2:
                depends = tuple((init, *non_fma_prod_markers))
                contrib = dabseg.add_call(ftypes.ContributeFmaProd2(), acall.args, depends=depends)
            elif is_fma_prod(call):
                contrib = dabseg.add_call(ftypes.ContributeToFmaProd(), (arg,), depends=(init,))
            elif acall.is_a(ftypes.FinishFmaProd):
                contrib = dabseg.add_call(ftypes.ContributeFmaProd(), (arg,), depends=(init,))
            else:
                contrib = dabseg.add_call(ftypes.Contribute(call.func), (arg,), depends=(init,))
            contrib.add_relation(marker, "marker")
            marker.add_relation(contrib, "contrib")
            contribs.append(contrib)

        # Finalize
        if is_fma_prod(call):
            assert call.n_args > 2
            finish = dabseg.add_call(ftypes.FinishFmaProd(), (init[0],), depends=tuple(contribs))
        else:
            finish = dabseg.add_call(ftypes.FinishAccumulate(), (init[0],), depends=tuple(contribs))

        finish.add_relation(init, "starter")
        init.add_relation(finish, "finish")

        dabseg.rebind(finish, call)
