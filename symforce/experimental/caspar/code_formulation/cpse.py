# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from __future__ import annotations

import typing as T
from itertools import combinations

from sortedcontainers import SortedList

from .dabseg import Call
from .dabseg import Dabseg
from .dabseg import Val
from .dabseg import ValId
from .ftypes import Func


def both_in(pair: tuple[ValId, ValId], sorted_set: SortedList[ValId]) -> bool:
    return pair[0] in sorted_set and pair[1] in sorted_set


def any_in(pair: tuple[ValId, ValId], sorted_set: SortedList[ValId]) -> bool:
    return pair[0] in sorted_set or pair[1] in sorted_set


def find_union(*argsets: SortedList[ValId]) -> SortedList[ValId]:
    iters = [iter(argset) for argset in argsets]
    values = [next(it) for it in iters]
    target = values[0]
    common: SortedList[ValId] = SortedList()
    try:
        while True:
            for i, it_i in enumerate(iters):
                while values[i] < target:
                    values[i] = next(it_i)
                if values[i] > target:
                    target = values[i]
                    break
            else:
                common.add(target)
                for j, it_j in enumerate(iters):
                    values[j] = next(it_j)
                    target = values[0]
    except StopIteration:
        pass
    return common


def replace_with(
    idx: ValId, to_replace: SortedList[ValId], argset: SortedList[ValId]
) -> SortedList[ValId]:
    out = SortedList([idx])
    i = 0
    for v in argset:
        if i < len(to_replace) and v == to_replace[i]:
            i += 1
        else:
            out.add(v)
    return out


OFFSET = 2**31


def _cpse_impl(
    argsets: list[SortedList[ValId]],
) -> tuple[list[SortedList[ValId]], list[SortedList[ValId]]]:
    """
    Iteratively finds and eliminates common partial subexpressions from the given sets of arguments.
    """
    tmp_vars: list[SortedList[ValId]] = []
    pair_to_uses: dict[tuple[ValId, ValId], set[int]] = {}
    for i, argset_i in enumerate(argsets):
        for pair in combinations(argset_i, 2):
            pair_to_uses.setdefault(pair, set()).add(i)
    pair_to_union: dict[tuple[ValId, ValId], SortedList[ValId]] = {}
    while (longest := max(map(len, pair_to_uses.values()), default=0)) >= 2:
        best_pair = (ValId(-1), ValId(-1))
        best_union: SortedList[ValId] = SortedList()
        for pair in (pair for pair, uses in pair_to_uses.items() if len(uses) == longest):
            if both_in(pair, best_union):
                continue
            if pair not in pair_to_union:
                union = find_union(*[argsets[i] for i in pair_to_uses[pair]])
                pair_to_union[pair] = union
            else:
                union = pair_to_union[pair]
            if len(best_union) < len(union):
                best_pair = pair
                best_union = union
        new_idx = ValId(len(tmp_vars) + OFFSET)
        tmp_vars.append(best_union)
        uses = pair_to_uses.pop(best_pair)
        for pair in list(pair_to_uses.keys()):
            if any_in(pair, best_union):
                pair_to_uses.pop(pair)
        for use in uses:
            argsets[use] = replace_with(new_idx, best_union, argsets[use])
        for i in uses:
            for pair in combinations(argsets[i], 2):
                pair_to_uses.setdefault(pair, set()).add(i)
    return argsets, tmp_vars


def do_cpse(dabseg: Dabseg, ftype: T.Type[Func]) -> None:
    """
    Perform Partial Common Subexpression Elimination on a Dabseg for a given
    commutative and associative function type.

    Example:
    1) r0=a+b+c, r1=a+c+d+e, r2=a+c+e           (original sums)
    2) r0=r3+b,  r1=r3+d+e,  r2=r3+e,  r3=a+c   (eliminate a+c)
    3) r0=r3+b,  r1=r2+d,    r2=r3+e,  r3=a+c   (eliminate r3+e)
                                                (done)
    """
    calls = [call for call in dabseg.call_iter() if call.is_a(ftype)]

    def get_args(call: Call) -> T.Iterator[ValId]:
        for arg in call.args:
            if arg.call.is_a(ftype):
                yield from get_args(arg.call)
            else:
                yield arg.id

    argsets = [SortedList(get_args(call)) for call in calls]

    argsets, tmp_varsets = _cpse_impl(argsets)
    tmp_map: dict[ValId, ValId] = {}

    def map_var(v: ValId) -> Val:
        return dabseg.val(tmp_map.get(v, v))

    unique = {argset[0]: func for argset, func in zip(argsets, calls) if len(argset) == 1}

    for i, tmp_varset in enumerate(tmp_varsets):
        tmp_id = ValId(i + OFFSET)
        if tmp_id in unique:
            func = unique[tmp_id]
        else:
            func = dabseg.add_call(ftype(), tuple(map(map_var, tmp_varset)), fix_accumulator=False)
        tmp_map[tmp_id] = func[0].id
    for func, argset in zip(calls, argsets):
        if len(argset) == 1:
            continue
        func_new = dabseg.add_call(ftype(), tuple(map(map_var, argset)), fix_accumulator=False)
        dabseg.rebind(func_new, func)
