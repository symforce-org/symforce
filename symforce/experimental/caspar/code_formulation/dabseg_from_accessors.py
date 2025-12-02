# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

from itertools import product

import symforce.symbolic as sf
from symforce.ops.lie_group_ops import LieGroupOps as Ops

from ..memory import accessors
from ..memory.pair import is_pair
from . import ftypes
from . import function_fixers
from .cpse import do_cpse
from .dabseg import Call
from .dabseg import Dabseg
from .dabseg import Val
from .dabseg import set_debug_graph
from .ftypes import EXPR_TO_FUNC


def expr_to_val(dabseg: Dabseg, expr: sf.Basic, expr_map: dict[sf.Basic, Val]) -> Val:
    if (out := expr_map.get(expr)) is not None:
        return out

    if expr.is_Number or isinstance(expr, (int, float)):
        return expr_map.setdefault(expr, dabseg.add_call(ftypes.Store(data=float(expr)))[0])  # type: ignore[arg-type]

    func: Call
    if expr.is_Symbol:
        raise ValueError(f"Symbol {expr} not found in arg_values")
    elif isinstance(expr, sf.Piecewise):
        args = list(expr.args)
        if len(args) != 4:
            raise ValueError("Only support Piecewise with 1 condition")
        if not args[-1]:
            raise ValueError("Last argument must be True")
        true_val = expr_to_val(dabseg, expr.args[0], expr_map)
        false_val = expr_to_val(dabseg, expr.args[2], expr_map)
        comp_str = ftypes.Ternary.compmap[type(expr.args[1])]
        comp_var0, comp_var1 = (expr_to_val(dabseg, arg, expr_map) for arg in expr.args[1].args)
        func = dabseg.add_call(
            ftypes.Ternary(data=comp_str), args=(comp_var0, comp_var1, true_val, false_val)
        )
    else:
        var_args = tuple(expr_to_val(dabseg, arg, expr_map) for arg in expr.args)
        func = dabseg.add_call(EXPR_TO_FUNC[type(expr)](), args=var_args)
    return expr_map.setdefault(expr, func[0])


def make_dabseg(
    inputs: list[accessors._ReadAccessor],
    outputs: list[accessors._WriteAccessor],
) -> Dabseg:
    expr_map: dict[sf.Basic, Val] = {}
    dabseg = Dabseg()

    for arg in inputs:
        if is_pair(arg.storage):
            storage0 = Ops.to_storage(arg.storage[0])
            storage1 = Ops.to_storage(arg.storage[1])
            for i, indices in enumerate(arg.chunk_indices):
                call = dabseg.add_call(
                    ftypes.READ_FUNCS[len(indices) * 2](
                        data=(arg.name, i), custom_code=arg.read_template(i)
                    )
                )
                for (storage, el), var in zip(product((storage0, storage1), indices), call.outs):
                    expr_map[storage[el]] = var
                    var._str = storage[el].name  # noqa: SLF001

        else:
            storage = Ops.to_storage(arg.storage)

            for i, indices in enumerate(arg.chunk_indices):
                call = dabseg.add_call(
                    ftypes.READ_FUNCS[len(indices)](
                        data=(arg.name, i), custom_code=arg.read_template(i)
                    ),
                )
                for el, var in zip(indices, call.outs):
                    expr_map[storage[el]] = var
                    var._str = storage[el].name  # noqa: SLF001

    leaves = []
    for out in outputs:
        if is_pair(out.storage):
            expr_lists = [Ops.to_storage(out.storage[0]), Ops.to_storage(out.storage[1])]
            for i, indices in enumerate(out.chunk_indices):
                expr_args = [exprs[i] for exprs in expr_lists for i in indices]
                if out.SKIP_IF_ALL_ZERO and all(e.is_zero for e in expr_args):
                    continue
                call = dabseg.add_call(
                    ftypes.WRITE_FUNCS[len(indices) * 2](
                        data=(out.name, i), custom_code=out.write_template(i)
                    ),
                    tuple((expr_to_val(dabseg, a, expr_map) for a in expr_args)),
                )
                leaves.append(call)
        else:
            exprs = Ops.to_storage(out.storage)
            for i, indices in enumerate(out.chunk_indices):
                expr_args = [exprs[j] for j in indices]
                if out.SKIP_IF_ALL_ZERO and all(e.is_zero for e in expr_args):
                    continue
                call = dabseg.add_call(
                    ftypes.WRITE_FUNCS[len(indices)](
                        data=(out.name, i), custom_code=out.write_template(i)
                    ),
                    tuple((expr_to_val(dabseg, a, expr_map) for a in expr_args)),
                )
                leaves.append(call)
    dabseg.set_finalize(leaves)
    for call in dabseg.call_iter():
        call = function_fixers.fix_pow(dabseg, call)
        call = function_fixers.unpack_square(dabseg, call)
    set_debug_graph(dabseg)
    do_cpse(dabseg, ftypes.Prod)
    do_cpse(dabseg, ftypes.Sum)
    function_fixers.split_accumulators(dabseg)
    return dabseg.clean()
