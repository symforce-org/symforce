# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import typing as T

import numpy as np

import symforce.symbolic as sf
from symforce.ops.lie_group_ops import LieGroupOps as Ops

from ..memory import accessors
from . import function_fixers
from . import function_types as ftypes
from .function_types import EXPR_TO_FUNC
from .function_types import Func
from .function_types import Var


def expr_to_var(expr: sf.Basic, expr_map: dict[sf.Basic, Var]) -> Var:
    if (out := expr_map.get(expr)) is not None:
        return out

    if expr.is_Number or isinstance(expr, (int, float)):
        return expr_map.setdefault(expr, ftypes.Store(data=float(expr))[0])  # type: ignore[arg-type]

    func: Func
    if expr.is_Symbol:
        raise ValueError(f"Symbol {expr} not found in arg_values")
    if expr in expr_map:
        return expr_map[expr]
    elif isinstance(expr, sf.Piecewise):
        args = list(expr.args)
        assert len(args) == 4, "Only support Piecewise with 1 condition"
        assert args[-1], "Last argument must be True"
        true_val = expr_to_var(expr.args[0], expr_map)
        false_val = expr_to_var(expr.args[2], expr_map)
        comp_str = ftypes.Ternary.compmap[type(expr.args[1])]
        comp_var0, comp_var1 = (expr_to_var(arg, expr_map) for arg in expr.args[1].args)
        func = ftypes.Ternary(comp_var0, comp_var1, true_val, false_val, data=comp_str)
    else:
        var_args = [expr_to_var(arg, expr_map) for arg in expr.args]
        func = EXPR_TO_FUNC[type(expr)](*var_args)
    return expr_map.setdefault(expr, func[0])


class ComputeGraphOptimizer:
    """
    Class used to reformulate a compute graph for better code performance.
    """

    def __init__(
        self,
        inputs: list[accessors._ReadAccessor],
        outputs: list[accessors._WriteAccessor],
    ):
        expr_map: dict[sf.Basic, Var] = {}

        for arg_id, arg in enumerate(inputs):
            storage = Ops.to_storage(arg.storage)

            for i, indices in enumerate(arg.chunk_indices):
                func = ftypes.READ_FUNCS[len(indices) - 1](
                    data=(arg.name, i),
                    custom_code=arg.read_template(i),
                    unique_id=arg_id,
                )
                for el, var in zip(indices, func.outs):
                    expr_map[storage[el]] = var
                    var._str = storage[el].name  # noqa: SLF001

        self.root_funcs = []
        for out in outputs:
            out_vars = [expr_to_var(expr, expr_map) for expr in Ops.to_storage(out.storage)]

            for i, indices in enumerate(out.chunk_indices):
                func = ftypes.WRITE_FUNCS[len(indices) - 1](
                    *(out_vars[i] for i in indices),
                    data=(out.name, i),
                    custom_code=out.write_template(i),
                )
                self.root_funcs.append(func)
        self.fix_pow()
        self.expand_prods()
        self.collect_pows()
        self.make_unique()
        self.extract_shared_terms(ftypes.Sum)
        self.extract_shared_terms(ftypes.Prod)
        self.fix_minus()
        self.fix_div()

        self.fix_ldexp()
        self.fix_sincos()
        self.fix_norms()
        self.fix_fma()
        self.split_store()
        self.split_accumulators()
        self.make_unique()
        self.check_unique()

    def make_unique(self) -> None:
        """
        Make sure there are no duplicate instances of functions in the graph.
        """
        to_visit: list[Func] = list(self.root_funcs)
        unique_arg: dict[Var, Var] = {}
        while to_visit:
            func = to_visit.pop(-1)
            update = False
            for arg in func.args:
                if arg not in unique_arg:
                    unique_arg[arg] = arg
                    to_visit.append(arg.func)
                    continue
                if unique_arg[arg] is not arg:
                    update = True
            if update:
                func.update_args(*(unique_arg[a] for a in func.args))

    def check_unique(self) -> None:
        """
        Check that all functions are unique. This is a debug function.
        """
        for func in self.funcs():
            for out in func.outs:
                assert out.func is func

            for arg in func.args:
                assert arg.func[arg.idx] is arg

        to_visit: list[Func] = list(self.root_funcs)
        unique_funcs: dict[Func, Func] = {}
        while to_visit:
            func = to_visit.pop(-1)
            assert all(func[i].func is func for i in range(func.n_outs))
            if func in unique_funcs:
                assert unique_funcs[func] is func
                continue
            unique_funcs[func] = func
            to_visit.extend(v.func for v in func.args)

    def funcs(
        self,
        ftype: T.Type[Func] | None = None,
        to_visit: T.Iterable[Func] | None = None,
        visited: set[int] | None = None,
    ) -> T.Generator[Func, None, None]:
        """
        Depth-first traversal of the function graph.
        """
        visited = visited or set()
        to_visit = to_visit or self.root_funcs
        for func in to_visit:
            if id(func) in visited:
                continue
            visited.add(id(func))
            yield from self.funcs(ftype, (arg.func for arg in func.args), visited)
            if ftype is None or isinstance(func, ftype):
                yield func

    def vars(self) -> T.Generator[Var, None, None]:
        """
        Generator for all variables in the function graph.
        """
        visited: set[Var] = set()
        for func in self.funcs():
            for arg in func.outs:
                if arg in visited:
                    continue
                visited.add(arg)
                yield arg

    def contribs(self) -> dict[Var, list[Func]]:
        """
        Return a dictionary of all variables and the functions using them.
        """
        contribs: dict[Var, list[Func]] = {}
        for func in self.funcs():
            for arg in func.args:
                contribs.setdefault(arg, []).append(func)
        return contribs

    def fix_pow(self) -> None:  # a**-(2/3) -> rcbrt(a)**2
        for power in self.funcs(ftypes.Pow):
            new_pow = function_fixers.fix_pow(power)
            if new_pow is not power:
                new_pow.rebind(power)

    def expand_prods(self) -> None:  # a*(b*c) -> a*b*c
        def prod_gen(arg: Var) -> T.Generator[Var, None, None]:
            if not arg.func.is_prod():
                yield arg
            else:
                for inner_arg in arg.func.args:
                    yield from prod_gen(inner_arg)

        for prod in self.funcs(ftypes.Prod):
            args = [b for a in prod.args for b in prod_gen(a)]
            new_prod = ftypes.Prod(*args)
            new_prod.rebind(prod)

    def collect_pows(self) -> None:  # sqrt(a)*sqrt(b) -> sqrt(a*b)
        ptypes = [ftypes.Square, ftypes.Rcp, ftypes.Sqrt, ftypes.RSqrt, ftypes.Cbrt, ftypes.RCbrt]
        to_check = list(self.funcs(ftypes.Prod))
        while to_check:
            prod = to_check.pop(0)
            args = []
            for ptype in ptypes:
                instances = [p for p in prod.args if isinstance(p.func, ptype)]
                if len(instances) == 0:
                    continue
                elif len(instances) == 1:
                    args.append(instances[0])
                else:
                    base = ftypes.Prod(*[p.func.args[0] for p in instances])[0]
                    to_check.append(base.func)
                    args.append(ptype(base)[0])

            common: dict[Var, list[Var]] = {}
            for arg in (a for a in prod.args if a.func.is_pow()):
                common.setdefault(arg.func.args[1], []).append(arg.func.args[0])
            for exp, bases in common.items():
                if len(bases) == 1:
                    args.append(ftypes.Pow(bases[0], exp)[0])
                else:
                    base = ftypes.Prod(*bases)[0]
                    to_check.append(base.func)
                    args.append(ftypes.Pow(base, exp)[0])

            args += [a for a in prod.args if not isinstance(a.func, ftypes.Exponent)]
            if len(args) == 1:
                new_prod = args[0].func
            else:
                new_prod = ftypes.Prod(*args)
            new_prod.rebind(prod)

    def fix_ldexp(self) -> None:  # a*2**k -> ldexp(a, k)
        new_func: Func
        for func in self.funcs(ftypes.Prod):
            if arg := next((a for a in func.args if a.func.is_store()), None):
                if (exp := np.log2(np.abs(arg.func.data))) != 0 and np.mod(exp, 1.0) == 0:
                    new_prod_args = [a for a in func.args if a is not arg]
                    if len(new_prod_args) == 1:
                        inner_arg = new_prod_args[0]
                    else:
                        inner_arg = ftypes.Prod(*new_prod_args)[0]
                    new_func = ftypes.Ldexp(inner_arg, data=int(exp))
                    if float(arg.func.data) < 0.0:
                        new_func = ftypes.Neg(new_func[0])
                    assert func[0].func is func
                    new_func.rebind(func)

    def extract_shared_terms(self, ftype: T.Type[Func]) -> None:  # cse with better search
        funcs = list(self.funcs(ftype))
        new_funcs = function_fixers.make_accumulators_shared(funcs, ftype)
        for func, new_func in zip(funcs, new_funcs):
            if new_func.n_args == 1:
                new_func = ftype(*new_func.args[0].func.args)
            if func is not new_func:
                new_func.rebind(func)
        self.make_unique()
        self.check_unique()

    def fix_div(self) -> None:  # a*rcp(b) -> a/b iff rcp(b) is only used here
        for var, contribs in self.contribs().items():
            if not (len(contribs) == 1 and contribs[0].is_prod() and var.func.is_rcp()):
                continue
            others = [a for a in contribs[0].args if a != var]
            new_prod_var = ftypes.Prod(*others)[0] if len(others) > 1 else others[0]
            new_div = ftypes.Div(new_prod_var, var.func.args[0])
            new_div.rebind(contribs[0])

    def fix_minus(self) -> None:  # a + -1 * b * c -> a - (b * c)
        for func in self.funcs(ftypes.Prod):
            funcs = (a.func for a in func.args)
            neg = next((f for f in funcs if f.is_store() and f.data == -1), None)
            if neg is not None:
                args = [a for a in func.args if a.func is not neg]
                if len(args) == 1:
                    new_var = args[0]
                else:
                    new_var = ftypes.Prod(*(a for a in func.args if a.func is not neg))[0]
                new_neg = ftypes.Neg(new_var)
                new_neg.rebind(func)

        for func in self.funcs(ftypes.Sum):
            negs = [a for a in func.args if a.func.is_neg()]
            if not negs:
                continue
            other = [a for a in func.args if not a.func.is_neg()]
            if len(negs) == 1:
                neg_part = negs[0].func.args[0]
            else:
                neg_part = ftypes.Sum(*(a.func.args[0] for a in negs))[0]

            if len(other) == 0:
                new_minus: Func = ftypes.Neg(neg_part)
            elif len(other) == 1:
                new_minus = ftypes.Minus(other[0], neg_part)
            else:
                other_part = ftypes.Sum(*other)[0]
                new_minus = ftypes.Minus(other_part, neg_part)
            new_minus.rebind(func)

    def fix_sincos(self) -> None:  # sin(a), cos(a) -> sincos(a)
        sin: dict[Var, Func] = {}
        cos: dict[Var, Func] = {}
        shared = {}
        for func in self.funcs():
            if func.is_sin():
                if func.args[0] in cos:
                    shared[func.args[0]] = (func, cos.pop(func.args[0]))
                else:
                    sin[func.args[0]] = func
            elif func.is_cos():
                if func.args[0] in sin:
                    shared[func.args[0]] = (sin.pop(func.args[0]), func)
                else:
                    cos[func.args[0]] = func

        for base, (s, c) in shared.items():
            new_sincos = ftypes.SinCos(base)
            new_sincos.rebind(s, 0, 0)
            new_sincos.rebind(c, 1, 0)

    def fix_norms(self) -> None:  # e.g. rsqrt(a**2 + b**2 + c**2) -> rnorm3(a, b, c)
        # TODO(Emil Martens): don't do rnorm if norm is used, use 1/norm instead (or vice versa)
        done_rsqrt: dict[Func, Func] = {}
        new_func: Func
        for root_typ in [ftypes.RSqrt, ftypes.Sqrt]:
            for root in self.funcs(root_typ):
                if not (inner := root.args[0].func).is_sum():
                    continue
                if root_typ is ftypes.Sqrt and inner in done_rsqrt:
                    new_func = ftypes.Rcp(done_rsqrt[inner][0])
                    new_func.rebind(root)
                    continue
                if all(
                    a.func.is_square() or (a.func.is_store() and float(a.func.data) >= 0.0)
                    for a in inner.args
                ):
                    store_vals = [float(a.func.data) for a in inner.args if a.func.is_store()]
                    new_lits = [ftypes.Store(data=v**0.5)[0] for v in store_vals]
                    other = [a.func.args[0] for a in inner.args if not a.func.is_store()]
                    norm_tyb = ftypes.Norm if root_typ is ftypes.Sqrt else ftypes.RNorm
                    new_func = norm_tyb(*new_lits, *other)
                    new_func.rebind(root)
                    if root_typ is ftypes.RSqrt:
                        done_rsqrt[inner] = new_func

    def fix_fma(self) -> None:  # see notes of ftypes.Fma
        new_sum: Func
        contribs = self.contribs()
        for sum_ in list(self.funcs(ftypes.Sum)):
            fma_prod_args: list[Var] = []
            other_args: list[Var] = []
            for arg in sum_.args:
                if (arg.func.is_prod() or arg.func.is_square()) and len(contribs[arg]) == 1:
                    if arg.func.is_square():
                        fma_prod_args.append(ftypes.FmaProd(arg.func.args[0], arg.func.args[0])[0])
                    else:
                        fma_prod_args.append(ftypes.FmaProd(*arg.func.args)[0])
                else:
                    other_args.append(arg)
            if not fma_prod_args:
                continue

            if (
                len(other_args) == 1
                and len(fma_prod_args) == 1
                and fma_prod_args[0].func.n_args == 2
            ):
                new_sum = ftypes.Fma3(*(a for a in fma_prod_args[0].func.args), other_args[0])
            elif len(other_args) == 0:
                new_sum = ftypes.FmaNone(*other_args, *fma_prod_args)
            else:
                new_sum = ftypes.Fma(*other_args, *fma_prod_args)
            new_sum.rebind(sum_)

    def split_store(self) -> None:  # split stores into unique stores
        nstores = 0
        for func in list(self.funcs()):
            if not any(a.func.is_store() for a in func.args):
                continue
            args = []
            for arg in func.args:
                if arg.func.is_store() and arg:
                    assert arg.func.unique_id == 0
                    nstores += 1
                    new_store = ftypes.Store(data=arg.func.data, unique_id=nstores)
                    args.append(new_store[0])
                else:
                    args.append(arg)
            new_func = func.__class__(*args, data=func.data)
            for i in range(func.n_outs):
                new_func.rebind(func, i, i)

    def split_accumulators(self) -> None:  # split accumulators into unique accumulators
        todo = [f for f in self.funcs() if f.is_accumulator()]
        for i, func in enumerate(todo):
            args = []
            for arg in func.args:
                if not arg.func.is_fmaprod():
                    args.append(ftypes.Contribute(arg, unique_id=i)[0])
                else:
                    args.append(arg)
            func.__class__(*args).rebind(func)
