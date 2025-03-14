# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import logging
from collections import Counter

from symforce import typing as T

from . import function_types
from .compute_graph_sorting_helpers import FuncOrderingData
from .compute_graph_sorting_helpers import VarOrderingData
from .compute_graph_sorting_helpers import argstr
from .function_types import Func
from .function_types import Var


class ComputeGraphSorter:
    """
    Class to find a topological sorting of a compute graph that minimizes register pressure.

    This method is based on the paper:
    https://inria.hal.science/hal-01956260/document

    This implementation adds support for accumulators, FMA, and multi-output functions.
    It also uses a different cost function that appears to yield better results.
    """

    def __init__(self, funcs: list[Func]):
        self.args: set[Var] = {arg for func in funcs for arg in func.outs}

        for arg in self.args:
            arg.vod = VarOrderingData(arg)

        for i, func in enumerate(funcs):
            func.fod = FuncOrderingData(func, index=i)
            for arg in func.args:
                assert arg in self.args
                arg.vod.missing_contribs.update([func])
        for func in funcs:
            for out in func.outs:
                if not out.vod.missing_contribs:
                    logging.warning("Missing contribs: " + str(out))

            if func.is_accumulator():
                assert len(func.args) > 1
                for arg in func.args:
                    arg.func.fod.contrib_parent_acc = func

        self.funcs = funcs
        self.ready: set[Func] = set()
        for f in (f for f in funcs if not f.is_store()):
            self.check_if_ready(f)

        self.ops: list = []
        self.max_registers = 0
        self.current_stack: list[int] = []
        self.ssa_count = 0

        self.reorder()

    def allocate(self, variables: list[Var]) -> None:
        """
        Add a variable to stack.
        """

        for var in variables:
            assert not var.vod.live and var.vod.register == -1

            var.vod.live = True
            var.vod.turn = self.turn
            if self.current_stack:
                var.vod.register = self.current_stack.pop(-1)
            else:
                var.vod.register = self.max_registers
                self.max_registers += 1

            var.vod.register_ssa = self.ssa_count
            self.ssa_count += 1

            if not var.vod.missing_contribs:
                self.pop_stack(var)

    def pop_stack(self, var: Var) -> None:
        """
        Remove a variable from stack.
        """
        if var.func.is_store():
            return
        assert var.vod.is_live()
        assert var.vod.register != -1
        var.vod.live = False
        self.current_stack.append(var.vod.register)

    def use_var(self, func: Func, var: Var) -> None:
        """
        Use a variable in a function. If it is the last missing contrib, pop the variable.
        """
        assert func in var.vod.missing_contribs
        var.vod.missing_contribs -= Counter([func])

        if not var.vod.missing_contribs:
            self.pop_stack(var)

        elif len(var.vod.missing_contribs) == 1:
            next(iter(var.vod.missing_contribs)).fod.update_freeable(var)

    def use_vars(self, func: Func, varlist: T.Sequence[Var]) -> None:
        for var in varlist:
            self.use_var(func, var)

    def check_if_ready(self, func: Func) -> None:
        """
        Check if a function is ready to be called.
        """
        assert func.fod.is_not_ready()
        if (
            func.is_fmaprod()
            and func.fod.contrib_parent_acc.is_fma()
            and func.fod.contrib_parent_acc.fod.acc_count == 0
        ):
            return

        if not func.fod.missing_args:
            func.fod.mark_ready()
            self.ready.add(func)

    def do_func(self, func: Func) -> None:
        """
        Do a function.
        """
        self.use_vars(func, func.args)
        self.allocate(func.outs)
        for out in func.outs:
            self.finish_var(out)

    def finish_var(self, out: Var) -> None:
        """
        Finish a variable, making it available to other functions.
        """
        if len(out.vod.missing_contribs) == 1:
            next(iter(out.vod.missing_contribs)).fod.update_freeable(out)
        for contrib in out.vod.missing_contribs:
            for var in (var for var in contrib.args if var.vod.is_live()):
                contrib.fod.update_aff(var)
        if len(out.vod.missing_contribs) == 1:
            for contrib in out.vod.missing_contribs:
                for var in (var for var in contrib.args if var.vod.is_live()):
                    contrib.fod.update_freeable(var)

        out.vod.live = True
        for contrib in out.vod.missing_contribs.copy():
            contrib.fod.missing_args.remove(out)
            self.check_if_ready(contrib)

    def do_contribute(self, func: function_types.Func) -> None:
        """
        Do a contribution.

        Different behaviour if its first, last or in between contribution.
        The first and last contribution produces no code.
        """

        acc = func.fod.contrib_parent_acc
        acc.fod.acc_count += 1
        acc.fod.missing_args.remove(func[0])

        if acc.fod.acc_count == 1:
            for arg in acc.fod.missing_args:
                arg.func.fod.update_aff(acc[0])
                if arg.func.is_fmaprod():
                    self.check_if_ready(arg.func)

        elif acc.fod.acc_count == 2:
            assert isinstance(acc.fod.acc_prev_contrib, function_types.Func)
            self.use_vars(acc.fod.acc_prev_contrib, acc.fod.acc_prev_contrib.args)
            self.allocate(acc.outs)
            for arg in acc.fod.missing_args:
                arg.func.fod.update_freeable(acc[0])
            func.fod.contrib_prev_var = acc.fod.acc_prev_contrib.args[0]
        else:
            func.fod.contrib_prev_var = acc[0]

        if 1 < acc.fod.acc_count < acc.n_args:
            self.use_vars(func, func.args)

        if acc.fod.acc_count == acc.n_args:
            func.fod.contrib_is_last = True
            self.check_if_ready(acc)

        acc.fod.acc_prev_contrib = func

    def do_accumulator(self, acc: Func) -> None:
        """
        Finnish an accumulator.

        Accumulators are called after all contributions are done.
        """
        assert isinstance(acc.fod.acc_prev_contrib, Func)
        if not (last_contrib := acc.fod.acc_prev_contrib).is_fmaprod():
            self.use_vars(last_contrib, last_contrib.args)
        self.finish_var(acc.outs[0])

    def do_fmaprod(self, func: Func) -> None:
        """
        Do an fma product.

        This consest of both finishing the product and contributing to an fma.
        """
        fma = func.fod.contrib_parent_acc
        fma.fod.acc_count += 1
        fma.fod.missing_args.remove(func[0])

        if func.n_args > 2:
            assert isinstance(func.fod.acc_prev_contrib, Func)
            self.use_vars(func.fod.acc_prev_contrib, func.fod.acc_prev_contrib.args)
            self.use_var(fma, func[0])
        else:
            self.use_vars(func, func.args)

        if fma.fod.acc_count == 1:
            assert fma.is_fma_none()
            self.allocate(fma.outs)
            for arg in fma.fod.missing_args:
                arg.func.fod.update_aff(fma[0])
                arg.func.fod.update_freeable(fma[0])

        elif fma.fod.acc_count == 2 and not fma.fod.acc_prev_contrib.is_fmaprod():  # type: ignore[union-attr]
            assert isinstance(fma.fod.acc_prev_contrib, Func)

            self.allocate(fma.outs)
            for arg in fma.fod.missing_args:
                arg.func.fod.update_freeable(fma[0])
            self.use_vars(fma.fod.acc_prev_contrib, fma.fod.acc_prev_contrib.args)
            func.fod.contrib_prev_var = fma.fod.acc_prev_contrib.args[0]
        else:
            func.fod.contrib_prev_var = fma[0]

        if fma.fod.acc_count == fma.n_args:
            func.fod.contrib_is_last = True
            self.check_if_ready(fma)

        fma.fod.acc_prev_contrib = func

    def reorder(self) -> None:
        """
        Main function for reordering the Funcs.
        """
        for turn in range(sum(not f.is_store() for f in self.funcs)):
            self.turn = turn

            it = iter(self.ready)
            func = next(it)
            for other in self.ready:
                if func.fod.worse_than(other.fod):
                    func = other
            self.ready.remove(func)
            if func.is_fmaprod():
                self.do_fmaprod(func)
            elif func.is_contrib():
                self.do_contribute(func)
            elif func.is_accumulator():
                self.do_accumulator(func)
            else:
                self.do_func(func)
            func.fod.mark_done()
            self.ops.append(func)

        assert len(self.current_stack) == self.max_registers, "Dangeling variables"

    @staticmethod
    def print_func(func: Func) -> str:
        """
        Returns the cuda code line for a single Func.
        """
        outs = func.outs
        args: list[Var]
        if func.is_fmaprod():
            acc = func.fod.contrib_parent_acc
            outs = func.fod.contrib_parent_acc.outs
            assert acc.is_fma_none() or func.fod.contrib_prev_var is not None
            if func.n_args > 2:
                assert isinstance(func.fod.acc_prev_contrib, Func)
                args = [func.fod.acc_prev_contrib.args[0], func[0]]
            else:
                args = [*func.args]
            if func.fod.contrib_prev_var is not None:
                args.append(func.fod.contrib_prev_var)

        elif func.is_fma_any():
            assert isinstance(func.fod.acc_prev_contrib, Func)
            if func.fod.acc_prev_contrib.is_fmaprod():
                return "// FMA"
            args = [func[0], func.fod.acc_prev_contrib.args[0]]
        elif func.is_contrib():
            if func.fod.contrib_prev_var is None or func.fod.contrib_is_last:
                return "// contrib"
            args = [func.fod.contrib_prev_var, func.args[0]]
            outs = func.fod.contrib_parent_acc.outs
        elif func.is_accumulator():
            assert isinstance(func.fod.acc_prev_contrib, Func)
            args = [func[0], func.fod.acc_prev_contrib.args[0]]
        else:
            args = list(func.args)

        args_strs = [argstr(a) for a in args]
        outs_strs = [argstr(a) for a in outs]
        return func.assign_code(outs_strs, args_strs)

    def get_lines(self) -> list[str]:
        """
        Returns the cuda code lines for all the reordered Funcs.
        """
        lines = [self.print_func(func) for func in self.ops]
        return [line for line in lines if not line.startswith("//")]
