# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import typing as T
from collections import Counter

from ..memory.dtype import DType
from . import ftypes
from .dabseg import Call
from .dabseg import CallId
from .dabseg import Dabseg
from .dabseg import Val
from .dabseg import ValId
from .dabseg import set_debug_graph
from .ftypes import Func


class DabsegSorter:
    def __init__(self, dabseg: Dabseg, dtype: DType):
        dabseg = dabseg.clean()
        self.dtype = dtype
        self.dabseg = dabseg
        set_debug_graph(dabseg)
        self.calls = calls = list(self.dabseg.call_iter())

        self.ARG_COUNTERS = {f.id: Counter(a.id for a in f.args) for f in calls}

        self.lines: list[str] = []
        self.missing: Counter[ValId] = Counter()

        for call in self.calls:
            self.missing.update(arg.id for arg in call.args)
        self.fired: set[CallId] = set()
        self.regs: list[str] = []
        self.allocation: dict[ValId, str] = {}
        self.n_allocated = 0
        self.accumulator_starters: dict[CallId, Call] = {}
        self.contrib_counts: Counter[CallId] = Counter()
        self.contrib_marker_counts: Counter[CallId] = Counter()
        self.fmaprod_last: dict[CallId, Call] = {}

        # This is currently just a depth-first ordering of the calls,
        # see earlier version a for a more sophisticated ordering.
        self.best_ordering = tuple(self.dabseg.func_id_iter())
        for cid in self.best_ordering:
            self.do_call(dabseg.call(cid))

    def use_args(self, call: Call) -> None:
        for arg, count in self.ARG_COUNTERS[call.id].items():
            self.missing[arg] -= count
            if self.missing[arg] == 0:
                self.regs.append(self.allocation[arg])
                del self.missing[arg]

    def allocate_outs(self, call: Call) -> None:
        for out in call.outs:
            if self.regs:
                self.allocation[out.id] = self.regs.pop()
            else:
                self.allocation[out.id] = f"r{self.n_allocated}"
                self.n_allocated += 1

    def free_unused_outs(self, call: Call) -> None:
        for out in call.outs:
            if self.missing[out.id] == 0:
                self.regs.append(self.allocation[out.id])
                del self.allocation[out.id]

    def do_call(self, call: Call, skip_eval: bool = False) -> None:
        out = self.get_call_args(call, skip_eval)
        if out is not None:
            func, outs, args = out

            self.allocate_outs(call)
            out_strs = [self.allocation[out.id] for out in outs]
            arg_strs = [self.allocation[arg.id] for arg in args]
            line = func.assign_code(out_strs, arg_strs, self.dtype)
            if line:
                self.lines.append(line)
        self.free_unused_outs(call)

    def get_call_args(
        self, call: Call, skip_eval: bool = False
    ) -> T.Optional[T.Tuple[Func, list[Val], list[Val]]]:
        if call.id in self.fired:
            return None

        self.fired.add(call.id)
        self.use_args(call)
        if skip_eval:
            return None
        elif call.is_start_accumulator:
            picked = [m for m in call.depends if m.id in self.fired]
            picked = sorted(picked, key=lambda m: m.is_fmaprod)
            args: list[Val] = []
            for p in picked[:2]:
                if p.is_a(ftypes.FmaProdMarker):
                    assert len(picked) == 2
                    final_contrib = self.fmaprod_last.pop(p.args[0].call.id)
                    args.extend((final_contrib.args[0], p.args[0]))
                    self.do_call(final_contrib.relation("free"), skip_eval=True)
                else:
                    args.extend(p.args)
            self.do_call(picked[0].relation("contrib"), skip_eval=True)
            self.do_call(picked[1].relation("contrib"), skip_eval=True)
            self.contrib_counts[call.id] = 2
            return call.func, call.outs, args

        elif call.is_contribute:
            start = call.depends[0]
            n = len(start.depends)
            self.contrib_counts[start.id] += 1
            if call.is_a(ftypes.ContributeToFmaProd):
                fma = call.relation("fma")
                if self.contrib_counts[start.id] < n:
                    self.do_call(call.relation("free"), skip_eval=True)
                elif fma.id in self.fired:
                    self.do_call(call.relation("free"), skip_eval=True)
                    return call.func, [fma[0]], [fma[0], call.args[0], call.depends[0][0]]
                else:
                    self.fmaprod_last[start.relation("finish").id] = call
                    return None
            return call.func, [start[0]], [start[0], *call.args]

        elif call.is_contrib_marker:
            self.contrib_marker_counts[call.relation("starter").id] += 1
            if self.contrib_marker_counts[call.relation("starter").id] == 2:
                self.do_call(call.relation("starter"))
            return None
        else:
            return call.func, list(call.outs), list(call.args)


def get_lines(dabseg: Dabseg, dtype: DType) -> tuple[list[str], int]:
    """
    Translates dabseg calls into lines of code.
    Returns the lines of code and the required number of registers.
    """
    sorter = DabsegSorter(dabseg, dtype)
    return sorter.lines, sorter.n_allocated
