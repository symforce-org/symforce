# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from dataclasses import field

from .function_types import Func
from .function_types import Var


@dataclass
class FuncOrderingData:
    """
    Class to store data used for ordering functions in the code generation.
    """

    func: Func
    index: int = field()  # index of function, used to ensure deterministic ordering

    missing_args: set[Var] = field(init=False)

    acc_count: int = field(default=0)
    acc_prev_contrib: Func | None = field(default=None)

    contrib_parent_acc: Func = field(init=False)
    contrib_prev_var: Var | None = field(default=None)
    contrib_is_last: bool = field(default=False)

    fma_target: Var = field(init=False)
    fma_parent: Func = field(init=False)
    fma_waiting_prods: list[Func] = field(default_factory=list)

    state: int = field(default=0)  # 0: not ready, 1: ready, 2: done

    freeable: list[set[Var]] = field(init=False)

    aff: list[set[Var]] = field(init=False)

    def __post_init__(self) -> None:
        self.missing_args = set(a for a in self.func.args if not a.func.is_store())
        self.freeable = [set() for i in range(4)]
        self.aff = [set() for i in range(4)]

    def worse_than(self, other: "FuncOrderingData") -> bool:
        for i in range(4):
            if len(self.freeable[i]) != len(other.freeable[i]):
                return len(self.freeable[i]) < len(other.freeable[i])
            if len(self.aff[i]) != len(other.aff[i]):
                return len(self.aff[i]) < len(other.aff[i])
        if self.index != other.index:
            return self.index < other.index
        return False

    def key(self) -> tuple:
        return (self.freeable, self.aff)

    def is_not_ready(self) -> bool:
        return self.state == 0

    def is_ready(self) -> bool:
        return self.state == 1

    def is_finished(self) -> bool:
        return self.state == 2

    def mark_ready(self) -> None:
        self.state = 1

    def mark_done(self) -> None:
        self.state = 2

    def update_freeable(self, var: Var, index: int = 0) -> None:
        self.freeable[index].add(var)
        new_index = index + (len(self.missing_args) > 1)
        if new_index < 4:
            for arg in self.missing_args:
                arg.func.fod.update_freeable(var, index=new_index)

    def update_aff(self, var: Var, index: int = 0) -> None:
        self.aff[index].add(var)
        new_index = index + (len(self.missing_args) > 1)
        if new_index < 4:
            for arg in self.missing_args:
                arg.func.fod.update_aff(var, index=new_index)


@dataclass
class VarOrderingData:
    """
    Class to store data used for ordering variables in the code generation.
    """

    var: Var
    live: bool = field(default=False)
    missing_contribs: Counter[Func] = field(default_factory=Counter)
    register: int = field(default=-1)
    register_ssa: int = field(default=-1)
    turn: int = field(default=-1)

    def is_live(self) -> bool:
        return self.live


def argstr(var: Var) -> str:
    if var.func.is_store():
        assert var.vod.register == -1
        return f"{var.func.data:.8e}f"
    elif var.func.is_contrib() and var.func.args[0].func.is_store():
        assert var.vod.register == -1
        return f"{var.func.args[0].func.data:.8e}f"
    else:
        assert var.vod.register_ssa != -1
        assert var.vod.register != -1
        return f"r{var.vod.register}"
