# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

from sortedcontainers import SortedSet

from symforce import typing as T

from .ftypes import Func


class Dabseg:
    """
    Directed Acyclic Bipartite Expression Graph.
    This class represents a bipartite directed acyclic graph containing calls and values.
    """

    def __init__(self) -> None:
        self._val_data: list[ValData] = []
        self._call_data: list[CallData] = []
        self.data_to_id: dict[CallData, CallId] = {}
        self.val_rebinds: dict[ValId, ValId] = {}
        self.call_rebinds: dict[CallId, CallId] = {}
        self.unique_used: set[ValId] = set()
        self.final_call_ids: SortedSet[CallId] = SortedSet()

    def call_data(self, call_id: CallId) -> CallData:
        return self._call_data[call_id]

    def val_data(self, call_id: ValId) -> ValData:
        return self._val_data[call_id]

    def call(self, call_id: CallId, *, rebind: bool = True) -> Call:
        if rebind:
            while (new_idx := self.call_rebinds.get(call_id)) is not None:
                call_id = new_idx
        assert call_id < len(self._call_data)
        return Call(dabseg=self, call_id=call_id)

    def val(self, val_id: ValId, *, rebind: bool = True) -> Val:
        if rebind:
            while (new_idx := self.val_rebinds.get(val_id)) is not None:
                val_id = new_idx
        assert val_id < len(self._val_data)
        return Val(dabseg=self, val_id=val_id)

    def add_call(
        self,
        func: Func,
        args: tuple[Val, ...] = (),
        *,
        depends: tuple[Call, ...] = (),
        fix_accumulator: bool = True,
    ) -> Call:
        return self.add_call_data(
            CallData(
                func=func,
                arg_ids=tuple(a.id for a in args),
                depends=tuple(call.id for call in depends),
                relations={},
            ),
            fix_accumulator=fix_accumulator,
        )

    def add_call_data(
        self,
        cdata: CallData,
        fix_accumulator: bool = False,
    ) -> Call:
        if fix_accumulator:
            cdata = self.satisfy_accumulator(cdata)
        if cdata.is_unique or (cid := self.data_to_id.get(cdata)) is None:
            new_id = CallId(len(self._call_data))
            out_ids = tuple(ValId(len(self._val_data) + i) for i in range(cdata.func.n_outs))
            self._val_data.extend(ValData(new_id, OutIdx(i)) for i in range(cdata.func.n_outs))
            cdata.finalize(out_ids, new_id)
            self._call_data.append(cdata)
            self.data_to_id[cdata] = new_id

            return self.call(new_id)
        else:
            return self.call(cid)

    def satisfy_unique(self, data: CallData) -> CallData:
        """
        Make sure all the arguments are used exactly once.
        If an argument is used multiple times, a duplicate is created.
        """

        def inner(val_id: ValId) -> ValId:
            val = self.val(val_id)
            if not val.call.is_unique or val_id not in self.unique_used:
                return val.id
            if val.call.n_outs != 1:
                raise ValueError("Cannot satisfy unique for calls with multiple outputs.")

            new_data = self.call_data(val.call.id).copy()
            new_data.id = CallId(-1)
            new_func = self.add_call_data(new_data)
            return new_func.outs[0].id

        data.arg_ids = tuple(inner(arg_id) for arg_id in data.arg_ids)
        return data

    def satisfy_accumulator(self, data: CallData) -> CallData:
        """
        Nested functions are expanded and arguments are sorted, ensuring a canonical representation.
        """
        if not data.func.is_accumulator:
            return data

        def inner(var_id: ValId) -> T.Iterator[ValId]:
            val = self.val(var_id)
            if val.call.func is data.func:
                yield from (a for arg in val.call.args for a in inner(arg.id))
            else:
                yield var_id

        data.arg_ids = tuple(sorted(fixed for arg_id in data.arg_ids for fixed in inner(arg_id)))
        return data

    def set_finalize(self, funcs: list[Call]) -> None:
        """
        Set the final calls representing the leaves of the graph.
        These calls represent the output when generating code.
        """
        self.final_call_ids = SortedSet(f.id for f in funcs)

    def call_iter(self) -> T.Iterator[Call]:
        return (self.call(f) for f in self.func_id_iter())

    def func_id_iter(
        self,
    ) -> T.Iterator[CallId]:
        """
        Iterate over the function ids in the graph.
        """
        visited: set[CallId] = set()
        relations: set[CallId] = set()

        def parent_iter(call_id: CallId, lv: int = 0) -> T.Iterator[CallId]:
            if call_id in visited:
                return
            visited.add(call_id)
            for parent in self.call(call_id).parents():
                yield from parent_iter(parent.id, lv + 1)
            yield call_id
            relations.update(rel.id for rel in self.call(call_id).relations.values())

        yield from (fid for root_id in self.final_call_ids for fid in parent_iter(root_id))
        yield from (fid for rel in relations for fid in parent_iter(rel))

    def val_iter(self) -> T.Iterator[Val]:
        """
        Iterate over the values in the graph.
        """
        yield from (v for func in self.call_iter() for v in func.outs)

    def rebind(self, new_call: Call, old_call: Call) -> Call:
        """
        Rebind a call to another call. Used when transforming the graph.
        """
        if new_call.id == old_call.id:
            return new_call
        assert old_call.id not in self.call_rebinds
        self.call_rebinds[old_call.id] = new_call.id
        for new_val, old_val in zip(new_call.outs, old_call.outs):
            assert old_val.id not in self.val_rebinds
            assert old_val.id != new_val.id
            self.val_rebinds[old_val.id] = new_val.id
        return new_call

    def clean(self) -> Dabseg:
        """
        Clean the graph by removing nodes that do not contribute to any of the final calls.
        """
        out = Dabseg()
        vmap: dict[ValId, ValId] = {}
        cmap: dict[CallId, CallId] = {}
        for call in self.call_iter():
            new_func = out.add_call_data(
                CallData(
                    func=call.func,
                    arg_ids=tuple((vmap[v.id] for v in call.args)),
                    depends=tuple(cmap[dep.id] for dep in call.depends),
                    relations=self.call_data(call.id).relations,
                )
            )
            vmap.update({v_old.id: v_new.id for v_old, v_new in zip(call.outs, new_func.outs)})
            cmap.update({call.id: new_func.id})
        out.final_call_ids = SortedSet(cmap[fid] for fid in self.final_call_ids)

        for call_data in out._call_data:
            call_data.relations = {k: cmap[fid] for k, fid in call_data.relations.items()}

        return out

    def __repr__(self) -> str:
        return "\n".join(self.call(CallId(i)).to_str(1) for i in range(len(self._call_data)))


debug_graph: list[Dabseg | None] = [None]


def set_debug_graph(graph: Dabseg) -> None:
    debug_graph[0] = graph


def clear_debug_graph() -> None:
    debug_graph[0] = None


class CallId(int):
    def __repr__(self) -> str:
        if debug_graph[0] is not None:
            return f"ID_{(debug_graph[0].call(self).to_str(1))}"
        return f"C{int(self)}"


class ValId(int):
    def __repr__(self) -> str:
        if debug_graph[0] is not None:
            return f"ID_{(debug_graph[0].val(self).to_str(1))}"
        return f"V{int(self)}"


class OutIdx(int):
    def __repr__(self) -> str:
        return f"O{int(self)}"


@dataclass(frozen=True)
class ValData:
    call_id: CallId
    out_idx: OutIdx


class CallData:
    func: Func
    arg_ids: tuple[ValId, ...]
    depends: tuple[CallId, ...]
    relations: dict[str, CallId]
    out_ids: tuple[ValId, ...]
    id: CallId

    def __init__(
        self,
        func: Func,
        arg_ids: tuple[ValId, ...],
        depends: tuple[CallId, ...],
        relations: dict[str, CallId],
    ) -> None:
        assert isinstance(func, Func)
        self.func = func
        self.arg_ids = arg_ids
        self.depends = depends
        self.relations = relations

    def finalize(self, outs: tuple[ValId, ...], fid: CallId) -> None:
        self.out_ids = outs
        self.id = fid

    @property
    def is_unique(self) -> bool:
        return self.func.is_unique

    def copy(self) -> CallData:
        return CallData(
            func=self.func,
            arg_ids=self.arg_ids,
            depends=self.depends,
            relations=self.relations,
        )

    def __hash__(self) -> int:
        if self.is_unique:
            return hash((self.func, self.arg_ids, self.id))
        return hash((self.func, self.arg_ids))

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, CallData)
            and self.func == value.func
            and self.arg_ids == value.arg_ids
        )


class Val:
    """
    A class representing a value in the expression tree.
    The value is defined by the function it belongs to and the index of the output.
    """

    _funcid: CallId
    _dabseg: Dabseg
    _id: ValId

    _str: T.Optional[str] = None

    def __init__(self, *, dabseg: Dabseg, val_id: ValId) -> None:
        self._dabseg = dabseg
        self._id = ValId(val_id)

    @property
    def dabseg(self) -> Dabseg:
        return self._dabseg

    @property
    def vdata(self) -> ValData:
        return self._dabseg.val_data(self._id)

    @property
    def id(self) -> ValId:
        return self._id

    @property
    def call(self) -> Call:
        return self._dabseg.call(self.vdata.call_id)

    @property
    def idx(self) -> int:
        return self.vdata.out_idx

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Val) and self.call == other.call and self.idx == other.idx

    def to_str(self, depth: int) -> str:
        if self._str is None:
            return self.call.to_str(depth) + (f"[{self.idx:d}]" if self.idx else "[]")
        return self._str

    def __repr__(self) -> str:
        return self.to_str(1)


class Call:
    """
    A parent class representing a function in the expression tree.
    The function is defined by its arguments and the data it holds.
    """

    _dabseg: Dabseg
    _id: CallId

    def __init__(self, *, dabseg: Dabseg, call_id: CallId) -> None:
        self._dabseg = dabseg
        self._id = CallId(call_id)

    @property
    def _fdata(self) -> CallData:
        return self._dabseg.call_data(self._id)

    @property
    def id(self) -> CallId:
        return self._id

    @property
    def args(self) -> tuple[Val, ...]:
        return tuple(self._dabseg.val(i) for i in self._fdata.arg_ids)

    @property
    def n_args(self) -> int:
        return len(self._fdata.arg_ids)

    @property
    def outs(self) -> list[Val]:
        return [self._dabseg.val(i, rebind=False) for i in self._fdata.out_ids]

    @property
    def n_outs(self) -> int:
        return self._fdata.func.n_outs

    @property
    def is_accumulator(self) -> bool:
        return self._fdata.func.is_accumulator

    @property
    def is_fmaprod(self) -> bool:
        return self._fdata.func.is_fmaprod

    @property
    def is_contribute(self) -> bool:
        return self._fdata.func.is_contribute

    @property
    def is_contrib_marker(self) -> bool:
        return self._fdata.func.is_contrib_marker

    @property
    def is_start_accumulator(self) -> bool:
        return self._fdata.func.is_start_accumulator

    @property
    def is_finish_accumulator(self) -> bool:
        return self._fdata.func.is_finish_accumulator

    @property
    def is_unique(self) -> bool:
        return self._fdata.func.is_unique

    @property
    def depends(self) -> tuple[Call, ...]:
        return tuple(self._dabseg.call(cid) for cid in self._fdata.depends)

    def relation(self, key: str) -> Call:
        return self._dabseg.call(self._fdata.relations[key])

    @property
    def relations(self) -> dict[str, Call]:
        return {k: self._dabseg.call(v) for k, v in self._fdata.relations.items()}

    @property
    def func(self) -> Func:
        return self._fdata.func

    def parents(
        self,
        include_depends: bool = True,
    ) -> list[Call]:
        """
        Returns the functions that need to be fired before this function.
        """
        iters = (
            (a.call.id for a in self.args),
            ((call.id for call in self.depends) if include_depends else []),
        )
        return [self._dabseg.call(v) for v in SortedSet(v for it in iters for v in it)]

    def add_relation(self, call: Call, key: str) -> None:
        assert key not in self._fdata.relations
        self._fdata.relations[key] = call.id

    def __getitem__(self, idx: int) -> Val:
        return self.outs[idx]

    def to_str(self, depth: int = 1) -> str:
        fstring = f"{self.func.__class__.__name__}#{self.id:d}"
        if self.func.data is not None:
            fstring += f"<{self.func.data}>"
        if depth == 0:
            return fstring
        else:
            arg_strs = [a.to_str(depth - 1) for a in self.args]
            return f"{fstring}({','.join(arg_strs)})"

    def __repr__(self) -> str:
        return self.to_str(2)

    def is_a(self, cls: T.Type[Func]) -> bool:
        return isinstance(self.func, cls)
