# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

from textwrap import indent

import symforce.symbolic as sf
from symforce import jacobian_helpers
from symforce import typing as T
from symforce.ops import StorageOps as Ops
from symforce.ops.interfaces import Storage

StorageT = T.TypeVar("StorageT", bound=T.Storable)


class Pair(T.Generic[StorageT]):
    def __init__(self, first: StorageT, second: StorageT):
        if type(first) is not type(second):
            raise ValueError("First and second must be of the same type")
        self.data = [first, second]

    def __iter__(self) -> T.Iterable[StorageT]:
        return iter(self.data)

    def __getitem__(self, idx: int) -> StorageT:
        return self.data[idx]

    def __repr__(self) -> str:
        return f"Pair(\n{indent(repr(self[0]), '  ')},\n{indent(repr(self[1]), '  ')}\n)"


def is_pair(thing: T.Union[T.Type[Storage], T.Type[Pair], Storage, Pair]) -> bool:
    return thing is Pair or T.get_origin(thing) is Pair or isinstance(thing, Pair)


def get_symbolic(storage_or_pair: T.Union[Storage, Pair], name: str) -> T.Union[Storage, Pair]:
    if is_pair(storage_or_pair):
        StorageT = get_memtype(storage_or_pair)
        return Pair(*(Ops.symbolic(StorageT, f"{name}_{k}") for k in ["first", "second"]))
    return Ops.symbolic(storage_or_pair, name)


def jacobians(
    fx: Storage, storage_or_pair: T.Union[Storage, Pair]
) -> T.Union[Pair[sf.Matrix], sf.Matrix]:
    diff = lambda fx, x: jacobian_helpers.tangent_jacobians(fx, [x])[0]
    if is_pair(storage_or_pair):
        storage_or_pair = T.cast(Pair, storage_or_pair)
        return Pair(diff(fx, storage_or_pair[0]), diff(fx, storage_or_pair[1]))
    return diff(fx, storage_or_pair)


def get_elements(storage_or_pair: T.Union[Storage, Pair]) -> T.List:
    if is_pair(storage_or_pair):
        storage_or_pair = T.cast(Pair, storage_or_pair)
        return [
            *Ops.to_storage(storage_or_pair[0]),
            *Ops.to_storage(storage_or_pair[1]),
        ]
    return Ops.to_storage(storage_or_pair)


def get_memtype(
    storage_or_pair: T.Union[T.Type[Storage], T.Type[Pair], Storage, Pair],
) -> T.Type[Storage]:
    if is_pair(storage_or_pair):
        if storage_or_pair is Pair:
            raise ValueError("Cannot get type from unannotated Pair")
        elif T.get_origin(storage_or_pair) is Pair:
            return T.get_args(storage_or_pair)[0]
        elif isinstance(storage_or_pair, Pair):
            return storage_or_pair.data[0].__class__
    if isinstance(storage_or_pair, type):
        storage_or_pair = T.cast(T.Type[Storage], storage_or_pair)
        return storage_or_pair
    storage_or_pair = T.cast(Storage, storage_or_pair)
    return storage_or_pair.__class__
