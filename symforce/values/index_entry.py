# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import dataclasses
import sys

from symforce import python_util
from symforce import typing as T


@dataclasses.dataclass
class IndexEntry:
    """
    Contains the structural information needed to reconstruct a single value of
    a `Values` from storage in method `Values.from_storage_index`

    Meant to be a python parallel to index_entry_t in symforce.lcm

    Attributes:
        For `entry: IndexEntry = v.index()[key]` for `Values v` and string `key`
        offset:
            The index of `StorageOps.to_storage(v)` at which
            `StorageOps.to_storage(v[key])` begins
        storage_dim:
            The length of `StorageOps.to_storage(v[key])`
        shape:
            If datatype() is np.ndarray or sf.Matrix, it's the shape of `v[key]`.
            Otherwise, it's None
        item_index:
            `v[key].index()` if datatype() is Values,
            if datatype() is list or tuple, is dict `d` where `d[f"{key}_{i}"]`
            equals the `IndexEntry` of `v[key][i]`, and
            otherwise is None
    """

    offset: int
    storage_dim: int
    # We do not store the datatype as an ordinary field because types are not serializable. Still,
    # we set the stored_datatype to be an InitVar so that we can translate it into a serializable
    # format.
    stored_datatype: dataclasses.InitVar[T.Type]
    # _module and _qualname are private because they need to be of a very particular format for
    # the method datatype to work. To support this, we mark them as not being init fields and
    # instead generate them in __post_init__ from stored_datatype. Together, they represent the
    # stored_datatype in a serializeable format.
    _module: str = dataclasses.field(init=False)
    _qualname: str = dataclasses.field(init=False)
    shape: T.Optional[T.Tuple[int, ...]] = None
    # T.Any should actually be T.Dict[str, IndexEntry], but mypy does not yet support
    # recursive types: https://github.com/python/mypy/issues/731
    item_index: T.Optional[T.Dict[str, T.Any]] = None

    def __post_init__(self, stored_datatype: T.Type) -> None:
        self._module = stored_datatype.__module__
        self._qualname = stored_datatype.__qualname__

    def datatype(self) -> T.Type:
        """
        Returns the type indexed by self

        Example:
            IndexEntry(offset, storage_dim, stored_datatype).datatype() returns stored_datatype

        Precondition:
            The datatype stored must have had its module loaded (i.e., if the stored datatype is
            sf.rot3, symforce.geo must have been imported).
            The datatype must also be accesible from the module (dynamically created types do not
            do this. For example, the sf.Matrix types with more than 10 rows or columns)
        """
        assert "<locals>" not in self._qualname.split("."), (
            f"Datatype {self._qualname} must be accesible from the module: dynamically created"
            + " types do not do this. For example, the sf.Matrix types with more than 10 rows or"
            + " or columns."
        )
        return python_util.getattr_recursive(sys.modules[self._module], self._qualname.split("."))
