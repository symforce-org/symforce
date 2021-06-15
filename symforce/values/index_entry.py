import dataclasses

from symforce import types as T


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
        datatype:
            "Values" if `v[key]` is an instance of `Values`, else
            "np.ndarray" if `v[key]` is an instance of `numpy.ndarray`, else
            "Scalar" if `v[key]` is an instance of `symforce.sympy.Expr`,
                `symforce.sympy.Symbol`, `int`, or `float`, else
            "Matrix" if `v[key]` is an instance of `geo.Matrix`, else
            `v[key].__class__.__name__` if `v[key]` is an instance of `Storage`, else
            "List" if `v[key]` is an instance of `list` or `tuple
        shape:
            If datatype is "Matrix" or "np.ndarray", it's the shape of `v[key]` in
            the traditional sense.
            Otherwise, it's None
        item_index:
            `v[key].index()` if datatype is "Values",
            if datatype is "List" is dict `d` where `d[f"{key}_{i}"]` equals the `IndexEntry`
            of `v[key][i]`, and
            otherwise is None
    """

    offset: int
    storage_dim: int
    datatype: str
    shape: T.Optional[T.Tuple[int, ...]] = None
    # Actually a T.Dict[str, IndexEntry], but mypy does not yet support
    # recursive types: https://github.com/python/mypy/issues/731
    item_index: T.Optional[T.Dict[str, T.Any]] = None
