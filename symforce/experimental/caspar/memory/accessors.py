# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import logging
import textwrap
from abc import abstractmethod

from symforce import typing as T
from symforce.ops import StorageOps

from ..memory.dtype import DType
from .layouts import caspar_size
from .layouts import chunk_dim
from .layouts import get_default_caspar_layout
from .pair import get_memtype
from .pair import is_pair

PYOBJ = "pybind11::object"


class Common:
    """
    Helper class to make code templating easier.
    """

    def __init__(self, accessor: Accessor, idx: int):
        self.name = accessor.name
        self.idx_name = accessor.idx_name
        self.offset = sum(chunk_dim(chunk) for chunk in accessor.chunk_indices[:idx])
        self.dim = len(accessor.chunk_indices[idx])
        self.args = ", ".join([f"{{arg_{i}}}" for i in range(self.dim)])
        self.elem_indices = accessor.chunk_indices[idx]
        self.vec_dim = caspar_size(self.dim)

        # Template argument types
        self.storage_t = accessor.storage_t
        self.kernel_t = accessor.kernel_t

        # Vector type depends on operation type and dimension
        is_read = isinstance(accessor, _ReadAccessor)
        if self.dim == 1:
            self.VecT = self.kernel_t.lower() if is_read else self.storage_t.lower()
        else:
            base_type = self.kernel_t if is_read else self.storage_t
            self.VecT = f"{base_type}{caspar_size(self.dim)}"
        self.SharedT = self.kernel_t.lower() if is_read else self.storage_t.lower()
        # Template arguments string for function calls
        self.template_args = (
            f"<{accessor.block_size}, {self.kernel_t}, {self.storage_t}, {self.VecT}>"
        )

        # Legacy fields for backward compatibility
        self.LocalT = self.kernel_t


class DtypeKwargs(T.TypedDict):
    """Helper class for dtype-related kwargs."""

    dtype: DType
    kernel_dtype: DType


class _UsingSharedMem:
    """
    An accessor that needs shared scratch memory.
    """

    EXTRA_DATA = 0


class _UsingIndexData:
    """
    An accessor that use indexing data.
    """


class Accessor:
    """
    Parent class for all memory accessors.
    """

    KERNEL_SIG_TEMPLATE: T.ClassVar[dict[str, str]]
    KERNEL_INIT_LINES_TEMPLATE: T.ClassVar[str] = ""

    PY_SIG_TEMPLATE: T.ClassVar[dict[str, str]]
    PY_ARGS_TEMPLATE: T.ClassVar[list[str]]

    SKIP_IF_ALL_ZERO: T.ClassVar[bool] = False

    def __init__(
        self,
        name: str,
        storage: T.StorableOrType,
        *,
        dtype: DType,
        kernel_dtype: DType,
        reuse_indices_from: str | None = None,
        offset: int | None = None,
        stride: int | None = None,
        default: T.Any | None = None,
        after: str | None = None,
        block_size: int = 1024,
    ):
        assert isinstance(self, _Pairwise) == is_pair(storage)
        assert isinstance(kernel_dtype, DType)
        if dtype != kernel_dtype:
            raise NotImplementedError("Currently caspar only supports homogenious scalar types.")
        self.storage = storage
        self.offset = offset
        self.stride = stride
        self.default = default
        self.after = after
        self.idx_name = name if reuse_indices_from is None else reuse_indices_from
        self.kernel_t = kernel_dtype
        self.storage_t = dtype
        self.chunk_indices = self.get_layout(storage)
        self.block_size = block_size
        params = {
            "name": name,
            "idx_name": self.idx_name,
            "block_size": self.block_size,
            "storage_t": dtype.lower(),
            "storage_t_capitalized": dtype.capitalize(),
        }

        self.kernel_sig = {
            k.format(**params): v.format(**params) for k, v in self.KERNEL_SIG_TEMPLATE.items()
        }
        self.prep_lines = self.KERNEL_INIT_LINES_TEMPLATE.format(**params)
        self.py_sig = {k.format(**params): v for k, v in self.PY_SIG_TEMPLATE.items()}
        self.py_args = [arg.format(**params) for arg in self.PY_ARGS_TEMPLATE]

        self.name = name
        if reuse_indices_from is not None:
            assert isinstance(self, _UsingIndexData)
            self.kernel_sig.pop(f"{self.idx_name}_indices")
            self.py_sig.pop(f"{self.idx_name}_indices")
            self.py_args = self.py_args[:-1]
            self.prep_lines = ""

    def get_signature(self, name: str) -> dict[str, str]:
        params = {"name": name, "idx_name": self.idx_name, "block_size": self.block_size}
        return {k.format(**params): v for k, v in self.KERNEL_SIG_TEMPLATE.items()}

    def get_layout(self, storage: T.StorableOrType) -> list[list[int]]:
        return get_default_caspar_layout(storage, self.storage_t)

    def pre_kernel_code(self) -> str:  # noqa: PLR6301
        return ""

    def pre_calc_code(self) -> str:  # noqa: PLR6301
        return ""

    def post_calc_code(self) -> str:  # noqa: PLR6301
        return ""

    def shared_size_req(self) -> int:
        if isinstance(self, _UsingSharedMem):
            shared_t = self.kernel_t if isinstance(self, _ReadAccessor) else self.storage_t
            return (
                max((caspar_size(len(c)) for c in self.chunk_indices), default=0)
                * (self.block_size + self.EXTRA_DATA)
                * shared_t.byte_size()
            )
        return 0

    def stride_string(self) -> str:
        if self.stride is None:
            return ""
        return f"* ({self.stride})"

    def offset_string(self) -> str:
        if self.offset is None:
            return ""
        return f"+ ({self.offset})"


class _ReadAccessor(Accessor):
    """
    An accessor that reads data.
    """

    @abstractmethod
    def read_template(self, idx: int) -> str:
        raise NotImplementedError


class _WriteAccessor(Accessor):
    """
    An accessor that writes data.
    """

    @abstractmethod
    def write_template(self, idx: int) -> str:
        raise NotImplementedError


class _AddAccessor(_WriteAccessor):
    """
    An accessor that adds data.
    """

    SKIP_IF_ALL_ZERO = True


class _Sequential:
    """
    Accessor for sequential read/write memory access.

    Each thread accesses the element at its global thread index.
    """

    KERNEL_SIG_TEMPLATE = {"{name}": "{storage_t}*", "{name}_num_alloc": "unsigned int"}
    PY_SIG_TEMPLATE = {"{name}": PYOBJ}
    PY_ARGS_TEMPLATE = ["As{storage_t_capitalized}Ptr({name})", "GetNumCols({name})"]


class ReadSequential(_Sequential, _ReadAccessor):
    def read_template(self, idx: int) -> str:
        c = Common(self, idx)
        idx_str = f" global_thread_idx {self.stride_string()} {self.offset_string()}"
        return f"read_idx_{c.dim}{c.template_args}({c.name}, {c.offset}*{c.name}_num_alloc, {idx_str}, {c.args});"


class WriteSequential(_Sequential, _WriteAccessor):
    def write_template(self, idx: int) -> str:
        c = Common(self, idx)
        idx_str = f" global_thread_idx {self.stride_string()} {self.offset_string()}"
        return f"write_idx_{c.dim}{c.template_args}({c.name}, {c.offset}*{c.name}_num_alloc, {idx_str}, {c.args});"


class _Strided:
    KERNEL_SIG_TEMPLATE = {
        "{name}": "{storage_t}*",
        "{name}_num_alloc": "unsigned int",
        "{name}_offset": "int",
        "{name}_stride": "int",
    }
    PY_SIG_TEMPLATE = {"{name}": PYOBJ, "{name}_offset": "int", "{name}_stride": "int"}
    PY_ARGS_TEMPLATE = [
        "As{storage_t_capitalized}Ptr({name})",
        "GetNumCols({name})",
        "{name}_offset",
        "{name}_stride",
    ]


class ReadStrided(_Strided, _ReadAccessor):
    def read_template(self, idx: int) -> str:
        c = Common(self, idx)
        idx_str = f" global_thread_idx * {c.name}_stride + {c.name}_offset"
        return f"read_idx_{c.dim}{c.template_args}({c.name}, {c.offset}*{c.name}_num_alloc, {idx_str}, {c.args});"


class ReadStridedWithDefault(_Strided, _ReadAccessor):
    def read_template(self, idx: int) -> str:
        c = Common(self, idx)
        idx_str = f" global_thread_idx * {c.name}_stride + {c.name}_offset"
        defaults = ",".join(
            f"{float(StorageOps.to_storage(self.default)[i]):06f}f" for i in c.elem_indices
        )
        return (
            f"read_idx_with_default_{c.dim}{c.template_args}({c.name}, {c.offset}*{c.name}_num_alloc, "
            f"{c.name}_num_alloc, {idx_str}, {c.args}, {defaults});"
        )


class WriteStrided(_Strided, _WriteAccessor):
    def write_template(self, idx: int) -> str:
        c = Common(self, idx)
        idx_str = f" global_thread_idx * {c.name}_stride + {c.name}_offset"
        return f"write_idx_{c.dim}{c.template_args}({c.name}, {c.offset}*{c.name}_num_alloc, {idx_str}, {c.args});"


class AddSequential(_Sequential, _AddAccessor):
    """
    Accessor for sequentially adding to the output.

    Each thread reads, increments and writes to the element at its global thread index.
    """

    def write_template(self, idx: int) -> str:
        c = Common(self, idx)
        return f"add_idx_{c.dim}{c.template_args}({c.name}, {c.offset}*{c.name}_num_alloc, global_thread_idx, {c.args});"


class _Indexed(_UsingIndexData):
    """
    Accessor for indexed read/write memory access.

    Each thread reads/writes the element at the index specified by the input array.

    Not optimized for shared memory access or coalescing.
    """

    KERNEL_SIG_TEMPLATE = {
        "{name}": "{storage_t}*",
        "{name}_num_alloc": "unsigned int",
        "{idx_name}_indices": "unsigned int*",
    }
    KERNEL_INIT_LINES_TEMPLATE = (
        "unsigned int {idx_name}_idx = {idx_name}_indices[global_thread_idx];"
    )

    PY_SIG_TEMPLATE = {"{name}": PYOBJ, "{idx_name}_indices": PYOBJ}
    PY_ARGS_TEMPLATE = [
        "As{storage_t_capitalized}Ptr({name})",
        "GetNumCols({name})",
        "AsUintPtr({idx_name}_indices)",
    ]


class ReadIndexed(_Indexed, _ReadAccessor):
    def read_template(self, idx: int) -> str:
        c = Common(self, idx)
        return f"read_idx_{c.dim}{c.template_args}({c.name}, {c.offset}*{c.name}_num_alloc, {c.idx_name}_idx, {c.args});"


class WriteIndexed(_Indexed, _WriteAccessor):
    def write_template(self, idx: int) -> str:
        c = Common(self, idx)
        return f"write_idx_{c.dim}{c.template_args}({c.name}, {c.offset}*{c.name}_num_alloc, {c.idx_name}_idx, {c.args});"


class AddIndexed(_Indexed, _AddAccessor):
    """
    Accessor for adding to indexed elements.

    Each thread reads, increments and writes to the index specified by the input array.
    This accessor does not use atomic operations, so the indices have to be unique.

    Not optimized for shared memory access or coalescing.
    """

    def write_template(self, idx: int) -> str:
        c = Common(self, idx)
        return f"add_idx_{c.dim}{c.template_args}({c.name}, {c.offset}*{c.name}_num_alloc, {c.name}_idx, {c.args});"


class ReadShared(_UsingIndexData, _UsingSharedMem, _ReadAccessor):
    """
    Accessor for shared memory read access.

    You need to generate the shared indices using the `lib.shared_indices` function.
    All reads within a block are sorted (for better coalescence), read once and distributed within the block.
    There is a small overhead compared to `Indexed` access.
    """

    KERNEL_SIG_TEMPLATE = {
        "{name}": "{storage_t}*",
        "{name}_num_alloc": "unsigned int",
        "{idx_name}_indices": "SharedIndex*",
    }

    KERNEL_INIT_LINES_TEMPLATE = textwrap.dedent("""
    __shared__ SharedIndex {idx_name}_indices_loc[{block_size}];
    {idx_name}_indices_loc[threadIdx.x] = (global_thread_idx<problem_size ?
                                   {idx_name}_indices[global_thread_idx] :
                                   SharedIndex{{0xffffffff, 0xffff, 0xffff}});
    """).strip()

    PY_SIG_TEMPLATE = {"{name}": PYOBJ, "{idx_name}_indices": PYOBJ}
    PY_ARGS_TEMPLATE = [
        "As{storage_t_capitalized}Ptr({name})",
        "GetNumCols({name})",
        "reinterpret_cast<SharedIndex*>(AsUint2Ptr({idx_name}_indices))",
    ]

    def read_template(self, idx: int) -> str:
        c = Common(self, idx)
        return f"""
        }}}};
        load_shared<{c.dim}, {c.kernel_t}, {c.storage_t}>({c.name}, {c.offset}*{c.name}_num_alloc, {c.idx_name}_indices_loc, ({c.SharedT}*)inout_shared);
        if (global_thread_idx < problem_size) {{{{
        read_shared_{c.dim}<{c.kernel_t}>(({c.SharedT}*)inout_shared, {c.idx_name}_indices_loc[threadIdx.x].target, {c.args});
        }}}};
        __syncthreads();
        if (global_thread_idx < problem_size) {{{{
        """.strip()


class ReadUnique(_UsingSharedMem, _ReadAccessor):
    """
    Accessor for shared memory read access.
    You need to generate the shared indices using the `lib.shared_indices` function.
    All reads within a block are sorted (for better coalesence), read once and distributed within the block.
    There is a small overhead compared to `Indexed` access.
    """

    KERNEL_SIG_TEMPLATE = {"{name}": "const {storage_t}* const"}
    PY_SIG_TEMPLATE = {"{name}": PYOBJ}
    PY_ARGS_TEMPLATE = ["As{storage_t_capitalized}Ptr({name})"]

    def read_template(self, idx: int) -> str:
        c = Common(self, idx)
        return f"""
        }}}};
        load_unique<{c.dim}, {c.kernel_t}, {c.storage_t}>({c.name}, {c.offset}, ({c.SharedT}*)inout_shared);
        if (global_thread_idx < problem_size) {{{{
        read_shared_{c.dim}<{c.kernel_t}>(({c.SharedT}*)inout_shared, 0, {c.args});
        }}}};
        __syncthreads();
        if (global_thread_idx < problem_size) {{{{
        """.strip()


class AddSharedSum(_UsingIndexData, _UsingSharedMem, _AddAccessor):
    """
    Accessor for shared sum memory write access.

    Each thread adds to the value at a give index.
    You need to generate the shared indices from the indices using the `lib.shared_indices` function.
    All writes within a block are sorted (for better coalesence), written once and distributed within the block.

    Equivalent to: ``(for i, k in enumerate(indices)): out[k] += values[i]``
    """

    KERNEL_SIG_TEMPLATE = {
        "{name}": "{storage_t}* const",
        "{name}_num_alloc": "unsigned int",
        "{idx_name}_indices": "SharedIndex*",
    }

    KERNEL_INIT_LINES_TEMPLATE = textwrap.dedent("""
    __shared__ SharedIndex {idx_name}_indices_loc[{block_size}];
    {idx_name}_indices_loc[threadIdx.x] = (global_thread_idx<problem_size ?
                                   {idx_name}_indices[global_thread_idx] :
                                   SharedIndex{{0xffffffff, 0xffff, 0xffff}});
    """).strip()

    PY_SIG_TEMPLATE = {"{name}": PYOBJ, "{idx_name}_indices": PYOBJ}
    PY_ARGS_TEMPLATE = [
        "As{storage_t_capitalized}Ptr({name})",
        "GetNumCols({name})",
        "reinterpret_cast<SharedIndex*>(AsUint2Ptr({idx_name}_indices))",
    ]

    def write_template(self, idx: int) -> str:
        c = Common(self, idx)
        return f"""
        write_sum_{c.dim}<{c.kernel_t}, {c.storage_t}>(({c.SharedT}*)inout_shared, {c.args});
        }}}};
        flush_sum_shared<{c.dim}, {c.storage_t}>({c.name}, {c.offset}*{c.name}_num_alloc, {c.idx_name}_indices_loc, ({c.SharedT}*)inout_shared);
        if (global_thread_idx < problem_size) {{{{
        """.strip()


class WriteBlockSum(_UsingSharedMem, _WriteAccessor):
    """
    Accessor for summation over the block.

    Each block writes to one element.

    To do a full reduction, the user needs to calculate the final sum from the ``n // 1024``
    elements.

    This class does not use atomic add when writing to the output.
    You need to generate the shared indices using the `lib.shared_indices` function.

    Equivalent to: ``(for i, k in enumerate(indices)): out[(k//1024)] += values[i]``
    """

    KERNEL_SIG_TEMPLATE = {
        "{name}": "{storage_t}* const",
        "{name}_num_alloc": "unsigned int",
    }

    PY_SIG_TEMPLATE = {"{name}": PYOBJ}
    PY_ARGS_TEMPLATE = [
        "As{storage_t_capitalized}Ptr({name})",
        "GetNumCols({name})",
    ]

    def write_template(self, idx: int) -> str:
        c = Common(self, idx)
        return f"""
        write_sum_{c.dim}<{c.kernel_t}, {c.storage_t}>(({c.SharedT}*)inout_shared, {c.args});
        }}}};
        flush_sum_block<{c.dim}, {c.storage_t}>({c.name}, ({c.SharedT}*)inout_shared, global_thread_idx<problem_size);
        if (global_thread_idx < problem_size) {{{{
        """.strip()


class AddBlockSum(_UsingSharedMem, _AddAccessor):
    """
    Accessor for summation over the block.

    Each block adds to one element.

    To do a full reduction, the user needs to calculate the final sum from the ``n // 1024``
    elements.

    This class does not use atomic add when writing to the output.
    You need to generate the shared indices using the `lib.shared_indices` function.

    Equivalent to: ``(for i, k in enumerate(indices)): out[(k//1024)] += values[i]``
    """

    KERNEL_SIG_TEMPLATE = {
        "{name}": "{storage_t}*",
        "{name}_num_alloc": "unsigned int",
    }

    PY_SIG_TEMPLATE = {"{name}": PYOBJ}
    PY_ARGS_TEMPLATE = [
        "As{storage_t_capitalized}Ptr({name})",
        "GetNumCols({name})",
    ]

    def write_template(self, idx: int) -> str:
        c = Common(self, idx)
        return f"""
        write_sum_{c.dim}<{c.kernel_t}, {c.storage_t}>(({c.SharedT}*)inout_shared, {c.args});
        }}}};
        flush_sum_block_add<{c.dim}, {c.storage_t}>({c.name}, ({c.SharedT}*)inout_shared, global_thread_idx<problem_size);
        if (global_thread_idx < problem_size) {{{{
        """.strip()


class AddSum(_UsingSharedMem, _AddAccessor):
    """
    Accessor for adding global sum.
    """

    KERNEL_SIG_TEMPLATE = {
        "{name}": "{storage_t}* const",
    }

    PY_SIG_TEMPLATE = {"{name}": PYOBJ}
    PY_ARGS_TEMPLATE = [
        "As{storage_t_capitalized}Ptr({name})",
    ]
    EXTRA_DATA = 32 - 1024

    def write_template(self, idx: int) -> str:
        c = Common(self, idx)
        return f"""
        }}}};
        sum_store<{c.storage_t}>({self.name}_local, ({c.SharedT}*)inout_shared, {self.chunk_indices[idx][0]}, global_thread_idx < problem_size, {c.args});
        if (global_thread_idx < problem_size) {{{{
        """.strip()

    def get_layout(self, storage: T.StorableOrType) -> list[list[int]]:  # noqa: PLR6301
        return [[i] for i in range(StorageOps.storage_dim(storage))]

    def pre_calc_code(self) -> str:
        if StorageOps.storage_dim(self.storage) > 0:
            return f"__shared__ {self.storage_t} {self.name}_local[{StorageOps.storage_dim(self.storage)}];"
        return ""

    def post_calc_code(self) -> str:
        if StorageOps.storage_dim(self.storage) > 0:
            return f"sum_flush_final<{self.storage_t}>({self.name}_local, {self.name}, {StorageOps.storage_dim(self.storage)});"
        return ""


class WriteSum(AddSum):
    """
    Accessor for writing global sum.
    """

    def pre_kernel_code(self) -> str:
        c = Common(self, 0)
        dim = caspar_size(self.storage)
        return f"cudaMemset({c.name}, 0, {dim}*sizeof({self.storage_t}));\n"


class _Pairwise(_UsingSharedMem): ...


class ReadPair(_Pairwise, _ReadAccessor):
    """
    Accessor to read the element corresponding to the current thread and the next thread.
    """

    KERNEL_SIG_TEMPLATE = {"{name}": "{storage_t}* const", "{name}_num_alloc": "unsigned int"}
    PY_SIG_TEMPLATE = {"{name}": PYOBJ}
    PY_ARGS_TEMPLATE = ["As{storage_t_capitalized}Ptr({name})", "GetNumCols({name})"]

    def read_template(self, idx: int) -> str:
        c = Common(self, idx)
        return f"""
        }}}};
        read_and_shuffle_{c.dim}{c.template_args}(({c.SharedT}*)inout_shared,
        {c.name}, {c.offset}*{c.name}_num_alloc, global_thread_idx,
        threadIdx.x == {self.block_size - 1},
        global_thread_idx <= problem_size,
        {",".join(f"{{arg_{i}}}" for i in range(c.dim * 2))});
        if (global_thread_idx < problem_size) {{{{
        """.strip()


class ReadPairStridedWithDefault(_Pairwise, _ReadAccessor):
    """
    Accessor to read the element corresponding to the current thread and the next thread.
    """

    KERNEL_SIG_TEMPLATE = {
        "{name}": "{storage_t}* const",
        "{name}_num_alloc": "unsigned int",
        "{name}_offset": "int",
        "{name}_stride": "int",
    }
    PY_SIG_TEMPLATE = {"{name}": PYOBJ, "{name}_offset": "int", "{name}_stride": "int"}
    PY_ARGS_TEMPLATE = [
        "As{storage_t_capitalized}Ptr({name})",
        "GetNumCols({name})",
        "{name}_offset",
        "{name}_stride",
    ]

    def read_template(self, idx: int) -> str:
        c = Common(self, idx)
        return f"""
        }}}};
        read_and_shuffle_with_default_{c.dim}{c.template_args}(({c.SharedT}*)inout_shared,
        {c.name}, {c.offset}*{c.name}_num_alloc,
        global_thread_idx * {c.name}_stride + {c.name}_offset,
        threadIdx.x == {self.block_size - 1},
        global_thread_idx <= problem_size,
        {c.name}_num_alloc,
        {c.name}_stride,
        {",".join(f"{{arg_{i}}}" for i in range(c.dim * 2))},
        {",".join(f"{float(StorageOps.to_storage(self.default)[i]):06f}f" for i in c.elem_indices)}
        );
        if (global_thread_idx < problem_size) {{{{
        """.strip()


class AddPair(_Pairwise, _AddAccessor):
    """
    Accessor for adding the pair to the element corresponding to the current thread
    and the next thread.
    """

    KERNEL_SIG_TEMPLATE = {"{name}": "{storage_t}* const", "{name}_num_alloc": "unsigned int"}
    PY_SIG_TEMPLATE = {"{name}": PYOBJ}
    PY_ARGS_TEMPLATE = ["As{storage_t_capitalized}Ptr({name})", "GetNumCols({name})"]
    EXTRA_DATA = 1

    def write_template(self, idx: int) -> str:
        c = Common(self, idx)
        template_args = f"<true, {self.block_size}, {c.kernel_t}, {c.storage_t}, {c.VecT}>"
        return f"""
        }}}};
        shuffle_and_write_{c.dim}{template_args}(({c.SharedT}*)inout_shared,
        {c.name}, {c.offset}*{c.name}_num_alloc,
        global_thread_idx, problem_size,
        {",".join(f"{{arg_{i}}}" for i in range(c.dim * 2))});
        if (global_thread_idx < problem_size) {{{{
        """.strip()


class WritePair(_Pairwise, _ReadAccessor):
    """
    Accessor for writing the pair to the element corresponding to the current thread
    and the next thread. The second element is added to the first element of the next thread.
    """

    KERNEL_SIG_TEMPLATE = {"{name}": "{storage_t}* const", "{name}_num_alloc": "unsigned int"}
    PY_SIG_TEMPLATE = {"{name}": PYOBJ}
    PY_ARGS_TEMPLATE = ["As{storage_t_capitalized}Ptr({name})", "GetNumCols({name})"]
    EXTRA_DATA = 1

    def pre_kernel_code(self) -> str:
        c = Common(self, 0)
        dim = caspar_size(get_memtype(self.storage))
        return f"cudaMemset({c.name}, 0, {c.name}_num_alloc*{dim}*sizeof({self.storage_t}));\n"

    def write_template(self, idx: int) -> str:
        c = Common(self, idx)
        template_args = f"<false, {self.block_size}, {c.kernel_t}, {c.storage_t}, {c.VecT}>"
        return f"""
        }}}};
        shuffle_and_write_{c.dim}{template_args}(({c.SharedT}*)inout_shared,
        {c.name}, {c.offset}*{c.name}_num_alloc,
        global_thread_idx, problem_size,
        {",".join(f"{{arg_{i}}}" for i in range(c.dim * 2))});
        if (global_thread_idx < problem_size) {{{{
        """.strip()


class _FactorAccessor(_ReadAccessor): ...


class _TunableAccessor(_FactorAccessor): ...


class _ConstAccessor(_FactorAccessor): ...


class TunableShared(ReadShared, _TunableAccessor):
    """
    Used in factors to define a tunable parameter that is shared between factors.
    """


class Tunable(TunableShared):
    def __init__(
        self,
        name: str,
        storage: T.StorableOrType,
        *,
        dtype: DType,
        kernel_dtype: DType,
        reuse_indices_from: str | None = None,
        offset: int | None = None,
        stride: int | None = None,
        default: T.Any | None = None,
        after: str | None = None,
        block_size: int = 1024,
    ) -> None:
        logging.warning("Tunable is deprecated, use TunableShared instead")
        super().__init__(
            name,
            storage,
            dtype=dtype,
            kernel_dtype=kernel_dtype,
            reuse_indices_from=reuse_indices_from,
            offset=offset,
            stride=stride,
            default=default,
            after=after,
            block_size=block_size,
        )


class TunableUnique(ReadUnique, _TunableAccessor):
    """
    Used in factors to define a tunable parameter that is shared between all factors.
    """


class TunablePair(ReadPair, _TunableAccessor):
    """
    Used in factors to define a tunable pair.
    That is factor[0] depends on arg[0] and arg[1], factor[1] depends on arg[1] and arg[2], etc.
    """


class ConstantSequential(ReadSequential, _ConstAccessor):
    """
    Used in factors to define constants that are unique to each factor.
    """


class Constant(ConstantSequential):
    def __init__(
        self,
        name: str,
        storage: T.StorableOrType,
        *,
        dtype: DType,
        kernel_dtype: DType,
        reuse_indices_from: str | None = None,
        offset: int | None = None,
        stride: int | None = None,
        default: T.Any | None = None,
        after: str | None = None,
        block_size: int = 1024,
    ) -> None:
        logging.warning("Constant is deprecated, use ConstantSequential instead")
        super().__init__(
            name,
            storage,
            dtype=dtype,
            kernel_dtype=kernel_dtype,
            reuse_indices_from=reuse_indices_from,
            offset=offset,
            stride=stride,
            default=default,
            after=after,
            block_size=block_size,
        )


class ConstantShared(ReadShared, _ConstAccessor):
    """
    Used in factors to define constants that are shared between factors.
    """


class ConstantIndexed(ReadIndexed, _ConstAccessor):
    """
    Used in factors to define constants that are indexed.
    """


class ConstantUnique(ReadUnique, _ConstAccessor):
    """
    Used in factors to define a constant that is shared between all factors.
    """
