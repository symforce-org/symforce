# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import symforce.symbolic as sf
from symforce import jacobian_helpers
from symforce import typing as T
from symforce.ops import LieGroupOps as Ops

from ..code_generation.kernel import Kernel
from ..memory import AddSharedSum
from ..memory import AddSum
from ..memory import ConstantIndexed
from ..memory import ConstantSequential
from ..memory import ConstantShared
from ..memory import ConstantUnique
from ..memory import ReadPair
from ..memory import ReadSequential
from ..memory import ReadShared
from ..memory import ReadUnique
from ..memory import TunablePair
from ..memory import TunableShared
from ..memory import TunableUnique
from ..memory import WriteSequential
from ..memory.accessors import AddPair
from ..memory.accessors import DtypeKwargs
from ..memory.accessors import _FactorAccessor
from ..memory.accessors import _ReadAccessor
from ..memory.accessors import _TunableAccessor
from ..memory.accessors import _WriteAccessor
from ..memory.dtype import DType
from ..memory.pair import Pair
from ..memory.pair import get_memtype
from ..memory.pair import get_symbolic


def get_diagonal_and_lower_triangle(mat: sf.Matrix) -> tuple[sf.Matrix, sf.Matrix]:
    """
    Returns the diagonal and lower triangular part of a matrix.
    """
    ltril_data = []
    for i in range(mat.shape[0]):
        for j in range(i + 1, mat.shape[1]):
            ltril_data.append(mat[i, j])
    diag_data = [mat[i, i] for i in range(mat.shape[0])]

    return sf.Matrix(diag_data), sf.Matrix(ltril_data)


def from_diagonal_and_lower_triangle(diag_data: sf.Matrix, ltril_data: sf.Matrix) -> sf.Matrix:
    """
    Reconstructs a matrix from its diagonal and lower triangular part.
    """
    n = len(diag_data)
    vec_loc = ltril_data.to_storage()
    mat = sf.Matrix(n, n)
    for i in range(n):
        for j in range(i + 1, mat.shape[0]):
            mat[i, j] = vec_loc.pop(0)
            mat[j, i] = mat[i, j]
    for i in range(n):
        mat[i, i] = diag_data[i]
    return sf.Matrix(mat)


def lower_tiangle_size(rows_and_columns: int) -> int:
    """
    Returns the number of elements in the lower triangular part of a matrix.
    """
    return rows_and_columns * (rows_and_columns - 1) // 2


def symbolic_diagonal_and_lower_triangle(
    rows_and_columns: int, name: str
) -> tuple[sf.Matrix, sf.Matrix]:
    """
    Returns symbolic variables for the diagonal and lower triangular part of a matrix.
    """
    tril_sym = sf.symbols(f"{name}_tril_0:{lower_tiangle_size(rows_and_columns)}")
    diag_sym = sf.symbols(f"{name}_diag_0:{rows_and_columns}")
    return sf.Matrix(diag_sym), sf.Matrix(tril_sym)


class Factor:
    """
    Class used to define a factor type used in a factor graph.

    A Factor object is essentially just a collection of kernels necessary for optimization.
    """

    def __init__(self, func: T.Callable, name: T.Optional[str], dtype: DType):  # noqa: PLR0915
        assert isinstance(dtype, DType)
        self.name = name or func.__name__
        self.func = func
        self.storage_t = dtype
        self.signature = T.get_type_hints(func, include_extras=True)  # type: ignore[call-arg]
        self.signature.pop("return", None)  # We don't care about the return annotation
        self.kernels: dict[str, Kernel] = {}
        for k, v in self.signature.items():
            if not (hasattr(v, "__metadata__") and hasattr(v, "__origin__")):
                raise ValueError(f"Argument {k} must be of type T.Annotated.")
            if not issubclass(v.__metadata__[0], _FactorAccessor):
                raise ValueError(f"Argument {k} must have an FactorAccessor descriptor.")

        self.fac_args = {k: get_symbolic(v.__origin__, k) for k, v in self.signature.items()}
        self.accessors = {k: v.__metadata__[0] for k, v in self.signature.items()}
        self.isnode = {k: issubclass(v, _TunableAccessor) for k, v in self.accessors.items()}
        self.isnodepair = {k: issubclass(v, TunablePair) for k, v in self.accessors.items()}
        self.isnodeuniq = {k: issubclass(v, TunableUnique) for k, v in self.accessors.items()}
        self.isnodeshared = {k: issubclass(v, TunableShared) for k, v in self.accessors.items()}
        self.isconstseq = {k: issubclass(v, ConstantSequential) for k, v in self.accessors.items()}
        self.isconstuniq = {k: issubclass(v, ConstantUnique) for k, v in self.accessors.items()}
        self.isconstshared = {k: issubclass(v, ConstantShared) for k, v in self.accessors.items()}
        self.isconstindexed = {k: issubclass(v, ConstantIndexed) for k, v in self.accessors.items()}
        self.arg_types: dict[str, T.Type[sf.Storage]] = {
            k: get_memtype(v.__origin__) for k, v in self.signature.items()
        }
        self.node_arg_types = {k: v for k, v in self.arg_types.items() if self.isnode[k]}
        self.const_arg_types = {k: v for k, v in self.arg_types.items() if not self.isnode[k]}

        self.solved_by_preconditioner = len(self.node_arg_types) == 1

        by_size = sorted(
            self.node_arg_types,
            key=lambda k: (self.isnodepair[k], Ops.tangent_dim(self.node_arg_types[k])),
        )
        self.prioritized_node = by_size[-1]
        self.prioritized_node_second = by_size[-2] if len(by_size) > 1 else by_size[-1]

        self.jnjtr_args = {
            k: self.node_arg_types[k]
            for k in sorted(
                self.node_arg_types,
                key=lambda k: k != self.prioritized_node_second,
            )
        }

        out = self.func(**self.fac_args)
        if out is None:
            raise ValueError("Factor functions cannot return None.")
        if isinstance(out, tuple):
            raise ValueError(
                "Factor functions cannot return a tuple. "
                "Use symforce.values.Values if you want to return multiple residual objects."
            )

        self.res = sf.Matrix(Ops.to_storage(out))
        self.res_dim = self.res.shape[0]

        self.jac_syms: dict[str, sf.Matrix] = {}
        self.jacs: dict[str, sf.Matrix] = {}
        self.jac_arg_types: dict[str, T.Type[sf.Storage]] = {}
        self.jac_name_map: dict[str, str] = {}
        self.tridig_Us: dict[str, sf.Matrix] = {}
        self.tridig_U_syms: dict[str, sf.Matrix] = {}
        for k, v in self.node_arg_types.items():
            if self.isnodepair[k]:
                j1 = get_jac(self.res, Ops.symbolic(v, f"{k}_first"))
                j2 = get_jac(self.res, Ops.symbolic(v, f"{k}_second"))
                tridig = j1.T * j2
                self.jac_syms[f"{k}_first"] = symbolic_with_const(j1, (f"{k}_jac_first"))
                self.jac_syms[f"{k}_second"] = symbolic_with_const(j2, (f"{k}_jac_second"))
                self.tridig_U_syms[f"{k}_tridig"] = symbolic_with_const(tridig, (f"{k}_tridig"))
                self.jacs[f"{k}_first"] = j1
                self.jacs[f"{k}_second"] = j2
                self.tridig_Us[f"{k}_tridig"] = tridig
                self.jac_arg_types[f"{k}_first"] = v
                self.jac_arg_types[f"{k}_second"] = v
                self.jac_name_map[f"{k}_first"] = k
                self.jac_name_map[f"{k}_second"] = k
            else:
                j = get_jac(self.res, Ops.symbolic(v, k))
                self.jac_syms[k] = symbolic_with_const(j, (f"{k}_jac"))
                self.jacs[k] = j
                self.jac_arg_types[k] = v
                self.jac_name_map[k] = k

        assert tuple(self.isnode) == tuple(self.fac_args)

    def add_kernel(
        self,
        args: tuple[str, ...],
        inputs: T.List[_ReadAccessor],
        outputs: T.List[_WriteAccessor],
    ) -> None:
        name = "_".join(args)
        self.kernels[name] = Kernel(
            name, inputs, outputs, dtype=self.storage_t, expose_to_python=False
        )

    def make_kernels(self) -> list[Kernel]:  # noqa: PLR0915
        accessor_kwargs = DtypeKwargs(dtype=self.storage_t, kernel_dtype=self.storage_t)
        inputs: list[_ReadAccessor] = [
            self.accessors[k](k, v, **accessor_kwargs) for k, v in self.fac_args.items()
        ]
        jac_args = self.jac_name_map

        jac_outs: list[T.Any] = []
        for k in self.node_arg_types:
            if self.isnodepair[k]:
                jac0, jac1 = self.jacs[f"{k}_first"], self.jacs[f"{k}_second"]
                tridig = self.tridig_Us[f"{k}_tridig"]
                diag0, tril0 = get_diagonal_and_lower_triangle(jac0.T * jac0)
                diag1, tril1 = get_diagonal_and_lower_triangle(jac1.T * jac1)
                if not self.solved_by_preconditioner:
                    jac_outs += (
                        WriteSequential(f"out_{k}_jac_first", dyn_part(jac0), **accessor_kwargs),
                        WriteSequential(f"out_{k}_jac_second", dyn_part(jac1), **accessor_kwargs),
                    )
                jac_outs += (
                    WriteSequential(f"out_{k}_tridig_U", dyn_part(tridig), **accessor_kwargs),
                    AddPair(
                        f"out_{k}_njtr",
                        Pair(-jac0.T * self.res, -jac1.T * self.res),
                        **accessor_kwargs,
                    ),
                    AddPair(f"out_{k}_precond_diag", Pair(diag0, diag1), **accessor_kwargs),
                    AddPair(f"out_{k}_precond_tril", Pair(tril0, tril1), **accessor_kwargs),
                )
            elif self.isnodeuniq[k]:
                jac = self.jacs[k]
                diag, tril = get_diagonal_and_lower_triangle(jac.T * jac)
                if not self.solved_by_preconditioner:
                    jac_outs.append(
                        WriteSequential(f"out_{k}_jac", dyn_part(jac), **accessor_kwargs)
                    )
                jac_outs += (
                    AddSum(f"out_{k}_njtr", -jac.T * self.res, **accessor_kwargs),
                    AddSum(f"out_{k}_precond_diag", diag, **accessor_kwargs),
                    AddSum(f"out_{k}_precond_tril", tril, **accessor_kwargs),
                )
            else:
                jac = self.jacs[k]
                diag, tril = get_diagonal_and_lower_triangle(jac.T * jac)
                if not self.solved_by_preconditioner:
                    jac_outs.append(
                        WriteSequential(f"out_{k}_jac", dyn_part(jac), **accessor_kwargs)
                    )
                jac_outs += (
                    AddSharedSum(
                        f"out_{k}_njtr", -jac.T * self.res, reuse_indices_from=k, **accessor_kwargs
                    ),
                    AddSharedSum(
                        f"out_{k}_precond_diag", diag, reuse_indices_from=k, **accessor_kwargs
                    ),
                    AddSharedSum(
                        f"out_{k}_precond_tril", tril, reuse_indices_from=k, **accessor_kwargs
                    ),
                )

        self.add_kernel(
            (self.name, "res_jac_first"),
            inputs,
            [
                WriteSequential("out_res", self.res, **accessor_kwargs),
                AddSum("out_rTr", self.res.T * self.res, **accessor_kwargs),
                *jac_outs,
            ],
        )
        self.add_kernel(
            (self.name, "res_jac"),
            inputs,
            [
                WriteSequential("out_res", self.res, **accessor_kwargs),
                *jac_outs,
            ],
        )
        self.add_kernel(
            (self.name, "score"),
            inputs,
            [
                AddSum("out_rTr", self.res.T * self.res, **accessor_kwargs),
            ],
        )

        njtr_syms = {
            k: sf.Matrix(Ops.tangent_dim(v), 1).symbolic(k) for k, v in self.jac_arg_types.items()
        }

        jtjnjtr_inputs: list[_ReadAccessor] = []
        for k in self.node_arg_types:
            if self.isnodepair[k]:
                jtjnjtr_inputs += (
                    ReadPair(
                        f"{k}_njtr",
                        Pair(njtr_syms[f"{k}_first"], njtr_syms[f"{k}_second"]),
                        **accessor_kwargs,
                    ),
                    ReadSequential(
                        f"{k}_jac_first", dyn_part(self.jac_syms[f"{k}_first"]), **accessor_kwargs
                    ),
                    ReadSequential(
                        f"{k}_jac_second", dyn_part(self.jac_syms[f"{k}_second"]), **accessor_kwargs
                    ),
                )
            elif self.isnodeuniq[k]:
                jtjnjtr_inputs += (
                    ReadUnique(f"{k}_njtr", njtr_syms[k], **accessor_kwargs),
                    ReadSequential(f"{k}_jac", dyn_part(self.jac_syms[k]), **accessor_kwargs),
                )
            else:
                jtjnjtr_inputs += (
                    ReadShared(f"{k}_njtr", njtr_syms[k], **accessor_kwargs),
                    ReadSequential(f"{k}_jac", dyn_part(self.jac_syms[k]), **accessor_kwargs),
                )

        def get_jnjtr(key: str) -> sf.Matrix:
            exclusion = {key} if not self.isnodepair[key] else {f"{key}_first", f"{key}_second"}
            keys = [k for k in self.jac_syms if k not in exclusion]
            start = sf.Matrix(self.res_dim, 1)
            return sum((self.jac_syms[k] * njtr_syms[k] for k in keys), start=start)

        jtjnjtr_outs: list[_WriteAccessor] = []
        for k in self.node_arg_types:
            if self.isnodepair[k]:
                jac0, jac1 = self.jac_syms[f"{k}_first"], self.jac_syms[f"{k}_second"]
                jtjnjtr_outs.append(
                    AddPair(
                        f"out_{k}_njtr",
                        Pair(jac0.T * get_jnjtr(k), jac1.T * get_jnjtr(k)),
                        **accessor_kwargs,
                    )
                )
            elif self.isnodeuniq[k]:
                jac = self.jac_syms[k]
                jtjnjtr_outs.append(
                    AddSum(f"out_{k}_njtr", jac.T * get_jnjtr(k), **accessor_kwargs)
                )
            else:
                jac = self.jac_syms[k]
                jtjnjtr_outs.append(
                    AddSharedSum(
                        f"out_{k}_njtr",
                        jac.T * get_jnjtr(k),
                        reuse_indices_from=f"{jac_args[k]}_njtr",
                        **accessor_kwargs,
                    )
                )

        self.add_kernel(
            (self.name, "jtjnjtr_direct"),
            jtjnjtr_inputs,
            jtjnjtr_outs,
        )

        self.res_sym = self.res.symbolic("res")
        self.jnjtr_sym = sf.Matrix(self.res_dim, 1).symbolic("jnjtr")

        return list(self.kernels.values())


def dyn_part(storage: sf.Storage) -> sf.Matrix:
    return sf.Matrix([i for i in storage.to_storage() if not i.is_number])


StorageT = T.TypeVar("StorageT", bound=T.Storable)


def symbolic_with_const(storage: StorageT, name: str) -> StorageT:
    return storage.from_storage(
        [
            v if v.is_number else sf.Symbol(name + "_" + str(i))
            for (i, v) in enumerate(storage.to_storage())
        ]
    )


def get_jac(fx: T.Element, x: T.Element) -> sf.Matrix:
    return jacobian_helpers.tangent_jacobians(fx, [x])[0]
