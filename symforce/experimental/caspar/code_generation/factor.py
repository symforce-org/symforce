# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import symforce.symbolic as sf
from symforce import typing as T
from symforce.jacobian_helpers import tangent_jacobians
from symforce.ops import LieGroupOps as Ops

from ..code_generation.kernel import Kernel
from ..memory import AddIndexed
from ..memory import AddSequential
from ..memory import AddSharedSum
from ..memory import ReadIndexed
from ..memory import ReadSequential
from ..memory import ReadShared
from ..memory import Tunable
from ..memory import WriteBlockSum
from ..memory import WriteIndexed
from ..memory import WriteSequential
from ..memory.accessors import _FactorAccessor
from ..memory.accessors import _ReadAccessor
from ..memory.accessors import _WriteAccessor


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
    vec_loc = [v for v in ltril_data]
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

    def __init__(self, func: T.Callable):
        self.name = func.__name__
        self.func = func
        self.signature = T.get_type_hints(func, include_extras=True)  # type: ignore[call-arg]
        self.signature.pop("return", None)  # We don't care about the return annotation

        for k, v in self.signature.items():
            if not (hasattr(v, "__metadata__") and hasattr(v, "__origin__")):
                raise ValueError(f"Argument {k} must be of type T.Annotated.")
            if not issubclass(v.__metadata__[0], _FactorAccessor):
                raise ValueError(f"Argument {k} must have an FactorAccessor descriptor.")

        self.fac_args = {k: Ops.symbolic(v.__origin__, k) for k, v in self.signature.items()}
        self.isnode = {k: issubclass(v.__metadata__[0], Tunable) for k, v in self.signature.items()}
        self.arg_types = {k: v.__origin__ for k, v in self.signature.items()}
        self.node_arg_types = {k: v for k, v in self.arg_types.items() if self.isnode[k]}
        self.const_arg_types = {k: v for k, v in self.arg_types.items() if not self.isnode[k]}

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

        by_size = sorted(self.node_arg_types, key=lambda k: Ops.tangent_dim(self.node_arg_types[k]))
        self.smallest_node = by_size[0]
        self.second_smallest_node = by_size[1] if len(by_size) > 1 else by_size[0]

        self.jnjtr_args = {
            k: self.node_arg_types[k]
            for k in sorted(
                self.node_arg_types,
                key=lambda k: (k == self.smallest_node, k == self.second_smallest_node),
            )
        }
        assert tuple(self.isnode) == tuple(self.fac_args)

    def make_kernels(self) -> list[Kernel]:
        kernels = []

        def kernel(name: str, inputs: list[_ReadAccessor], outputs: list[_WriteAccessor]) -> None:
            kernels.append(Kernel(name, inputs, outputs, expose_to_python=False))

        inputs: list[_ReadAccessor] = [
            (ReadShared if self.isnode[k] else ReadSequential)(k, v)
            for k, v in self.fac_args.items()
        ]

        jacs = {
            f"{k}": tangent_jacobians(self.res, [v])[0]
            for k, v in self.fac_args.items()
            if self.isnode[k]
        }

        kernel(
            f"{self.name}_res_jac_first",
            inputs,
            [
                WriteSequential("out_res", self.res),
                WriteBlockSum("out_rTr", self.res.T * self.res),
                *(WriteIndexed(f"out_{k}_jac", v) for k, v in jacs.items()),
            ],
        )
        kernel(
            f"{self.name}_res_jac",
            inputs,
            [
                WriteSequential("out_res", self.res),
                *(WriteIndexed(f"out_{k}_jac", v) for k, v in jacs.items()),
            ],
        )

        kernel(f"{self.name}_score", inputs, [WriteBlockSum("out_resTres", self.res.T * self.res)])

        res_sym = self.res.symbolic("res")
        jacs_syms = {k: v.symbolic(f"{k}_jac") for k, v in jacs.items()}
        jnjtr_sym = sf.Matrix(self.res_dim, 1).symbolic("jnjtr")

        for name, jac_sym in jacs_syms.items():
            njtr = -jac_sym.T * res_sym
            jtj = jac_sym.T * jac_sym
            precond_diag, precond_tril = get_diagonal_and_lower_triangle(jtj)
            njtr_sym = njtr.symbolic(f"{name}_njtr")

            kernel(
                f"{self.name}_{name}_njtr_precond",
                [
                    ReadIndexed("res", res_sym),
                    ReadSequential(f"{name}_jac", jac_sym),
                ],
                [
                    AddSharedSum(f"out_{name}_njtr", njtr),
                    AddSharedSum(
                        f"out_{name}_precond_diag", precond_diag, use_index=f"out_{name}_njtr"
                    ),
                    AddSharedSum(
                        f"out_{name}_precond_tril", precond_tril, use_index=f"out_{name}_njtr"
                    ),
                ],
            )

            out_t: T.Type[_WriteAccessor]
            if name == self.smallest_node:
                out_t = AddSequential
            elif name == self.second_smallest_node:
                out_t = WriteIndexed
            else:
                out_t = AddIndexed

            kernel(
                f"{self.name}_{name}_jnjtr",
                [
                    ReadSequential(f"{name}_jac", jac_sym),
                    ReadShared(f"{name}_njtr", njtr_sym),
                ],
                [out_t(f"out_{name}_jnjtr", jac_sym * njtr_sym)],
            )

            kernel(
                f"{self.name}_{name}_jtjnjtr",
                [
                    ReadSequential(f"{name}_jac", jac_sym),
                    (ReadSequential if name == self.smallest_node else ReadIndexed)(
                        "jnjtr", jnjtr_sym
                    ),
                ],
                [AddSharedSum(f"out_{name}_jtjnjtr", jac_sym.T * jnjtr_sym)],
            )
        return kernels
