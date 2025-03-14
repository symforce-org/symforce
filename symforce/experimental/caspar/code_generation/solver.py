# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import symforce.symbolic as sf
from symforce import typing as T
from symforce.ops import LieGroupOps as Ops
from symforce.ops.interfaces import LieGroup

from ..code_generation.factor import Factor
from ..code_generation.factor import from_diagonal_and_lower_triangle
from ..code_generation.factor import lower_tiangle_size
from ..code_generation.factor import symbolic_diagonal_and_lower_triangle
from ..code_generation.kernel import Kernel
from ..memory import AddSequential
from ..memory import ReadSequential
from ..memory import ReadUnique
from ..memory import WriteBlockSum
from ..memory import WriteSequential
from ..memory import caspar_size
from ..memory.accessors import _ReadAccessor
from ..memory.accessors import _WriteAccessor
from ..source.templates import env
from ..source.templates import write_if_different

if T.TYPE_CHECKING:
    from .. import CasparLibrary


def fac_key(fac: Factor, suffix: str = "") -> str:
    return f"facs__{fac.name}__{suffix}_"


def arg_key(fac: Factor, arg: str, suffix: str = "") -> str:
    return f"{fac_key(fac, 'args')[:-1]}__{arg}__{suffix}_"


def node_key(typ: LieGroup, suffix: str = "") -> str:
    return f"nodes__{typ.__name__}__{suffix}_"  # type: ignore[attr-defined]


def name_key(thing: LieGroup | Factor) -> str:
    if isinstance(thing, LieGroup) or isinstance(thing, type):
        return thing.__name__  # type: ignore[union-attr]
    elif isinstance(thing, Factor):
        return thing.name
    else:
        raise ValueError(f"Unknown type {type(thing)}")


def num_key(thing: LieGroup | Factor, suffix: str = "num_") -> str:
    return f"{name_key(thing)}_{suffix}"


def num_max_key(thing: LieGroup | Factor) -> str:
    return num_key(thing, "num_max_")


def num_blocks_key(thing: LieGroup | Factor) -> str:
    return num_key(thing, "bnum_")


def num_arg_key(thing: LieGroup | Factor) -> str:
    return num_key(thing, "num_max")


def bnum_sum(*things: LieGroup | Factor) -> str:
    return f"1 + {' + '.join([num_blocks_key(n) for n in things])}"


class MemDesc:
    """
    Memory descriptor used to make the memory layout of the generated solver.
    """

    def __init__(
        self,
        dim: int,
        num_key: str | int,
        is_caspar_data: bool = True,
        dtype: str = "float",
        alignment: int = 16,
    ):
        assert alignment in set((4, 16))
        self.dim = dim
        self.dim_real = caspar_size(dim) if is_caspar_data else dim
        self.num_key = num_key
        self.dtype = dtype
        self.alignment = alignment


class Solver:
    def __init__(self, caslib: CasparLibrary):
        """
        Class used to generate the solver code.

        The fields is what is used to generate the memory layout of the solver.
        The order of the fields is important as certain regions of the memory are reused.

        The Solver object also generates several kernels needed for the optimization.
        """

        self.struct_name = "GraphSolver"
        self.caslib = caslib
        self.fields: dict[str, MemDesc] = {}

        self.fields["marker__start_"] = MemDesc(0, 0, alignment=4)

        self.add_storage(self.fields)
        self.add_indices(self.fields)

        self.fields["marker__scratch_inout_"] = MemDesc(0, 0)
        self.fields["marker__scratch_sum_"] = MemDesc(
            1, bnum_sum(*caslib.factors, *caslib.node_types)
        )
        self.fields["marker__scratch_sum_end_"] = MemDesc(0, 0, alignment=4)

        self.add_solver_data(self.fields)

        self.fields["marker__end_"] = MemDesc(0, 0, alignment=4)

        for k in self.fields:
            assert k.endswith("_")
        self.size_contributors = [
            *sorted(caslib.node_types, key=lambda x: x.__name__),
            *caslib.factors,
        ]

    def add_storage(self, fields: dict[str, MemDesc]) -> None:
        for typ in self.caslib.node_types:
            sdim = Ops.storage_dim(typ)
            fields[node_key(typ, "storage_current")] = MemDesc(sdim, num_key(typ))
            fields[node_key(typ, "storage_check")] = MemDesc(sdim, num_key(typ))
            fields[node_key(typ, "storage_new_best")] = MemDesc(sdim, num_key(typ))

    def add_indices(self, fields: dict[str, MemDesc]) -> None:
        for fac in self.caslib.factors:
            for arg, typ in fac.arg_types.items():

                def idx(name: str) -> str:
                    return arg_key(fac, arg, f"idx_{name}")  # noqa: B023

                if fac.isnode[arg]:
                    fields[idx("shared")] = MemDesc(2, num_key(fac), False, "SharedIndex")
                    fields[idx("sorted")] = MemDesc(1, num_key(fac), False, "unsigned int")
                    fields[idx("sorted_shared")] = MemDesc(2, num_key(fac), False, "SharedIndex")
                    fields[idx("target")] = MemDesc(1, num_key(fac), False, "unsigned int")
                    fields[idx("argsort")] = MemDesc(1, num_key(fac), False, "unsigned int")
                    if arg != fac.smallest_node:
                        fields[idx("jp_target")] = MemDesc(1, num_key(fac), False, "unsigned int")
                else:
                    fields[arg_key(fac, arg, "data")] = MemDesc(Ops.storage_dim(typ), num_key(fac))

    def add_solver_data(self, fields: dict[str, MemDesc]) -> None:
        caslib = self.caslib
        for fac in caslib.factors:
            fields[fac_key(fac, "res")] = MemDesc(fac.res_dim, num_key(fac))
            fields[fac_key(fac, "res")] = MemDesc(fac.res_dim, num_key(fac))

        for fac, arg, typ in ((f, a, t) for f in caslib.factors for a, t in f.arg_types.items()):
            tdim = Ops.tangent_dim(typ)
            fields[arg_key(fac, arg, "jac")] = MemDesc(fac.res_dim * tdim, num_key(fac))

        fields["marker__res_tot_start_"] = MemDesc(0, 0)
        for i, fac in enumerate(caslib.factors):
            alignment = 16 if i == 0 else 4
            fields[fac_key(fac, "res_tot")] = MemDesc(1, num_blocks_key(fac), alignment=alignment)
        fields["marker__res_tot_end_"] = MemDesc(0, 0, alignment=4)

        for thing in [
            "z",
            "p_k_a",
            "p_k_b",
            "step_k_a",
            "step_k_b",
            "w",
        ]:
            for typ in caslib.node_types:
                fields[node_key(typ, thing)] = MemDesc(Ops.tangent_dim(typ), num_key(typ))

        # USED IN THE njtr_precond
        fields["marker__r_k_a_start_"] = MemDesc(0, 0)
        for typ in caslib.node_types:
            fields[node_key(typ, "r_k_a")] = MemDesc(Ops.tangent_dim(typ), num_key(typ))
        for typ in caslib.node_types:
            fields[node_key(typ, "r_k_b")] = MemDesc(Ops.tangent_dim(typ), num_key(typ))
        fields["marker__r_0_start_"] = MemDesc(0, 0)
        for typ in caslib.node_types:
            fields[node_key(typ, "r_0")] = MemDesc(Ops.tangent_dim(typ), num_key(typ))
        fields["marker__precond_start_"] = MemDesc(0, 0)
        for typ in caslib.node_types:
            ntril = lower_tiangle_size(Ops.tangent_dim(typ))
            fields[node_key(typ, "precond_diag")] = MemDesc(Ops.tangent_dim(typ), num_key(typ))
            fields[node_key(typ, "precond_tril")] = MemDesc(ntril, num_key(typ))
        fields["marker__precond_end_"] = MemDesc(0, 0, alignment=4)

        # USED IN THE jnjtr, jtjnjtr
        fields["marker__jp_start_"] = MemDesc(0, 0)
        for fac in caslib.factors:
            fields[fac_key(fac, "jp")] = MemDesc(fac.res_dim, num_key(fac))
        fields["marker__jp_end_"] = MemDesc(0, 0, alignment=4)

        for thing in [
            "alpha_numerator_tot",
            "alpha_denumerator_tot",
            "r_0_norm2_tot",
            "r_kp1_norm2_tot",
            "pred_decrease_tot",
            "beta_numerator_tot",
        ]:
            fields[f"marker__{thing}_start_"] = MemDesc(0, 0)
            for typ in caslib.node_types:
                fields[node_key(typ, thing)] = MemDesc(1, num_blocks_key(typ))
            fields[f"marker__{thing}_end_"] = MemDesc(0, 0, alignment=4)

        for thing in [
            "current_diag",
            "alpha_numerator",
            "alpha_denumerator",
            "alpha",
            "neg_alpha",
            "beta_numerator",
            "beta",
        ]:
            fields[f"solver__{thing}_"] = MemDesc(1, 1, alignment=4)

    @property
    def factors(self) -> list[Factor]:
        return self.caslib.factors

    @property
    def node_types(self) -> list[T.LieGroupElement]:
        return self.caslib.node_types

    def make_kernels(self) -> list[Kernel]:
        """
        Generates the kernels for each node type used in the solver.
        """

        kernels = []

        def kernel(name: str, inputs: list[_ReadAccessor], outputs: list[_WriteAccessor]) -> None:
            kernels.append(Kernel(name, inputs, outputs, expose_to_python=False))

        diag = sf.symbols("dampening")
        gain = sf.symbols("gain")

        for ntype in self.node_types:
            name = ntype.__name__

            state = Ops.symbolic(ntype, name)
            precond_diag, precond_tril = symbolic_diagonal_and_lower_triangle(
                Ops.tangent_dim(ntype), "precond"
            )
            njtr = sf.Matrix(Ops.tangent_dim(ntype), 1).symbolic("negJTres")
            p_k = sf.Matrix(Ops.tangent_dim(ntype), 1).symbolic(f"{name}_p")
            p_kp1 = sf.Matrix(Ops.tangent_dim(ntype), 1).symbolic(f"{name}_p_kp1")
            step_k = sf.Matrix(Ops.tangent_dim(ntype), 1).symbolic(f"{name}_step")
            step_kp1 = sf.Matrix(Ops.tangent_dim(ntype), 1).symbolic(f"{name}_step")
            r_k = sf.Matrix(Ops.tangent_dim(ntype), 1).symbolic(f"{name}_r_k")
            r_kp1 = sf.Matrix(Ops.tangent_dim(ntype), 1).symbolic(f"{name}_r")
            w = sf.Matrix(Ops.tangent_dim(ntype), 1).symbolic(f"{name}_w")

            precond = from_diagonal_and_lower_triangle(
                sf.Matrix([sf.Max(sf.epsilon(), x) for x in precond_diag]), precond_tril
            )
            eye = precond.eye()
            precond_safe = precond.multiply_elementwise(eye * diag + 1)
            normalized = sf.Matrix(precond_safe.mat.solve(njtr.mat, method="LU"))

            kernel(
                f"{name}_retract",
                [ReadSequential(name, state), ReadSequential("delta", step_k)],
                [WriteSequential(f"out_{name}_retracted", Ops.retract(state, step_k.to_storage()))],
            )
            kernel(
                f"{name}_normalize",
                [
                    ReadSequential("precond_diag", precond_diag),
                    ReadSequential("precond_tril", precond_tril),
                    ReadSequential("njtr", njtr),
                    ReadUnique("diag", diag),
                ],
                [WriteSequential("out_normalized", normalized)],
            )

            kernel(
                f"{name}_start_w",
                [
                    ReadSequential(f"{name}_precond_diag", precond_diag),
                    ReadUnique("diag", diag),
                    ReadSequential(f"{name}_p", p_k),
                ],
                [WriteSequential(f"out_{name}_w", diag * precond_diag.multiply_elementwise(p_k))],
            )
            kernel(
                f"{name}_start_w_contribute",
                [
                    ReadSequential(f"{name}_precond_diag", precond_diag),
                    ReadUnique("diag", diag),
                    ReadSequential(f"{name}_p", p_k),
                ],
                [AddSequential(f"out_{name}_w", diag * precond_diag.multiply_elementwise(p_k))],
            )

            kernel(
                f"{name}_alpha_numerator_denominator",
                [
                    ReadSequential(f"{name}_p_kp1", p_kp1),
                    ReadSequential(f"{name}_r_k", r_k),
                    ReadSequential(f"{name}_w", w),
                ],
                [
                    WriteBlockSum(f"{name}_total_ag", p_kp1.T * r_k),
                    WriteBlockSum(f"{name}_total_ac", p_kp1.T * w),
                ],
            )
            kernel(
                f"{name}_alpha_denumerator_or_beta_nummerator",
                [ReadSequential(f"{name}_p_kp1", p_kp1), ReadSequential(f"{name}_w", w)],
                [WriteBlockSum(f"{name}_out", p_kp1.T * w)],
            )

            kernel(
                f"{name}_update_r_first",
                [
                    ReadSequential(f"{name}_r_k", r_k),
                    ReadSequential(f"{name}_w", w),
                    ReadUnique("negalpha", gain),
                ],
                [
                    WriteSequential(f"out_{name}_r_kp1", r_k + w * gain),
                    WriteBlockSum(f"out_{name}_r_0_norm2_tot", r_k.squared_norm()),
                    WriteBlockSum(f"out_{name}_r_kp1_norm2_tot", (r_k + w * gain).squared_norm()),
                ],
            )
            kernel(
                f"{name}_update_r",
                [
                    ReadSequential(f"{name}_r_k", r_k),
                    ReadSequential(f"{name}_w", w),
                    ReadUnique("negalpha", gain),
                ],
                [
                    WriteSequential(f"out_{name}_r_kp1", r_k + w * gain),
                    WriteBlockSum(f"out_{name}_r_kp1_norm2_tot", (r_k + w * gain).squared_norm()),
                ],
            )

            kernel(
                f"{name}_update_step_first",
                [ReadSequential(f"{name}_p_kp1", p_kp1), ReadUnique("alpha", gain)],
                [WriteSequential(f"out_{name}_step_kp1", p_kp1 * gain)],
            )
            kernel(
                f"{name}_update_step_or_update_p",
                [
                    ReadSequential(f"{name}_step_k", step_k),
                    ReadSequential(f"{name}_p_kp1", p_kp1),
                    ReadUnique("alpha", gain),
                ],
                [WriteSequential(f"out_{name}_step_kp1", step_k + p_kp1 * gain)],
            )

            """
            From L(0) - L(h_{lm}) in METHODS FOR NON-LINEAR LEAST SQUARES PROBLEMS, p.25
            using # (r + njtr) = -2 J^T res  - J^T J step
            """
            kernel(
                f"{name}_pred_decrease",
                [
                    ReadSequential(f"{name}_step", step_kp1),
                    ReadSequential(f"{name}_precond_diag", precond_diag),
                    ReadUnique("diag", diag),
                    ReadSequential(f"{name}_r", r_kp1),
                    ReadSequential(f"{name}_njtr", njtr),
                ],
                [WriteBlockSum(f"out_{name}_pred_dec", step_kp1.T * (r_kp1 + njtr))],
            )
        return kernels

    def generate(self, out_dir: Path) -> None:
        kwargs = dict(
            solver=self,
            name_key=name_key,
            fac_key=fac_key,
            arg_key=arg_key,
            node_key=node_key,
            num_key=num_key,
            num_blocks_key=num_blocks_key,
            num_max_key=num_max_key,
            num_arg_key=num_arg_key,
            Ops=Ops,
        )
        header = env.get_template("solver.h.jinja").render(**kwargs)
        write_if_different(header, out_dir.joinpath("solver.h"))
        definition = env.get_template("solver.cc.jinja").render(**kwargs)
        write_if_different(definition, out_dir.joinpath("solver.cc"))
        definition = env.get_template("solver_pybinding.h.jinja").render(**kwargs)
        write_if_different(definition, out_dir.joinpath("solver_pybinding.h"))
