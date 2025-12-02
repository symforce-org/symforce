# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import logging
from pathlib import Path

import symforce.symbolic as sf
from symforce import typing as T
from symforce.codegen.codegen import WARNING_MESSAGE
from symforce.ops import LieGroupOps as Ops

from ..code_generation.factor import Factor
from ..code_generation.factor import dyn_part
from ..code_generation.factor import from_diagonal_and_lower_triangle
from ..code_generation.factor import lower_tiangle_size
from ..code_generation.factor import symbolic_diagonal_and_lower_triangle
from ..code_generation.kernel import Kernel
from ..memory import AddSequential
from ..memory import AddSum
from ..memory import ReadSequential
from ..memory import ReadUnique
from ..memory import WriteSequential
from ..memory.accessors import _ReadAccessor
from ..memory.accessors import _WriteAccessor
from ..source.templates import env
from ..source.templates import write_if_different
from .solver_utils import MemDesc
from .solver_utils import arg_key
from .solver_utils import fac_key
from .solver_utils import max_stacked_storage
from .solver_utils import name_key
from .solver_utils import node_key
from .solver_utils import num_arg_key
from .solver_utils import num_blocks_key
from .solver_utils import num_key
from .solver_utils import num_max_key
from .solver_utils import solver_key

if T.TYPE_CHECKING:
    from .. import CasparLibrary


class Solver:
    def __init__(self, caslib: CasparLibrary):
        """
        Class used to generate the solver code.

        The fields is what is used to generate the memory layout of the solver.
        The order of the fields is important as certain regions of the memory are reused.

        The Solver object also generates several kernels needed for the optimization.
        """
        if sf.epsilon() == 0:
            logging.warning(WARNING_MESSAGE)

        self.struct_name = "GraphSolver"
        self.caslib = caslib
        self.fields: dict[str, MemDesc] = {}
        self.at_least: dict[str, str] = {}
        self.fields["marker__start_"] = MemDesc(0, 0, alignment=16)

        self.add_storage(self.fields)
        self.add_indices(self.fields)

        self.fields["marker__scratch_inout_"] = MemDesc(0, 0)
        self.at_least["marker__scratch_inout_"] = max_stacked_storage(*caslib.node_types)

        self.add_solver_data(self.fields)

        for k in self.fields:
            assert k.endswith("_")
        self.size_contributors = [
            *sorted(caslib.node_types, key=lambda x: x.__name__),
            *caslib.factors,
        ]
        self.kernels: dict[str, Kernel] = {}

    def add_storage(self, fields: dict[str, MemDesc]) -> None:
        for typ in self.caslib.node_types:
            sdim = Ops.storage_dim(typ)
            fields[node_key(typ, "storage_current")] = MemDesc(sdim, num_key(typ))
            fields[node_key(typ, "storage_check")] = MemDesc(sdim, num_key(typ))
            fields[node_key(typ, "storage_new_best")] = MemDesc(sdim, num_key(typ))

    def add_indices(self, fields: dict[str, MemDesc]) -> None:
        for fac, arg, typ in [
            (f, a, t) for f in self.caslib.factors for a, t in f.arg_types.items()
        ]:
            if fac.isnode[arg]:
                fields[arg_key(fac, arg, "idx_shared")] = MemDesc(
                    2, num_key(fac), False, "SharedIndex"
                )
            elif fac.isconstseq[arg]:
                fields[arg_key(fac, arg, "data")] = MemDesc(Ops.storage_dim(typ), num_key(fac))
            elif fac.isconstuniq[arg]:
                fields[arg_key(fac, arg, "data")] = MemDesc(Ops.storage_dim(typ), 1)
            elif fac.isconstshared[arg]:
                fields[arg_key(fac, arg, "data")] = MemDesc(Ops.storage_dim(typ), num_key(fac))
                fields[arg_key(fac, arg, "idx_shared")] = MemDesc(
                    2, num_key(fac), False, "SharedIndex"
                )
            else:
                raise NotImplementedError

    def add_solver_data(self, fields: dict[str, MemDesc]) -> None:
        caslib = self.caslib
        for fac in caslib.factors:
            fields[fac_key(fac, "res")] = MemDesc(fac.res_dim, num_key(fac))

        for fac, arg in ((f, a) for f in caslib.factors for a in f.node_arg_types):
            if fac.isnodepair[arg]:
                fields[arg_key(fac, arg, "jac_first")] = MemDesc(
                    len(dyn_part(fac.jac_syms[f"{arg}_first"])), num_key(fac)
                )
                fields[arg_key(fac, arg, "jac_second")] = MemDesc(
                    len(dyn_part(fac.jac_syms[f"{arg}_second"])), num_key(fac)
                )
            else:
                fields[arg_key(fac, arg, "jac")] = MemDesc(
                    len(dyn_part(fac.jac_syms[arg])), num_key(fac)
                )
        for thing in ["z", "p_k_a", "p_k_b", "step_k_a", "step_k_b"]:
            for typ in caslib.node_types:
                fields[node_key(typ, thing)] = MemDesc(Ops.tangent_dim(typ), num_key(typ))

        fields["marker__w_start_"] = MemDesc(0, 0)
        for typ in caslib.node_types:
            fields[node_key(typ, "w")] = MemDesc(Ops.tangent_dim(typ), num_key(typ))
        fields["marker__w_end_"] = MemDesc(0, 0, alignment=4)

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
            "current_diag",
            "alpha_numerator",
            "alpha_denumerator",
            "alpha",
            "neg_alpha",
            "beta_numerator",
            "beta",
            "r_0_norm2_tot",
            "r_kp1_norm2_tot",
            "pred_decrease_tot",
            "res_tot",
        ]:
            fields[solver_key(thing)] = MemDesc(1, 1, alignment=4)

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

        def kernel(
            args: tuple[str, ...], inputs: list[_ReadAccessor], outputs: list[_WriteAccessor]
        ) -> None:
            name = "_".join(args)
            self.kernels[name] = Kernel(name, inputs, outputs, expose_to_python=False)

        diag = sf.symbols("dampening")
        gain = sf.symbols("gain")

        self.nonzero_jtj_patterns: dict[T.Type[sf.Storage], list[bool]] = {
            t: [False] * Ops.tangent_dim(t) ** 2 for t in self.node_types
        }

        def set_nonzero(typ: T.Type[sf.Storage], jac: sf.Matrix) -> None:
            for i, v in enumerate((jac.T * jac).to_storage()):
                self.nonzero_jtj_patterns[typ][i] |= not v.is_zero

        for fac in self.factors:
            for arg, typ in fac.node_arg_types.items():
                if fac.isnodepair[arg]:
                    set_nonzero(typ, fac.jac_syms[f"{arg}_first"])
                    set_nonzero(typ, fac.jac_syms[f"{arg}_second"])
                else:
                    set_nonzero(typ, fac.jac_syms[arg])

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
            w = sf.Matrix(Ops.tangent_dim(ntype), 1).symbolic(f"{name}_w")

            precond = from_diagonal_and_lower_triangle(precond_diag, precond_tril)
            precond = precond.from_storage(
                [
                    v if nz else 0
                    for v, nz in zip(precond.to_storage(), self.nonzero_jtj_patterns[ntype])
                ]
            )
            for i in range(precond.shape[0]):
                precond[i, i] = sf.Max(precond[i, i] * (1 + diag), sf.epsilon() ** 2)

            normalized = sf.Matrix(precond.mat.solve(njtr.mat, method="LDL"))

            kernel(
                (name, "retract"),
                [ReadSequential(name, state), ReadSequential("delta", step_k)],
                [WriteSequential(f"out_{name}_retracted", Ops.retract(state, step_k.to_storage()))],
            )
            kernel(
                (name, "normalize"),
                [
                    ReadSequential("precond_diag", precond_diag),
                    ReadSequential("precond_tril", precond_tril),
                    ReadSequential("njtr", njtr),
                    ReadUnique("diag", diag),
                ],
                [WriteSequential("out_normalized", normalized)],
            )

            kernel(
                (name, "start_w"),
                [
                    ReadSequential(f"{name}_precond_diag", precond_diag),
                    ReadUnique("diag", diag),
                    ReadSequential(f"{name}_p", p_k),
                ],
                [WriteSequential(f"out_{name}_w", diag * precond_diag.multiply_elementwise(p_k))],
            )
            kernel(
                (name, "start_w_contribute"),
                [
                    ReadSequential(f"{name}_precond_diag", precond_diag),
                    ReadUnique("diag", diag),
                    ReadSequential(f"{name}_p", p_k),
                ],
                [AddSequential(f"out_{name}_w", diag * precond_diag.multiply_elementwise(p_k))],
            )

            kernel(
                (name, "alpha_numerator_denominator"),
                [
                    ReadSequential(f"{name}_p_kp1", p_kp1),
                    ReadSequential(f"{name}_r_k", r_k),
                    ReadSequential(f"{name}_w", w),
                ],
                [
                    AddSum(f"{name}_total_ag", p_kp1.T * r_k),
                    AddSum(f"{name}_total_ac", p_kp1.T * w),
                ],
            )
            kernel(
                (name, "alpha_denumerator_or_beta_nummerator"),
                [ReadSequential(f"{name}_p_kp1", p_kp1), ReadSequential(f"{name}_w", w)],
                [AddSum(f"{name}_out", p_kp1.T * w)],
            )

            kernel(
                (name, "update_r_first"),
                [
                    ReadSequential(f"{name}_r_k", r_k),
                    ReadSequential(f"{name}_w", w),
                    ReadUnique("negalpha", gain),
                ],
                [
                    WriteSequential(f"out_{name}_r_kp1", r_k + w * gain),
                    AddSum(f"out_{name}_r_0_norm2_tot", r_k.squared_norm()),
                    AddSum(f"out_{name}_r_kp1_norm2_tot", (r_k + w * gain).squared_norm()),
                ],
            )
            kernel(
                (name, "update_r"),
                [
                    ReadSequential(f"{name}_r_k", r_k),
                    ReadSequential(f"{name}_w", w),
                    ReadUnique("negalpha", gain),
                ],
                [
                    WriteSequential(f"out_{name}_r_kp1", r_k + w * gain),
                    AddSum(f"out_{name}_r_kp1_norm2_tot", (r_k + w * gain).squared_norm()),
                ],
            )

            kernel(
                (name, "update_step_first"),
                [ReadSequential(f"{name}_p_kp1", p_kp1), ReadUnique("alpha", gain)],
                [WriteSequential(f"out_{name}_step_kp1", p_kp1 * gain)],
            )
            kernel(
                (name, "update_step_or_update_p"),
                [
                    ReadSequential(f"{name}_step_k", step_k),
                    ReadSequential(f"{name}_p_kp1", p_kp1),
                    ReadUnique("alpha", gain),
                ],
                [WriteSequential(f"out_{name}_step_kp1", step_k + p_kp1 * gain)],
            )

            """
            From L(0) - L(h_{lm}) in METHODS FOR NON-LINEAR LEAST SQUARES PROBLEMS, p.25

            """
            kernel(
                (name, "pred_decrease_times_two"),
                [
                    ReadSequential(f"{name}_step", step_kp1),
                    ReadSequential(f"{name}_precond_diag", precond_diag),
                    ReadUnique("diag", diag),
                    ReadSequential(f"{name}_njtr", njtr),
                ],
                [
                    AddSum(
                        f"out_{name}_pred_dec",
                        step_kp1.T * (njtr + diag * precond_diag.multiply_elementwise(step_kp1)),
                    )
                ],
            )
        return list(self.kernels.values())

    def generate(self, out_dir: Path) -> None:
        kwargs = dict(
            solver=self,
            name_key=name_key,
            fac_key=fac_key,
            arg_key=arg_key,
            node_key=node_key,
            solver_key=solver_key,
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
