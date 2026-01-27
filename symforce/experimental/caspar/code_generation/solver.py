# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------


import logging
from dataclasses import dataclass
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
from ..code_generation.factor import symbolic_with_const
from ..code_generation.kernel import Kernel
from ..memory import AddSequential
from ..memory import AddSum
from ..memory import Pair
from ..memory import ReadPairStridedWithDefault
from ..memory import ReadSequential
from ..memory import ReadStrided
from ..memory import ReadUnique
from ..memory import WriteSequential
from ..memory import WriteStrided
from ..memory.accessors import DtypeKwargs
from ..memory.accessors import _ReadAccessor
from ..memory.accessors import _WriteAccessor
from ..memory.special_square_matrices import LowerTriangularMatrix
from ..memory.special_square_matrices import UpperTriangularMatrix
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


@dataclass
class AnyEquation:
    """
    Type annotation placeholder. The actual implementation is defined in the Equation class below.
    """

    a: sf.Matrix
    b: sf.Matrix
    c: sf.Matrix
    d: sf.Matrix
    bL: LowerTriangularMatrix
    bU: UpperTriangularMatrix


class Solver:
    def __init__(self, caslib: "CasparLibrary"):
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
        self.fields: T.Dict[str, MemDesc] = {}
        # at_least values are C++ expression strings used in templates
        self.at_least: T.Dict[str, str] = {}
        self.storage_t = caslib.storage_t
        self.linear_t = caslib.storage_t
        self.fields["marker__start_"] = MemDesc(0, 0, self.storage_t, alignment=4)
        self.add_storage(self.fields)
        self.add_indices(self.fields)
        self.fields["marker__scratch_inout_"] = MemDesc(0, 0, self.storage_t)
        self.at_least["marker__scratch_inout_"] = max_stacked_storage(*caslib.node_types)

        self.add_solver_data(self.fields)
        self.diag = sf.symbols("dampening")

        for k in self.fields:
            assert k.endswith("_")
        self.size_contributors = [
            *sorted(caslib.node_types, key=lambda x: x.__name__),
            *caslib.factors,
        ]
        self.kernels: T.Dict[str, Kernel] = {}
        self.accessor_kwargs = DtypeKwargs(
            dtype=self.storage_t,
            kernel_dtype=self.linear_t,
        )

    def add_storage(self, fields: T.Dict[str, MemDesc]) -> None:
        for typ in self.caslib.node_types:
            sdim = Ops.storage_dim(typ)
            fields[node_key(typ, "storage_current")] = MemDesc(sdim, num_key(typ), self.storage_t)
            fields[node_key(typ, "storage_check")] = MemDesc(sdim, num_key(typ), self.storage_t)
            fields[node_key(typ, "storage_new_best")] = MemDesc(sdim, num_key(typ), self.storage_t)

    def add_indices(self, fields: T.Dict[str, MemDesc]) -> None:
        for fac, arg, typ in [
            (f, a, t) for f in self.caslib.factors for a, t in f.arg_types.items()
        ]:
            if fac.isnode[arg]:
                fields[arg_key(fac, arg, "idx_shared")] = MemDesc(
                    1, num_key(fac), "SharedIndex", False
                )
            elif fac.isconstseq[arg]:
                fields[arg_key(fac, arg, "data")] = MemDesc(
                    Ops.storage_dim(typ), num_key(fac), self.storage_t
                )
            elif fac.isconstuniq[arg]:
                fields[arg_key(fac, arg, "data")] = MemDesc(Ops.storage_dim(typ), 1, self.storage_t)
            elif fac.isconstshared[arg]:
                fields[arg_key(fac, arg, "data")] = MemDesc(
                    Ops.storage_dim(typ), num_key(fac), self.storage_t
                )
                fields[arg_key(fac, arg, "idx_shared")] = MemDesc(
                    1, num_key(fac), "SharedIndex", False
                )
            elif fac.isconstindexed[arg]:
                fields[arg_key(fac, arg, "data")] = MemDesc(
                    Ops.storage_dim(typ), num_key(fac), self.storage_t
                )
                fields[arg_key(fac, arg, "idx")] = MemDesc(1, num_key(fac), "unsigned int", False)
            else:
                raise ValueError(f"Unknown arg type for {fac}.{arg}: {typ}")

    def add_solver_data(self, fields: T.Dict[str, MemDesc]) -> None:
        caslib = self.caslib
        for fac in caslib.factors:
            fields[fac_key(fac, "res")] = MemDesc(fac.res_dim, num_key(fac), self.linear_t)

        for fac, arg, typ in (
            (f, a, t) for f in caslib.factors for a, t in f.node_arg_types.items()
        ):
            if fac.isnodepair[arg]:
                fields[arg_key(fac, arg, "jac_first")] = MemDesc(
                    len(dyn_part(fac.jac_syms[f"{arg}_first"])), num_key(fac), self.storage_t
                )
                fields[arg_key(fac, arg, "jac_second")] = MemDesc(
                    len(dyn_part(fac.jac_syms[f"{arg}_second"])), num_key(fac), self.storage_t
                )
                assert node_key(typ, "tridig_U") not in fields
                fields[node_key(typ, "tridig_U")] = MemDesc(
                    len(dyn_part(fac.tridig_U_syms[f"{arg}_tridig"])), num_key(typ), self.linear_t
                )
            else:
                fields[arg_key(fac, arg, "jac")] = MemDesc(
                    len(dyn_part(fac.jac_syms[arg])), num_key(fac), self.storage_t
                )
        for thing in ["z", "p", "step"]:
            for typ in caslib.node_types:
                fields[node_key(typ, thing)] = MemDesc(
                    Ops.tangent_dim(typ), num_key(typ), self.linear_t
                )
                fields[node_key(typ, f"{thing}_end_")] = MemDesc(0, 0, self.linear_t)

        fields["marker__w_start_"] = MemDesc(0, 0, self.linear_t)
        for typ in caslib.node_types:
            fields[node_key(typ, "w")] = MemDesc(Ops.tangent_dim(typ), num_key(typ), self.linear_t)

        fields["marker__w_end_"] = MemDesc(0, 0, self.linear_t, alignment=1)

        # USED IN THE njtr_precond
        fields["marker__r_0_start_"] = MemDesc(0, 0, self.linear_t)
        for typ in caslib.node_types:
            fields[node_key(typ, "r_0")] = MemDesc(
                Ops.tangent_dim(typ), num_key(typ), self.linear_t
            )
        fields["marker__r_0_end_"] = MemDesc(0, 0, self.linear_t)

        fields["marker__r_k_start_"] = MemDesc(0, 0, self.linear_t)
        for typ in caslib.node_types:
            fields[node_key(typ, "r_k")] = MemDesc(
                Ops.tangent_dim(typ), num_key(typ), self.linear_t
            )
        fields["marker__r_k_end_"] = MemDesc(0, 0, self.linear_t)

        fields["marker__Mp_start_"] = MemDesc(0, 0, self.linear_t)
        for typ in caslib.node_types:
            fields[node_key(typ, "Mp")] = MemDesc(Ops.tangent_dim(typ), num_key(typ), self.linear_t)
        fields["marker__Mp_end_"] = MemDesc(0, 0, self.linear_t)

        fields["marker__precond_start_"] = MemDesc(0, 0, self.linear_t)
        for typ in caslib.node_types:
            ntril = lower_tiangle_size(Ops.tangent_dim(typ))
            fields[node_key(typ, "precond_diag")] = MemDesc(
                Ops.tangent_dim(typ), num_key(typ), self.linear_t
            )
            fields[node_key(typ, "precond_tril")] = MemDesc(ntril, num_key(typ), self.linear_t)
        fields["marker__precond_end_"] = MemDesc(0, 0, self.linear_t, alignment=1)

        # USED IN THE jnjtr, jtjnjtr
        fields["marker__jp_start_"] = MemDesc(0, 0, self.linear_t)
        for fac in caslib.factors:
            fields[fac_key(fac, "jp")] = MemDesc(fac.res_dim, num_key(fac), self.linear_t)
        fields["marker__jp_end_"] = MemDesc(0, 0, self.linear_t, alignment=1)

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
            fields[solver_key(thing)] = MemDesc(1, 1, self.linear_t, alignment=1)

    @property
    def factors(self) -> T.List[Factor]:
        return self.caslib.factors

    @property
    def node_types(self) -> T.List[T.LieGroupElement]:
        return self.caslib.node_types

    def add_kernel(
        self,
        name_args: T.Tuple[str, ...],
        inputs: T.List[_ReadAccessor],
        outputs: T.List[_WriteAccessor],
        block_size: int = 1024,
    ) -> None:
        name = "_".join(name_args)
        self.kernels[name] = Kernel(
            name,
            inputs,
            outputs,
            dtype=self.linear_t,
            expose_to_python=False,
            block_size=block_size,
        )

    def add_tridiagonal_solver_kernels_for_node(self, ntype: T.Type[sf.Storage]) -> None:  # noqa: PLR0914
        """
        Cyclic reduction based on:
        https://people.inf.ethz.ch/gander/papers/cyclic.pdf
        https://research.nvidia.com/sites/default/files/pubs/2010-01_Fast-Tridiagonal-Solvers/Zhang_Fast_2009.pdf
        """
        tdim = Ops.tangent_dim(ntype)
        Uk = self.tridig_Uk_syms[ntype]
        Ukp1 = self.tridig_Ukp1_syms[ntype]
        precond_diag, precond_tril, precond = self.preconds[ntype]

        class SizedMatrix(sf.Matrix):
            SHAPE = (tdim, tdim)

        class SizedVector(sf.Matrix):
            SHAPE = (tdim, 1)

        class SizedLowerTriangularMatrix(LowerTriangularMatrix):
            SHAPE = tdim

        class SizedUpperTriangularMatrix(UpperTriangularMatrix):
            SHAPE = tdim

        @dataclass
        class Equation:
            """
            Stores the problem:
            a * x_km1 + b * x_k + c * x_kp1 = d

            bL and bU are the LU decomposition of b.

            Equation used in cyclic reduction described in 2.1 in the following paper:
            https://research.nvidia.com/sites/default/files/pubs/2010-01_Fast-Tridiagonal-Solvers/Zhang_Fast_2009.pdf
            """

            a: SizedMatrix
            b: SizedMatrix
            c: SizedMatrix
            d: SizedVector
            bL: SizedLowerTriangularMatrix
            bU: SizedUpperTriangularMatrix

        self.fields[node_key(ntype, "tridig_eq")] = MemDesc(
            Ops.storage_dim(Equation), num_key(ntype), self.linear_t
        )
        lu = precond.LU()
        precondL, precondU = lu[0], lu[1]
        self.add_kernel(
            (ntype.__name__, "tridig_make_eq"),
            [
                ReadSequential("precond_diag", precond_diag, **self.accessor_kwargs),
                ReadSequential("precond_tril", precond_tril, **self.accessor_kwargs),
                ReadPairStridedWithDefault(
                    "tridig_U",
                    Pair(dyn_part(Uk), dyn_part(Ukp1)),
                    default=dyn_part(Uk).zero(),
                    **self.accessor_kwargs,
                ),
                ReadSequential("njtr", self.njtrs[ntype], **self.accessor_kwargs),
                ReadUnique("diag", self.diag, **self.accessor_kwargs),
            ],
            [
                WriteSequential(
                    "eq",
                    Equation(
                        SizedMatrix(Uk.T),
                        SizedMatrix(precond),
                        SizedMatrix(Ukp1),
                        SizedVector(self.njtrs[ntype]),
                        SizedLowerTriangularMatrix(precondL),
                        SizedUpperTriangularMatrix(precondU),
                    ),
                    **self.accessor_kwargs,
                )
            ],
        )

        eqkm1 = Ops.symbolic(Equation, "eqkm1")
        eqkp1 = Ops.symbolic(Equation, "eqkp1")
        eqk: Equation = Ops.symbolic(Equation, "eqk")
        eq_zero = Equation(
            eqk.a.zero(),
            eqk.b.eye(),
            eqk.c.zero(),
            eqk.d.zero(),
            SizedLowerTriangularMatrix(eqk.b.eye()),
            SizedUpperTriangularMatrix(eqk.b.eye()),
        )

        k1 = eqkm1.bL.mat().T.solve(eqkm1.bU.mat().T.solve(eqk.a.T)).T
        k2 = eqkp1.bL.mat().T.solve(eqkp1.bU.mat().T.solve(eqk.c.T)).T
        a_next = -k1 * eqkm1.a
        b_next = eqk.b - k1 * eqkm1.c - k2 * eqkp1.a
        lu_next = b_next.LU()
        bL_next, bU_next = lu_next[0], lu_next[1]
        c_next = -k2 * eqkp1.c
        d_next = eqk.d - k1 * eqkm1.d - k2 * eqkp1.d
        self.add_kernel(
            (ntype.__name__, "CR_down"),
            [
                ReadStrided("eqk", eqk, block_size=64, **self.accessor_kwargs),
                ReadPairStridedWithDefault(
                    "eq", Pair(eqkm1, eqkp1), default=eq_zero, block_size=64, **self.accessor_kwargs
                ),
            ],
            [
                WriteStrided(
                    "eq_marked",
                    Equation(
                        SizedMatrix(a_next),
                        SizedMatrix(b_next),
                        SizedMatrix(c_next),
                        SizedVector(d_next),
                        SizedLowerTriangularMatrix(bL_next),
                        SizedUpperTriangularMatrix(bU_next),
                    ),
                    after="eqk",
                    block_size=64,
                    **self.accessor_kwargs,
                ),
            ],
            block_size=64,
        )
        solu2 = sf.Matrix.block_matrix(
            [
                [eqkm1.b.eye(), eqkm1.bU.mat().solve(eqkm1.bL.mat().solve(eqkm1.c))],
                [eqk.bU.mat().solve(eqk.bL.mat().solve(eqk.a)), eqk.b.eye()],
            ]
        ).mat.solve(
            sf.Matrix.block_matrix(
                [
                    [eqkm1.bU.mat().mat.solve(eqkm1.bL.mat().mat.solve(eqkm1.d.mat))],
                    [eqk.bU.mat().mat.solve(eqk.bL.mat().mat.solve(eqk.d.mat))],
                ],
            ).mat,
        )

        self.add_kernel(
            (ntype.__name__, "solve_two"),
            [
                ReadPairStridedWithDefault(
                    "eq", Pair(eqkm1, eqk), default=eq_zero, block_size=32, **self.accessor_kwargs
                ),
            ],
            [
                WriteStrided("solu2km1", solu2[:tdim], block_size=32, **self.accessor_kwargs),
                WriteStrided("solu2k", solu2[tdim:], block_size=32, **self.accessor_kwargs),
            ],
            block_size=32,
        )
        xkm1 = sf.Matrix(tdim, 1).symbolic("xkm1")
        xkp1 = sf.Matrix(tdim, 1).symbolic("xkp1")
        rhs = eqk.d - eqk.a * xkm1 - eqk.c * xkp1
        soluk = sf.Matrix(eqk.bU.mat().mat.solve(eqk.bL.mat().mat.solve(rhs.mat)))
        self.add_kernel(
            (ntype.__name__, "CR_up"),
            [
                ReadStrided("eqk", eqk, block_size=32, **self.accessor_kwargs),
                ReadPairStridedWithDefault(
                    "x",
                    Pair(xkm1, xkp1),
                    default=xkm1.zero(),
                    block_size=32,
                    **self.accessor_kwargs,
                ),
            ],
            [
                WriteStrided("soluk", soluk, after="x", block_size=32, **self.accessor_kwargs),
            ],
            block_size=32,
        )

    def make_kernels(self) -> T.List[Kernel]:  # noqa: PLR0915
        """
        Generates the kernels for each node type used in the solver.
        """

        akws = self.accessor_kwargs
        gain = sf.symbols("gain")

        self.nonzero_jtj_patterns: T.Dict[T.Type[sf.Storage], T.List[bool]] = {
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

        self.preconds: T.Dict[T.Type[sf.Storage], T.Tuple[sf.Matrix, sf.Matrix, sf.Matrix]] = {}
        self.njtrs: T.Dict[T.Type[sf.Storage], sf.Matrix] = {}
        for ntype in self.node_types:
            precond_diag, precond_tril = symbolic_diagonal_and_lower_triangle(
                Ops.tangent_dim(ntype), "precond"
            )

            precond = from_diagonal_and_lower_triangle(precond_diag, precond_tril)
            precond = precond.from_storage(
                [
                    v if nz else 0
                    for v, nz in zip(precond.to_storage(), self.nonzero_jtj_patterns[ntype])
                ]
            )
            for i in range(precond.shape[0]):
                precond[i, i] = precond[i, i] * (1 + self.diag) + sf.epsilon() * self.diag
            self.preconds[ntype] = precond_diag, precond_tril, precond
            self.njtrs[ntype] = sf.Matrix(Ops.tangent_dim(ntype), 1).symbolic("negJTres")
        self.tridig_Uk_syms: T.Dict[T.Type[sf.Storage], sf.Matrix] = {}
        self.tridig_Ukp1_syms: T.Dict[T.Type[sf.Storage], sf.Matrix] = {}
        for fac in self.factors:
            for arg, typ in fac.node_arg_types.items():
                if fac.isnodepair[arg]:
                    if typ in self.tridig_Uk_syms:
                        raise ValueError(f"Only one factor can use Pairwise {typ}")
                    _Uk = fac.tridig_U_syms[f"{arg}_tridig"]
                    self.tridig_Uk_syms[typ] = symbolic_with_const(_Uk, "Uk")
                    self.tridig_Ukp1_syms[typ] = symbolic_with_const(_Uk, "Ukp1")

        for ntype in self.node_types:
            name = ntype.__name__

            state = Ops.symbolic(ntype, name)
            p_k = sf.Matrix(Ops.tangent_dim(ntype), 1).symbolic(f"{name}_p")
            p_kp1 = sf.Matrix(Ops.tangent_dim(ntype), 1).symbolic(f"{name}_p_kp1")
            step_k = sf.Matrix(Ops.tangent_dim(ntype), 1).symbolic(f"{name}_step")
            step_kp1 = sf.Matrix(Ops.tangent_dim(ntype), 1).symbolic(f"{name}_step")
            r_k = sf.Matrix(Ops.tangent_dim(ntype), 1).symbolic(f"{name}_r_k")
            z = sf.Matrix(Ops.tangent_dim(ntype), 1).symbolic(f"{name}_z")
            w = sf.Matrix(Ops.tangent_dim(ntype), 1).symbolic(f"{name}_w")
            mp = sf.Matrix(Ops.tangent_dim(ntype), 1).symbolic(f"{name}_Mp")
            njtr = self.njtrs[ntype]
            precond_diag, precond_tril, precond = self.preconds[ntype]
            normalized = sf.Matrix(precond.mat.solve(self.njtrs[ntype].mat, method="LDL"))
            beta = sf.Symbol(f"{name}_beta")
            self.add_kernel(
                (name, "retract"),
                [ReadSequential(name, state, **akws), ReadSequential("delta", step_k, **akws)],
                [
                    WriteSequential(
                        f"out_{name}_retracted", Ops.retract(state, step_k.to_storage()), **akws
                    )
                ],
            )
            if ntype in self.tridig_Uk_syms:
                self.add_tridiagonal_solver_kernels_for_node(ntype)

            self.add_kernel(
                (name, "normalize"),
                [
                    ReadSequential("precond_diag", precond_diag, **akws),
                    ReadSequential("precond_tril", precond_tril, **akws),
                    ReadSequential("njtr", njtr, **akws),
                    ReadUnique("diag", self.diag, **akws),
                ],
                [WriteSequential("out_normalized", normalized, **akws)],
            )

            self.add_kernel(
                (name, "start_w"),
                [
                    ReadSequential(f"{name}_precond_diag", precond_diag, **akws),
                    ReadUnique("diag", self.diag, **akws),
                    ReadSequential(f"{name}_p", p_k, **akws),
                ],
                [
                    WriteSequential(
                        f"out_{name}_w", self.diag * precond_diag.multiply_elementwise(p_k), **akws
                    )
                ],
            )
            self.add_kernel(
                (name, "start_w_contribute"),
                [
                    ReadSequential(f"{name}_precond_diag", precond_diag, **akws),
                    ReadUnique("diag", self.diag, **akws),
                    ReadSequential(f"{name}_p", p_k, **akws),
                ],
                [
                    AddSequential(
                        f"out_{name}_w", self.diag * precond_diag.multiply_elementwise(p_k), **akws
                    )
                ],
            )

            self.add_kernel(
                (name, "alpha_numerator_denominator"),
                [
                    ReadSequential(f"{name}_p_kp1", p_kp1, **akws),
                    ReadSequential(f"{name}_r_k", r_k, **akws),
                    ReadSequential(f"{name}_w", w, **akws),
                ],
                [
                    AddSum(f"{name}_total_ag", p_kp1.T * r_k, **akws),
                    AddSum(f"{name}_total_ac", p_kp1.T * w, **akws),
                ],
            )
            self.add_kernel(
                (name, "alpha_denumerator_or_beta_nummerator"),
                [
                    ReadSequential(f"{name}_p_kp1", p_kp1, **akws),
                    ReadSequential(f"{name}_w", w, **akws),
                ],
                [AddSum(f"{name}_out", p_kp1.T * w, **akws)],
            )

            self.add_kernel(
                (name, "update_r_first"),
                [
                    ReadSequential(f"{name}_r_k", r_k, **akws),
                    ReadSequential(f"{name}_w", w, **akws),
                    ReadUnique("negalpha", gain, **akws),
                ],
                [
                    WriteSequential(
                        f"out_{name}_r_kp1", r_k + w * gain, after=f"{name}_r_k", **akws
                    ),
                    AddSum(f"out_{name}_r_0_norm2_tot", r_k.squared_norm(), **akws),
                    AddSum(f"out_{name}_r_kp1_norm2_tot", (r_k + w * gain).squared_norm(), **akws),
                ],
            )
            self.add_kernel(
                (name, "update_r"),
                [
                    ReadSequential(f"{name}_r_k", r_k, **akws),
                    ReadSequential(f"{name}_w", w, **akws),
                    ReadUnique("negalpha", gain, **akws),
                ],
                [
                    WriteSequential(
                        f"out_{name}_r_kp1", r_k + w * gain, after=f"{name}_r_k", **akws
                    ),
                    AddSum(f"out_{name}_r_kp1_norm2_tot", (r_k + w * gain).squared_norm(), **akws),
                ],
            )

            self.add_kernel(
                (name, "update_step_first"),
                [ReadSequential(f"{name}_p_kp1", p_kp1, **akws), ReadUnique("alpha", gain, **akws)],
                [WriteSequential(f"out_{name}_step_kp1", p_kp1 * gain, **akws)],
            )
            self.add_kernel(
                (name, "update_step"),
                [
                    ReadSequential(f"{name}_step_k", step_k, **akws),
                    ReadSequential(f"{name}_p_kp1", p_kp1, **akws),
                    ReadUnique("alpha", gain, **akws),
                ],
                [
                    WriteSequential(
                        f"out_{name}_step_kp1",
                        step_k + p_kp1 * gain,
                        after=f"{name}_step_k",
                        **akws,
                    )
                ],
            )
            self.add_kernel(
                (name, "update_p"),
                [
                    ReadSequential(f"{name}_z", z, **akws),
                    ReadSequential(f"{name}_p_k", p_k, **akws),
                    ReadUnique("beta", beta, **akws),
                ],
                [
                    WriteSequential(
                        f"out_{name}_p_kp1", z + p_k * beta, after=f"{name}_p_k", **akws
                    ),
                ],
            )
            self.add_kernel(
                (name, "update_Mp"),
                [
                    ReadSequential(f"{name}_r_k", r_k, **akws),
                    ReadSequential(f"{name}_Mp", mp, **akws),
                    ReadUnique("beta", beta, **akws),
                ],
                [
                    WriteSequential(
                        f"out_{name}_Mp_kp1", r_k + mp * beta, after=f"{name}_Mp", **akws
                    ),
                    WriteSequential(f"out_{name}_w", r_k + mp * beta, after=f"{name}_Mp", **akws),
                ],
            )

            """
            From L(0) - L(h_{lm}) in METHODS FOR NON-LINEAR LEAST SQUARES PROBLEMS, p.25

            """
            self.add_kernel(
                (name, "pred_decrease_times_two"),
                [
                    ReadSequential(f"{name}_step", step_kp1, **akws),
                    ReadSequential(f"{name}_precond_diag", precond_diag, **akws),
                    ReadUnique("diag", self.diag, **akws),
                    ReadSequential(f"{name}_njtr", njtr, **akws),
                ],
                [
                    AddSum(
                        f"out_{name}_pred_dec",
                        step_kp1.T
                        * (njtr + self.diag * precond_diag.multiply_elementwise(step_kp1)),
                        **akws,
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
