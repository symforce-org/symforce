# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from symforce import typing as T
from symforce.experimental import caspar
from symforce.experimental.caspar.memory.dtype import DType
from symforce.ops import LieGroupOps as Ops

from ..code_generation.factor import Factor
from ..code_generation.kernel import Kernel
from ..code_generation.solver import Solver
from ..code_generation.solver import num_arg_key
from ..memory import caspar_size
from ..memory.accessors import Accessor
from ..memory.accessors import _ReadAccessor
from ..memory.accessors import _WriteAccessor
from ..memory.layouts import get_default_caspar_layout
from ..memory.pair import get_memtype
from ..memory.pair import get_symbolic
from ..source.templates import copy_if_different
from ..source.templates import env
from ..source.templates import write_if_different


def insert_sorted_unique(
    container: list[T.LieGroupElementOrType], items: T.Iterable[T.LieGroupElementOrType]
) -> None:
    for item in map(get_memtype, items):
        lo = 0
        hi = len(container)
        name = item.__name__
        while lo < hi:
            mid = (lo + hi) // 2
            if container[mid].__name__ < name:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(container) or container[lo] != item:
            container.insert(lo, item)


class CasparLibrary:
    """
    The CasLib class is the main entry point for generating and compiling caspar libraries.
    """

    def __init__(self, name: str = "caspar_lib", dtype: DType = DType.FLOAT):
        self.kernels: list[Kernel] = []
        self.factors: list[Factor] = []
        self.name = name
        self.storage_t = dtype
        # Keep sorted for deterministic template output
        self.node_types: list[T.LieGroupElement] = []
        self.exposed_types: list[T.LieGroupElement] = []

    def add_kernel(self, func: T.Callable) -> T.Callable:
        signature = T.get_type_hints(func, include_extras=True)  # type: ignore[call-arg]
        for k, v in signature.items():
            if k == "return":
                continue
            if not (hasattr(v, "__metadata__") and hasattr(v, "__origin__")):
                raise ValueError(f"Argument {k} must be of type T.Annotated.")
            if not issubclass(v.__metadata__[0], Accessor):
                raise ValueError(f"Argument {k} must have an Accessor descriptor.")

        ret_type = signature.pop("return")
        in_types = [v.__origin__ for v in signature.values()]
        in_syms = {k: get_symbolic(ST, k) for k, ST in zip(signature, in_types)}
        output = func(**in_syms)

        if isinstance(output, tuple):
            out_syms = {f"out_{i}": v for i, v in enumerate(output)}
            out_types = [v.__origin__ for v in ret_type.__args__]
            out_accessors = {f"out_{i}": a.__metadata__[0] for i, a in enumerate(ret_type.__args__)}
        else:
            out_syms = {"out": output}
            out_types = [ret_type.__origin__]
            out_accessors = {"out": ret_type.__metadata__[0]}
        defaults = {
            k: v.__metadata__[1] if len(v.__metadata__) == 2 else None for k, v in signature.items()
        }
        inputs = [
            signature[k].__metadata__[0](
                k, v, dtype=self.storage_t, kernel_dtype=self.storage_t, default=defaults[k]
            )
            for k, v in in_syms.items()
        ]
        outputs = [
            out_accessors[k](k, v, dtype=self.storage_t, kernel_dtype=self.storage_t)
            for k, v in out_syms.items()
        ]
        self.add_kernel_from_accessors(func.__name__, inputs, outputs)
        insert_sorted_unique(self.exposed_types, (get_memtype(t) for t in (*in_types, *out_types)))
        return func

    def add_kernel_from_accessors(
        self,
        name: str,
        inputs: T.List[_ReadAccessor],
        outputs: T.List[_WriteAccessor],
        expose_to_python: bool = True,
    ) -> None:
        kernel = Kernel(
            name, inputs, outputs, dtype=self.storage_t, expose_to_python=expose_to_python
        )
        self.kernels.append(kernel)
        insert_sorted_unique(self.exposed_types, [get_memtype(acc.storage) for acc in inputs])
        insert_sorted_unique(self.exposed_types, [get_memtype(acc.storage) for acc in outputs])

    def add_factor(
        self,
        func_or_name: T.Union[T.Callable, str],
        name: str | None = None,
    ) -> T.Callable:
        if isinstance(func_or_name, str):
            return lambda func: self.add_factor(func, func_or_name)
        else:
            factor = Factor(func_or_name, name, dtype=self.storage_t)
            self.factors.append(factor)
            insert_sorted_unique(self.node_types, factor.node_arg_types.values())
            insert_sorted_unique(self.exposed_types, factor.arg_types.values())
            return func_or_name

    def generate(self, out_dir: Path, use_symlinks: bool = False) -> None:
        out_dir.mkdir(exist_ok=True, parents=True)

        for fac in self.factors:
            self.kernels.extend(fac.make_kernels())

        self.generate_castype_mappings(out_dir)
        self.generate_links(out_dir, use_symlinks)
        if solver := (Solver(self) if self.factors else None):
            self.kernels.extend(solver.make_kernels())
            solver.generate(out_dir)
        self.generate_binding_file(out_dir, solver)
        self.generate_stubs(out_dir, solver)
        self.generate_buildfiles(out_dir)
        self.generate_kernels(out_dir)

    @staticmethod
    def compile(out_dir: Path, debug: bool = False) -> None:
        import subprocess

        build_dir = out_dir / "build"
        build_dir.mkdir(exist_ok=True)
        logging.info(f"Compiling {out_dir}")
        if debug:
            subprocess.run(["cmake", "-DCMAKE_BUILD_TYPE=Debug", ".."], cwd=build_dir, check=True)
        else:
            subprocess.run(["cmake", "-DCMAKE_BUILD_TYPE=Release", ".."], cwd=build_dir, check=True)
        subprocess.run(["make", "-j"], cwd=build_dir, check=True)

    # This function has no annotated return type, so that type inference in IDEs can get more
    # specific typing than ModuleType
    # TODO(aaron): Does pyright actually require we leave this annotation off to deduce more
    # specifically?
    def import_lib(self, out_dir: Path):  # type: ignore[no-untyped-def]
        """
        This function makes generating and importing the library easier in some build systems.

        It is however recommended to use the standard: from {...} import lib
        This function will create a temporary symlink that is imported as ..generated.tmp.lib for
        type checking. This file link is overwritten every time this function is called.
        """
        import importlib.util
        from typing import TYPE_CHECKING

        tmp_pyi_path = Path(caspar.__file__).parent / "tmp/lib.pyi"
        tmp_pyi_path.parent.mkdir(exist_ok=True, parents=True)
        if tmp_pyi_path.exists():
            tmp_pyi_path.unlink()
        shutil.copy(out_dir / f"{self.name}.pyi", tmp_pyi_path)
        if TYPE_CHECKING:
            # This import is useful for type inference in IDEs, or in local projects where a
            # generated caspar library is present here.  It won't always be present, and
            # won't be present in CI, so we ignore this if it's missing.
            from ..tmp import lib  # type: ignore[import-not-found,unused-ignore]
        else:
            import sysconfig as s

            lib_name = f"{self.name}{s.get_config_var('EXT_SUFFIX') or s.get_config_var('SO')}"
            spec = importlib.util.spec_from_file_location(self.name, out_dir / lib_name)
            lib = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lib)
        return lib

    def generate_stubs(self, out_dir: Path, solver: Solver | None) -> None:
        pyi = env.get_template("lib.pyi.jinja").render(
            caslib=self, solver=solver, num_arg_key=num_arg_key
        )
        write_if_different(pyi, out_dir.joinpath(f"{self.name}.pyi"))

    def generate_kernels(self, out_dir: Path) -> None:
        for kernel in sorted(self.kernels, key=lambda k: k.name != "imu_residual_res_jac_first"):
            logging.info(f"Generating kernel {kernel.name}")
            kernel.generate(out_dir)

    @staticmethod
    def generate_links(out_dir: Path, use_symlinks: bool = True) -> None:
        for f in Path(caspar.__file__).parent.glob("source/runtime/*"):
            f_new = out_dir / f.name
            if use_symlinks:
                if f_new.exists():
                    f_new.unlink()
                f_new.symlink_to(f)
            else:
                copy_if_different(f, f_new)

    def generate_castype_mappings(self, out_dir: Path) -> None:
        """
        Generates code to perform mapping between stacked format (array of structs)
        and the caspar layout of the corresponding types.
        """
        kwargs = dict(
            caslib=self,
            get_layout=get_default_caspar_layout,
            caspar_size=caspar_size,
            Ops=Ops,
            len=len,
        )
        definition = env.get_template("caspar_mappings.cu.jinja").render(**kwargs)
        write_if_different(definition, out_dir.joinpath("caspar_mappings.cu"))
        definition = env.get_template("caspar_mappings.h.jinja").render(**kwargs)
        write_if_different(definition, out_dir.joinpath("caspar_mappings.h"))
        definition = env.get_template("caspar_mappings_pybinding.h.jinja").render(**kwargs)
        write_if_different(definition, out_dir.joinpath("caspar_mappings_pybinding.h"))

    def generate_binding_file(self, out_dir: Path, solver: Solver | None) -> None:
        binding = env.get_template("pybinding.cc.jinja").render(caslib=self, solver=solver)
        write_if_different(binding, out_dir.joinpath("pybinding.cc"))

    def generate_buildfiles(self, out_dir: Path) -> None:
        for template in env.list_templates(filter_func=lambda t: t.startswith("buildfiles")):
            content = env.get_template(template).render(caslib=self)
            write_if_different(content, out_dir.joinpath(Path(template).stem))
