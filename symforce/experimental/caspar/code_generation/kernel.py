# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import re
from pathlib import Path

from ..code_formulation.compute_graph_optimizer import ComputeGraphOptimizer
from ..code_formulation.compute_graph_sorter import ComputeGraphSorter
from ..memory.accessors import Accessor
from ..memory.accessors import _ReadAccessor
from ..memory.accessors import _WriteAccessor
from ..source.templates import env
from ..source.templates import write_if_different

EMPTY_BLOCK_PATTERN = re.compile(r"\s*if \(global_thread_idx < problem_size\)\s*\{\n\s*\};\s*\n")


class Kernel:
    """
    Class representing a single kernel in the caspar library.
    """

    def __init__(
        self,
        name: str,
        inputs: list[_ReadAccessor],
        outputs: list[_WriteAccessor],
        expose_to_python: bool = True,
    ):
        self.name = name

        self.inputs = inputs
        self.outputs = outputs
        self.accessors: list[Accessor] = [*inputs, *outputs]
        self.shared_size_req = max(acc.shared_size_req() for acc in self.accessors)
        self.expose_to_python = expose_to_python

    def generate(self, out_dir: Path) -> None:
        funcset = ComputeGraphOptimizer(self.inputs, self.outputs)
        ordering = ComputeGraphSorter(list(funcset.funcs()))
        self.code_lines = ordering.get_lines()
        self.registers = [f"r{i}" for i in range(ordering.max_registers)]

        code = env.get_template("kernel.cu.jinja").render(kernel=self)
        code = EMPTY_BLOCK_PATTERN.sub("", code)
        write_if_different(code, out_dir.joinpath(f"kernel_{self.name}.cu"))

        header = env.get_template("kernel.h.jinja").render(kernel=self)
        write_if_different(header, out_dir.joinpath(f"kernel_{self.name}.h"))

    def __repr__(self) -> str:
        return f"Kernel [{self.name}]"
