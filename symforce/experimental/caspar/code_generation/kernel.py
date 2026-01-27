# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import re
from pathlib import Path

from ..code_formulation.dabseg_from_accessors import make_dabseg
from ..code_formulation.dabseg_sorter import get_lines
from ..memory.accessors import Accessor
from ..memory.accessors import _ReadAccessor
from ..memory.accessors import _WriteAccessor
from ..memory.dtype import DType
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
        dtype: DType,
        *,
        expose_to_python: bool = True,
        block_size: int = 1024,
    ):
        self.name = name

        self.inputs = inputs
        self.outputs = outputs
        self.accessors: list[Accessor] = [*inputs, *outputs]
        self.shared_size_req = max(acc.shared_size_req() for acc in self.accessors)
        self.expose_to_python = expose_to_python
        self.kernel_t = dtype
        self.block_size = block_size

    def generate(self, out_dir: Path) -> None:
        dabseg = make_dabseg(self.inputs, self.outputs)
        self.code_lines, n_registers = get_lines(dabseg, dtype=self.kernel_t)
        self.registers = [f"r{i}" for i in range(n_registers)]

        code = env.get_template("kernel.cu.jinja").render(kernel=self)
        code = EMPTY_BLOCK_PATTERN.sub("", code)
        write_if_different(code, out_dir.joinpath(f"kernel_{self.name}.cu"))

        header = env.get_template("kernel.h.jinja").render(kernel=self)
        write_if_different(header, out_dir.joinpath(f"kernel_{self.name}.h"))

    def __repr__(self) -> str:
        return f"Kernel [{self.name}]"
