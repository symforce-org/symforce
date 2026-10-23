# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import logging
import re
from pathlib import Path

from symforce.python_util import parts_to_pascal

from ..code_formulation.dabseg_from_accessors import make_dabseg
from ..code_formulation.dabseg_sorter import get_lines
from ..memory.accessors import Accessor
from ..memory.accessors import AddSharedSum
from ..memory.accessors import ReadShared
from ..memory.accessors import _ReadAccessor
from ..memory.accessors import _WriteAccessor
from ..memory.dtype import DType
from ..source.templates import env
from ..source.templates import write_if_different

EMPTY_BLOCK_PATTERN = re.compile(r"\s*if \(global_thread_idx < problem_size\)\s*\{\n\s*\};\s*\n")


# sizeof(SharedIndex) in source/runtime/shared_indices.h: {uint32_t unique; uint16_t target, argsort}.
SHARED_INDEX_BYTES = 8
SHARED_ALIGNMENT = 16
# Shared memory per block available without opt-in on every GPU, and the largest opt-in limit of
# any (sm_90).
DEFAULT_SHARED_BYTES_PER_BLOCK = 48 * 1024
MAX_SHARED_BYTES_PER_BLOCK = 227 * 1024


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


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

        # Dynamic shared memory layout: the scratch buffer, then one block of shared indices per
        # shared-index accessor that owns its indices.
        offset = _align(self.shared_size_req, SHARED_ALIGNMENT)
        self.shared_index_offsets: dict[str, int] = {}
        for acc in self.accessors:
            if isinstance(acc, (ReadShared, AddSharedSum)) and acc.prep_lines:
                self.shared_index_offsets[acc.idx_name] = offset
                offset = _align(offset + SHARED_INDEX_BYTES * block_size, SHARED_ALIGNMENT)
        self.dynamic_shared_bytes = offset
        if self.dynamic_shared_bytes > MAX_SHARED_BYTES_PER_BLOCK:
            shared_args = ", ".join(self.shared_index_offsets) or "none"
            raise ValueError(
                f"Kernel {name} needs {self.dynamic_shared_bytes} bytes of shared memory per block "
                f"({self.shared_size_req} scratch + {SHARED_INDEX_BYTES * block_size} per shared "
                f"argument: {shared_args}), more than any GPU supports "
                f"({MAX_SHARED_BYTES_PER_BLOCK}). Use fewer shared (Tunable/Shared) arguments, "
                "e.g. indexed constants, or a smaller dtype."
            )
        if self.dynamic_shared_bytes > DEFAULT_SHARED_BYTES_PER_BLOCK:
            logging.warning(
                "Kernel %s needs %d bytes of shared memory per block, above the %d every GPU "
                "supports; it will only launch on GPUs with a larger opt-in limit (e.g. 64 KB on "
                "sm_75, 99 KB on sm_86, 163 KB on sm_80).",
                name,
                self.dynamic_shared_bytes,
                DEFAULT_SHARED_BYTES_PER_BLOCK,
            )

    def generate(self, out_dir: Path) -> None:
        dabseg = make_dabseg(self.inputs, self.outputs)
        self.code_lines, n_registers = get_lines(dabseg, dtype=self.kernel_t)
        self.registers = [f"r{i}" for i in range(n_registers)]

        code = env.get_template("kernel.cu.jinja").render(
            kernel=self, parts_to_pascal=parts_to_pascal
        )
        code = EMPTY_BLOCK_PATTERN.sub("", code)
        write_if_different(code, out_dir.joinpath(f"kernel_{self.name}.cu"))

        header = env.get_template("kernel.h.jinja").render(
            kernel=self, parts_to_pascal=parts_to_pascal
        )
        write_if_different(header, out_dir.joinpath(f"kernel_{self.name}.h"))

    def __repr__(self) -> str:
        return f"Kernel [{self.name}]"
