# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import filecmp
import shutil
from pathlib import Path

import jinja2

from symforce.codegen.format_util import format_cpp

template_dir = Path(__file__).parent

env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(template_dir),
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=jinja2.StrictUndefined,
)


def write_if_different(content: str, target: Path) -> None:
    if target.suffix in {".cpp", ".cc", ".cu", ".h", ".cuh"}:
        formatted = format_cpp(content, str(target))
    else:
        formatted = content
    if not target.exists() or formatted != target.read_text():
        target.write_text(formatted)


def copy_if_different(other: Path, target: Path) -> None:
    if not target.exists() or not filecmp.cmp(other, target, shallow=False):
        shutil.copy2(other, target)
