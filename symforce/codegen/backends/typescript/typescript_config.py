# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path
from sympy.printing.codeprinter import CodePrinter

from symforce import typing as T
from symforce.codegen.codegen_config import CodegenConfig


CURRENT_DIR = Path(__file__).parent


@dataclass
class TypescriptConfig(CodegenConfig):
    """
    Code generation config for the Typescript backend.

    Args:
    """


    @classmethod
    def backend_name(cls) -> str:
        return "typescript"

    @classmethod
    def template_dir(cls) -> Path:
        return CURRENT_DIR / "templates"

    def templates_to_render(self, generated_file_name: str) -> T.List[T.Tuple[str, str]]:
        return [
            ("function/FUNCTION.ts.jinja", f"{generated_file_name}.ts"),
            ("function/__init__.ts.jinja", "__init__.ts"),
        ]

    def printer(self) -> CodePrinter:
        from symforce.codegen.backends.typescript import typescript_code_printer

        return typescript_code_printer.TypescriptCodePrinter()
