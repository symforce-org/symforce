# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import functools
import operator

from symforce import typing as T
from symforce.codegen.backends.python.python_config import PythonConfig
from symforce.codegen.codegen import Codegen
from symforce.codegen.codegen import GeneratedPaths
from symforce.codegen.template_util import RenderTemplateConfig
from symforce.codegen.template_util import render_template


def generate_module_init(
    codegen_objects: T.Iterable[Codegen],
    generated_paths: T.Iterable[GeneratedPaths],
    template_config: RenderTemplateConfig = RenderTemplateConfig(),
) -> None:
    """
    Generate an __init__.py for a directory of generated functions that imports all the functions
    for convenience

    Args:
        specs: The Codegen objects for the generated functions
        generated_pathses: The generated paths for the functions, returned by
            Codegen.generate_function
        template_config: The configuration for rendering the template
    """
    if not functools.reduce(operator.eq, (paths.function_dir for paths in generated_paths)):
        raise ValueError("All generated paths must have the same function directory")

    render_template(
        template_path="function/module_init.py.jinja",
        data={"specs": codegen_objects},
        config=template_config,
        template_dir=PythonConfig.template_dir(),
        output_path=next(iter(generated_paths)).function_dir / "__init__.py",
    )
