# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import tempfile
from pathlib import Path

from symforce import codegen
from symforce import logger
from symforce import python_util
from symforce.codegen import cam_package_codegen
from symforce.codegen import template_util


def generate(config: codegen.CodegenConfig, output_dir: Path = None) -> Path:
    """
    Generate the sym util package for the given language.
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = Path(
            tempfile.mkdtemp(prefix=f"sf_codegen_{type(config).__name__.lower()}_", dir="/tmp")
        )
        logger.debug(f"Creating temp directory: {output_dir}")

    # Subdirectory for everything we'll generate
    package_dir = output_dir / "sym" / "util"
    templates = template_util.TemplateList()

    if isinstance(config, codegen.CppConfig):
        template_dir = config.template_dir()
        templates.add(
            template_path="typedefs.h.jinja",
            output_path=package_dir / "typedefs.h",
            data={},
            template_dir=template_dir,
        )

        templates.add(
            template_path="type_ops.h.jinja",
            output_path=package_dir / "type_ops.h",
            data=dict(
                python_util=python_util,
                camera_cal_class_names=cam_package_codegen.camera_cal_class_names(),
            ),
            template_dir=template_dir,
        )

        for filename in ("epsilon.h", "epsilon.cc"):
            templates.add(
                template_path=f"{filename}.jinja",
                output_path=package_dir / filename,
                data={},
                template_dir=template_dir,
            )
    else:
        # sym/util is currently C++ only
        pass

    templates.render()

    return output_dir
