import os
import tempfile

from symforce import logger
from symforce import codegen
from symforce.codegen import template_util


def generate(config: codegen.CodegenConfig, output_dir: str = None) -> str:
    """
    Generate the sym util package for the given language.
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = tempfile.mkdtemp(
            prefix=f"sf_codegen_{type(config).__name__.lower()}_", dir="/tmp"
        )
        logger.debug(f"Creating temp directory: {output_dir}")

    # Subdirectory for everything we'll generate
    package_dir = os.path.join(output_dir, "sym", "util")
    templates = template_util.TemplateList()

    if isinstance(config, codegen.CppConfig):
        templates.add(
            template_path=os.path.join(template_util.CPP_TEMPLATE_DIR, "typedefs.h.jinja"),
            output_path=os.path.join(package_dir, "typedefs.h"),
            data={},
        )

        templates.add(
            template_path=os.path.join(template_util.CPP_TEMPLATE_DIR, "type_ops.h.jinja"),
            output_path=os.path.join(package_dir, "type_ops.h"),
            data={},
        )

        templates.add(
            template_path=os.path.join(template_util.CPP_TEMPLATE_DIR, "epsilon.h.jinja"),
            output_path=os.path.join(package_dir, "epsilon.h"),
            data={},
        )

        templates.add(
            template_path=os.path.join(template_util.CPP_TEMPLATE_DIR, "epsilon.cc.jinja"),
            output_path=os.path.join(package_dir, "epsilon.cc"),
            data={},
        )
    else:
        # sym/util is currently C++ only
        pass

    templates.render()

    return output_dir
