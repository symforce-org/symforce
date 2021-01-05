import os
import tempfile

from symforce import logger
from symforce.codegen import CodegenMode
from symforce.codegen import template_util


def generate(mode: CodegenMode, output_dir: str = None) -> str:
    """
    Generate the sym util package for the given language.
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix=f"sf_codegen_{mode.name}_", dir="/tmp")
        logger.debug(f"Creating temp directory: {output_dir}")

    # Subdirectory for everything we'll generate
    package_dir = os.path.join(output_dir, "sym", "util")
    templates = template_util.TemplateList()

    if mode == CodegenMode.CPP:
        templates.add(
            template_path=os.path.join(template_util.CPP_TEMPLATE_DIR, "typedefs.h.jinja"),
            output_path=os.path.join(package_dir, "typedefs.h"),
            data={},
        )
    else:
        # sym/util is currently C++ only
        pass

    templates.render()

    return output_dir
