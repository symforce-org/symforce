import os

from symforce import logger
from symforce import types as T

from ..codegen_util import CodegenMode
from .. import function_codegen
from .. import geo_package_common
from .. import template_util

CURRENT_DIR = os.path.dirname(__file__)
TEMPLATE_DIR = os.path.join(template_util.CPP_TEMPLATE_DIR, "geo_package")


def generate(output_dir, types=geo_package_common.DEFAULT_TYPES):
    # type: (str, T.Tuple) -> None
    """
    Generate the runtime geometry package in C++.

    Args:
        output_dir (str):
        types (tuple(type)): Classes to generate
    """
    # Subdirectory for everything we'll generate
    package_dir = os.path.join(output_dir, "geo")
    logger.info('Creating C++ package at: "{}"'.format(package_dir))

    templates = template_util.TemplateList()

    # Build up templates for each type
    for cls in types:
        data = geo_package_common.class_data(cls, mode=CodegenMode.CPP)

        for path in (
            "CLASS.h",
            "CLASS.cc",
            "ops/impl/CLASS/storage_ops.h",
            "ops/impl/CLASS/storage_ops.cc",
            "ops/impl/CLASS/group_ops.h",
            "ops/impl/CLASS/group_ops.cc",
            "ops/impl/CLASS/lie_group_ops.h",
            "ops/impl/CLASS/lie_group_ops.cc",
        ):
            template_path = os.path.join(TEMPLATE_DIR, path) + ".jinja"
            output_path = os.path.join(package_dir, path).replace("CLASS", cls.__name__.lower())
            templates.add(template_path, output_path, data)

    # Concept headers
    for template_name in ("storage_ops.h", "group_ops.h", "lie_group_ops.h"):
        templates.add(
            os.path.join(TEMPLATE_DIR, "ops", template_name) + ".jinja",
            os.path.join(package_dir, "ops", template_name),
            dict(),
        )

    # Test example
    for name in ("geo_package_cpp_test.cc", "Makefile"):
        templates.add(
            os.path.join(TEMPLATE_DIR, "example", name) + ".jinja",
            os.path.join(output_dir, "example", name),
            dict(
                function_codegen.common_data(),
                all_types=types,
                include_dir=output_dir,
                eigen_include_dir=os.path.realpath(
                    os.path.join(
                        CURRENT_DIR, "../***REMOVED***/include/eigen3/"
                    )
                ),
                lib_dir=os.path.join(output_dir, "example"),
            ),
        )

    templates.render()
