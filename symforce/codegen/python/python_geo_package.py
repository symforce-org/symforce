import os

from symforce import logger
from symforce import types as T

from ..codegen_util import CodegenMode
from .. import function_codegen
from .. import geo_package_common
from .. import template_util

TEMPLATE_DIR = os.path.join(template_util.PYTHON_TEMPLATE_DIR, "geo_package")


def generate(output_dir, types=geo_package_common.DEFAULT_TYPES):
    # type: (str, T.Tuple) -> None
    """
    Generate the runtime geometry package in Python.

    Args:
        output_dir (str):
        types (tuple(type)): Classes to generate
    """
    # Subdirectory for everything we'll generate
    package_dir = os.path.join(output_dir, "geo")
    logger.info('Creating Python package at: "{}"'.format(package_dir))

    templates = template_util.TemplateList()

    # Build up templates for each type
    for cls in types:
        data = geo_package_common.class_data(cls, mode=CodegenMode.PYTHON2)

        for path in (
            "CLASS.py",
            "ops/__init__.py",
            "ops/impl/__init__.py",
            "ops/impl/CLASS/__init__.py",
            "ops/impl/CLASS/group_ops.py",
            "ops/impl/CLASS/lie_group_ops.py",
        ):
            template_path = os.path.join(TEMPLATE_DIR, path) + ".jinja"
            output_path = os.path.join(package_dir, path).replace("CLASS", cls.__name__.lower())
            templates.add(template_path, output_path, data)

    # Package init
    templates.add(
        os.path.join(TEMPLATE_DIR, "__init__.py.jinja"),
        os.path.join(package_dir, "__init__.py"),
        dict(function_codegen.common_data(), all_types=types,),
    )

    # Test example
    for name in ("geo_package_python_test.py",):
        templates.add(
            os.path.join(TEMPLATE_DIR, "example", name) + ".jinja",
            os.path.join(output_dir, "example", name),
            dict(function_codegen.common_data(), all_types=types,),
        )

    templates.render()
