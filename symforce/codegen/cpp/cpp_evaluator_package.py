import os

from symforce import logger
from symforce import sympy as sm
from symforce import types as T

from .. import codegen_util
from .. import template_util
from ..evaluator_package_common import EvaluatorCodegenSpec

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates", "evaluator_package")


def generate_evaluator(spec):
    # type: (EvaluatorCodegenSpec) -> T.Dict[str, T.Any]
    """
    Generate everything needed to numerically compute the output values in C++,
    including types, serialization, and execution code. Returns a dictionary of output
    quantities.
    """
    # Subdirectory for everything we'll generate
    package_dir = os.path.join(spec.output_dir, spec.name)
    logger.info('Creating C++ package at: "{}"'.format(package_dir))

    # List of (template_path, output_path, data)
    templates = template_util.TemplateList()

    # Default data for templates
    data = {"spec": spec, "types_package": "{}/types".format(spec.name)}

    # -------------------------------------------------------------------------
    # Emit types
    # -------------------------------------------------------------------------
    for typename in spec.types:
        templates.add(
            os.path.join(TEMPLATE_DIR, "types/type.h.jinja"),
            os.path.join(package_dir, "types", "{}.h".format(typename)),
            dict(data, typename=typename),
        )

    # -------------------------------------------------------------------------
    # Emit StorageOps operations for types
    # -------------------------------------------------------------------------
    for typename in spec.types:
        templates.add(
            os.path.join(TEMPLATE_DIR, "storage_ops/storage_ops_type.h.jinja"),
            os.path.join(package_dir, "storage_ops/{}.h".format(typename)),
            dict(data, typename=typename),
        )

    # Init that contains traits definition and imports all types
    templates.add(
        os.path.join(TEMPLATE_DIR, "storage_ops/storage_ops.h.jinja"),
        os.path.join(package_dir, "storage_ops.h"),
        dict(data, typenames=spec.types.keys()),
    )

    # -------------------------------------------------------------------------
    # Emit evaluator module
    # -------------------------------------------------------------------------
    # Convert the output expressions into code, with CSE
    intermediate_terms, output_terms = codegen_util.print_code(
        input_symbols=spec.input_values_recursive,
        output_exprs=spec.output_values_recursive,
        mode=spec.mode,
    )

    # CC file contains generated code
    templates.add(
        os.path.join(TEMPLATE_DIR, "evaluator.cc.jinja"),
        os.path.join(package_dir, "evaluator.cc"),
        dict(data, intermediate_terms=intermediate_terms, output_terms=output_terms),
    )

    # Header exposes a nice wrapper class
    templates.add(
        os.path.join(TEMPLATE_DIR, "evaluator.h.jinja"),
        os.path.join(package_dir, "evaluator.h"),
        data,
    )

    # -------------------------------------------------------------------------
    # Emit usage example
    # -------------------------------------------------------------------------
    templates.add(
        os.path.join(TEMPLATE_DIR, "example/example.cc.jinja"),
        os.path.join(package_dir, "example/example.cc"),
        data,
    )

    templates.add(
        os.path.join(TEMPLATE_DIR, "example/Makefile.jinja"),
        os.path.join(package_dir, "example/Makefile"),
        data,
    )

    # -------------------------------------------------------------------------
    # Render and return
    # -------------------------------------------------------------------------
    templates.render()

    return {
        "generated_files": [v[1] for v in templates.items],
        "package_dir": package_dir,
    }
