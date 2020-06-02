import os

from symforce import logger
from symforce import sympy as sm
from symforce import types as T

from .. import codegen_util
from .. import template_util
from ..evaluator_package_common import EvaluatorCodegenSpec

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates", "evaluator_package")

NUMPY_DTYPE_FROM_SCALAR_TYPE = {"double": "numpy.float64", "float": "numpy.float32"}


def generate_evaluator(spec):
    # type: (EvaluatorCodegenSpec) -> T.Dict[str, T.Any]
    """
    Generate everything needed to numerically compute the output values in python,
    including types, serialization, and execution code. Returns a dictionary of
    output quantities, including a live dynamic import of the package.
    """
    # Subdirectory for everything we'll generate
    package_dir = os.path.join(spec.output_dir, spec.name)
    logger.info('Creating Python package at: "{}"'.format(package_dir))

    # List of (template_path, output_path, data)
    templates = template_util.TemplateList()

    # Default data for templates
    data = {
        "spec": spec,
        "types_package": "{}.types".format(spec.name),
        "np_scalar_types": NUMPY_DTYPE_FROM_SCALAR_TYPE,
    }

    # -------------------------------------------------------------------------
    # Emit types
    # -------------------------------------------------------------------------
    for typename in spec.types:
        templates.add(
            os.path.join(TEMPLATE_DIR, "types", "type.py.jinja"),
            os.path.join(package_dir, "types", "{}.py".format(typename)),
            dict(data, typename=typename),
        )

    # Init that imports all types
    templates.add(
        os.path.join(TEMPLATE_DIR, "types", "__init__.py.jinja"),
        os.path.join(package_dir, "types", "__init__.py"),
        dict(data, typenames=spec.types.keys()),
    )

    # -------------------------------------------------------------------------
    # Emit StorageOps operations for types
    # -------------------------------------------------------------------------
    for typename in spec.types:
        templates.add(
            os.path.join(TEMPLATE_DIR, "storage_ops", "storage_ops_type.py.jinja"),
            os.path.join(package_dir, "storage_ops", "{}.py".format(typename)),
            dict(data, typename=typename),
        )

    # Init that contains a registration function to add ops to types
    templates.add(
        os.path.join(TEMPLATE_DIR, "storage_ops", "__init__.py.jinja"),
        os.path.join(package_dir, "storage_ops", "__init__.py"),
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

    templates.add(
        os.path.join(TEMPLATE_DIR, "evaluator.py.jinja"),
        os.path.join(package_dir, "evaluator.py"),
        dict(data, intermediate_terms=intermediate_terms, output_terms=output_terms),
    )

    # Overall package init
    templates.add(
        os.path.join(TEMPLATE_DIR, "__init__.py.jinja"),
        os.path.join(package_dir, "__init__.py"),
        data,
    )

    # -------------------------------------------------------------------------
    # Emit usage example
    # -------------------------------------------------------------------------
    templates.add(
        os.path.join(TEMPLATE_DIR, "example", "example.py.jinja"),
        os.path.join(package_dir, "example", "example.py"),
        data,
    )

    # -------------------------------------------------------------------------
    # Render and return
    # -------------------------------------------------------------------------
    templates.render()

    return {
        "generated_files": [entry.output_path for entry in templates.items],
        "package_dir": package_dir,
    }
