from __future__ import absolute_import

import os
import tempfile
import textwrap
import collections

from symforce import ops
from symforce import geo
from symforce import logger
from symforce import python_util
from symforce import sympy as sm
from symforce import types as T
from symforce.codegen import Codegen
from symforce.codegen import CodegenMode
from symforce.codegen import codegen_util
from symforce.codegen import template_util

CURRENT_DIR = os.path.dirname(__file__)
# Default geo types to generate
DEFAULT_GEO_TYPES = (geo.Rot2, geo.Pose2, geo.Rot3, geo.Pose3)


def make_storage_ops_funcs(cls, mode):
    # type: (T.Type, codegen_util.CodegenMode) -> T.List[Codegen]
    """
    Create func spec arguments for storage_ops on the given class.
    """
    storage_vec = geo.M(range(ops.StorageOps.storage_dim(cls)))
    return [
        Codegen.function(
            name="ToStorage",
            func=ops.StorageOps.to_storage,
            input_types=[cls],
            mode=mode,
            docstring=ops.StorageOps.to_storage.__doc__,
        ),
        Codegen.function(
            name="FromStorage",
            func=(lambda vec: ops.StorageOps.from_storage(cls, vec)),
            input_types=[storage_vec],
            mode=mode,
            docstring=ops.StorageOps.from_storage.__doc__,
        ),
    ]


def make_group_ops_funcs(cls, mode):
    # type: (T.Type, codegen_util.CodegenMode) -> T.List[Codegen]
    """
    Create func spec arguments for group ops on the given class.
    """
    identity = Codegen.function(
        name="Identity", func=(lambda: ops.GroupOps.identity(cls)), input_types=[], mode=mode
    )

    inverse = Codegen.function(
        name="Inverse",
        func=ops.GroupOps.inverse,
        input_types=[cls],
        mode=mode,
        docstring=ops.GroupOps.inverse.__doc__,
    )

    compose = Codegen.function(
        name="Compose",
        func=ops.GroupOps.compose,
        input_types=[cls, cls],
        mode=mode,
        docstring=ops.GroupOps.compose.__doc__,
    )

    between = Codegen.function(
        name="Between", func=ops.GroupOps.between, input_types=[cls, cls], mode=mode
    )

    return [
        identity,
        inverse,
        compose,
        between,
        inverse.create_with_derivatives(
            name="InverseWithJacobian", use_product_manifold_for_pose3=False
        ),
        compose.create_with_derivatives(
            name="ComposeWithJacobians", use_product_manifold_for_pose3=False
        ),
        between.create_with_derivatives(
            name="BetweenWithJacobians", use_product_manifold_for_pose3=False
        ),
    ]


def make_lie_group_ops_funcs(cls, mode):
    # type: (T.Type, codegen_util.CodegenMode) -> T.List[Codegen]
    """
    Create func spec arguments for lie group ops on the given class.
    """
    tangent_vec = geo.M(range(ops.LieGroupOps.tangent_dim(cls)))
    return [
        Codegen.function(
            name="FromTangent",
            func=(lambda vec, epsilon: ops.LieGroupOps.from_tangent(cls, vec, epsilon)),
            input_types=[tangent_vec, sm.Symbol],
            mode=mode,
            docstring=ops.LieGroupOps.from_tangent.__doc__,
        ),
        Codegen.function(
            name="ToTangent",
            func=ops.LieGroupOps.to_tangent,
            input_types=[cls, sm.Symbol],
            mode=mode,
            docstring=ops.LieGroupOps.to_tangent.__doc__,
        ),
        Codegen.function(
            name="Retract",
            func=ops.LieGroupOps.retract,
            input_types=[cls, tangent_vec, sm.Symbol],
            mode=mode,
            docstring=ops.LieGroupOps.retract.__doc__,
        ),
        Codegen.function(
            name="LocalCoordinates",
            func=ops.LieGroupOps.local_coordinates,
            input_types=[cls, cls, sm.Symbol],
            mode=mode,
            docstring=ops.LieGroupOps.local_coordinates.__doc__,
        ),
    ]


def geo_class_data(cls, mode):
    # type: (T.Type, CodegenMode) -> T.Dict[str, T.Any]
    """
    Data for template generation of this class. Contains all useful info for
    class-specific templates.
    """
    data = Codegen.common_data()
    data["cls"] = cls

    data["specs"] = collections.defaultdict(list)

    for func in make_storage_ops_funcs(cls, mode):
        data["specs"]["StorageOps"].append(func)

    for func in make_group_ops_funcs(cls, mode):
        data["specs"]["GroupOps"].append(func)

    for func in make_lie_group_ops_funcs(cls, mode):
        data["specs"]["LieGroupOps"].append(func)

    data["doc"] = textwrap.dedent(cls.__doc__).strip() if cls.__doc__ else ""
    data["is_lie_group"] = hasattr(cls, "from_tangent")

    return data


def generate(mode, output_dir=None, gen_example=True):
    # type: (CodegenMode, str, bool) -> str
    """
    Generate the geo package for the given language.

    TODO(hayk): Take scalar_type list here.
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="sf_codegen_{}_".format(mode.name), dir="/tmp")
        logger.debug("Creating temp directory: {}".format(output_dir))
    # Subdirectory for everything we'll generate
    package_dir = os.path.join(output_dir, "geo")
    templates = template_util.TemplateList()

    # First generate the sym/util package as it's a dependency of the geo package
    from symforce.codegen import sym_util_package_codegen

    sym_util_package_codegen.generate(mode=CodegenMode.CPP, output_dir=output_dir)

    if mode == CodegenMode.PYTHON2:
        logger.info('Creating Python package at: "{}"'.format(package_dir))
        template_dir = os.path.join(template_util.PYTHON_TEMPLATE_DIR, "geo_package")

        # Build up templates for each type
        for cls in DEFAULT_GEO_TYPES:
            data = geo_class_data(cls, mode=CodegenMode.PYTHON2)

            for path in (
                "CLASS.py",
                "ops/__init__.py",
                "ops/CLASS/__init__.py",
                "ops/CLASS/group_ops.py",
                "ops/CLASS/lie_group_ops.py",
            ):
                template_path = os.path.join(template_dir, path) + ".jinja"
                output_path = os.path.join(package_dir, path).replace("CLASS", cls.__name__.lower())
                templates.add(template_path, output_path, data)

        # Package init
        templates.add(
            os.path.join(template_dir, "__init__.py.jinja"),
            os.path.join(package_dir, "__init__.py"),
            dict(Codegen.common_data(), all_types=DEFAULT_GEO_TYPES,),
        )

        # Test example
        if gen_example:
            for name in ("geo_package_python_test.py",):
                templates.add(
                    os.path.join(template_dir, "example", name) + ".jinja",
                    os.path.join(output_dir, "example", name),
                    dict(Codegen.common_data(), all_types=DEFAULT_GEO_TYPES,),
                )

    elif mode == CodegenMode.CPP:
        logger.info('Creating C++ package at: "{}"'.format(package_dir))
        template_dir = os.path.join(template_util.CPP_TEMPLATE_DIR, "geo_package")

        # Build up templates for each type
        for cls in DEFAULT_GEO_TYPES:
            data = geo_class_data(cls, mode=CodegenMode.CPP)

            for path in (
                "CLASS.h",
                "CLASS.cc",
                "ops/CLASS/storage_ops.h",
                "ops/CLASS/storage_ops.cc",
                "ops/CLASS/group_ops.h",
                "ops/CLASS/group_ops.cc",
                "ops/CLASS/lie_group_ops.h",
                "ops/CLASS/lie_group_ops.cc",
            ):
                template_path = os.path.join(template_dir, path) + ".jinja"
                output_path = os.path.join(package_dir, path).replace("CLASS", cls.__name__.lower())
                templates.add(template_path, output_path, data)

        # Render non geo type specific templates
        for template_name in python_util.files_in_dir(
            os.path.join(template_dir, "ops"), relative=True
        ):
            if "CLASS" in template_name:
                continue

            if not template_name.endswith(".jinja"):
                continue

            templates.add(
                os.path.join(template_dir, "ops", template_name),
                os.path.join(package_dir, "ops", template_name[: -len(".jinja")]),
                dict(Codegen.common_data()),
            )

        # Test example
        if gen_example:
            for name in ("geo_package_cpp_test.cc", "Makefile"):
                templates.add(
                    os.path.join(template_dir, "example", name) + ".jinja",
                    os.path.join(output_dir, "example", name),
                    dict(
                        Codegen.common_data(),
                        all_types=DEFAULT_GEO_TYPES,
                        include_dir=output_dir,
                        symforce_include_dir=os.path.join(CURRENT_DIR, "../../"),
                        eigen_include_dir=os.path.realpath(
                            os.path.join(
                                CURRENT_DIR, "***REMOVED***/include/eigen3/"
                            )
                        ),
                        lib_dir=os.path.join(output_dir, "example"),
                    ),
                )

    else:
        raise NotImplementedError('Unknown mode: "{}"'.format(mode))

    templates.render()

    return output_dir
