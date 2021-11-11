import os
import tempfile
import textwrap
import collections
import functools

from symforce import ops
from symforce import geo
from symforce import logger
from symforce import python_util
from symforce import sympy as sm
from symforce import typing as T
from symforce.codegen import Codegen
from symforce.codegen import CodegenConfig
from symforce.codegen import CppConfig
from symforce.codegen import PythonConfig
from symforce.codegen import codegen_util
from symforce.codegen import template_util
from symforce import path_util

CURRENT_DIR = os.path.dirname(__file__)
# Default geo types to generate
DEFAULT_GEO_TYPES = (geo.Rot2, geo.Pose2, geo.Rot3, geo.Pose3)


def make_storage_ops_funcs(cls: T.Type, config: CodegenConfig) -> T.List[Codegen]:
    """
    Create func spec arguments for storage_ops on the given class.
    """
    storage_vec = geo.M(ops.StorageOps.storage_dim(cls), 1)
    return [
        Codegen.function(
            func=ops.StorageOps.to_storage,
            input_types=[cls],
            config=config,
            docstring=ops.StorageOps.to_storage.__doc__,
        ),
        Codegen.function(
            name="from_storage",
            func=(lambda vec: ops.StorageOps.from_storage(cls, vec)),
            input_types=[storage_vec],
            config=config,
            docstring=ops.StorageOps.from_storage.__doc__,
        ),
    ]


def make_group_ops_funcs(cls: T.Type, config: CodegenConfig) -> T.List[Codegen]:
    """
    Create func spec arguments for group ops on the given class.
    """
    identity = Codegen.function(
        name="identity", func=(lambda: ops.GroupOps.identity(cls)), input_types=[], config=config
    )

    inverse = Codegen.function(
        func=ops.GroupOps.inverse,
        input_types=[cls],
        config=config,
        docstring=ops.GroupOps.inverse.__doc__,
    )

    compose = Codegen.function(
        func=ops.GroupOps.compose,
        input_types=[cls, cls],
        config=config,
        docstring=ops.GroupOps.compose.__doc__,
    )

    between = Codegen.function(func=ops.GroupOps.between, input_types=[cls, cls], config=config)

    return [
        identity,
        inverse,
        compose,
        between,
        inverse.create_with_jacobians(),
        compose.create_with_jacobians(),
        between.create_with_jacobians(),
    ]


def make_lie_group_ops_funcs(cls: T.Type, config: CodegenConfig) -> T.List[Codegen]:
    """
    Create func spec arguments for lie group ops on the given class.
    """
    tangent_vec = geo.M(list(range(ops.LieGroupOps.tangent_dim(cls))))
    return [
        Codegen.function(
            name="from_tangent",
            func=(lambda vec, epsilon: ops.LieGroupOps.from_tangent(cls, vec, epsilon)),
            input_types=[tangent_vec, sm.Symbol],
            config=config,
            docstring=ops.LieGroupOps.from_tangent.__doc__,
        ),
        Codegen.function(
            func=ops.LieGroupOps.to_tangent,
            input_types=[cls, sm.Symbol],
            config=config,
            docstring=ops.LieGroupOps.to_tangent.__doc__,
        ),
        Codegen.function(
            func=ops.LieGroupOps.retract,
            input_types=[cls, tangent_vec, sm.Symbol],
            config=config,
            docstring=ops.LieGroupOps.retract.__doc__,
        ),
        Codegen.function(
            func=ops.LieGroupOps.local_coordinates,
            input_types=[cls, cls, sm.Symbol],
            config=config,
            docstring=ops.LieGroupOps.local_coordinates.__doc__,
        ),
    ]


def geo_class_common_data(cls: T.Type, config: CodegenConfig) -> T.Dict[str, T.Any]:
    """
    Data for template generation of this class. Contains all useful info common
    to all class-specific templates.
    """
    data = Codegen.common_data()
    data["cls"] = cls

    data["specs"] = collections.defaultdict(list)

    for func in make_storage_ops_funcs(cls, config):
        data["specs"]["StorageOps"].append(func)

    for func in make_group_ops_funcs(cls, config):
        data["specs"]["GroupOps"].append(func)

    for func in make_lie_group_ops_funcs(cls, config):
        data["specs"]["LieGroupOps"].append(func)

    data["doc"] = textwrap.dedent(cls.__doc__).strip() if cls.__doc__ else ""
    data["is_lie_group"] = hasattr(cls, "from_tangent")

    return data


def _matrix_type_aliases() -> T.Dict[T.Type, T.Dict[str, str]]:
    """
    Returns a dictionary d where d[datatype] is a mapping
    between C++ types and their type aliases that are used in
    the generated code of type datatype.
    """
    return {
        geo.Rot2: {"Eigen::Matrix<Scalar, 2, 1>": "Vector2"},
        geo.Rot3: {"Eigen::Matrix<Scalar, 3, 1>": "Vector3"},
        geo.Pose2: {"Eigen::Matrix<Scalar, 2, 1>": "Vector2"},
        geo.Pose3: {"Eigen::Matrix<Scalar, 3, 1>": "Vector3"},
    }


def _custom_generated_methods(config: CodegenConfig) -> T.Dict[T.Type, T.List[Codegen]]:
    """
    Returns a dictionary d where d[datatype] is a list of codegened functions
    we wish to be added to type datatype's generated code.

    Args:
        config (CodegenConfig): Specifies the target language of the codegened functions.
    """

    def pose2_inverse_compose(self: geo.Pose2, point: geo.Vector2) -> geo.Vector2:
        return self.inverse() * point

    def pose3_inverse_compose(self: geo.Pose3, point: geo.Vector3) -> geo.Vector3:
        return self.inverse() * point

    def codegen_mul(group: T.Type, multiplicand_type: T.Type) -> Codegen:
        """
        A helper to generate a Codegen object for groups with the method __mul__ taking
        an instance of group and composing it with an instance of multiplicand_type
        """
        return Codegen.function(
            func=group.__mul__,
            name="compose",
            input_types=[group, multiplicand_type],
            config=config,
        )

    return {
        geo.Rot2: [
            codegen_mul(geo.Rot2, geo.Vector2),
            Codegen.function(func=geo.Rot2.to_rotation_matrix, config=config),
        ],
        geo.Rot3: [
            codegen_mul(geo.Rot3, geo.Vector3),
            Codegen.function(func=geo.Rot3.to_rotation_matrix, config=config),
            Codegen.function(
                func=functools.partial(geo.Rot3.random_from_uniform_samples, pi=sm.pi),
                name="random_from_uniform_samples",
                config=config,
            ),
            Codegen.function(func=geo.Rot3.from_yaw_pitch_roll, config=config),
            Codegen.function(
                func=lambda ypr: geo.Rot3.from_yaw_pitch_roll(*ypr),
                input_types=[geo.V3],
                name="from_yaw_pitch_roll",
                config=config,
            ),
        ],
        geo.Pose2: [
            codegen_mul(geo.Pose2, geo.Vector2),
            Codegen.function(func=pose2_inverse_compose, name="inverse_compose", config=config),
        ],
        geo.Pose3: [
            codegen_mul(geo.Pose3, geo.Vector3),
            Codegen.function(func=pose3_inverse_compose, name="inverse_compose", config=config),
        ],
    }


def generate(config: CodegenConfig, output_dir: str = None) -> str:
    """
    Generate the geo package for the given language.

    TODO(hayk): Take scalar_type list here.
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = tempfile.mkdtemp(
            prefix=f"sf_codegen_{type(config).__name__.lower()}_", dir="/tmp"
        )
        logger.debug(f"Creating temp directory: {output_dir}")
    # Subdirectory for everything we'll generate
    package_dir = os.path.join(output_dir, "sym")
    templates = template_util.TemplateList()

    matrix_type_aliases = _matrix_type_aliases()
    custom_generated_methods = _custom_generated_methods(config)

    if isinstance(config, PythonConfig):
        logger.info(f'Creating Python package at: "{package_dir}"')
        template_dir = os.path.join(template_util.PYTHON_TEMPLATE_DIR, "geo_package")

        # Build up templates for each type

        for cls in DEFAULT_GEO_TYPES:
            data = geo_class_common_data(cls, config)
            data["matrix_type_aliases"] = matrix_type_aliases[cls]
            data["custom_generated_methods"] = custom_generated_methods[cls]

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
            dict(Codegen.common_data(), all_types=DEFAULT_GEO_TYPES),
        )

        # Test example
        for name in ("geo_package_python_test.py",):
            templates.add(
                os.path.join(template_dir, "example", name) + ".jinja",
                os.path.join(output_dir, "example", name),
                dict(Codegen.common_data(), all_types=DEFAULT_GEO_TYPES),
            )

    elif isinstance(config, CppConfig):
        # First generate the sym/util package as it's a dependency of the geo package
        from symforce.codegen import sym_util_package_codegen

        sym_util_package_codegen.generate(config, output_dir=output_dir)

        logger.info(f'Creating C++ package at: "{package_dir}"')
        template_dir = os.path.join(template_util.CPP_TEMPLATE_DIR, "geo_package")

        # Build up templates for each type
        for cls in DEFAULT_GEO_TYPES:
            data = geo_class_common_data(cls, config)
            data["matrix_type_aliases"] = matrix_type_aliases[cls]
            data["custom_generated_methods"] = custom_generated_methods[cls]

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
        for name in ("geo_package_cpp_test.cc",):
            templates.add(
                os.path.join(template_dir, "..", "tests", name) + ".jinja",
                os.path.join(output_dir, "tests", name),
                dict(
                    Codegen.common_data(),
                    all_types=DEFAULT_GEO_TYPES,
                    cpp_geo_types=[
                        f"sym::{cls.__name__}<{scalar}>"
                        for cls in DEFAULT_GEO_TYPES
                        for scalar in data["scalar_types"]
                    ],
                    cpp_matrix_types=[
                        f"sym::Vector{i}<{scalar}>"
                        for i in range(1, 10)
                        for scalar in data["scalar_types"]
                    ],
                ),
            )

    else:
        raise NotImplementedError(f'Unknown config type: "{config}"')

    # LCM type_t
    templates.add(
        os.path.join(template_util.LCM_TEMPLATE_DIR, "symforce_types.lcm.jinja"),
        os.path.join(package_dir, "..", "lcmtypes", "lcmtypes", "symforce_types.lcm"),
        {},
    )

    templates.render()

    # Codegen for LCM type_t

    # NOTE(aaron): The lcm-gen syntax for enums is different from skymarshal, so we use skymarshal
    # because this LCM type is also used in ***REMOVED***
    prev_use_skymarshal = codegen_util.USE_SKYMARSHAL
    try:
        codegen_util.USE_SKYMARSHAL = True
        codegen_util.generate_lcm_types(
            os.path.join(package_dir, "..", "lcmtypes", "lcmtypes"), ["symforce_types.lcm"]
        )
    finally:
        codegen_util.USE_SKYMARSHAL = prev_use_skymarshal

    return output_dir
