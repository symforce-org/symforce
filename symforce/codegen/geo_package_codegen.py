# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import collections
import functools
import tempfile
import textwrap
from pathlib import Path

import symforce.symbolic as sf
from symforce import logger
from symforce import python_util
from symforce import typing as T
from symforce.codegen import Codegen
from symforce.codegen import CodegenConfig
from symforce.codegen import CppConfig
from symforce.codegen import PythonConfig
from symforce.codegen import codegen_util
from symforce.codegen import lcm_types_codegen
from symforce.codegen import template_util
from symforce.codegen.ops_codegen_util import make_group_ops_funcs
from symforce.codegen.ops_codegen_util import make_lie_group_ops_funcs

# Default geo types to generate
DEFAULT_GEO_TYPES = (sf.Rot2, sf.Pose2, sf.Rot3, sf.Pose3)


def geo_class_common_data(cls: T.Type, config: CodegenConfig) -> T.Dict[str, T.Any]:
    """
    Data for template generation of this class. Contains all useful info common
    to all class-specific templates.
    """
    data = Codegen.common_data()
    data["cls"] = cls

    data["specs"] = collections.defaultdict(list)

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
        sf.Rot2: {"Eigen::Matrix<Scalar, 2, 1>": "Vector2"},
        sf.Rot3: {"Eigen::Matrix<Scalar, 3, 1>": "Vector3"},
        sf.Pose2: {"Eigen::Matrix<Scalar, 2, 1>": "Vector2"},
        sf.Pose3: {"Eigen::Matrix<Scalar, 3, 1>": "Vector3"},
    }


def _custom_generated_methods(config: CodegenConfig) -> T.Dict[T.Type, T.List[Codegen]]:
    """
    Returns a dictionary d where d[datatype] is a list of codegened functions
    we wish to be added to type datatype's generated code.

    Args:
        config (CodegenConfig): Specifies the target language of the codegened functions.
    """

    def pose2_inverse_compose(self: sf.Pose2, point: sf.Vector2) -> sf.Vector2:
        return self.inverse() * point

    def pose3_inverse_compose(self: sf.Pose3, point: sf.Vector3) -> sf.Vector3:
        return self.inverse() * point

    def codegen_mul(group: T.Type, multiplicand_type: T.Type) -> Codegen:
        """
        A helper to generate a Codegen object for groups with the method __mul__ taking
        an instance of group and composing it with an instance of multiplicand_type
        """
        return Codegen.function(
            func=group.__mul__,
            name="compose_with_point",
            input_types=[group, multiplicand_type],
            config=config,
        )

    rot3_functions = [
        codegen_mul(sf.Rot3, sf.Vector3),
        Codegen.function(func=sf.Rot3.to_rotation_matrix, config=config),
        Codegen.function(
            func=functools.partial(sf.Rot3.random_from_uniform_samples, pi=sf.pi),
            name="random_from_uniform_samples",
            config=config,
        ),
        Codegen.function(
            # TODO(aaron): We currently can't generate custom methods with defaults - fix this, and
            # pass epsilon as an argument with a default
            func=lambda self: sf.V3(self.to_yaw_pitch_roll(epsilon=0)),
            input_types=[sf.Rot3],
            name="to_yaw_pitch_roll",
            config=config,
        ),
        Codegen.function(func=sf.Rot3.from_yaw_pitch_roll, config=config),
    ]

    # TODO(brad): We don't currently generate this in python because python (unlike C++)
    # has no function overloading, and we already generate a from_yaw_pitch_roll which
    # instead takes yaw, pitch, and roll as seperate arguments. Figure out how to allow
    # this overload to better achieve parity between C++ and python.
    if isinstance(config, PythonConfig):
        pass
        # rot3_functions.insert(2, Codegen.function(func=sf.Rot3.from_rotation_matrix, config=config))
    else:
        rot3_functions.append(
            Codegen.function(
                func=lambda ypr: sf.Rot3.from_yaw_pitch_roll(*ypr),
                input_types=[sf.V3],
                name="from_yaw_pitch_roll",
                config=config,
            ),
        )

    def pose_getter_methods(pose_type: T.Type) -> T.List[Codegen]:
        def rotation(self: T.Any) -> T.Any:
            """
            Returns the rotational component of this pose.
            """
            return self.R

        def position(self: T.Any) -> T.Any:
            """
            Returns the positional component of this pose.
            """
            return self.t

        return [
            Codegen.function(func=rotation, input_types=[pose_type], config=config),
            Codegen.function(func=position, input_types=[pose_type], config=config),
        ]

    return {
        sf.Rot2: [
            codegen_mul(sf.Rot2, sf.Vector2),
            Codegen.function(func=sf.Rot2.from_angle, config=config),
            Codegen.function(func=sf.Rot2.to_rotation_matrix, config=config),
        ],
        sf.Rot3: rot3_functions,
        sf.Pose2: pose_getter_methods(sf.Pose2)
        + [
            codegen_mul(sf.Pose2, sf.Vector2),
            Codegen.function(func=pose2_inverse_compose, name="inverse_compose", config=config),
            Codegen.function(func=sf.Pose2.to_homogenous_matrix, config=config),
        ],
        sf.Pose3: pose_getter_methods(sf.Pose3)
        + [
            codegen_mul(sf.Pose3, sf.Vector3),
            Codegen.function(func=pose3_inverse_compose, name="inverse_compose", config=config),
            Codegen.function(func=sf.Pose3.to_homogenous_matrix, config=config),
        ],
    }


def generate(config: CodegenConfig, output_dir: Path = None) -> Path:
    """
    Generate the geo package for the given language.

    TODO(hayk): Take scalar_type list here.
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = Path(
            tempfile.mkdtemp(prefix=f"sf_codegen_{type(config).__name__.lower()}_", dir="/tmp")
        )
        logger.debug(f"Creating temp directory: {output_dir}")
    # Subdirectory for everything we'll generate
    package_dir = output_dir / "sym"
    template_dir = config.template_dir()
    templates = template_util.TemplateList(template_dir)

    matrix_type_aliases = _matrix_type_aliases()
    custom_generated_methods = _custom_generated_methods(config)

    if isinstance(config, PythonConfig):
        logger.debug(f'Creating Python package at: "{package_dir}"')

        # Build up templates for each type

        for cls in DEFAULT_GEO_TYPES:
            data = geo_class_common_data(cls, config)
            data["matrix_type_aliases"] = matrix_type_aliases[cls]
            data["custom_generated_methods"] = custom_generated_methods[cls]
            if cls == sf.Pose2:
                data["imported_classes"] = [sf.Rot2]
            elif cls == sf.Pose3:
                data["imported_classes"] = [sf.Rot3]

            for base_dir, relative_path in (
                ("geo_package", "CLASS.py"),
                (".", "ops/CLASS/__init__.py"),
                (".", "ops/CLASS/group_ops.py"),
                (".", "ops/CLASS/lie_group_ops.py"),
            ):
                template_path = Path(base_dir, relative_path + ".jinja")
                output_path = package_dir / relative_path.replace("CLASS", cls.__name__.lower())
                templates.add(template_path, data, output_path=output_path)

        templates.add(
            template_path=Path("ops", "__init__.py.jinja"),
            output_path=package_dir / "ops" / "__init__.py",
            data={},
        )

        # Package init
        if config.namespace_package:
            templates.add(
                template_path=Path("function", "namespace_init.py.jinja"),
                data=dict(pkg_namespace="sym"),
                output_path=package_dir / "__init__.py",
            )
        templates.add(
            template_path=Path("geo_package", "__init__.py.jinja"),
            data=dict(
                Codegen.common_data(),
                all_types=DEFAULT_GEO_TYPES,
                numeric_epsilon=sf.numeric_epsilon,
            ),
            output_path=package_dir / ("_init.py" if config.namespace_package else "__init__.py"),
        )

        # Test example
        for name in ("geo_package_python_test.py",):
            templates.add(
                template_path=Path("tests", name + ".jinja"),
                data=dict(Codegen.common_data(), all_types=DEFAULT_GEO_TYPES),
                output_path=output_dir / "tests" / name,
            )

    elif isinstance(config, CppConfig):
        # First generate the sym/util package as it's a dependency of the geo package
        from symforce.codegen import sym_util_package_codegen

        sym_util_package_codegen.generate(config, output_dir=output_dir)

        logger.debug(f'Creating C++ package at: "{package_dir}"')

        # Build up templates for each type
        for cls in DEFAULT_GEO_TYPES:
            data = geo_class_common_data(cls, config)
            data["matrix_type_aliases"] = matrix_type_aliases[cls]
            data["custom_generated_methods"] = custom_generated_methods[cls]

            for base_dir, relative_path in (
                ("geo_package", "CLASS.h"),
                ("geo_package", "CLASS.cc"),
                (".", "ops/CLASS/storage_ops.h"),
                (".", "ops/CLASS/storage_ops.cc"),
                (".", "ops/CLASS/group_ops.h"),
                (".", "ops/CLASS/group_ops.cc"),
                (".", "ops/CLASS/lie_group_ops.h"),
                (".", "ops/CLASS/lie_group_ops.cc"),
            ):
                template_path = Path(base_dir, f"{relative_path}.jinja")
                output_path = package_dir / relative_path.replace("CLASS", cls.__name__.lower())
                templates.add(template_path, data, output_path=output_path)

        # Render non geo type specific templates
        for template_name in python_util.files_in_dir(
            template_dir / "geo_package" / "ops", relative=True
        ):
            if "CLASS" in template_name:
                continue

            if not template_name.endswith(".jinja"):
                continue

            templates.add(
                template_path=Path("geo_package", "ops", template_name),
                data=dict(Codegen.common_data()),
                output_path=package_dir / "ops" / template_name[: -len(".jinja")],
            )

        # Test example
        for name in ("geo_package_cpp_test.cc",):
            templates.add(
                template_path=Path("tests", name + ".jinja"),
                output_path=output_dir / "tests" / name,
                data=dict(
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
        template_path="symforce_types.lcm.jinja",
        data=lcm_types_codegen.lcm_symforce_types_data(),
        template_dir=template_util.LCM_TEMPLATE_DIR,
        output_path=package_dir / ".." / "lcmtypes" / "lcmtypes" / "symforce_types.lcm",
    )

    templates.render()

    # Codegen for LCM type_t
    codegen_util.generate_lcm_types(
        package_dir / ".." / "lcmtypes" / "lcmtypes", ["symforce_types.lcm"]
    )

    return output_dir
