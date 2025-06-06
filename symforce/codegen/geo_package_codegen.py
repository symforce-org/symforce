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
from symforce.codegen.ops_codegen_util import make_manifold_ops_funcs


def geo_class_common_data(cls: T.Type, config: CodegenConfig) -> T.Dict[str, T.Any]:
    """
    Data for template generation of this class. Contains all useful info common
    to all class-specific templates.
    """
    data = Codegen.common_data()
    data["cls"] = cls

    data["specs"] = collections.defaultdict(list)

    # Group + Lie Group Functions
    if cls in sf.GROUP_GEO_TYPES:
        data["is_group"] = True
        data["is_lie_group"] = True
        for func in make_group_ops_funcs(cls, config):
            data["specs"]["GroupOps"].append(func)
        for func in make_lie_group_ops_funcs(cls, config):
            data["specs"]["LieGroupOps"].append(func)
    else:
        data["is_group"] = False
        data["is_lie_group"] = False

    # Manifold Functions
    # Note(chet): Currently all geo types are valid manifolds.
    data["is_manifold"] = True
    for func in make_manifold_ops_funcs(cls, config):
        data["specs"]["LieGroupOps"].append(func)

    data["doc"] = textwrap.dedent(cls.__doc__).strip() if cls.__doc__ else ""

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
        sf.Unit3: {"Eigen::Matrix<Scalar, 3, 1>": "Vector3"},
    }


def _custom_generated_methods(config: CodegenConfig) -> T.Dict[T.Type, T.List[Codegen]]:
    """
    Returns a dictionary d where d[datatype] is a list of codegened functions
    we wish to be added to type datatype's generated code.

    Args:
        config (CodegenConfig): Specifies the target language of the codegened functions.
    """

    def inverse_compose(self: T.Any, point: sf.Matrix) -> sf.Matrix:
        """
        Returns ``self.inverse() * point``

        This is more efficient than calling the generated inverse and compose methods separately, if
        doing this for one point.
        """
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

    def to_yaw_pitch_roll(self: sf.Rot3) -> sf.V3:
        return sf.V3(self.to_yaw_pitch_roll(epsilon=0))

    to_yaw_pitch_roll.__doc__ = sf.Rot3.to_yaw_pitch_roll.__doc__

    def from_yaw(yaw: T.Scalar) -> sf.Rot3:
        """Construct from yaw angle in radians"""
        return sf.Rot3.from_yaw_pitch_roll(yaw=yaw)

    def from_pitch(pitch: T.Scalar) -> sf.Rot3:
        """Construct from pitch angle in radians"""
        return sf.Rot3.from_yaw_pitch_roll(pitch=pitch)

    def from_roll(roll: T.Scalar) -> sf.Rot3:
        """Construct from roll angle in radians"""
        return sf.Rot3.from_yaw_pitch_roll(roll=roll)

    rot3_functions = (
        [
            codegen_mul(sf.Rot3, sf.Vector3),
            Codegen.function(func=sf.Rot3.to_tangent_norm, config=config),
            Codegen.function(func=sf.Rot3.to_rotation_matrix, config=config),
            Codegen.function(
                func=functools.partial(sf.Rot3.random_from_uniform_samples, pi=sf.pi), config=config
            ),
            Codegen.function(
                # TODO(aaron): We currently can't generate custom methods with defaults - fix this, and
                # pass epsilon as an argument with a default
                func=to_yaw_pitch_roll,
                config=config,
            ),
            Codegen.function(func=sf.Rot3.from_yaw_pitch_roll, config=config),
            Codegen.function(func=from_yaw, config=config),
            Codegen.function(func=from_pitch, config=config),
            Codegen.function(func=from_roll, config=config),
        ]
        + (
            # TODO(brad): We don't currently generate this in python because python (unlike C++)
            # has no function overloading, and we already generate a from_yaw_pitch_roll which
            # instead takes yaw, pitch, and roll as seperate arguments. Figure out how to allow
            # this overload to better achieve parity between C++ and python.
            [
                Codegen.function(
                    func=lambda ypr: sf.Rot3.from_yaw_pitch_roll(*ypr),
                    input_types=[sf.V3],
                    name="from_yaw_pitch_roll",
                    config=config,
                )
            ]
            if isinstance(config, CppConfig)
            else []
        )
        + (
            # In C++, we do this with Eigen
            [Codegen.function(func=sf.Rot3.from_angle_axis, config=config)]
            if isinstance(config, PythonConfig)
            else []
        )
        + [
            Codegen.function(func=sf.Rot3.from_two_unit_vectors, config=config),
        ]
    )

    def pose_getter_methods(pose_type: T.Type) -> T.List[Codegen]:
        def rotation_storage(self: T.Any) -> T.Any:
            """
            Returns the rotational component of this pose.
            """
            return sf.Matrix(self.R.to_storage())

        def position(self: T.Any) -> T.Any:
            """
            Returns the positional component of this pose.
            """
            return self.t

        return [
            Codegen.function(func=rotation_storage, input_types=[pose_type], config=config),
            Codegen.function(func=position, input_types=[pose_type], config=config),
        ]

    unit3_functions = [
        Codegen.function(func=sf.Unit3.basis, config=config),
        Codegen.function(func=sf.Unit3.to_unit_vector, config=config),
        Codegen.function(func=sf.Unit3.random_from_uniform_samples, config=config),
    ]
    # TODO(chet): because the generated C++ config has normalization default option for class
    # initialization, these functions are custom written. For Python config, the normalization
    # default does not exist and so generating these functions automatically is not-redundant.
    if isinstance(config, PythonConfig):
        unit3_functions += [
            Codegen.function(func=sf.Unit3.from_vector, config=config),
            Codegen.function(func=sf.Unit3.from_unit_vector, config=config),
        ]

    return {
        sf.Rot2: [
            codegen_mul(sf.Rot2, sf.Vector2),
            Codegen.function(func=sf.Rot2.from_angle, config=config),
            Codegen.function(func=sf.Rot2.to_rotation_matrix, config=config),
            Codegen.function(func=sf.Rot2.from_rotation_matrix, config=config),
            Codegen.function(
                func=functools.partial(sf.Rot2.random_from_uniform_sample, pi=sf.pi), config=config
            ),
        ],
        sf.Rot3: rot3_functions,
        sf.Pose2: pose_getter_methods(sf.Pose2)
        + [
            codegen_mul(sf.Pose2, sf.Vector2),
            Codegen.function(
                func=inverse_compose, config=config, input_types=[sf.Pose2, sf.Vector2]
            ),
            Codegen.function(func=sf.Pose2.to_homogenous_matrix, config=config),
        ],
        sf.Pose3: pose_getter_methods(sf.Pose3)
        + [
            codegen_mul(sf.Pose3, sf.Vector3),
            Codegen.function(
                func=inverse_compose, config=config, input_types=[sf.Pose3, sf.Vector3]
            ),
            Codegen.function(func=sf.Pose3.to_homogenous_matrix, config=config),
        ],
        sf.Unit3: unit3_functions,
    }


def generate(config: CodegenConfig, output_dir: T.Optional[Path] = None) -> Path:
    """
    Generate the geo package for the given language.
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

        for cls in sf.GEO_TYPES:
            data = geo_class_common_data(cls, config)
            data["matrix_type_aliases"] = matrix_type_aliases.get(cls, {})
            data["custom_generated_methods"] = custom_generated_methods.get(cls, {})
            if cls == sf.Pose2:
                data["imported_classes"] = [sf.Rot2]
            elif cls in {sf.Pose3}:
                data["imported_classes"] = [sf.Rot3]

            for base_dir, relative_path in (
                ("geo_package", "CLASS.py"),
                (".", "ops/CLASS/__init__.py"),
                (".", "ops/CLASS/group_ops.py"),
                (".", "ops/CLASS/lie_group_ops.py"),
            ):
                template_path = Path(base_dir, relative_path + ".jinja")
                output_path = package_dir / relative_path.replace("CLASS", cls.__name__.lower())
                templates.add(
                    template_path, data, config.render_template_config, output_path=output_path
                )

        templates.add(
            template_path=Path("ops", "__init__.py.jinja"),
            output_path=package_dir / "ops" / "__init__.py",
            data={},
            config=config.render_template_config,
        )

        # Package init
        templates.add(
            template_path=Path("geo_package", "__init__.py.jinja"),
            data=dict(
                Codegen.common_data(),
                all_types=sf.GEO_TYPES,
                numeric_epsilon=sf.numeric_epsilon,
            ),
            config=config.render_template_config,
            output_path=package_dir / "__init__.py",
        )

        # Test example
        for name in ("geo_package_python_test.py",):
            templates.add(
                template_path=Path("tests", name + ".jinja"),
                data=dict(
                    Codegen.common_data(),
                    all_types=sf.GEO_TYPES,
                    group_geo_types=sf.GROUP_GEO_TYPES,
                ),
                config=config.render_template_config,
                output_path=output_dir / "tests" / name,
            )

    elif isinstance(config, CppConfig):
        # First generate the sym/util package as it's a dependency of the geo package
        from symforce.codegen import sym_util_package_codegen

        sym_util_package_codegen.generate(config, output_dir=output_dir)

        logger.debug(f'Creating C++ package at: "{package_dir}"')

        # Build up templates for each type
        for cls in sf.GEO_TYPES:
            data = geo_class_common_data(cls, config)
            data["matrix_type_aliases"] = matrix_type_aliases.get(cls, {})
            data["custom_generated_methods"] = custom_generated_methods.get(cls, {})

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
                templates.add(
                    template_path, data, config.render_template_config, output_path=output_path
                )

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
                config=config.render_template_config,
                output_path=package_dir / "ops" / template_name[: -len(".jinja")],
            )

        # Test example
        for name in ("geo_package_cpp_test.cc",):
            templates.add(
                template_path=Path("tests", name + ".jinja"),
                output_path=output_dir / "tests" / name,
                data=dict(
                    Codegen.common_data(),
                    all_types=sf.GEO_TYPES,
                    cpp_geo_types=[
                        f"sym::{cls.__name__}<{scalar}>"
                        for cls in sf.GEO_TYPES
                        for scalar in data["scalar_types"]
                    ],
                    cpp_group_geo_types=[
                        f"sym::{cls.__name__}<{scalar}>"
                        for cls in sf.GROUP_GEO_TYPES
                        for scalar in data["scalar_types"]
                    ],
                    cpp_matrix_types=[
                        f"sym::Vector{i}<{scalar}>"
                        for i in range(1, 10)
                        for scalar in data["scalar_types"]
                    ],
                ),
                config=config.render_template_config,
            )

        templates.add(
            template_path=Path("geo_package/all_geo_types.h.jinja"),
            data=Codegen.common_data(),
            config=config.render_template_config,
            output_path=package_dir / "all_geo_types.h",
        )
    else:
        raise NotImplementedError(f'Unknown config type: "{config}"')

    # LCM type_t
    templates.add(
        template_path="symforce_types.lcm.jinja",
        data=lcm_types_codegen.lcm_symforce_types_data(),
        config=config.render_template_config,
        template_dir=template_util.LCM_TEMPLATE_DIR,
        output_path=package_dir / ".." / "lcmtypes" / "lcmtypes" / "symforce_types.lcm",
    )

    templates.render()

    # Codegen for LCM type_t
    codegen_util.generate_lcm_types(
        package_dir / ".." / "lcmtypes" / "lcmtypes", ["symforce_types.lcm"]
    )

    return output_dir
