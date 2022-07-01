# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from pathlib import Path
import tempfile
import textwrap
import collections
import functools

from symforce import logger
from symforce import python_util
import symforce.symbolic as sf
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
            name="compose",
            input_types=[group, multiplicand_type],
            config=config,
        )

    return {
        sf.Rot2: [
            codegen_mul(sf.Rot2, sf.Vector2),
            Codegen.function(func=sf.Rot2.from_angle, config=config),
            Codegen.function(func=sf.Rot2.to_rotation_matrix, config=config),
        ],
        sf.Rot3: [
            codegen_mul(sf.Rot3, sf.Vector3),
            Codegen.function(func=sf.Rot3.to_rotation_matrix, config=config),
            Codegen.function(
                func=functools.partial(sf.Rot3.random_from_uniform_samples, pi=sf.pi),
                name="random_from_uniform_samples",
                config=config,
            ),
            Codegen.function(func=sf.Rot3.from_yaw_pitch_roll, config=config),
            Codegen.function(
                func=lambda ypr: sf.Rot3.from_yaw_pitch_roll(*ypr),
                input_types=[sf.V3],
                name="from_yaw_pitch_roll",
                config=config,
            ),
        ],
        sf.Pose2: [
            codegen_mul(sf.Pose2, sf.Vector2),
            Codegen.function(func=pose2_inverse_compose, name="inverse_compose", config=config),
        ],
        sf.Pose3: [
            codegen_mul(sf.Pose3, sf.Vector3),
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
    package_dir = Path(output_dir, "sym")
    templates = template_util.TemplateList()

    matrix_type_aliases = _matrix_type_aliases()
    custom_generated_methods = _custom_generated_methods(config)

    if isinstance(config, PythonConfig):
        logger.info(f'Creating Python package at: "{package_dir}"')
        template_dir = config.template_dir()

        # Build up templates for each type

        for cls in DEFAULT_GEO_TYPES:
            data = geo_class_common_data(cls, config)
            data["matrix_type_aliases"] = matrix_type_aliases[cls]
            data["custom_generated_methods"] = custom_generated_methods[cls]

            for base_dir, relative_path in (
                ("geo_package", "CLASS.py"),
                (".", "ops/CLASS/__init__.py"),
                (".", "ops/CLASS/group_ops.py"),
                (".", "ops/CLASS/lie_group_ops.py"),
            ):
                template_path = template_dir / base_dir / (relative_path + ".jinja")
                output_path = package_dir / relative_path.replace("CLASS", cls.__name__.lower())
                templates.add(template_path, output_path, data)

        templates.add(
            template_path=template_dir / "ops" / "__init__.py.jinja",
            output_path=package_dir / "ops" / "__init__.py",
            data={},
        )

        # Package init
        templates.add(
            template_dir / "geo_package" / "__init__.py.jinja",
            package_dir / "__init__.py",
            dict(Codegen.common_data(), all_types=DEFAULT_GEO_TYPES),
        )

        # Test example
        for name in ("geo_package_python_test.py",):
            templates.add(
                template_dir / "tests" / (name + ".jinja"),
                Path(output_dir, "tests", name),
                dict(Codegen.common_data(), all_types=DEFAULT_GEO_TYPES),
            )

    elif isinstance(config, CppConfig):
        # First generate the sym/util package as it's a dependency of the geo package
        from symforce.codegen import sym_util_package_codegen

        sym_util_package_codegen.generate(config, output_dir=output_dir)

        logger.info(f'Creating C++ package at: "{package_dir}"')
        template_dir = config.template_dir()

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
                template_path = template_dir / base_dir / f"{relative_path}.jinja"
                output_path = package_dir / relative_path.replace("CLASS", cls.__name__.lower())
                templates.add(template_path, output_path, data)

        # Render non geo type specific templates
        for template_name in python_util.files_in_dir(
            template_dir / "geo_package" / "ops", relative=True
        ):
            if "CLASS" in template_name:
                continue

            if not template_name.endswith(".jinja"):
                continue

            templates.add(
                template_dir / "geo_package" / "ops" / template_name,
                package_dir / "geo_package" / "ops" / template_name[: -len(".jinja")],
                dict(Codegen.common_data()),
            )

        # Test example
        for name in ("geo_package_cpp_test.cc",):
            templates.add(
                template_dir / "tests" / (name + ".jinja"),
                Path(output_dir, "tests", name),
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
        template_util.LCM_TEMPLATE_DIR / "symforce_types.lcm.jinja",
        package_dir / ".." / "lcmtypes" / "lcmtypes" / "symforce_types.lcm",
        lcm_types_codegen.lcm_symforce_types_data(),
    )

    templates.render()

    # Codegen for LCM type_t
    codegen_util.generate_lcm_types(
        str(package_dir / ".." / "lcmtypes" / "lcmtypes"), ["symforce_types.lcm"]
    )

    return output_dir
