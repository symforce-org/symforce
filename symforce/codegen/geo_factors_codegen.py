# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import os

import symforce.symbolic as sf
from symforce import ops
from symforce import python_util
from symforce import typing as T
from symforce.codegen import Codegen
from symforce.codegen import CppConfig

TYPES = (sf.Rot2, sf.Rot3, sf.V3, sf.Pose2, sf.Pose3)


def get_between_factor_docstring(between_argument_name: str) -> str:
    return """
    Residual that penalizes the difference between between(a, b) and {a_T_b}.

    In vector space terms that would be:
        (b - a) - {a_T_b}

    In lie group terms:
        local_coordinates({a_T_b}, between(a, b))
        to_tangent(compose(inverse({a_T_b}), compose(inverse(a), b)))

    Args:
        sqrt_info: Square root information matrix to whiten residual. This can be computed from
                   a covariance matrix as the cholesky decomposition of the inverse. In the case
                   of a diagonal it will contain 1/sigma values. Must match the tangent dim.
    """.format(
        a_T_b=between_argument_name
    )


def get_prior_docstring() -> str:
    return """
    Residual that penalizes the difference between a value and prior (desired / measured value).

    In vector space terms that would be:
        prior - value

    In lie group terms:
        to_tangent(compose(inverse(value), prior))

    Args:
        sqrt_info: Square root information matrix to whiten residual. This can be computed from
                   a covariance matrix as the cholesky decomposition of the inverse. In the case
                   of a diagonal it will contain 1/sigma values. Must match the tangent dim.
    """


def between_factor(
    a: T.Element, b: T.Element, a_T_b: T.Element, sqrt_info: sf.Matrix, epsilon: sf.Scalar = 0
) -> sf.Matrix:
    assert type(a) == type(b) == type(a_T_b)  # pylint: disable=unidiomatic-typecheck
    assert sqrt_info.rows == sqrt_info.cols == ops.LieGroupOps.tangent_dim(a)

    # Compute error
    tangent_error = ops.LieGroupOps.local_coordinates(
        a_T_b, ops.LieGroupOps.between(a, b), epsilon=epsilon
    )

    # Apply noise model
    residual = sqrt_info * sf.M(tangent_error)

    return residual


def prior_factor(
    value: T.Element, prior: T.Element, sqrt_info: sf.Matrix, epsilon: sf.Scalar = 0
) -> sf.Matrix:
    assert type(value) == type(prior)  # pylint: disable=unidiomatic-typecheck
    assert sqrt_info.rows == sqrt_info.cols == ops.LieGroupOps.tangent_dim(value)

    # Compute error
    tangent_error = ops.LieGroupOps.local_coordinates(prior, value, epsilon=epsilon)

    # Apply noise model
    residual = sqrt_info * sf.M(tangent_error)

    return residual


def get_function_code(codegen: Codegen, cleanup: bool = True) -> str:
    """
    Return just the function code from a Codegen object.
    """
    # Codegen
    data = codegen.generate_function()

    # Read
    assert codegen.name is not None
    filename = "{}.h".format(codegen.name)
    with open(os.path.join(data.function_dir, filename)) as f:
        func_code = f.read()

    # Cleanup
    if cleanup:
        python_util.remove_if_exists(data.output_dir)

    return func_code


def get_filename(codegen: Codegen) -> str:
    """
    Helper to get appropriate filename
    """
    assert codegen.name is not None
    return codegen.name + ".h"


def get_between_factors(types: T.Sequence[T.Type]) -> T.Dict[str, str]:
    """
    Compute
    """
    files_dict: T.Dict[str, str] = {}
    for cls in types:
        tangent_dim = ops.LieGroupOps.tangent_dim(cls)
        between_codegen = Codegen.function(
            func=between_factor,
            input_types=[cls, cls, cls, sf.M(tangent_dim, tangent_dim), sf.Symbol],
            output_names=["res"],
            config=CppConfig(),
            docstring=get_between_factor_docstring("a_T_b"),
        ).with_linearization(name=f"between_factor_{cls.__name__.lower()}", which_args=["a", "b"])
        files_dict[get_filename(between_codegen)] = get_function_code(between_codegen)

        prior_codegen = Codegen.function(
            func=prior_factor,
            input_types=[cls, cls, sf.M(tangent_dim, tangent_dim), sf.Symbol],
            output_names=["res"],
            config=CppConfig(),
            docstring=get_prior_docstring(),
        ).with_linearization(name=f"prior_factor_{cls.__name__.lower()}", which_args=["value"])
        files_dict[get_filename(prior_codegen)] = get_function_code(prior_codegen)

    return files_dict


def get_pose3_extra_factors(files_dict: T.Dict[str, str]) -> None:
    """
    Generates factors specific to Poses which penalize individual components

    This includes factors for only the position or rotation components of a Pose.  This can't be
    done by wrapping the other generated functions because we need jacobians with respect to the
    full pose.
    """

    def between_factor_pose3_rotation(
        a: sf.Pose3, b: sf.Pose3, a_R_b: sf.Rot3, sqrt_info: sf.Matrix33, epsilon: sf.Scalar = 0
    ) -> sf.Matrix:
        # NOTE(aaron): This should be equivalent to between_factor(a.R, b.R, a_R_b), but we write it
        # this way for explicitness and symmetry with between_factor_pose3_position, where the two
        # are not equivalent
        tangent_error = ops.LieGroupOps.local_coordinates(
            a_R_b, ops.LieGroupOps.between(a, b).R, epsilon=epsilon
        )

        return sqrt_info * sf.M(tangent_error)

    def between_factor_pose3_position(
        a: sf.Pose3,
        b: sf.Pose3,
        a_t_b: sf.Vector3,
        sqrt_info: sf.Matrix33,
        epsilon: sf.Scalar = 0,
    ) -> sf.Matrix:
        # NOTE(aaron): This is NOT the same as between_factor(a.t, b.t, a_t_b, sqrt_info, epsilon)
        # between_factor(a.t, b.t, a_t_b) would be penalizing the difference in the global frame
        # (and expecting a_t_b to be in the global frame), we want to penalize the position
        # component of between_factor(a, b, a_T_b), which is in the `a` frame
        tangent_error = ops.LieGroupOps.local_coordinates(
            a_t_b, ops.LieGroupOps.between(a, b).t, epsilon=epsilon
        )

        return sqrt_info * sf.M(tangent_error)

    def between_factor_pose3_translation_norm(
        a: sf.Pose3,
        b: sf.Pose3,
        translation_norm: sf.Scalar,
        sqrt_info: sf.Matrix11,
        epsilon: sf.Scalar = 0,
    ) -> sf.Matrix:
        """
        Residual that penalizes the difference between translation_norm and (a.t - b.t).norm().

        Args:
            sqrt_info: Square root information matrix to whiten residual. In this one dimensional case
                    this is just 1/sigma.
        """
        error = translation_norm - (a.t - b.t).norm(epsilon)
        return sqrt_info * sf.M([error])

    def prior_factor_pose3_rotation(
        value: sf.Pose3, prior: sf.Rot3, sqrt_info: sf.Matrix33, epsilon: sf.Scalar = 0
    ) -> sf.Matrix:
        return prior_factor(value.R, prior, sqrt_info, epsilon)

    def prior_factor_pose3_position(
        value: sf.Pose3, prior: sf.Vector3, sqrt_info: sf.Matrix33, epsilon: sf.Scalar = 0
    ) -> sf.Matrix:
        return prior_factor(value.t, prior, sqrt_info, epsilon)

    between_rotation_codegen = Codegen.function(
        func=between_factor_pose3_rotation,
        output_names=["res"],
        config=CppConfig(),
        docstring=get_between_factor_docstring("a_R_b"),
    ).with_linearization(name="between_factor_pose3_rotation", which_args=["a", "b"])

    between_position_codegen = Codegen.function(
        func=between_factor_pose3_position,
        output_names=["res"],
        config=CppConfig(),
        docstring=get_between_factor_docstring("a_t_b"),
    ).with_linearization(name="between_factor_pose3_position", which_args=["a", "b"])

    between_translation_norm_codegen = Codegen.function(
        func=between_factor_pose3_translation_norm, output_names=["res"], config=CppConfig()
    ).with_linearization(name="between_factor_pose3_translation_norm", which_args=["a", "b"])

    prior_rotation_codegen = Codegen.function(
        func=prior_factor_pose3_rotation,
        output_names=["res"],
        config=CppConfig(),
        docstring=get_prior_docstring(),
    ).with_linearization(name="prior_factor_pose3_rotation", which_args=["value"])

    prior_position_codegen = Codegen.function(
        func=prior_factor_pose3_position,
        output_names=["res"],
        config=CppConfig(),
        docstring=get_prior_docstring(),
    ).with_linearization(name="prior_factor_pose3_position", which_args=["value"])

    files_dict[get_filename(between_rotation_codegen)] = get_function_code(between_rotation_codegen)
    files_dict[get_filename(between_position_codegen)] = get_function_code(between_position_codegen)
    files_dict[get_filename(between_translation_norm_codegen)] = get_function_code(
        between_translation_norm_codegen
    )
    files_dict[get_filename(prior_rotation_codegen)] = get_function_code(prior_rotation_codegen)
    files_dict[get_filename(prior_position_codegen)] = get_function_code(prior_position_codegen)


def generate(output_dir: str) -> None:
    """
    Prior factors and between factors for C++.
    """
    # Compute code
    files_dict = get_between_factors(types=TYPES)
    get_pose3_extra_factors(files_dict)

    # Create output dir
    factors_dir = os.path.join(output_dir, "factors")
    os.makedirs(factors_dir)

    # Write out
    for filename, code in files_dict.items():
        with open(os.path.join(factors_dir, filename), "w") as f:
            f.write(code)
