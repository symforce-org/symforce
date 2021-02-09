import logging
import os
import tempfile

import symforce
from symforce import logger
from symforce import geo
from symforce import ops
from symforce import python_util
from symforce import types as T
from symforce import sympy as sm
from symforce.test_util import TestCase, slow_on_sympy
from symforce.codegen import CodegenMode
from symforce.codegen import Codegen
from symforce.values import Values

SYMFORCE_DIR = os.path.dirname(os.path.dirname(__file__))

TYPES = (geo.Rot2, geo.Rot3, geo.V3, geo.Pose2, geo.Pose3)

Element = T.Any


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
    a: Element, b: Element, a_T_b: Element, sqrt_info: geo.Matrix, epsilon: T.Scalar = 0,
) -> T.Tuple[geo.Matrix, geo.Matrix]:
    assert type(a) == type(b) == type(a_T_b)
    assert sqrt_info.rows == sqrt_info.cols == ops.LieGroupOps.tangent_dim(a)

    # Compute error
    tangent_error = ops.LieGroupOps.local_coordinates(
        a_T_b, ops.LieGroupOps.between(a, b), epsilon=epsilon
    )

    # Apply noise model
    residual = sqrt_info * geo.M(tangent_error)

    # Compute derivative
    jacobian = residual.jacobian(a).row_join(residual.jacobian(b))

    return residual, jacobian


def prior_factor(
    value: Element, prior: Element, sqrt_info: geo.Matrix, epsilon: T.Scalar = 0,
) -> T.Tuple[geo.Matrix, geo.Matrix]:
    assert type(value) == type(prior)
    assert sqrt_info.rows == sqrt_info.cols == ops.LieGroupOps.tangent_dim(value)

    # Compute error
    tangent_error = ops.LieGroupOps.local_coordinates(prior, value, epsilon=epsilon)

    # Apply noise model
    residual = sqrt_info * geo.M(tangent_error)

    # Compute derivative
    jacobian = residual.jacobian(value)

    return residual, jacobian


def get_function_code(codegen: Codegen, cleanup: bool = True) -> str:
    """
    Return just the function code from a Codegen object.
    """
    # Codegen
    data = codegen.generate_function()

    # Read
    assert codegen.name is not None
    filename = "{}.h".format(python_util.camelcase_to_snakecase(codegen.name))
    with open(os.path.join(data["cpp_function_dir"], filename)) as f:
        func_code = f.read()

    # Cleanup
    if cleanup:
        python_util.remove_if_exists(data["output_dir"])

    return func_code


def get_filename(codegen: Codegen) -> str:
    """
    Helper to get appropriate filename
    """
    assert codegen.name is not None
    return python_util.camelcase_to_snakecase(codegen.name) + ".h"


def get_between_factors(types: T.Sequence[T.Type]) -> T.Dict[str, str]:
    """
    Compute
    """
    files_dict: T.Dict[str, str] = {}
    for cls in types:
        tangent_dim = ops.LieGroupOps.tangent_dim(cls)
        between_codegen = Codegen.function(
            name=f"BetweenFactor{cls.__name__}",
            func=between_factor,
            input_types=[cls, cls, cls, geo.M(tangent_dim, tangent_dim), sm.Symbol],
            output_names=["res", "jac"],
            mode=CodegenMode.CPP,
            docstring=get_between_factor_docstring("a_T_b"),
        )
        files_dict[get_filename(between_codegen)] = get_function_code(between_codegen)

        prior_codegen = Codegen.function(
            name=f"PriorFactor{cls.__name__}",
            func=prior_factor,
            input_types=[cls, cls, geo.M(tangent_dim, tangent_dim), sm.Symbol],
            output_names=["res", "jac"],
            mode=CodegenMode.CPP,
            docstring=get_prior_docstring(),
        )
        files_dict[get_filename(prior_codegen)] = get_function_code(prior_codegen)

    return files_dict


def get_pose3_extra_factors(files_dict: T.Dict[str, str]) -> None:
    """
    Generates factors specific to Poses which penalize individual components

    This includes factors for only the position or rotation components of a Pose, or for both, but
    with the residual and sqrt information on the product manifold.  This can't be done by
    wrapping the other generated functions because we need jacobians with respect to the full pose,
    and for the product manifold version we want to specify the full covariance with correlations
    between rotation and position.
    """

    def between_factor_pose3_rotation(
        a: geo.Pose3, b: geo.Pose3, a_R_b: geo.Rot3, sqrt_info: geo.Matrix33, epsilon: T.Scalar = 0,
    ) -> T.Tuple[geo.Matrix, geo.Matrix]:
        tangent_error = ops.LieGroupOps.local_coordinates(
            a_R_b, ops.LieGroupOps.between(a, b).R, epsilon=epsilon
        )

        residual = sqrt_info * geo.M(tangent_error)
        jacobian = residual.jacobian(a).row_join(residual.jacobian(b))
        return residual, jacobian

    def between_factor_pose3_position(
        a: geo.Pose3,
        b: geo.Pose3,
        a_t_b: geo.Matrix31,
        sqrt_info: geo.Matrix33,
        epsilon: T.Scalar = 0,
    ) -> T.Tuple[geo.Matrix, geo.Matrix]:
        tangent_error = ops.LieGroupOps.local_coordinates(
            a_t_b, ops.LieGroupOps.between(a, b).t, epsilon=epsilon
        )

        residual = sqrt_info * geo.M(tangent_error)
        jacobian = residual.jacobian(a).row_join(residual.jacobian(b))
        return residual, jacobian

    def between_factor_pose3_product(
        a: geo.Pose3,
        b: geo.Pose3,
        a_T_b: geo.Pose3,
        sqrt_info: geo.Matrix66,
        epsilon: T.Scalar = 0,
    ) -> T.Tuple[geo.Matrix, geo.Matrix]:
        a_T_b_est = ops.LieGroupOps.between(a, b)

        product_a_T_b = Values()
        product_a_T_b["R"] = a_T_b.R
        product_a_T_b["t"] = a_T_b.t

        product_a_T_b_est = Values()
        product_a_T_b_est["R"] = a_T_b_est.R
        product_a_T_b_est["t"] = a_T_b_est.t

        tangent_error = ops.LieGroupOps.local_coordinates(
            product_a_T_b, product_a_T_b_est, epsilon=epsilon
        )

        residual = sqrt_info * geo.M(tangent_error)
        jacobian = residual.jacobian(a).row_join(residual.jacobian(b))
        return residual, jacobian

    def prior_factor_pose3_rotation(
        value: geo.Pose3, prior: geo.Rot3, sqrt_info: geo.Matrix33, epsilon: T.Scalar = 0,
    ) -> T.Tuple[geo.Matrix, geo.Matrix]:
        tangent_error = ops.LieGroupOps.local_coordinates(prior, value.R, epsilon=epsilon)
        residual = sqrt_info * geo.M(tangent_error)
        jacobian = residual.jacobian(value)
        return residual, jacobian

    def prior_factor_pose3_position(
        value: geo.Pose3, prior: geo.Matrix31, sqrt_info: geo.Matrix33, epsilon: T.Scalar = 0,
    ) -> T.Tuple[geo.Matrix, geo.Matrix]:
        tangent_error = ops.LieGroupOps.local_coordinates(prior, value.t, epsilon=epsilon)
        residual = sqrt_info * geo.M(tangent_error)
        jacobian = residual.jacobian(value)
        return residual, jacobian

    def prior_factor_pose3_product(
        value: geo.Pose3, prior: geo.Pose3, sqrt_info: geo.Matrix66, epsilon: T.Scalar = 0,
    ) -> T.Tuple[geo.Matrix, geo.Matrix]:
        product_value = Values()
        product_value["R"] = value.R
        product_value["t"] = value.t

        product_prior = Values()
        product_prior["R"] = prior.R
        product_prior["t"] = prior.t

        tangent_error = ops.LieGroupOps.local_coordinates(
            product_prior, product_value, epsilon=epsilon
        )
        residual = sqrt_info * geo.M(tangent_error)
        jacobian = residual.jacobian(value)
        return residual, jacobian

    between_rotation_codegen = Codegen.function(
        name=f"BetweenFactor{geo.Pose3.__name__}Rotation",
        func=between_factor_pose3_rotation,
        output_names=["res", "jac"],
        mode=CodegenMode.CPP,
        docstring=get_between_factor_docstring("a_R_b"),
    )

    between_position_codegen = Codegen.function(
        name=f"BetweenFactor{geo.Pose3.__name__}Position",
        func=between_factor_pose3_position,
        output_names=["res", "jac"],
        mode=CodegenMode.CPP,
        docstring=get_between_factor_docstring("a_t_b"),
    )

    between_pose_codegen = Codegen.function(
        name=f"BetweenFactor{geo.Pose3.__name__}Product",
        func=between_factor_pose3_product,
        output_names=["res", "jac"],
        mode=CodegenMode.CPP,
        docstring=get_between_factor_docstring("a_T_b"),
    )

    prior_rotation_codegen = Codegen.function(
        name=f"PriorFactor{geo.Pose3.__name__}Rotation",
        func=prior_factor_pose3_rotation,
        output_names=["res", "jac"],
        mode=CodegenMode.CPP,
        docstring=get_prior_docstring(),
    )

    prior_position_codegen = Codegen.function(
        name=f"PriorFactor{geo.Pose3.__name__}Position",
        func=prior_factor_pose3_position,
        output_names=["res", "jac"],
        mode=CodegenMode.CPP,
        docstring=get_prior_docstring(),
    )

    prior_pose_codegen = Codegen.function(
        name=f"PriorFactor{geo.Pose3.__name__}Product",
        func=prior_factor_pose3_product,
        output_names=["res", "jac"],
        mode=CodegenMode.CPP,
        docstring=get_prior_docstring(),
    )

    files_dict[get_filename(between_rotation_codegen)] = get_function_code(between_rotation_codegen)
    files_dict[get_filename(between_position_codegen)] = get_function_code(between_position_codegen)
    files_dict[get_filename(between_pose_codegen)] = get_function_code(between_pose_codegen)
    files_dict[get_filename(prior_rotation_codegen)] = get_function_code(prior_rotation_codegen)
    files_dict[get_filename(prior_position_codegen)] = get_function_code(prior_position_codegen)
    files_dict[get_filename(prior_pose_codegen)] = get_function_code(prior_pose_codegen)


class SymFactorCodegenTest(TestCase):
    """
    Test generating some derivatives of geo types that are commonly used to make factors.
    """

    @slow_on_sympy
    def test_factor_codegen(self) -> None:
        """
        Prior factors and between factors for C++.
        """
        output_dir = tempfile.mkdtemp(prefix="sf_factor_codegen_test_", dir="/tmp")
        logger.debug(f"Creating temp directory: {output_dir}")

        try:
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

            logger.info(f"Wrote factors out at: {factors_dir}")

            # Only test on SymEngine backend
            if symforce.get_backend() == "symengine":
                self.compare_or_update_directory(
                    actual_dir=factors_dir,
                    expected_dir=os.path.join(SYMFORCE_DIR, "gen", "cpp", "sym", "factors"),
                )

        finally:
            if logger.level != logging.DEBUG:
                python_util.remove_if_exists(output_dir)


if __name__ == "__main__":
    TestCase.main()
