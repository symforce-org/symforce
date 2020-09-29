import logging
import os
import tempfile

from symforce import logger
from symforce import geo
from symforce import ops
from symforce import python_util
from symforce import types as T
from symforce import sympy as sm
from symforce.test_util import TestCase
from symforce.codegen import CodegenMode
from symforce.codegen import Codegen

SYMFORCE_DIR = os.path.dirname(os.path.dirname(__file__))

TYPES = (geo.Rot2, geo.Rot3, geo.V3)

Element = T.Any


def between_factor(
    a,  # type: Element
    b,  # type: Element
    a_T_b,  # type: Element
    sqrt_info,  # type: geo.Matrix
    epsilon=0,  # type: T.Scalar
):
    # type: (...) -> T.Tuple[geo.Matrix, geo.Matrix]
    """
    Residual that penalizes the difference between(a, b) and a_T_b.

    In vector space terms that would be:
        (b - a) - a_T_b

    In lie group terms:
        local_coordinates(a_T_b, between(a, b))
        to_tangent(compose(inverse(a_T_b), compose(inverse(a), b)))

    Args:
        sqrt_info: Square root information matrix to whiten residual. This can be computed from
                   a covariance matrix as the cholesky decomposition of the inverse. In the case
                   of a diagonal it will contain 1/sigma values. Must match the tangent dim.
    """
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
    value,  # type: Element
    prior,  # type: Element
    sqrt_info,  # type: geo.Matrix
    epsilon=0,  # type: T.Scalar
):
    # type: (...) -> T.Tuple[geo.Matrix, geo.Matrix]
    """
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
    assert type(value) == type(prior)
    assert sqrt_info.rows == sqrt_info.cols == ops.LieGroupOps.tangent_dim(value)

    # Compute error
    tangent_error = ops.LieGroupOps.local_coordinates(prior, value, epsilon=epsilon)

    # Apply noise model
    residual = sqrt_info * geo.M(tangent_error)

    # Compute derivative
    jacobian = residual.jacobian(value)

    return residual, jacobian


def get_function_code(codegen, cleanup=True):
    # type: (Codegen, bool) -> str
    """
    Return just the function code from a Codegen object.
    """
    # Codegen
    data = codegen.generate_function()

    # Read
    filename = "{}.h".format(python_util.camelcase_to_snakecase(codegen.name))
    with open(os.path.join(data["cpp_function_dir"], filename)) as f:
        func_code = f.read()

    # Cleanup
    if cleanup:
        python_util.remove_if_exists(data["output_dir"])

    return func_code


def get_between_factors(types):
    # type: (T.Sequence[T.Type]) -> T.Dict[str, str]
    """
    Compute
    """
    files_dict = {}  # type: T.Dict[str, str]
    for cls in types:
        # Helper to get appropriate filename
        get_filename = lambda cg: python_util.camelcase_to_snakecase(cg.name) + ".h"

        tangent_dim = ops.LieGroupOps.tangent_dim(cls)
        between_codegen = Codegen.function(
            name="BetweenFactor{}".format(cls.__name__),
            func=between_factor,
            input_types=[cls, cls, cls, geo.M(tangent_dim, tangent_dim), sm.Symbol],
            output_names=["res", "jac"],
            mode=CodegenMode.CPP,
        )
        files_dict[get_filename(between_codegen)] = get_function_code(between_codegen)

        prior_codegen = Codegen.function(
            name="PriorFactor{}".format(cls.__name__),
            func=prior_factor,
            input_types=[cls, cls, geo.M(tangent_dim, tangent_dim), sm.Symbol],
            output_names=["res", "jac"],
            mode=CodegenMode.CPP,
        )
        files_dict[get_filename(prior_codegen)] = get_function_code(prior_codegen)

    return files_dict


class SymFactorCodegenTest(TestCase):
    """
    Test generating some derivatives of geo types that are commonly used to make factors.
    """

    def test_factor_codegen(self):
        # type: () -> None
        """
        Prior factors and between factors for C++.
        """
        output_dir = tempfile.mkdtemp(prefix="sf_factor_codegen_test_", dir="/tmp")
        logger.debug("Creating temp directory: {}".format(output_dir))

        try:
            # Compute code
            files_dict = get_between_factors(types=TYPES)

            # Create output dir
            factors_dir = os.path.join(output_dir, "factors")
            os.makedirs(factors_dir)

            # Write out
            for filename, code in files_dict.items():
                with open(os.path.join(factors_dir, filename), "w") as f:
                    f.write(code)

            logger.info("Wrote factors out at: {}".format(factors_dir))

            self.compare_or_update_directory(
                actual_dir=factors_dir,
                expected_dir=os.path.join(SYMFORCE_DIR, "gen", "cpp", "sym", "factors"),
            )

        finally:
            if logger.level != logging.DEBUG:
                python_util.remove_if_exists(output_dir)


if __name__ == "__main__":
    TestCase.main()
