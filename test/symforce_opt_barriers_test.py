# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from symforce import util
from symforce.opt.barrier_functions import max_linear_barrier
from symforce.opt.barrier_functions import max_power_barrier
from symforce.opt.barrier_functions import min_linear_barrier
from symforce.opt.barrier_functions import min_max_centering_power_barrier
from symforce.opt.barrier_functions import min_max_linear_barrier
from symforce.opt.barrier_functions import min_max_power_barrier
from symforce.opt.barrier_functions import min_power_barrier
from symforce.opt.barrier_functions import symmetric_power_barrier
from symforce.test_util import TestCase


class SymforceOptBarriersTest(TestCase):
    def test_symmetric_power_barrier(self) -> None:
        """
        Tests symmetric_power_barrier at key points for different powers.
        """
        dist_zero_to_nominal = 0.5
        x_nominal = 2.5
        error_nominal = 1.5
        powers = [0.5, 1, 2]
        for power in powers:
            with self.subTest(power=power):
                # Check center
                center_error = symmetric_power_barrier(
                    0, x_nominal, error_nominal, dist_zero_to_nominal, power
                )
                self.assertEqual(center_error, 0.0)

                # Check corners
                x_left_corner = -x_nominal + dist_zero_to_nominal
                x_right_corner = -x_left_corner
                left_corner_error = symmetric_power_barrier(
                    x_left_corner, x_nominal, error_nominal, dist_zero_to_nominal, power
                )
                right_corner_error = symmetric_power_barrier(
                    x_right_corner, x_nominal, error_nominal, dist_zero_to_nominal, power
                )
                self.assertEqual(left_corner_error, 0.0)
                self.assertEqual(right_corner_error, 0.0)

                # Check nominal point
                left_nominal_error = symmetric_power_barrier(
                    -x_nominal, x_nominal, error_nominal, dist_zero_to_nominal, power
                )
                right_nominal_error = symmetric_power_barrier(
                    x_nominal, x_nominal, error_nominal, dist_zero_to_nominal, power
                )
                self.assertEqual(left_nominal_error, error_nominal)
                self.assertEqual(right_nominal_error, error_nominal)

                # Check curve shape
                x = x_nominal + 1
                x_zero_error = x_nominal - dist_zero_to_nominal
                expected_error = (
                    error_nominal * ((x - x_zero_error) / (x_nominal - x_zero_error)) ** power
                )
                error = symmetric_power_barrier(
                    x, x_nominal, error_nominal, dist_zero_to_nominal, power
                )
                self.assertEqual(error, expected_error)

    def test_min_max_power_barriers(self) -> None:
        """
        Tests min_max_power_barrier, max_power_barrier, min_power_barrier, and their linear
        equivalents (with power=1)
        """
        x_nominal_lower = -20.0
        x_nominal_upper = -10.0
        error_nominal = 1.5
        dist_zero_to_nominal = 0.5

        min_max_power_barrier_helper = lambda x, power: min_max_power_barrier(
            x, x_nominal_lower, x_nominal_upper, error_nominal, dist_zero_to_nominal, power
        )
        max_power_barrier_helper = lambda x, power: max_power_barrier(
            x, x_nominal_upper, error_nominal, dist_zero_to_nominal, power
        )
        min_power_barrier_helper = lambda x, power: min_power_barrier(
            x, x_nominal_lower, error_nominal, dist_zero_to_nominal, power
        )

        powers = [0.5, 1, 2]
        for power in powers:
            with self.subTest(power=power):
                # Check center
                center = (x_nominal_lower + x_nominal_upper) / 2
                self.assertEqual(min_max_power_barrier_helper(center, power), 0.0)
                self.assertEqual(max_power_barrier_helper(center, power), 0.0)
                self.assertEqual(min_power_barrier_helper(center, power), 0.0)

                # Check corners
                left_corner = x_nominal_lower + dist_zero_to_nominal
                right_corner = x_nominal_upper - dist_zero_to_nominal
                self.assertEqual(min_max_power_barrier_helper(left_corner, power), 0.0)
                self.assertEqual(min_max_power_barrier_helper(right_corner, power), 0.0)
                self.assertEqual(max_power_barrier_helper(right_corner, power), 0.0)
                self.assertEqual(min_power_barrier_helper(left_corner, power), 0.0)

                # Check nominal point
                self.assertEqual(
                    min_max_power_barrier_helper(x_nominal_lower, power), error_nominal
                )
                self.assertEqual(
                    min_max_power_barrier_helper(x_nominal_upper, power), error_nominal
                )
                self.assertEqual(max_power_barrier_helper(x_nominal_upper, power), error_nominal)
                self.assertEqual(min_power_barrier_helper(x_nominal_lower, power), error_nominal)

                # Check curve shape
                x = x_nominal_upper + 1.0
                expected_error = (
                    error_nominal * ((x - right_corner) / dist_zero_to_nominal) ** power
                )
                self.assertEqual(min_max_power_barrier_helper(x, power), expected_error)
                self.assertEqual(max_power_barrier_helper(x, power), expected_error)
                self.assertEqual(
                    min_power_barrier_helper(x_nominal_lower - 1, power), expected_error
                )

        # Check that min_max_linear_barrier returns the same error for several points

        min_max_linear_barrier_helper = lambda x: min_max_linear_barrier(
            x, x_nominal_lower, x_nominal_upper, error_nominal, dist_zero_to_nominal
        )
        max_linear_barrier_helper = lambda x: max_linear_barrier(
            x, x_nominal_upper, error_nominal, dist_zero_to_nominal
        )
        min_linear_barrier_helper = lambda x: min_linear_barrier(
            x, x_nominal_lower, error_nominal, dist_zero_to_nominal
        )

        barrier_map = (
            (min_max_power_barrier_helper, min_max_linear_barrier_helper),
            (max_power_barrier_helper, max_linear_barrier_helper),
            (min_power_barrier_helper, min_linear_barrier_helper),
        )
        for power_barrier, linear_barrier in barrier_map:
            with self.subTest(power_barrier=power_barrier):
                for x in [-42.0, -20.0, -19.5, -15.0, -10.5, -10.0, 42.0]:
                    expected_error = power_barrier(x, 1.0)
                    error = linear_barrier(x)
                    self.assertEqual(error, expected_error)

    def test_centering_barrier(self) -> None:
        """
        Tests min_max_centering_power_barrier
        """
        x_nominal_lower = -20.0
        x_nominal_upper = -10.0
        error_nominal = 1.5
        dist_zero_to_nominal = 0.5
        centering_scale = 0.5

        centering_barrier_helper = lambda x, power: min_max_centering_power_barrier(
            x=x,
            x_nominal_lower=x_nominal_lower,
            x_nominal_upper=x_nominal_upper,
            error_nominal=error_nominal,
            dist_zero_to_nominal=dist_zero_to_nominal,
            power=power,
            centering_scale=centering_scale,
        )

        powers = [0.5, 1, 2]
        for power in powers:
            with self.subTest(power=power):
                # Check center
                center = (x_nominal_lower + x_nominal_upper) / 2
                self.assertEqual(centering_barrier_helper(center, power), 0.0)

                # Check corners
                left_corner = x_nominal_lower + dist_zero_to_nominal
                right_corner = x_nominal_upper - dist_zero_to_nominal
                self.assertNotEqual(centering_barrier_helper(left_corner, power), 0.0)
                self.assertNotEqual(centering_barrier_helper(right_corner, power), 0.0)

                # Check nominal point
                self.assertEqual(centering_barrier_helper(x_nominal_lower, power), error_nominal)
                self.assertEqual(centering_barrier_helper(x_nominal_upper, power), error_nominal)

                # Check curve shape on bounding curve
                x_bounding = x_nominal_upper + 1
                expected_error_bounding = (
                    error_nominal * ((x_bounding - right_corner) / dist_zero_to_nominal) ** power
                )
                self.assertEqual(
                    centering_barrier_helper(x_bounding, power), expected_error_bounding
                )

                # Check curve shape on centering curve
                x_centering = center + 0.1
                expected_error_centering = (
                    centering_scale
                    * error_nominal
                    * ((x_centering - center) / (x_nominal_upper - center)) ** power
                )
                self.assertStorageNear(
                    centering_barrier_helper(x_centering, power), expected_error_centering
                )

    def test_not_singular(self) -> None:
        """
        Tests that the x=0 singularity is handled
        """

        def f(x: sf.Scalar, y: sf.Scalar, d: sf.Scalar, epsilon: sf.Scalar) -> sf.Scalar:
            return max_power_barrier(
                x=x, x_nominal=2, error_nominal=1, dist_zero_to_nominal=d, power=y, epsilon=epsilon
            )

        def f_deriv(x: sf.Scalar, y: sf.Scalar, d: sf.Scalar, epsilon: sf.Scalar) -> sf.Scalar:
            return f(x, y, d, epsilon).diff(x)

        f_deriv_numeric = util.lambdify(f_deriv)
        with self.assertRaises(ZeroDivisionError):
            f_deriv_numeric(x=0, y=1, d=1, epsilon=0)
        self.assertEqual(f_deriv_numeric(x=0, y=1, d=1, epsilon=sf.numeric_epsilon), 0)


if __name__ == "__main__":
    TestCase.main()
