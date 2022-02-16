# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce.test_util import TestCase
from symforce.opt.barrier_functions import (
    symmetric_power_barrier,
    min_max_power_barrier,
    min_max_linear_barrier,
)


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

    def test_min_max_power_barrier(self) -> None:
        """
        Tests min_max_power_barrier and min_max_linear_barrier
        """
        x_nominal_lower = -20.0
        x_nominal_upper = -10.0
        error_nominal = 1.5
        dist_zero_to_nominal = 0.5
        powers = [0.5, 1, 2]
        for power in powers:
            with self.subTest(power=power):
                # Check center
                center = (x_nominal_lower + x_nominal_upper) / 2
                center_error = min_max_power_barrier(
                    x=center,
                    x_nominal_lower=x_nominal_lower,
                    x_nominal_upper=x_nominal_upper,
                    error_nominal=error_nominal,
                    dist_zero_to_nominal=dist_zero_to_nominal,
                    power=power,
                )
                self.assertEqual(center_error, 0.0)

                # Check corners
                left_corner = x_nominal_lower + dist_zero_to_nominal
                left_error = min_max_power_barrier(
                    x=left_corner,
                    x_nominal_lower=x_nominal_lower,
                    x_nominal_upper=x_nominal_upper,
                    error_nominal=error_nominal,
                    dist_zero_to_nominal=dist_zero_to_nominal,
                    power=power,
                )
                right_corner = x_nominal_upper - dist_zero_to_nominal
                right_error = min_max_power_barrier(
                    x=right_corner,
                    x_nominal_lower=x_nominal_lower,
                    x_nominal_upper=x_nominal_upper,
                    error_nominal=error_nominal,
                    dist_zero_to_nominal=dist_zero_to_nominal,
                    power=power,
                )
                self.assertEqual(left_error, 0.0)
                self.assertEqual(right_error, 0.0)

                # Check nominal point
                left_nominal_error = min_max_power_barrier(
                    x=x_nominal_lower,
                    x_nominal_lower=x_nominal_lower,
                    x_nominal_upper=x_nominal_upper,
                    error_nominal=error_nominal,
                    dist_zero_to_nominal=dist_zero_to_nominal,
                    power=power,
                )
                right_nominal_error = min_max_power_barrier(
                    x=x_nominal_upper,
                    x_nominal_lower=x_nominal_lower,
                    x_nominal_upper=x_nominal_upper,
                    error_nominal=error_nominal,
                    dist_zero_to_nominal=dist_zero_to_nominal,
                    power=power,
                )
                self.assertEqual(left_nominal_error, error_nominal)
                self.assertEqual(right_nominal_error, error_nominal)

                # Check curve shape
                x = x_nominal_upper + 1.0
                expected_error = (
                    error_nominal * ((x - right_corner) / dist_zero_to_nominal) ** power
                )
                error = min_max_power_barrier(
                    x=x,
                    x_nominal_lower=x_nominal_lower,
                    x_nominal_upper=x_nominal_upper,
                    error_nominal=error_nominal,
                    dist_zero_to_nominal=dist_zero_to_nominal,
                    power=power,
                )
                self.assertEqual(error, expected_error)

        # Check that min_max_linear_barrier returns the same error for several points
        for x in [-42.0, -20.0, -19.5, -15, -10.5, -10.0, 42.0]:
            expected_error = min_max_power_barrier(
                x=x,
                x_nominal_lower=x_nominal_lower,
                x_nominal_upper=x_nominal_upper,
                error_nominal=error_nominal,
                dist_zero_to_nominal=dist_zero_to_nominal,
                power=1,
            )
            error = min_max_linear_barrier(
                x=x,
                x_nominal_lower=x_nominal_lower,
                x_nominal_upper=x_nominal_upper,
                error_nominal=error_nominal,
                dist_zero_to_nominal=dist_zero_to_nominal,
            )
            self.assertEqual(error, expected_error)


if __name__ == "__main__":
    TestCase.main()
