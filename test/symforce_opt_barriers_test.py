# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce.test_util import TestCase
from symforce.opt.barrier_functions import symmetric_power_barrier


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


if __name__ == "__main__":
    TestCase.main()
