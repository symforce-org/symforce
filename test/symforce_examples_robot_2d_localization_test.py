# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_symbol()

from symforce.examples.robot_2d_localization import robot_2d_localization
from symforce.opt.optimizer import Optimizer
from symforce.test_util import TestCase


class Robot2DLocalizationTest(TestCase):
    def test_optimize(self) -> None:
        """
        Test the equivalent of robot_2d_localization.main, but without plotting or printing.
        Implemented separately here since 1) the sequence of operations is simple in terms of the
        functions in the example and 2) we want extra checks here and no plotting or printing
        """
        # Create a problem setup and initial guess
        initial_values, num_poses, num_landmarks = robot_2d_localization.build_initial_values()

        # Create factors
        factors = robot_2d_localization.build_factors(
            num_poses=num_poses, num_landmarks=num_landmarks
        )

        # Select the keys to optimize - the rest will be held constant
        optimized_keys = [f"poses[{i}]" for i in range(num_poses)]

        # Create the optimizer
        optimizer = Optimizer(
            factors=factors,
            optimized_keys=optimized_keys,
            debug_stats=True,  # Return problem stats for every iteration
            params=Optimizer.Params(verbose=True),  # Customize optimizer behavior
        )

        # Solve and return the result
        result = optimizer.optimize(initial_values)

        self.assertAlmostEqual(result.iterations[0].new_error, 6.396357319110695)
        self.assertLess(result.error(), 1e-3)
        self.assertEqual(result.status, Optimizer.Status.SUCCESS)


if __name__ == "__main__":
    Robot2DLocalizationTest.main()
