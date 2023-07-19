# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_symbol()

from symforce.examples.robot_3d_localization import robot_3d_localization
from symforce.opt.optimizer import Optimizer
from symforce.test_util import TestCase


class Robot3DLocalizationTest(TestCase):
    def test_optimize(self) -> None:
        """
        Test the equivalent of robot_3d_localization.main, but without plotting or printing.
        Implemented separately here since 1) the sequence of operations is simple in terms of the
        functions in the example and 2) we want extra checks here and no plotting or printing
        """
        values, num_landmarks = robot_3d_localization.build_values(robot_3d_localization.NUM_POSES)

        # Create factors
        factors = robot_3d_localization.build_factors(
            num_poses=robot_3d_localization.NUM_POSES, num_landmarks=num_landmarks
        )

        # Select the keys to optimize - the rest will be held constant
        optimized_keys = [f"world_T_body[{i}]" for i in range(robot_3d_localization.NUM_POSES)]

        # Create the optimizer
        optimizer = Optimizer(
            factors=factors,
            optimized_keys=optimized_keys,
            # Return problem stats for every iteration
            debug_stats=True,
            # Customize optimizer behavior
            params=Optimizer.Params(verbose=True, initial_lambda=1e4, lambda_down_factor=1 / 2.0),
        )

        # Solve and return the result
        result = optimizer.optimize(values)

        self.assertAlmostEqual(result.iterations[0].new_error, 463700.5576620833)
        self.assertLess(result.error(), 140)
        self.assertEqual(result.status, Optimizer.Status.SUCCESS)


if __name__ == "__main__":
    Robot3DLocalizationTest.main()
