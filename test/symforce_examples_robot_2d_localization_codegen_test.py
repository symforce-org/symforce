# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from pathlib import Path

from symforce.examples.robot_2d_localization import robot_2d_localization
from symforce.test_util import TestCase
from symforce.test_util import symengine_only

SYMFORCE_DIR = Path(__file__).parent.parent


class Robot2DLocalizationCodegenTest(TestCase):
    # Only use the symengine output, to save the trouble of juggling both
    @symengine_only
    def test_generate(self) -> None:
        output_dir = Path(self.make_output_dir("symforce_robot_2d_localization_example"))

        robot_2d_localization.generate_bearing_residual_code(output_dir)
        robot_2d_localization.generate_odometry_residual_code(output_dir)

        self.compare_or_update_directory(
            actual_dir=output_dir,
            expected_dir=SYMFORCE_DIR / "symforce" / "examples" / "robot_2d_localization" / "gen",
        )


if __name__ == "__main__":
    Robot2DLocalizationCodegenTest.main()
