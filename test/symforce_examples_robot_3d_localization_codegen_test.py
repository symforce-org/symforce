import os
from pathlib import Path

from symforce.examples.robot_3d_localization.robot_3d_localization import generate
from symforce.test_util import TestCase, symengine_only

CURRENT_DIR = Path(__file__).parent
SYMFORCE_DIR = CURRENT_DIR.parent

BASE_DIRNAME = "symforce_robot_3d_localization_example"


class Robot3DScanMatchingCodegenTest(TestCase):
    # This one is so impossibly slow on SymPy that we just disable it
    @symengine_only
    def test_generate(self) -> None:
        output_dir = Path(self.make_output_dir(BASE_DIRNAME))

        generate(output_dir)

        self.compare_or_update_directory(
            actual_dir=output_dir,
            expected_dir=os.fspath(
                SYMFORCE_DIR / "symforce" / "examples" / "robot_3d_localization" / "gen"
            ),
        )


if __name__ == "__main__":
    Robot3DScanMatchingCodegenTest.main()
