import os

from symforce.examples.bundle_adjustment_fixed_size.generate_fixed_problem import (
    FixedBundleAdjustmentProblem,
)
from symforce.test_util import TestCase, slow_on_sympy

CURRENT_DIR = os.path.dirname(__file__)
SYMFORCE_DIR = os.path.join(CURRENT_DIR, "..")

BASE_DIRNAME = "symforce_bundle_adjustment_example"


class BundleAdjustmentExampleCodegenTest(TestCase):
    @slow_on_sympy
    def test_generate_example_fixed(self) -> None:
        output_dir = self.make_output_dir(BASE_DIRNAME)

        FixedBundleAdjustmentProblem(2, 20).generate(output_dir=output_dir)

        self.compare_or_update_directory(
            actual_dir=output_dir,
            expected_dir=os.path.join(
                SYMFORCE_DIR, "symforce", "examples", "bundle_adjustment_fixed_size", "gen"
            ),
        )


if __name__ == "__main__":
    BundleAdjustmentExampleCodegenTest.main()
