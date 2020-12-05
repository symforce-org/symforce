import os
import tempfile

from symforce import logger
from symforce.examples.bundle_adjustment.generate_fixed_problem import FixedBundleAdjustmentProblem
from symforce.examples.bundle_adjustment.generate_dynamic_problem import (
    DynamicBundleAdjustmentProblem,
)
from symforce.test_util import TestCase, slow_on_sympy
from symforce.test_util.test_case_mixin import SymforceTestCaseMixin

CURRENT_DIR = os.path.dirname(__file__)
SYMFORCE_DIR = os.path.join(CURRENT_DIR, "..")


class BundleAdjustmentExampleCodegenTest(TestCase, SymforceTestCaseMixin):
    @slow_on_sympy
    def test_generate_example(self) -> None:
        base_dirname = "symforce_bundle_adjustment_example"
        output_dir = tempfile.mkdtemp(prefix=base_dirname, dir="/tmp")
        logger.debug(f"Creating temp directory: {output_dir}")

        FixedBundleAdjustmentProblem(2, 20).generate(output_dir=output_dir)

        DynamicBundleAdjustmentProblem().generate(output_dir=output_dir)

        self.compare_or_update_directory(
            actual_dir=output_dir,
            expected_dir=os.path.join(
                SYMFORCE_DIR, "symforce", "examples", "bundle_adjustment", "gen"
            ),
        )


if __name__ == "__main__":
    BundleAdjustmentExampleCodegenTest.main()
