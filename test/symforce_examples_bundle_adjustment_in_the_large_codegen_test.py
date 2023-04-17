# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_symbol()

from symforce import path_util
from symforce.examples.bundle_adjustment_in_the_large.bundle_adjustment_in_the_large import generate
from symforce.test_util import TestCase
from symforce.test_util import sympy_only


class BundleAdjustmentInTheLargeCodegenTest(TestCase):
    """
    Generate code for the BundleAdjustmentInTheLarge example
    """

    @sympy_only
    def test_generate(self) -> None:
        output_dir = self.make_output_dir("sf_examples_bundle_adjustment_in_the_large_codegen_test")

        generate(output_dir)

        self.compare_or_update_directory(
            actual_dir=output_dir,
            expected_dir=(
                path_util.symforce_data_root()
                / "symforce"
                / "examples"
                / "bundle_adjustment_in_the_large"
                / "gen"
            ),
        )


if __name__ == "__main__":
    BundleAdjustmentInTheLargeCodegenTest.main()
