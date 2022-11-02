# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_symbol()

from symforce import path_util
from symforce.examples.custom_factor_generation import generate_factors
from symforce.test_util import TestCase
from symforce.test_util import symengine_only
from symforce.test_util.test_case_mixin import SymforceTestCaseMixin

BASE_DIRNAME = "symforce_custom_factor_generation_example"


class CustomFactorGenerationExampleCodegenTest(TestCase, SymforceTestCaseMixin):
    @symengine_only
    def test_generate_factors(self) -> None:
        output_dir = self.make_output_dir(BASE_DIRNAME)

        generate_factors.generate(output_dir=output_dir)

        self.compare_or_update_directory(
            actual_dir=output_dir,
            expected_dir=(
                path_util.symforce_data_root()
                / "symforce"
                / "examples"
                / "custom_factor_generation"
                / "gen"
            ),
        )


if __name__ == "__main__":
    CustomFactorGenerationExampleCodegenTest.main()
