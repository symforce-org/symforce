# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_symbol()

from symforce import path_util
from symforce.benchmarks.inverse_compose_jacobian import generate_inverse_compose_jacobian
from symforce.benchmarks.matrix_multiplication import generate_matrix_multiplication_benchmark
from symforce.test_util import TestCase
from symforce.test_util import sympy_only

BENCHMARKS_DIR = path_util.symforce_data_root() / "symforce" / "benchmarks"


class SymforceBenchmarksCodegenTest(TestCase):
    """
    Generates code used by the benchmarks
    """

    @sympy_only
    def test_generate_inverse_compose(self) -> None:
        """
        Tests:
            Generates code for the inverse_compose_jacobian benchmark
        """
        output_dir = self.make_output_dir("sf_benchmarks_codegen_test")
        generate_inverse_compose_jacobian.generate(output_dir)
        self.compare_or_update_directory(
            actual_dir=output_dir, expected_dir=BENCHMARKS_DIR / "inverse_compose_jacobian" / "gen"
        )

    @sympy_only
    def test_generate_matrix_multiplication(self) -> None:
        """
        Tests:
            Generates code for the matrix_multiplication benchmark
        """
        output_dir = self.make_output_dir("sf_benchmarks_codegen_test")

        generate_matrix_multiplication_benchmark.generate(output_dir)

        self.compare_or_update_directory(
            actual_dir=output_dir, expected_dir=BENCHMARKS_DIR / "matrix_multiplication" / "gen"
        )


if __name__ == "__main__":
    TestCase.main()
