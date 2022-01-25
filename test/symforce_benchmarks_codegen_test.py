from pathlib import Path

from symforce.test_util import TestCase, sympy_only
from symforce.benchmarks.inverse_compose_jacobian import generate_inverse_compose_jacobian
from symforce.benchmarks.matrix_multiplication import generate_matrix_multiplication_benchmark


BENCHMARKS_DIR = Path(__file__).parent.parent / "symforce" / "benchmarks"


class SymforceBenchmarksCodegenTest(TestCase):
    @sympy_only
    def test_generate_inverse_compose(self) -> None:
        output_dir = Path(self.make_output_dir("sf_benchmarks_codegen_test"))
        generate_inverse_compose_jacobian.generate(output_dir)
        self.compare_or_update_directory(
            actual_dir=output_dir, expected_dir=BENCHMARKS_DIR / "inverse_compose_jacobian" / "gen"
        )

    @sympy_only
    def test_generate_matrix_multiplication(self) -> None:
        output_dir = Path(self.make_output_dir("sf_benchmarks_codegen_test"))

        generate_matrix_multiplication_benchmark.generate_matrices(output_dir)
        generate_matrix_multiplication_benchmark.generate_tests(output_dir)

        self.compare_or_update_directory(
            actual_dir=output_dir, expected_dir=BENCHMARKS_DIR / "matrix_multiplication" / "gen"
        )


if __name__ == "__main__":
    TestCase.main()
