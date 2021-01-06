import logging
import math
import os
import sys
import tempfile

import symforce
from symforce import ops
from symforce import logger
from symforce import python_util
from symforce import sympy as sm
from symforce import types as T
from symforce.test_util import TestCase, slow_on_sympy
from symforce.codegen import Codegen
from symforce.codegen import CodegenMode
from symforce.codegen import codegen_util
from symforce.codegen import geo_package_codegen


SYMFORCE_DIR = os.path.dirname(os.path.dirname(__file__))
TEST_DATA_DIR = os.path.join(
    SYMFORCE_DIR, "test", "symforce_function_codegen_test_data", symforce.get_backend()
)


class SymforceGeoCodegenTest(TestCase):
    """
    Test symforce.codegen.geo_package_codegen.
    """

    def test_codegen_python(self) -> None:
        """
        Test Python code generation from geometry types.
        """
        output_dir = tempfile.mkdtemp(prefix="sf_geo_package_codegen_test_", dir="/tmp")
        logger.debug(f"Creating temp directory: {output_dir}")

        try:
            geo_package_codegen.generate(mode=CodegenMode.PYTHON2, output_dir=output_dir)

            # Test against checked-in geo package (only on SymEngine)
            if symforce.get_backend() == "symengine":
                self.compare_or_update_directory(
                    actual_dir=os.path.join(output_dir, "geo"),
                    expected_dir=os.path.join(SYMFORCE_DIR, "gen", "python", "geo"),
                )

            # Compare against the checked-in test itself
            expected_code_file = os.path.join(SYMFORCE_DIR, "test", "geo_package_python_test.py")
            generated_code_file = os.path.join(output_dir, "example", "geo_package_python_test.py")
            self.compare_or_update_file(expected_code_file, generated_code_file)

            # Run generated example / test from disk in a standalone process
            python_util.execute_subprocess(["python", generated_code_file])

            # Also hot load package directly in to this process
            geo_pkg = codegen_util.load_generated_package(os.path.join(output_dir, "geo"))

            # Test something basic from the hot loaded package
            rot = geo_pkg.Rot3.from_tangent([math.pi / 2, 0, 0])
            rot_inv = rot.inverse()
            identity_expected = rot * rot_inv
            identity_actual = geo_pkg.Rot3.identity()
            for i in range(len(identity_expected.data)):
                self.assertAlmostEqual(identity_expected.data[i], identity_actual.data[i], places=7)

        finally:
            if logger.level != logging.DEBUG:
                python_util.remove_if_exists(output_dir)

    @slow_on_sympy
    def test_codegen_cpp(self) -> None:
        """
        Test C++ code generation from geometry types.
        """
        output_dir = tempfile.mkdtemp(prefix="sf_geo_package_codegen_test_", dir="/tmp")
        logger.debug(f"Creating temp directory: {output_dir}")

        try:
            geo_package_codegen.generate(mode=CodegenMode.CPP, output_dir=output_dir)

            # Test against checked-in geo package (only on SymEngine)
            if symforce.get_backend() == "symengine":
                self.compare_or_update_directory(
                    actual_dir=os.path.join(output_dir, "geo"),
                    expected_dir=os.path.join(SYMFORCE_DIR, "gen", "cpp", "geo"),
                )

            # Generate functions for testing tangent_D_storage numerical derivatives
            for cls in geo_package_codegen.DEFAULT_GEO_TYPES:
                tangent_D_storage_codegen = Codegen.function(
                    name="Tangent_D_Storage",
                    func=ops.LieGroupOps.tangent_D_storage,
                    input_types=[cls, sm.Symbol],
                    mode=CodegenMode.CPP,
                )
                tangent_D_storage_codegen.generate_function(
                    # Underscore here because of how python_util.camelcase_to_snakecase works
                    output_dir=output_dir,
                    generated_file_name="Tangent_DStorage" + cls.__name__,
                )

            self.compare_or_update_directory(
                actual_dir=os.path.join(output_dir, "cpp/symforce/sym"),
                expected_dir=os.path.join(TEST_DATA_DIR, "tangent_d_storage"),
            )

            # Compare against the checked-in test itself
            expected_code_file = os.path.join(TEST_DATA_DIR, "example", "geo_package_cpp_test.cc")
            generated_code_file = os.path.join(output_dir, "example", "geo_package_cpp_test.cc")
            self.compare_or_update_file(expected_code_file, generated_code_file)

            # Compile and run the test
            if not self.UPDATE:
                example_dir = os.path.join(output_dir, "example")
                TestCase.compile_and_run_cpp(example_dir, "geo_package_cpp_test")

        finally:
            if logger.level != logging.DEBUG:
                python_util.remove_if_exists(output_dir)


if __name__ == "__main__":
    TestCase.main()
