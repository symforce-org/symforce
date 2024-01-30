# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import asyncio
import math
import sys
from pathlib import Path

import numpy as np

import symforce

symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from symforce import codegen
from symforce import ops
from symforce import path_util
from symforce import python_util
from symforce.codegen import cam_package_codegen
from symforce.codegen import codegen_util
from symforce.codegen import geo_factors_codegen
from symforce.codegen import slam_factors_codegen
from symforce.codegen import sym_util_package_codegen
from symforce.codegen import template_util
from symforce.slam.imu_preintegration.generate import generate_manifold_imu_preintegration
from symforce.test_util import TestCase
from symforce.test_util import symengine_only

SYMFORCE_DIR = path_util.symforce_data_root()
TEST_DATA_DIR = (
    SYMFORCE_DIR / "test" / "symforce_function_codegen_test_data" / symforce.get_symbolic_api()
)


class SymforceGenCodegenTest(TestCase):
    """
    Generate everything that goes into symforce/gen
    """

    def generate_cam_example_function(self, output_dir: Path) -> None:
        def pixel_to_ray_and_back(
            pixel: sf.Vector2, cam: sf.LinearCameraCal, epsilon: sf.Scalar = 0
        ) -> sf.Vector2:
            """
            Transform a given pixel into a ray and project the ray back to
            pixel coordinates. Input and output should match.
            """
            camera_ray, _ = cam.camera_ray_from_pixel(pixel, epsilon)
            reprojected_pixel, _ = cam.pixel_from_camera_point(camera_ray, epsilon)
            return reprojected_pixel

        # Create the specification
        namespace = "cam_function_codegen_test"
        my_func = codegen.Codegen.function(func=pixel_to_ray_and_back, config=codegen.CppConfig())
        my_func.generate_function(output_dir=output_dir, namespace=namespace)

        # Compare against the checked-in test data
        self.compare_or_update_directory(
            actual_dir=output_dir / "cpp/symforce/cam_function_codegen_test",
            expected_dir=(
                TEST_DATA_DIR / "symforce_gen_codegen_test_data" / "cam_function_codegen_test"
            ),
        )

    def generate_tangent_d_storage_functions(self, output_dir: Path) -> None:
        for cls in sf.GEO_TYPES:
            tangent_D_storage_codegen = codegen.Codegen.function(
                func=ops.LieGroupOps.tangent_D_storage,
                input_types=[cls],
                config=codegen.CppConfig(),
            )
            tangent_D_storage_codegen.generate_function(
                output_dir=output_dir,
                generated_file_name="tangent_d_storage_" + cls.__name__.lower(),
            )

        # Compare against the checked-in test data
        self.compare_or_update_directory(
            actual_dir=output_dir / "cpp/symforce/sym",
            expected_dir=TEST_DATA_DIR / "symforce_gen_codegen_test_data" / "tangent_d_storage",
        )

    def test_gen_package_codegen_python(self) -> None:
        """
        Test Python code generation
        """
        output_dir = self.make_output_dir("sf_gen_codegen_test_")

        config = codegen.PythonConfig()
        cam_package_codegen.generate(config=config, output_dir=output_dir)
        template_util.render_template(
            template_dir=config.template_dir(),
            template_path="setup.py.jinja",
            output_path=output_dir / "setup.py",
            data=dict(
                package_name="symforce-sym",
                version=symforce.__version__,
                description="generated numerical python package",
                long_description="generated numerical python package",
            ),
            config=config.render_template_config,
        )

        # Test against checked-in geo package (only on SymEngine)
        if symforce.get_symbolic_api() == "symengine":
            self.compare_or_update_directory(
                actual_dir=output_dir / "sym", expected_dir=SYMFORCE_DIR / "gen" / "python" / "sym"
            )
            self.compare_or_update_file(
                new_file=output_dir / "setup.py", path=SYMFORCE_DIR / "gen" / "python" / "setup.py"
            )

        # Compare against the checked-in tests
        for test_name in ("cam_package_python_test.py", "geo_package_python_test.py"):
            generated_code_file = output_dir / "tests" / test_name

            self.compare_or_update_file(
                path=SYMFORCE_DIR / "test" / test_name, new_file=generated_code_file
            )

            # Run generated example / test from disk in a standalone process
            current_python = sys.executable
            asyncio.run(
                python_util.execute_subprocess(
                    [current_python, str(generated_code_file)], log_stdout=False
                )
            )

        # Also hot load package directly in to this process
        geo_pkg = codegen_util.load_generated_package("sym", output_dir / "sym")

        # Test something basic from the hot loaded package
        rot = geo_pkg.Rot3.from_tangent(np.array([math.pi / 2, 0, 0]))
        rot_inv = rot.inverse()
        identity_expected = rot * rot_inv
        identity_actual = geo_pkg.Rot3.identity()
        for i, expected_data in enumerate(identity_expected.data):
            self.assertAlmostEqual(expected_data, identity_actual.data[i], places=7)

    # This is so slow on sympy that we disable it entirely
    @symengine_only
    def test_gen_package_codegen_cpp(self) -> None:
        """
        Test C++ code generation
        """
        config = codegen.CppConfig()
        output_dir = self.make_output_dir("sf_gen_codegen_test_")

        # Prior factors, between factors, and SLAM factors for C++.
        geo_factors_codegen.generate(output_dir / "sym")
        slam_factors_codegen.generate(output_dir / "sym")
        generate_manifold_imu_preintegration(
            config=config,
            output_dir=output_dir / "sym" / "factors" / "internal",
        )

        # Generate typedefs.h
        sym_util_package_codegen.generate(config=config, output_dir=output_dir)

        # Generate cam package, geo package, and tests
        # This calls geo_package_codegen.generate internally
        cam_package_codegen.generate(config=config, output_dir=output_dir)

        # Check against existing generated package (only on SymEngine)
        self.compare_or_update_directory(
            actual_dir=output_dir / "sym", expected_dir=SYMFORCE_DIR / "gen" / "cpp" / "sym"
        )

        # Generate functions for testing tangent_D_storage numerical derivatives
        self.generate_tangent_d_storage_functions(output_dir)

        # Generate function that uses camera object as argument
        self.generate_cam_example_function(output_dir)

        # Compare against the checked-in tests
        for test_name in (
            "cam_package_cpp_test.cc",
            "cam_function_codegen_cpp_test.cc",
            "geo_package_cpp_test.cc",
        ):
            self.compare_or_update_file(
                path=SYMFORCE_DIR / "test" / test_name, new_file=output_dir / "tests" / test_name
            )


if __name__ == "__main__":
    TestCase.main()
