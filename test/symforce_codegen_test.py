import logging
import os

from symforce import geo
from symforce import logger
from symforce import python_util
from symforce import sympy as sm
from symforce import types as T
from symforce.codegen import CodegenMode
from symforce.codegen import Codegen
from symforce.codegen import geo_package_codegen
from symforce.codegen import codegen_util
from symforce.test_util import TestCase
from symforce.values import Values

SYMFORCE_DIR = os.path.dirname(os.path.dirname(__file__))

# Test function
def az_el_from_point(nav_T_cam, nav_t_point, epsilon=0):
    # type: (geo.Pose3, geo.Matrix, T.Scalar) -> geo.Matrix
    """
    Transform a nav point into azimuth / elevation angles in the
    camera frame.

    Args:
        nav_T_cam (geo.Pose3): camera pose in the world
        nav_t_point (geo.Matrix): nav point
        epsilon (Scalar): small number to avoid singularities

    Returns:
        geo.Matrix: (azimuth, elevation)
    """
    cam_t_point = nav_T_cam.inverse() * nav_t_point
    x, y, z = cam_t_point
    theta = sm.atan2_safe(y, x, epsilon=epsilon)
    phi = sm.pi / 2 - sm.acos(z / (cam_t_point.norm() + epsilon))
    return geo.V2(theta, phi)


class SymforceCodegenTest(TestCase):
    """
    Test symforce.codegen.EvaluatorCodegen.
    """

    @staticmethod
    def build_values():
        # type: () -> T.Tuple[Values, Values]
        """
        Create some example input/output values.
        """
        inputs = Values()
        x, y = sm.symbols("x y")
        inputs.add(x)
        inputs.add(y)

        inputs["rot"] = geo.Rot3().symbolic("rot")

        # Scalar
        inputs.add(sm.Symbol("constants.epsilon"))

        with inputs.scope("states"):
            # Array element, turns into std::array
            inputs["p"] = geo.V2.symbolic("p")

            # Vector element, turns into Eigen::Vector
            # inputs.add(sm.Symbol('q(0)'))

        outputs = Values()
        outputs["foo"] = x ** 2 + inputs["rot"].q.w
        outputs["bar"] = inputs.attr.constants.epsilon + sm.sin(inputs.attr.y) + x ** 2

        return inputs, outputs

    # -------------------------------------------------------------------------
    # Python
    # -------------------------------------------------------------------------

    def test_codegen_python(self):
        # type: () -> None
        """
        Test python code generation.
        """
        inputs, outputs = self.build_values()

        for scalar_type in ("double", "float"):
            python_func = Codegen(
                "codegen_test_python", inputs, outputs, CodegenMode.PYTHON2, scalar_type=scalar_type
            )
            codegen_data = python_func.generate_function()
            geo_package_codegen.generate(
                mode=CodegenMode.PYTHON2, output_dir=codegen_data["output_dir"]
            )

            geo_pkg = codegen_util.load_generated_package(
                os.path.join(codegen_data["output_dir"], "geo")
            )
            types_module = codegen_util.load_generated_package(
                os.path.join(codegen_data["output_dir"], "codegen_test_python_types")
            )

            x = 2.0
            y = -5.0
            rot = geo_pkg.Rot3()
            states = types_module.states_t()
            states.p = [1.0, 2.0]
            constants = types_module.constants_t()
            constants.epsilon = 1e-8

            gen_module = codegen_util.load_generated_package(codegen_data["output_dir"])
            foo, bar = gen_module.codegen_test_python(x, y, rot, constants, states)
            self.assertNear(foo, x ** 2 + rot.data[3])
            self.assertNear(bar, constants.epsilon + sm.sin(y) + x ** 2)

            if scalar_type == "double":
                test_data_dir = os.path.join(
                    SYMFORCE_DIR, "test", "symforce_function_codegen_test_data"
                )
                # Compare the function file
                expected_code_file = os.path.join(test_data_dir, "codegen_test_python.py")
                output_function = os.path.join(codegen_data["output_dir"], "codegen_test_python.py")
                with open(output_function) as f:
                    code = f.read()
                    self.compare_or_update(expected_code_file, code)

                # Compare the generated types
                self.compare_or_update_directory(
                    actual_dir=os.path.join(
                        codegen_data["output_dir"], "codegen_test_python_types"
                    ),
                    expected_dir=os.path.join(test_data_dir, "codegen_test_python_types"),
                )

            # Clean up
            if logger.level != logging.DEBUG:
                python_util.remove_if_exists(codegen_data["output_dir"])

    def test_function_codegen_python(self):
        # type: () -> None

        # Create the specification
        az_el_codegen = Codegen.function(
            name="az_el_from_point",
            func=az_el_from_point,
            input_types=[geo.Pose3, geo.V3, sm.Symbol],
            mode=CodegenMode.PYTHON2,
        )
        az_el_codegen_data = az_el_codegen.generate_function()

        # Compare to expected
        expected_code_file = os.path.join(
            SYMFORCE_DIR, "test", "symforce_function_codegen_test_data", "az_el_from_point.py"
        )
        output_function = os.path.join(az_el_codegen_data["output_dir"], "az_el_from_point.py")
        with open(output_function) as f:
            code = f.read()
            self.compare_or_update(expected_code_file, code)

        # Clean up
        if logger.level != logging.DEBUG:
            python_util.remove_if_exists(az_el_codegen_data["output_dir"])

    # -------------------------------------------------------------------------
    # C++
    # -------------------------------------------------------------------------

    def test_codegen_cpp(self):
        # type: () -> None
        """
        Test C++ code generation.
        """
        inputs, outputs = self.build_values()

        for scalar_type in ("double", "float"):
            cpp_func = Codegen(
                "CodegenTestCpp", inputs, outputs, CodegenMode.CPP, scalar_type=scalar_type
            )
            codegen_data = cpp_func.generate_function()

            if scalar_type == "double":
                test_data_dir = os.path.join(
                    SYMFORCE_DIR, "test", "symforce_function_codegen_test_data"
                )
                # Compare the function file
                expected_code_file = os.path.join(test_data_dir, "codegen_test_cpp.h")
                output_function = os.path.join(codegen_data["output_dir"], "codegen_test_cpp.h")
                with open(output_function) as f:
                    code = f.read()
                    self.compare_or_update(expected_code_file, code)

                # Compare the generated types
                self.compare_or_update_directory(
                    actual_dir=os.path.join(codegen_data["output_dir"], "codegen_test_cpp_types"),
                    expected_dir=os.path.join(test_data_dir, "codegen_test_cpp_types"),
                )

                try:
                    TestCase.compile_and_run_cpp(
                        os.path.join(SYMFORCE_DIR, "test"), "codegen_function_test"
                    )
                finally:
                    if logger.level != logging.DEBUG:
                        python_util.remove_if_exists(
                            os.path.join(SYMFORCE_DIR, "test", "codegen_function_test")
                        )
                        python_util.remove_if_exists(
                            os.path.join(SYMFORCE_DIR, "test", "libsymforce_geo.so")
                        )

            # Clean up
            if logger.level != logging.DEBUG:
                python_util.remove_if_exists(codegen_data["output_dir"])

    def test_function_codegen_cpp(self):
        # type: () -> None

        # Create the specification
        az_el_codegen = Codegen.function(
            name="AzElFromPoint",
            func=az_el_from_point,
            input_types=[geo.Pose3, geo.V3, sm.Symbol],
            mode=CodegenMode.CPP,
        )
        az_el_codegen_data = az_el_codegen.generate_function()

        # Compare to expected
        expected_code_file = os.path.join(
            SYMFORCE_DIR, "test", "symforce_function_codegen_test_data", "az_el_from_point.h"
        )
        output_function = os.path.join(az_el_codegen_data["output_dir"], "az_el_from_point.h")
        with open(output_function) as f:
            code = f.read()
            self.compare_or_update(expected_code_file, code)

        # Clean up
        if logger.level != logging.DEBUG:
            python_util.remove_if_exists(az_el_codegen_data["output_dir"])


if __name__ == "__main__":
    TestCase.main()
