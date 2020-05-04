import logging
import os
import tempfile

import symforce
from symforce import geo
from symforce import logger
from symforce import python_util
from symforce import sympy as sm
from symforce.test_util import TestCase
from symforce.codegen import FunctionCodegen
from symforce.codegen import CodegenMode


DATA_DIR = os.path.join(os.path.dirname(__file__), "symforce_function_codegen_test_data")


# Test function
def az_el_from_point(nav_T_cam, nav_t_point, epsilon=0):
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


class SymforceFunctionCodegenTest(TestCase):
    """
    Test symforce.codegen.function_codegen.
    """

    def test_codegen_cpp(self):
        # Create the specification
        spec = FunctionCodegen(
            name="AzElFromPoint",
            func=az_el_from_point,
            arg_types=[geo.Pose3, geo.V3(), sm.Symbol],
            return_type=geo.V2(),
        )

        # Emit function code
        code = spec.render(mode=CodegenMode.CPP)

        # Compare to expected
        expected_code_file = os.path.join(
            DATA_DIR, "az_el_from_point_{}.cc".format(symforce.get_backend())
        )
        self.compare_or_update(expected_code_file, code)

    def test_codegen_python(self):
        # Create the specification
        spec = FunctionCodegen(
            name="az_el_from_point",
            func=az_el_from_point,
            arg_types=[geo.Pose3, geo.V3(), sm.Symbol],
            return_type=geo.V2(),
        )

        # Emit function code
        code = spec.render(mode=CodegenMode.PYTHON2)

        # Compare to expected
        expected_code_file = os.path.join(
            DATA_DIR, "az_el_from_point_{}.py".format(symforce.get_backend())
        )
        self.compare_or_update(expected_code_file, code)

    def test_no_docstring(self):
        # Function with no docstring
        def func_no_docstring(vector3, scalar):
            return scalar * vector3

        # Create the specification
        spec = FunctionCodegen(
            name="Foobar",
            func=func_no_docstring,
            arg_types=[geo.V3(), sm.Symbol],
            return_type=geo.V3(),
        )

        # Emit function code
        code = spec.render(mode=CodegenMode.CPP)

        # Compare to expected
        expected_code_file = os.path.join(DATA_DIR, "test_no_docstring.cc")
        self.compare_or_update(expected_code_file, code)


if __name__ == "__main__":
    TestCase.main()
