import collections
import logging
import tempfile
import sys
import os

import symforce
from symforce import geo
from symforce import ops
from symforce import logger
from symforce import python_util
from symforce import sympy as sm
from symforce import types as T
from symforce.codegen import CodegenMode
from symforce.codegen import Codegen
from symforce.codegen import geo_package_codegen
from symforce.codegen import codegen_util
from symforce.test_util import TestCase, slow_on_sympy
from symforce.values import Values

SYMFORCE_DIR = os.path.dirname(os.path.dirname(__file__))
TEST_DATA_DIR = os.path.join(
    SYMFORCE_DIR, "test", "symforce_function_codegen_test_data", symforce.get_backend()
)

# Test function
def az_el_from_point(
    nav_T_cam: geo.Pose3, nav_t_point: geo.Vector3, epsilon: T.Scalar = 0
) -> geo.Matrix:
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
    x, y, z = cam_t_point.to_flat_list()
    theta = sm.atan2_safe(y, x, epsilon=epsilon)
    phi = sm.pi / 2 - sm.acos(z / (cam_t_point.norm() + epsilon))
    return geo.V2(theta, phi)


class SymforceCodegenTest(TestCase):
    """
    Test symforce.codegen.EvaluatorCodegen.
    """

    @staticmethod
    def build_values() -> T.Tuple[Values, Values]:
        """
        Create some example input/output values.
        """
        inputs = Values()
        x, y = sm.symbols("x y")
        inputs.add(x)
        inputs.add(y)

        inputs["rot"] = geo.Rot3().symbolic("rot")

        # Test lists of objects, scalars, and Values
        inputs["rot_vec"] = [
            geo.Rot3().symbolic("rot1"),
            geo.Rot3().symbolic("rot2"),
            geo.Rot3().symbolic("rot3"),
        ]
        inputs["scalar_vec"] = [
            sm.Symbol("scalar1"),
            sm.Symbol("scalar2"),
            sm.Symbol("scalar3"),
        ]
        inputs["list_of_lists"] = [
            ops.StorageOps.symbolic(inputs["rot_vec"], "rot_vec1"),
            ops.StorageOps.symbolic(inputs["rot_vec"], "rot_vec2"),
            ops.StorageOps.symbolic(inputs["rot_vec"], "rot_vec3"),
        ]
        inputs_copy = inputs.copy()
        inputs["values_vec"] = [
            inputs_copy.symbolic("inputs_copy1"),
            inputs_copy.symbolic("inputs_copy2"),
            inputs_copy.symbolic("inputs_copy3"),
        ]
        inputs["values_vec_2D"] = [
            [inputs_copy.symbolic("inputs_copy11")],
            [inputs_copy.symbolic("inputs_copy12")],
        ]

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
        # Test outputing lists of objects, scalars, and Values
        outputs["scalar_vec_out"] = ops.GroupOps.compose(inputs["scalar_vec"], inputs["scalar_vec"])
        outputs["values_vec_out"] = ops.GroupOps.compose(inputs["values_vec"], inputs["values_vec"])
        outputs["values_vec_2D_out"] = ops.GroupOps.compose(
            inputs["values_vec_2D"], inputs["values_vec_2D"]
        )

        return inputs, outputs

    # -------------------------------------------------------------------------
    # Python
    # -------------------------------------------------------------------------

    @slow_on_sympy
    def test_codegen_python(self) -> None:
        """
        Test python code generation.
        """
        inputs, outputs = self.build_values()

        for scalar_type in ("double", "float"):
            python_func = Codegen(
                inputs=inputs,
                outputs=outputs,
                mode=CodegenMode.PYTHON2,
                name="python_function",
                scalar_type=scalar_type,
            )
            shared_types = {
                "values_vec": "values_vec_t",
                "values_vec_out": "values_vec_t",
                "values_vec_2D": "values_vec_t",
                "values_vec_2D_out": "values_vec_t",
            }
            namespace = "codegen_python_test"
            codegen_data = python_func.generate_function(
                shared_types=shared_types, namespace=namespace
            )
            if scalar_type == "double":
                self.compare_or_update_directory(
                    actual_dir=codegen_data["output_dir"],
                    expected_dir=os.path.join(TEST_DATA_DIR, namespace + "_data"),
                )

            geo_package_codegen.generate(
                mode=CodegenMode.PYTHON2, output_dir=codegen_data["output_dir"]
            )

            geo_pkg = codegen_util.load_generated_package(
                os.path.join(codegen_data["output_dir"], "sym")
            )
            types_module = codegen_util.load_generated_package(
                os.path.join(codegen_data["python_types_dir"], namespace)
            )

            x = 2.0
            y = -5.0
            rot = geo_pkg.Rot3()
            rot_vec = [geo_pkg.Rot3(), geo_pkg.Rot3(), geo_pkg.Rot3()]
            scalar_vec = [1.0, 2.0, 3.0]
            list_of_lists = [rot_vec, rot_vec, rot_vec]
            values_vec = [
                types_module.values_vec_t(),
                types_module.values_vec_t(),
                types_module.values_vec_t(),
            ]
            values_vec_2D = [[types_module.values_vec_t()], [types_module.values_vec_t()]]

            states = types_module.states_t()
            states.p = [1.0, 2.0]
            constants = types_module.constants_t()
            constants.epsilon = 1e-8

            gen_module = codegen_util.load_generated_package(codegen_data["python_function_dir"])
            # TODO(nathan): Split this test into several different functions
            (
                foo,
                bar,
                scalar_vec_out,
                values_vec_out,
                values_vec_2D_out,
            ) = gen_module.python_function(
                x,
                y,
                rot,
                rot_vec,
                scalar_vec,
                list_of_lists,
                values_vec,
                values_vec_2D,
                constants,
                states,
            )
            self.assertNear(foo, x ** 2 + rot.data[3])
            self.assertNear(bar, constants.epsilon + sm.sin(y) + x ** 2)

            # Clean up
            if logger.level != logging.DEBUG:
                python_util.remove_if_exists(codegen_data["output_dir"])

    def test_function_codegen_python(self) -> None:

        # Create the specification
        az_el_codegen = Codegen.function(
            name="az_el_from_point", func=az_el_from_point, mode=CodegenMode.PYTHON2,
        )
        az_el_codegen_data = az_el_codegen.generate_function()

        # Compare to expected
        expected_code_file = os.path.join(TEST_DATA_DIR, "az_el_from_point.py")
        output_function = os.path.join(
            az_el_codegen_data["python_function_dir"], "az_el_from_point.py"
        )
        self.compare_or_update_file(expected_code_file, output_function)

        # Clean up
        if logger.level != logging.DEBUG:
            python_util.remove_if_exists(az_el_codegen_data["output_dir"])

    # -------------------------------------------------------------------------
    # C++
    # -------------------------------------------------------------------------

    @slow_on_sympy
    def test_codegen_cpp(self) -> None:
        """
        Test C++ code generation.
        """
        inputs, outputs = self.build_values()

        for scalar_type in ("double", "float"):
            cpp_func = Codegen(
                inputs, outputs, CodegenMode.CPP, "CodegenCppTest", scalar_type=scalar_type
            )
            shared_types = {
                "values_vec": "values_vec_t",
                "values_vec_out": "values_vec_t",
                "values_vec_2D": "values_vec_t",
                "values_vec_2D_out": "values_vec_t",
            }
            namespace = "codegen_cpp_test"
            codegen_data = cpp_func.generate_function(
                shared_types=shared_types, namespace=namespace
            )

            if scalar_type == "double":
                self.compare_or_update_directory(
                    actual_dir=os.path.join(codegen_data["output_dir"]),
                    expected_dir=os.path.join(TEST_DATA_DIR, namespace + "_data"),
                )

                if not self.UPDATE:
                    try:
                        TestCase.compile_and_run_cpp(
                            os.path.join(SYMFORCE_DIR, "test"),
                            "codegen_cpp_test",
                            make_args=(
                                "codegen_cpp_test",
                                f"SYMFORCE_TEST_BACKEND={symforce.get_backend()}",
                            ),
                        )
                    finally:
                        if logger.level != logging.DEBUG:
                            python_util.remove_if_exists(
                                os.path.join(SYMFORCE_DIR, "test", "codegen_cpp_test")
                            )
                            python_util.remove_if_exists(
                                os.path.join(SYMFORCE_DIR, "test", "libsymforce_geo.so")
                            )

            # Clean up
            if logger.level != logging.DEBUG:
                python_util.remove_if_exists(codegen_data["output_dir"])

    def test_function_codegen_cpp(self) -> None:

        # Create the specification
        az_el_codegen = Codegen.function(
            name="AzElFromPoint", func=az_el_from_point, mode=CodegenMode.CPP,
        )
        az_el_codegen_data = az_el_codegen.generate_function()

        # Compare to expected
        expected_code_file = os.path.join(TEST_DATA_DIR, "az_el_from_point.h")
        output_function = os.path.join(az_el_codegen_data["cpp_function_dir"], "az_el_from_point.h")
        self.compare_or_update_file(expected_code_file, output_function)

        # Clean up
        if logger.level != logging.DEBUG:
            python_util.remove_if_exists(az_el_codegen_data["output_dir"])

    def test_cpp_nan(self) -> None:
        inputs = Values()
        inputs["R1"] = geo.Rot3.symbolic("R1")
        inputs["e"] = sm.Symbol("e")
        dist_to_identity = geo.M(
            inputs["R1"].local_coordinates(geo.Rot3.identity(), epsilon=inputs["e"])
        ).squared_norm()
        dist_D_R1 = dist_to_identity.diff(inputs["R1"].q.w)  # type: ignore

        namespace = "codegen_nan_test"
        cpp_func = Codegen(
            name="IdentityDistJacobian",
            inputs=inputs,
            outputs=Values(dist_D_R1=dist_D_R1),
            return_key="dist_D_R1",
            mode=CodegenMode.CPP,
            scalar_type="double",
        )
        codegen_data = cpp_func.generate_function(namespace=namespace)

        # Compare the function file
        self.compare_or_update_directory(
            actual_dir=os.path.join(codegen_data["output_dir"]),
            expected_dir=os.path.join(TEST_DATA_DIR, namespace + "_data"),
        )

        if not self.UPDATE:
            try:
                TestCase.compile_and_run_cpp(
                    package_dir=os.path.join(SYMFORCE_DIR, "test"),
                    executable_names="codegen_nan_test",
                    make_args=(
                        "codegen_nan_test",
                        f"SYMFORCE_TEST_BACKEND={symforce.get_backend()}",
                    ),
                )
            finally:
                if logger.level != logging.DEBUG:
                    python_util.remove_if_exists(
                        os.path.join(SYMFORCE_DIR, "test", "codegen_nan_test")
                    )
                    python_util.remove_if_exists(
                        os.path.join(SYMFORCE_DIR, "test", "libsymforce_geo.so")
                    )
        # Clean up
        if logger.level != logging.DEBUG:
            python_util.remove_if_exists(codegen_data["output_dir"])

    @slow_on_sympy
    def test_multi_function_codegen_cpp(self) -> None:
        inputs, outputs_1 = self.build_values()
        outputs_2 = Values()
        outputs_2["foo"] = inputs["y"] ** 3 + inputs["x"]

        cpp_func_1 = Codegen(
            name="CodegenMultiFunctionTest1",
            inputs=Values(inputs=inputs),
            outputs=Values(outputs_1=outputs_1),
            mode=CodegenMode.CPP,
        )
        cpp_func_2 = Codegen(
            name="CodegenMultiFunctionTest2",
            inputs=Values(inputs=inputs),
            outputs=Values(outputs_2=outputs_2),
            mode=CodegenMode.CPP,
        )

        namespace = "codegen_multi_function_test"
        output_dir = tempfile.mkdtemp(prefix="sf_codegen_multiple_functions_", dir="/tmp")
        logger.debug(f"Creating temp directory: {output_dir}")

        shared_types = {
            "inputs.values_vec": "values_vec_t",
            "outputs_1.values_vec_out": "values_vec_t",
            "inputs.values_vec_2D": "values_vec_t",
            "outputs_1.values_vec_2D_out": "values_vec_t",
        }
        cpp_func_1.generate_function(
            output_dir=output_dir, shared_types=shared_types, namespace=namespace
        )
        shared_types["inputs"] = namespace + ".inputs_t"
        cpp_func_2.generate_function(
            output_dir=output_dir, shared_types=shared_types, namespace=namespace
        )

        # Compare the generated types
        self.compare_or_update_directory(
            output_dir, expected_dir=os.path.join(TEST_DATA_DIR, namespace + "_data"),
        )

        if not self.UPDATE:
            try:
                TestCase.compile_and_run_cpp(
                    package_dir=os.path.join(SYMFORCE_DIR, "test"),
                    executable_names="codegen_multi_function_test",
                    make_args=(
                        "codegen_multi_function_test",
                        f"SYMFORCE_TEST_BACKEND={symforce.get_backend()}",
                    ),
                )
            finally:
                if logger.level != logging.DEBUG:
                    python_util.remove_if_exists(
                        os.path.join(SYMFORCE_DIR, "test", "codegen_multi_function_test")
                    )
                    python_util.remove_if_exists(
                        os.path.join(SYMFORCE_DIR, "test", "libsymforce_geo.so")
                    )

        # Clean up
        if logger.level != logging.DEBUG:
            python_util.remove_if_exists(output_dir)

    @slow_on_sympy
    def test_sparse_matrix_codegen(self) -> None:
        """
        Tests:
            Generation of code that outputs a sparse matrix
        """
        # Start with a dense matrix
        dim = 100
        inputs = Values()
        inputs["matrix_in"] = geo.Matrix(dim, dim).symbolic("mat")

        # Output a sparse matrix
        outputs = Values()
        outputs["matrix_out"] = geo.Matrix(dim, dim).zero()
        for i in range(dim):
            outputs["matrix_out"][i, i] = inputs["matrix_in"][i, i]

        namespace = "codegen_sparse_matrix_test"
        output_dir = tempfile.mkdtemp(prefix="sf_codegen_multiple_functions_", dir="/tmp")
        logger.debug(f"Creating temp directory: {output_dir}")

        # Function that creates a sparse matrix
        get_sparse_func = Codegen(
            name="GetDiagonalSparse",
            inputs=inputs,
            outputs=outputs,
            return_key="matrix_out",
            mode=CodegenMode.CPP,
            scalar_type="double",
            sparse_matrices=["matrix_out"],
        )
        get_sparse_func_data = get_sparse_func.generate_function(
            output_dir=output_dir, namespace=namespace
        )

        # Function that generates both dense and sparse outputs
        multiple_outputs = Values()
        multiple_outputs["sparse_first"] = 2 * outputs["matrix_out"]
        multiple_outputs["dense"] = 3 * geo.Matrix44.matrix_identity()
        multiple_outputs["sparse_second"] = 4 * outputs["matrix_out"]
        multiple_outputs["result"] = geo.Matrix33().zero()

        get_dense_and_sparse_func = Codegen(
            name="GetMultipleDenseAndSparse",
            inputs=inputs,
            outputs=multiple_outputs,
            return_key="result",
            mode=CodegenMode.CPP,
            scalar_type="double",
            sparse_matrices=["sparse_first", "sparse_second"],
        )
        get_dense_and_sparse_func_data = get_dense_and_sparse_func.generate_function(
            output_dir=output_dir, namespace=namespace
        )

        # Function that updates sparse matrix without copying
        update_spase_func = Codegen(
            name="UpdateSparseMat",
            inputs=inputs,
            outputs=Values(updated_mat=2 * outputs["matrix_out"]),
            mode=CodegenMode.CPP,
            scalar_type="double",
            sparse_matrices=["updated_mat"],
        )
        update_spase_func_data = update_spase_func.generate_function(
            output_dir=output_dir, namespace=namespace
        )

        # Compare the function files
        self.compare_or_update_directory(
            output_dir, expected_dir=os.path.join(TEST_DATA_DIR, namespace + "_data"),
        )

        if not self.UPDATE:
            try:
                TestCase.compile_and_run_cpp(
                    package_dir=os.path.join(SYMFORCE_DIR, "test"),
                    executable_names="codegen_sparse_matrix_test",
                    make_args=(
                        "codegen_sparse_matrix_test",
                        f"SYMFORCE_TEST_BACKEND={symforce.get_backend()}",
                    ),
                )
            finally:
                if logger.level != logging.DEBUG:
                    python_util.remove_if_exists(
                        os.path.join(SYMFORCE_DIR, "test", "codegen_sparse_matrix_test")
                    )
        # Clean up
        if logger.level != logging.DEBUG:
            python_util.remove_if_exists(get_sparse_func_data["output_dir"])
            python_util.remove_if_exists(update_spase_func_data["output_dir"])

    def test_invalid_codegen_raises(self) -> None:
        """
        Tests:
            Codegen outputs must be a function of given inputs
            Codegen input/output names must be valid variable names
        """
        # Outputs have symbols not present in inputs
        x = sm.Symbol("x")
        y = sm.Symbol("y")
        inputs = Values(input=x)
        outputs = Values(output=x + y)
        self.assertRaises(AssertionError, Codegen, "test", inputs, outputs, CodegenMode.CPP)

        # Inputs or outputs have keys that aren't valid variable names
        invalid_name_values = Values()
        invalid_name_values["1"] = x
        valid_name_values = Values(x=x)
        self.assertRaises(
            AssertionError, Codegen, "test", invalid_name_values, valid_name_values, CodegenMode.CPP
        )
        self.assertRaises(
            AssertionError, Codegen, "test", valid_name_values, invalid_name_values, CodegenMode.CPP
        )
        name_with_spaces = Values()
        name_with_spaces[" spa ces "] = x
        self.assertRaises(
            AssertionError, Codegen, "test", name_with_spaces, valid_name_values, CodegenMode.CPP
        )
        self.assertRaises(
            AssertionError, Codegen, "test", valid_name_values, name_with_spaces, CodegenMode.CPP
        )

        # Inputs have non-unique symbols
        inputs = Values(in_x=x, in_y=x)
        outputs = Values(out_x=x)
        self.assertRaises(AssertionError, Codegen, "test", inputs, outputs, CodegenMode.CPP)

        # Inputs and outputs have non-unique keys
        inputs = Values(x=x)
        outputs = Values(x=x)
        self.assertRaises(AssertionError, Codegen, "test", inputs, outputs, CodegenMode.CPP)

    def test_name_deduction(self) -> None:
        """
        Tests:
            Codegen.function must create the right name for Python and C++
            Codegen.function should assert on trying to deduce the name from a lambda
        """

        def my_function(x: T.Scalar) -> T.Scalar:
            return x

        self.assertEqual(
            Codegen.function(func=my_function, mode=CodegenMode.PYTHON2).name, "my_function"
        )

        self.assertEqual(
            Codegen.function(func=my_function, mode=CodegenMode.CPP).name, "MyFunction"
        )

        # Can't automagically deduce name for lambda
        self.assertRaises(
            AssertionError,
            Codegen.function,
            func=lambda x: x,
            input_types=[T.Scalar],
            mode=CodegenMode.CPP,
        )

    def test_create_with_derivatives(self) -> None:
        """
        Tests:
            Codegen.create_with_derivatives

        TODO(aaron): Test STACKED_JACOBIAN and FULL_LINEARIZATION modes
        """
        output_dir = tempfile.mkdtemp(prefix="sf_codegen_create_with_derivatives_", dir="/tmp")

        # Let's pick Pose3 compose
        cls = geo.Pose3
        codegen = Codegen.function(
            name="Compose" + cls.__name__,
            func=ops.GroupOps.compose,
            input_types=[cls, cls],
            mode=CodegenMode.CPP,
        )

        codegens = collections.OrderedDict()

        # By default should return the value and have jacobians for each input arg
        codegens["value_and_all_jacs"] = codegen.create_with_derivatives()

        # All jacobians, no value - should return jacobians as output args
        codegens["all_jacs"] = codegen.create_with_derivatives(include_result=False)

        # First jacobian, no value - should return the jacobian
        codegens["jac_0"] = codegen.create_with_derivatives([0], include_result=False)

        # Second jacobian, no value - should return the jacobian
        codegens["jac_1"] = codegen.create_with_derivatives([1], include_result=False)

        # Value and first jacobian - should return the value
        codegens["value_and_jac_0"] = codegen.create_with_derivatives([0], include_result=True)

        # Value and second jacobian - should return the value
        codegens["value_and_jac_1"] = codegen.create_with_derivatives([1], include_result=True)

        # Generate all
        for codegen in codegens.values():
            codegen.generate_function(output_dir=output_dir)

        self.compare_or_update_directory(
            actual_dir=os.path.join(output_dir, "cpp/symforce/sym"),
            expected_dir=os.path.join(TEST_DATA_DIR, "create_with_derivatives"),
        )


if __name__ == "__main__":
    TestCase.main()
