import functools
import importlib.util
import logging
import numpy as np
import sys
import os
import unittest

import symforce
from symforce import cam
from symforce import geo
from symforce import logger
from symforce import ops
from symforce import python_util
from symforce import sympy as sm
from symforce import typing as T
from symforce import codegen
from symforce.codegen import geo_package_codegen
from symforce.codegen import codegen_util
from symforce.test_util import TestCase, slow_on_sympy, symengine_only
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
            python_func = codegen.Codegen(
                inputs=inputs,
                outputs=outputs,
                config=codegen.PythonConfig(),
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

            output_dir = self.make_output_dir("sf_codegen_python_test_")

            codegen_data = python_func.generate_function(
                shared_types=shared_types, namespace=namespace, output_dir=output_dir
            )
            if scalar_type == "double":
                self.compare_or_update_directory(
                    actual_dir=output_dir,
                    expected_dir=os.path.join(TEST_DATA_DIR, namespace + "_data"),
                )

            geo_package_codegen.generate(config=codegen.PythonConfig(), output_dir=output_dir)

            geo_pkg = codegen_util.load_generated_package(
                "sym", os.path.join(output_dir, "sym", "__init__.py")
            )
            values_vec_t = codegen_util.load_generated_lcmtype(
                namespace, "values_vec_t", os.path.join(codegen_data["python_types_dir"])
            )

            states_t = codegen_util.load_generated_lcmtype(
                namespace, "states_t", os.path.join(codegen_data["python_types_dir"])
            )

            constants_t = codegen_util.load_generated_lcmtype(
                namespace, "constants_t", os.path.join(codegen_data["python_types_dir"])
            )

            x = 2.0
            y = -5.0
            rot = geo_pkg.Rot3()
            rot_vec = [geo_pkg.Rot3(), geo_pkg.Rot3(), geo_pkg.Rot3()]
            scalar_vec = [1.0, 2.0, 3.0]
            list_of_lists = [rot_vec, rot_vec, rot_vec]
            values_vec = [
                values_vec_t(),
                values_vec_t(),
                values_vec_t(),
            ]
            values_vec_2D = [[values_vec_t()], [values_vec_t()]]

            states = states_t()
            states.p = [1.0, 2.0]
            constants = constants_t()
            constants.epsilon = 1e-8

            gen_module = codegen_util.load_generated_package(
                namespace, codegen_data["python_function_dir"]
            )
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

    def test_function_codegen_python(self) -> None:
        output_dir = self.make_output_dir("sf_codegen_function_codegen_python_")

        # Create the specification
        az_el_codegen = codegen.Codegen.function(
            func=az_el_from_point, config=codegen.PythonConfig()
        )
        az_el_codegen_data = az_el_codegen.generate_function(output_dir)

        # Compare to expected
        expected_code_file = os.path.join(TEST_DATA_DIR, "az_el_from_point.py")
        output_function = os.path.join(
            az_el_codegen_data["python_function_dir"], "az_el_from_point.py"
        )
        self.compare_or_update_file(expected_code_file, output_function)

    @unittest.skipIf(importlib.util.find_spec("numba") is None, "Requires numba")
    def test_function_codegen_python_numba(self) -> None:
        output_dir = self.make_output_dir("sf_codegen_numba_")

        # Create the specification
        def numba_test_func(x: geo.V3) -> geo.V2:
            return geo.V2(x[0, 0], x[1, 0])

        numba_test_func_codegen = codegen.Codegen.function(
            func=numba_test_func, config=codegen.PythonConfig(use_numba=True)
        )
        numba_test_func_codegen_data = numba_test_func_codegen.generate_function(output_dir)

        # Compare to expected
        expected_code_file = os.path.join(TEST_DATA_DIR, "numba_test_func.py")
        output_function = os.path.join(
            numba_test_func_codegen_data["python_function_dir"], "numba_test_func.py"
        )
        self.compare_or_update_file(expected_code_file, output_function)

        gen_module = codegen_util.load_generated_package(
            "sym", numba_test_func_codegen_data["python_function_dir"]
        )

        x = np.array([1, 2, 3])
        y = gen_module.numba_test_func(x)
        self.assertTrue((y == np.array([1, 2])).all())
        self.assertTrue(hasattr(gen_module.numba_test_func, "__numba__"))

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
            cpp_func = codegen.Codegen(
                inputs, outputs, codegen.CppConfig(), "codegen_cpp_test", scalar_type=scalar_type
            )
            shared_types = {
                "values_vec": "values_vec_t",
                "values_vec_out": "values_vec_t",
                "values_vec_2D": "values_vec_t",
                "values_vec_2D_out": "values_vec_t",
            }
            namespace = "codegen_cpp_test"
            output_dir = self.make_output_dir(f"sf_codegen_cpp_{scalar_type}_")
            codegen_data = cpp_func.generate_function(
                shared_types=shared_types, namespace=namespace, output_dir=output_dir
            )

            if scalar_type == "double":
                self.compare_or_update_directory(
                    actual_dir=os.path.join(output_dir),
                    expected_dir=os.path.join(TEST_DATA_DIR, namespace + "_data"),
                )

    def test_function_codegen_cpp(self) -> None:
        output_dir = self.make_output_dir("sf_codegen_function_codegen_cpp_")

        # Create the specification
        az_el_codegen = codegen.Codegen.function(func=az_el_from_point, config=codegen.CppConfig())
        az_el_codegen_data = az_el_codegen.generate_function(output_dir=output_dir)

        # Compare to expected
        expected_code_file = os.path.join(TEST_DATA_DIR, "az_el_from_point.h")
        output_function = os.path.join(az_el_codegen_data["cpp_function_dir"], "az_el_from_point.h")
        self.compare_or_update_file(expected_code_file, output_function)

    def test_cpp_nan(self) -> None:
        inputs = Values()
        inputs["R1"] = geo.Rot3.symbolic("R1")
        inputs["e"] = sm.Symbol("e")
        dist_to_identity = geo.M(
            inputs["R1"].local_coordinates(geo.Rot3.identity(), epsilon=inputs["e"])
        ).squared_norm()
        dist_D_R1 = dist_to_identity.diff(inputs["R1"].q.w)  # type: ignore

        namespace = "codegen_nan_test"
        output_dir = self.make_output_dir("sf_codegen_cpp_nan_")
        cpp_func = codegen.Codegen(
            name="identity_dist_jacobian",
            inputs=inputs,
            outputs=Values(dist_D_R1=dist_D_R1),
            return_key="dist_D_R1",
            config=codegen.CppConfig(),
            scalar_type="double",
        )
        codegen_data = cpp_func.generate_function(namespace=namespace, output_dir=output_dir)

        # Compare the function file
        self.compare_or_update_directory(
            actual_dir=os.path.join(codegen_data["output_dir"]),
            expected_dir=os.path.join(TEST_DATA_DIR, namespace + "_data"),
        )

    @slow_on_sympy
    def test_multi_function_codegen_cpp(self) -> None:
        inputs, outputs_1 = self.build_values()
        outputs_2 = Values()
        outputs_2["foo"] = inputs["y"] ** 3 + inputs["x"]

        cpp_func_1 = codegen.Codegen(
            name="codegen_multi_function_test1",
            inputs=Values(inputs=inputs),
            outputs=Values(outputs_1=outputs_1),
            config=codegen.CppConfig(),
        )
        cpp_func_2 = codegen.Codegen(
            name="codegen_multi_function_test2",
            inputs=Values(inputs=inputs),
            outputs=Values(outputs_2=outputs_2),
            config=codegen.CppConfig(),
        )

        namespace = "codegen_multi_function_test"
        output_dir = self.make_output_dir("sf_codegen_multiple_functions_")

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
        output_dir = self.make_output_dir("sf_codegen_multiple_functions_")

        # Function that creates a sparse matrix
        get_sparse_func = codegen.Codegen(
            name="get_diagonal_sparse",
            inputs=inputs,
            outputs=outputs,
            return_key="matrix_out",
            config=codegen.CppConfig(),
            scalar_type="double",
            sparse_matrices=["matrix_out"],
        )
        get_sparse_func_data = get_sparse_func.generate_function(
            output_dir=output_dir, namespace=namespace
        )

        # Function that generates both dense and sparse outputs
        multiple_outputs = Values()
        multiple_outputs["sparse_first"] = 2 * outputs["matrix_out"]
        multiple_outputs["dense"] = 3 * geo.Matrix44.eye()
        multiple_outputs["sparse_second"] = 4 * outputs["matrix_out"]
        multiple_outputs["result"] = geo.Matrix33().zero()

        get_dense_and_sparse_func = codegen.Codegen(
            name="get_multiple_dense_and_sparse",
            inputs=inputs,
            outputs=multiple_outputs,
            return_key="result",
            config=codegen.CppConfig(),
            scalar_type="double",
            sparse_matrices=["sparse_first", "sparse_second"],
        )
        get_dense_and_sparse_func_data = get_dense_and_sparse_func.generate_function(
            output_dir=output_dir, namespace=namespace
        )

        # Function that updates sparse matrix without copying
        update_spase_func = codegen.Codegen(
            name="update_sparse_mat",
            inputs=inputs,
            outputs=Values(updated_mat=2 * outputs["matrix_out"]),
            config=codegen.CppConfig(),
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
        self.assertRaises(
            AssertionError, codegen.Codegen, "test", inputs, outputs, codegen.CppConfig()
        )

        # Inputs have non-unique symbols
        inputs = Values(in_x=x, in_y=x)
        outputs = Values(out_x=x)
        self.assertRaises(
            AssertionError, codegen.Codegen, "test", inputs, outputs, codegen.CppConfig()
        )

        # Inputs and outputs have non-unique keys
        inputs = Values(x=x)
        outputs = Values(x=x)
        self.assertRaises(
            AssertionError, codegen.Codegen, "test", inputs, outputs, codegen.CppConfig()
        )

    def test_name_deduction(self) -> None:
        """
        Tests:
            Codegen.function must create the right name, which should be snake_case regardless of
                language
            Codegen.function should assert on trying to deduce the name from a lambda
        """

        def my_function(x: T.Scalar, y: T.Scalar) -> T.Scalar:
            return x

        self.assertEqual(
            codegen.Codegen.function(func=my_function, config=codegen.PythonConfig()).name,
            "my_function",
        )

        self.assertEqual(
            codegen.Codegen.function(func=my_function, config=codegen.CppConfig()).name,
            "my_function",
        )

        # Should still work with functools.partial
        self.assertEqual(
            codegen.Codegen.function(
                func=functools.partial(my_function, y=2), config=codegen.CppConfig()
            ).name,
            "my_function",
        )

        # Can't automagically deduce name for lambda
        self.assertRaises(
            AssertionError,
            codegen.Codegen.function,
            func=lambda x: x,
            input_types=[T.Scalar],
            config=codegen.CppConfig(),
        )

    def test_with_jacobians(self) -> None:
        """
        Tests:
            Codegen.with_jacobians

        TODO(aaron): Test STACKED_JACOBIAN and FULL_LINEARIZATION modes
        """
        output_dir = self.make_output_dir("sf_codegen_with_jacobians_")

        # Let's pick Pose3 compose
        cls = geo.Pose3
        codegen_function = codegen.Codegen.function(
            name="compose_" + cls.__name__.lower(),
            func=ops.GroupOps.compose,
            input_types=[cls, cls],
            config=codegen.CppConfig(),
        )

        codegens = {}

        # By default should return the value and have jacobians for each input arg
        codegens["value_and_all_jacs"] = codegen_function.with_jacobians()

        # All jacobians, no value - should return jacobians as output args
        codegens["all_jacs"] = codegen_function.with_jacobians(include_results=False)

        # First jacobian, no value - should return the jacobian
        codegens["jac_0"] = codegen_function.with_jacobians(["a"], include_results=False)

        # Second jacobian, no value - should return the jacobian
        codegens["jac_1"] = codegen_function.with_jacobians(["b"], include_results=False)

        # Value and first jacobian - should return the value
        codegens["value_and_jac_0"] = codegen_function.with_jacobians(["a"], include_results=True)

        # Value and second jacobian - should return the value
        codegens["value_and_jac_1"] = codegen_function.with_jacobians(["b"], include_results=True)

        # Generate all
        for codegen_function in codegens.values():
            codegen_function.generate_function(output_dir=output_dir)

        self.assertEqual(
            len(os.listdir(os.path.join(output_dir, "cpp/symforce/sym"))), len(codegens)
        )

        self.compare_or_update_directory(
            actual_dir=os.path.join(output_dir, "cpp/symforce/sym"),
            expected_dir=os.path.join(TEST_DATA_DIR, "with_jacobians"),
        )

    # This test generates a lot of code, and it isn't really valuable to test on sympy as well
    # because it's very unlikely to have differences there, so we only do it for symengine
    @symengine_only
    def test_with_jacobians_multiple_outputs(self) -> None:
        """
        Tests:
            Codegen.with_jacobians

        Test with_jacobians with multiple outputs
        """
        output_dir = self.make_output_dir("sf_codegen_with_jacobians_multiple_outputs_")

        # Let's make a simple function with two outputs
        def cross_and_distance(
            a: geo.V3, b: geo.V3, epsilon: T.Scalar
        ) -> T.Tuple[geo.V3, T.Scalar]:
            return a.cross(b), (a - b).norm(epsilon=epsilon)

        codegen_function = codegen.Codegen.function(
            func=cross_and_distance,
            config=codegen.CppConfig(),
            output_names=["cross", "distance"],
            return_key="cross",
        )

        # -------------------------------------------------------------------
        # Jacobians w.r.t. first output
        # -------------------------------------------------------------------
        specs_first: T.List[T.Dict[str, T.Any]] = [
            # By default should return the value and have jacobians for the first output (cross) wrt
            # each input arg
            dict(args=dict(), return_key="cross", outputs=5),
            # All jacobians for first output, no value - should return distance and jacobians for
            # cross as output args
            dict(args=dict(include_results=False), return_key=None, outputs=4),
            # First jacobian for first output, no value - should return distance and jacobian as
            # output args
            dict(args=dict(which_args=["a"], include_results=False), return_key=None, outputs=2),
            # Second jacobian for first output, no value - should return distance and jacobian as
            # output args
            dict(args=dict(which_args=["b"], include_results=False), return_key=None, outputs=2),
            # Value and first jacobian for first output - should return the value
            dict(args=dict(which_args=["a"], include_results=True), return_key="cross", outputs=3),
            # Value and second jacobian for first output - should return the value
            dict(args=dict(which_args=["b"], include_results=True), return_key="cross", outputs=3),
        ]

        # -------------------------------------------------------------------
        # Jacobians w.r.t. second output
        # -------------------------------------------------------------------
        specs_second: T.List[T.Dict[str, T.Any]] = [
            # Should return the value and have jacobians for the second output (distance) wrt each
            # input arg
            dict(args=dict(), return_key="cross", outputs=5),
            # All jacobians for second output, no value - should return cross, and jacobians for
            # distance as output args
            dict(args=dict(include_results=False), return_key="cross", outputs=4),
            # First jacobian for second output, no value - should return cross, and jacobian as
            # output arg
            dict(args=dict(which_args=["a"], include_results=False), return_key="cross", outputs=2),
            # Second jacobian for second output, no value - should return cross, and jacobian as
            # output arg
            dict(args=dict(which_args=["b"], include_results=False), return_key="cross", outputs=2),
            # Value and first jacobian for second output - should return cross
            dict(args=dict(which_args=["a"], include_results=True), return_key="cross", outputs=3),
            # Value and second jacobian for second output - should return cross
            dict(args=dict(which_args=["b"], include_results=True), return_key="cross", outputs=3),
        ]

        # -------------------------------------------------------------------
        # Jacobians w.r.t. both outputs
        # -------------------------------------------------------------------
        specs_both: T.List[T.Dict[str, T.Any]] = [
            # Should return the value and have jacobians for both outputs wrt each input arg
            dict(args=dict(), return_key="cross", outputs=8),
            # All jacobians for both outputs, no values - should return jacobians as output args
            dict(args=dict(include_results=False), return_key=None, outputs=6),
            # First jacobian for both outputs, no values - should return jacobians as output args
            dict(args=dict(which_args=["a"], include_results=False), return_key=None, outputs=2),
            # Second jacobian for both outputs, no values - should return jacobians as output args
            dict(args=dict(which_args=["b"], include_results=False), return_key=None, outputs=2),
            # Value and first jacobian for both outputs - should return cross
            dict(args=dict(which_args=["a"], include_results=True), return_key="cross", outputs=4),
            # Value and second jacobian for both outputs - should return cross
            dict(args=dict(which_args=["b"], include_results=True), return_key="cross", outputs=4),
        ]

        specs = (
            specs_first
            + [
                dict(
                    raw_spec,
                    base_name="cross_and_distance_second",
                    args=dict(**raw_spec["args"], which_results=[1]),
                )
                for raw_spec in specs_second
            ]
            + [
                dict(
                    raw_spec,
                    base_name="cross_and_distance_both",
                    args=dict(**raw_spec["args"], which_results=[0, 1]),
                )
                for raw_spec in specs_both
            ]
        )

        for spec in specs:
            with self.subTest(spec=spec):
                if "base_name" in spec:
                    codegen_function.name = spec["base_name"]

                curr_codegen = codegen_function.with_jacobians(**spec["args"])
                self.assertEqual(curr_codegen.return_key, spec["return_key"])
                self.assertEqual(len(curr_codegen.outputs), spec["outputs"])

                curr_codegen.generate_function(output_dir=output_dir)

        self.assertEqual(len(os.listdir(os.path.join(output_dir, "cpp/symforce/sym"))), len(specs))

        self.compare_or_update_directory(
            actual_dir=os.path.join(output_dir, "cpp/symforce/sym"),
            expected_dir=os.path.join(TEST_DATA_DIR, "with_jacobians_multiple_outputs"),
        )


if __name__ == "__main__":
    TestCase.main()
