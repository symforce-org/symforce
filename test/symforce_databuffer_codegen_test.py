# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
import numpy as np
from pathlib import Path

import symforce
from symforce import codegen
from symforce import sympy as sm
from symforce.codegen import codegen_util
from symforce.values import Values
from symforce import geo
from symforce.test_util import TestCase
from symforce import typing as T
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer

CURRENT_DIR = Path(__file__).parent
SYMFORCE_DIR = CURRENT_DIR.parent
TEST_DATA_DIR = SYMFORCE_DIR.joinpath(
    "test", "symforce_function_codegen_test_data", symforce.get_backend()
)


class SymforceDataBufferCodegenTest(TestCase):
    """
    Test databuffer codegen
    """

    def gen_code(self, output_dir: str) -> None:
        a, b = sm.symbols("a b")
        # make sure Databuffer works with whatever namestring works for symbol
        buffer = sm.DataBuffer("foo.Buffer")
        result = buffer[(a + b) * (b - a)] + buffer[b * b - a * a] + (a + b)

        inputs = Values()

        inputs["a"] = a
        inputs["b"] = b

        inputs["buffer"] = buffer

        outputs = Values(result=result)

        # default namespace is sym, which causes issues w/ linting when it's in the test folder
        namespace = "buffer_test"
        buffer_func_cpp = codegen.Codegen(
            inputs=inputs,
            outputs=outputs,
            config=codegen.CppConfig(),
            name="buffer_func",
            return_key="result",
        )

        buffer_func_cpp.generate_function(output_dir=output_dir, namespace=namespace)

        buffer_func_py = codegen.Codegen(
            inputs=inputs,
            outputs=outputs,
            config=codegen.PythonConfig(),
            name="buffer_func",
            return_key="result",
        )

        py_codegen_data = buffer_func_py.generate_function(
            output_dir=output_dir, namespace=namespace
        )

        # Also test that the generated python code runs
        gen_module = codegen_util.load_generated_package(
            "buffer_func",
            py_codegen_data["python_function_dir"],
        )

        a_numeric = 1.0
        b_numeric = 2.0
        buffer_numeric = np.array([0, 1, 2, 3])
        # 2 * buffer[b^2 - a^2] + (a+b)
        # 2 * buffer[3] + 3
        expected = 9
        result_numeric = gen_module.buffer_func(a_numeric, b_numeric, buffer_numeric)

        self.assertStorageNear(expected, result_numeric)

    def test_databuffer_codegen(self) -> None:
        output_dir = self.make_output_dir("sf_databuffer_codegen_test")

        self.gen_code(output_dir)

        self.compare_or_update_directory(
            actual_dir=output_dir,
            expected_dir=TEST_DATA_DIR / "databuffer_codegen_test_data",
        )

    def test_databuffer_factor(self) -> None:
        # make sure the implementation plays nicely with the python machinery

        # sample residual function that's a simple linear interpolation of the databuffer
        # assume that the scale = 1 for convenience
        def buffer_residual(x: T.Scalar, left_bound: T.Scalar, buffer: sm.DataBuffer) -> geo.V1:
            shifted_x = x - left_bound
            lower_idx = sm.floor(shifted_x)
            upper_idx = lower_idx + 1

            a1 = shifted_x - sm.floor(shifted_x)
            a0 = 1 - a1
            return geo.V1(a0 * buffer[lower_idx] + a1 * buffer[upper_idx])

        factors = [Factor(keys=["x", "left_bound", "buffer"], residual=buffer_residual)]
        optimizer = Optimizer(factors=factors, optimized_keys=["x"])
        initial_values = Values(epsilon=sm.default_epsilon)
        initial_values["left_bound"] = -2
        initial_values["buffer"] = np.array([-2, -1, 0, 1, 2])
        initial_values["x"] = 1.5

        result = optimizer.optimize(initial_values)

        self.assertAlmostEqual(result.optimized_values["x"], 0, places=9)
        self.assertAlmostEqual(result.error(), 0, places=9)


if __name__ == "__main__":
    TestCase.main()
