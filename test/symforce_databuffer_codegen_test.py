# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from pathlib import Path

import numpy as np

import symforce

symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from symforce import codegen
from symforce import typing as T
from symforce.codegen import codegen_util
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
from symforce.test_util import TestCase
from symforce.values import Values

CURRENT_DIR = Path(__file__).parent
SYMFORCE_DIR = CURRENT_DIR.parent
TEST_DATA_DIR = SYMFORCE_DIR.joinpath(
    "test", "symforce_function_codegen_test_data", symforce.get_symbolic_api()
)


class SymforceDataBufferCodegenTest(TestCase):
    """
    Test databuffer codegen
    """

    def gen_code(self, output_dir: str) -> None:
        a, b = sf.symbols("a b")
        # make sure Databuffer works with whatever namestring works for symbol
        buffer = sf.DataBuffer("foo.Buffer")
        result = buffer[(a + b) * (b - a)] + buffer[b * b - a * a] + (a + b)

        inputs = Values()

        inputs["buffer"] = buffer
        inputs["a"] = a
        inputs["b"] = b

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
        buffer_func = codegen_util.load_generated_function(
            "buffer_func",
            py_codegen_data.function_dir,
        )

        a_numeric = 1.0
        b_numeric = 2.0
        buffer_numeric = np.array([0, 1, 2, 3])
        # 2 * buffer[b^2 - a^2] + (a+b)
        # 2 * buffer[3] + 3
        expected = 9
        result_numeric = buffer_func(buffer_numeric, a_numeric, b_numeric)

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
        def buffer_residual(x: sf.Scalar, left_bound: sf.Scalar, buffer: sf.DataBuffer) -> sf.V1:
            shifted_x = x - left_bound
            lower_idx = sf.floor(shifted_x)
            upper_idx = lower_idx + 1

            a1 = shifted_x - sf.floor(shifted_x)
            a0 = 1 - a1
            return sf.V1(a0 * buffer[lower_idx] + a1 * buffer[upper_idx])

        factors = [Factor(keys=["x", "left_bound", "buffer"], residual=buffer_residual)]
        optimizer = Optimizer(factors=factors, optimized_keys=["x"])
        initial_values = Values(epsilon=sf.numeric_epsilon)
        initial_values["left_bound"] = -2
        initial_values["buffer"] = np.array([-2, -1, 0, 1, 2])
        initial_values["x"] = 1.5

        result = optimizer.optimize(initial_values)

        self.assertAlmostEqual(result.optimized_values["x"], 0, places=9)
        self.assertAlmostEqual(result.error(), 0, places=9)


if __name__ == "__main__":
    TestCase.main()
