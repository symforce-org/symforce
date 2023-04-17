# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from symforce import path_util
from symforce import typing as T
from symforce.codegen import Codegen
from symforce.codegen.backends.pytorch import PyTorchConfig
from symforce.test_util import TestCase
from symforce.test_util.backend_coverage_expressions import backend_test_function

TEST_DATA_DIR = (
    path_util.symforce_data_root()
    / "test"
    / "symforce_function_codegen_test_data"
    / symforce.get_symbolic_api()
    / "symforce_pytorch_codegen_test"
)


class SymforcePyTorchCodegenTest(TestCase):
    """
    Tests code generation with the pytorch backend
    """

    def test_codegen(self) -> None:
        output_dir = self.make_output_dir("symforce_pytorch_codegen_test_")

        def pytorch_func(
            a: sf.Scalar, b: sf.V1, c: sf.V3, d: sf.M22, e: sf.V5, f: sf.M66
        ) -> T.Tuple[sf.Scalar, sf.V1, sf.V3, sf.M22, sf.V5, sf.M66]:
            return a, b, c, d, e, f

        output_names = ("a_out", "b_out", "c_out", "d_out", "e_out", "f_out")

        # Generate a function for all combinations of argument types
        Codegen.function(
            pytorch_func, config=PyTorchConfig(), name="pytorch_func", output_names=output_names
        ).generate_function(output_dir, skip_directory_nesting=True)

        # Generate the symbolic backend test function
        Codegen.function(
            backend_test_function, config=PyTorchConfig(), name="backend_test_function"
        ).generate_function(output_dir, skip_directory_nesting=True)

        self.compare_or_update_directory(output_dir, TEST_DATA_DIR)


if __name__ == "__main__":
    SymforcePyTorchCodegenTest.main()
