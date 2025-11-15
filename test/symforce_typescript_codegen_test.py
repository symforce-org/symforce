# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_symbol()

import functools

import symforce.symbolic as sf
from symforce import path_util
from symforce.codegen import Codegen
from symforce.codegen.backends.typescript.typescript_config import TypeScriptConfig
from symforce.test_util import TestCase
from symforce.test_util.backend_coverage_expressions import backend_test_function

TEST_DATA_DIR = (
    path_util.symforce_data_root(__file__)
    / "test"
    / "symforce_function_codegen_test_data"
    / symforce.get_symbolic_api()
    / "symforce_typescript_codegen_test"
)


class SymforceTypeScriptCodegenTest(TestCase):
    """
    Tests code generation with the TypeScript backend
    """

    def test_codegen(self) -> None:
        def vector_multiply(mat33: sf.M33, vec3: sf.V3) -> sf.V3:
            return sf.V3(mat33 * vec3)

        def matrix_multiply(lhs: sf.M33, rhs: sf.M33) -> sf.M33:
            return sf.M33(lhs * rhs)

        output_dir = self.make_output_dir("symforce_typescript_codegen_test_")

        # Generate the vector/matrix function
        Codegen.function(
            vector_multiply,
            config=TypeScriptConfig(),
        ).generate_function(output_dir, skip_directory_nesting=True)

        # Generate the matrix multiplication function
        Codegen.function(
            matrix_multiply,
            config=TypeScriptConfig(),
        ).generate_function(output_dir, skip_directory_nesting=True)

        # Generate the backend test function
        Codegen.function(
            functools.partial(backend_test_function, []),
            config=TypeScriptConfig(),
            name="backend_test_function",
        ).generate_function(output_dir, skip_directory_nesting=True)

        self.compare_or_update_directory(output_dir, TEST_DATA_DIR)


if __name__ == "__main__":
    SymforceTypeScriptCodegenTest.main()
