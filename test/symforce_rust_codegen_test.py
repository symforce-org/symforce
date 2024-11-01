# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_symbol()

import itertools
import functools

import symforce.symbolic as sf
from symforce import path_util
from symforce import typing as T
from symforce.codegen import Codegen
from symforce.codegen import CodeGenerationException
from symforce.codegen.backends.rust import RustConfig
from symforce.codegen.backends.rust import ScalarType
from symforce.test_util import TestCase
from symforce.test_util.backend_coverage_expressions import backend_test_function

TEST_DATA_DIR = (
    path_util.symforce_data_root()
    / "test"
    / "symforce_function_codegen_test_data"
    / symforce.get_symbolic_api()
    / "symforce_rust_codegen_test"
    / "src"
)


class SymforceRustCodegenTest(TestCase):
    """
    Tests code generation with the Rust backend
    """

    def test_codegen(self) -> None:
        output_dir = self.make_output_dir("symforce_rust_codegen_test_")

        scalars = (ScalarType.FLOAT, ScalarType.DOUBLE)

        # Generate the symbolic backend test function
        for scalar in scalars:
            Codegen.function(
                functools.partial(backend_test_function, ()),
                config=RustConfig(scalar_type=scalar),
                name=f"backend_test_function_{scalar.value}",
            ).generate_function(output_dir, skip_directory_nesting=True)

        self.compare_or_update_directory(output_dir, TEST_DATA_DIR)


if __name__ == "__main__":
    SymforceRustCodegenTest.main()
