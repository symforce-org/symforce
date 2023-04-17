# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_symbol()

import itertools

import symforce.symbolic as sf
from symforce import path_util
from symforce import typing as T
from symforce.codegen import Codegen
from symforce.codegen import CodeGenerationException
from symforce.codegen.backends.cuda import CudaConfig
from symforce.codegen.backends.cuda import ScalarType
from symforce.test_util import TestCase
from symforce.test_util.backend_coverage_expressions import backend_test_function

TEST_DATA_DIR = (
    path_util.symforce_data_root()
    / "test"
    / "symforce_function_codegen_test_data"
    / symforce.get_symbolic_api()
    / "symforce_cuda_codegen_test"
)


class SymforceCudaCodegenTest(TestCase):
    """
    Tests code generation with the CUDA backend
    """

    def test_codegen(self) -> None:
        output_dir = self.make_output_dir("symforce_cuda_codegen_test_")

        def cuda_func(
            a: sf.Scalar, b: sf.V1, c: sf.V3, d: sf.M22, e: sf.V5, f: sf.M66, g: sf.DataBuffer
        ) -> T.Tuple[sf.Scalar, sf.V1, sf.V3, sf.M22, sf.V5, sf.M66]:
            return a, b, c, d, e, f + g[0]

        output_names = ("a_out", "b_out", "c_out", "d_out", "e_out", "f_out")

        scalars = (ScalarType.FLOAT, ScalarType.DOUBLE)
        inlines = (False, True)
        builtin_vector_variableses: T.Tuple[T.Set[str], ...] = (
            set(),
            {"b"},
            {"b", "c", "d", "b_out", "c_out", "d_out"},
        )

        # Generate functions for all combinations of scalars, inline, and argument types
        for scalar, inline, builtin_vector_variables in itertools.product(
            scalars, inlines, builtin_vector_variableses
        ):
            config = CudaConfig(
                scalar_type=scalar, inline=inline, builtin_vector_variables=builtin_vector_variables
            )
            name = f"cuda_func_{scalar.value}_{inline}_{'_'.join(sorted(builtin_vector_variables)) or 'empty'}"
            Codegen.function(
                cuda_func, config=config, name=name, output_names=output_names
            ).generate_function(output_dir, skip_directory_nesting=True)

        # Using CUDA vectors for arguments that are too big should fail
        with self.assertRaises(CodeGenerationException):
            Codegen.function(
                cuda_func,
                config=CudaConfig(builtin_vector_variables={"e"}),
                output_names=output_names,
            ).generate_function(output_dir, skip_directory_nesting=True)

        with self.assertRaises(CodeGenerationException):
            Codegen.function(
                cuda_func,
                config=CudaConfig(builtin_vector_variables={"f"}),
                output_names=output_names,
            ).generate_function(output_dir, skip_directory_nesting=True)

        # Generate functions for all combinations of return key
        for return_key in (None, "a_out", "b_out", "d_out"):
            Codegen.function(
                cuda_func,
                config=CudaConfig(
                    inline=False, builtin_vector_variables={"b_out", "c_out", "d_out"}
                ),
                name=f"cuda_func_vectors_return_{return_key}",
                output_names=output_names,
                return_key=return_key,
            ).generate_function(output_dir, skip_directory_nesting=True)

        # Returning 1) by pointer or 2) things that are too large for CUDA vectors should fail
        for return_key in ("b_out", "d_out", "e_out"):
            with self.assertRaises(CodeGenerationException):
                Codegen.function(
                    cuda_func,
                    config=CudaConfig(),
                    name=f"cuda_func_pointers_return_{return_key}",
                    output_names=output_names,
                    return_key=return_key,
                ).generate_function(output_dir, skip_directory_nesting=True)

        # Generate the symbolic backend test function
        for scalar in scalars:
            Codegen.function(
                backend_test_function,
                config=CudaConfig(inline=False, scalar_type=scalar),
                name=f"backend_test_function_{scalar.value}",
            ).generate_function(output_dir, skip_directory_nesting=True)

        self.compare_or_update_directory(output_dir, TEST_DATA_DIR)


if __name__ == "__main__":
    SymforceCudaCodegenTest.main()
