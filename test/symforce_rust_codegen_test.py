# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_symbol()

import functools
import shutil
import subprocess
import textwrap
from pathlib import Path

import symforce.symbolic as sf
from symforce import path_util
from symforce.codegen import Codegen
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
)


def write_cargo_toml(output_dir: Path) -> None:
    """Write the Cargo.toml file for the generated code."""
    (output_dir / "Cargo.toml").write_text(
        textwrap.dedent(
            """
            [package]
            name = "symforce_rust_codegen_test"
            version = "0.1.0"
            edition = "2021"

            [dependencies]
            nalgebra = "0"
            """
        )
    )


def write_lib_rs(output_dir: Path) -> None:
    """Write the lib.rs file that includes the generated functions."""
    (output_dir / "lib.rs").write_text(
        textwrap.dedent(
            """
            mod backend_test_function_float32;
            mod backend_test_function_float64;
            mod vector_matrix_fun;
            """
        )
    )


def cargo_clean(output_dir: Path) -> None:
    """Clean up at files that are generated by cargo build

    Note that 'cargo clean' doesn't clean the lock file so we do this manually.
    """
    lock_file = TEST_DATA_DIR / "Cargo.lock"
    if lock_file.exists():
        lock_file.unlink()

    rustc_json = TEST_DATA_DIR / ".rustc_info.json"
    if rustc_json.exists():
        rustc_json.unlink()

    # Remove the target directory
    target_dir = TEST_DATA_DIR / "target"
    if target_dir.exists():
        shutil.rmtree(target_dir)


class SymforceRustCodegenTest(TestCase):
    """
    Tests code generation with the Rust backend
    """

    def test_codegen(self) -> None:
        def rust_func(vec3: sf.V3, mat33: sf.M33) -> sf.Matrix31:
            return sf.Matrix31(mat33 * vec3)

        output_dir_base = self.make_output_dir("symforce_rust_codegen_test_")
        output_dir_src = output_dir_base / "src"

        scalars = (ScalarType.FLOAT, ScalarType.DOUBLE)

        # Generate the symbolic backend test function
        Codegen.function(
            rust_func,
            config=RustConfig(scalar_type=ScalarType.DOUBLE),
            name="vector_matrix_fun",
        ).generate_function(output_dir_src, skip_directory_nesting=True)

        # Generate the symbolic backend test function
        for scalar in scalars:
            Codegen.function(
                functools.partial(backend_test_function, []),
                config=RustConfig(scalar_type=scalar),
                name=f"backend_test_function_{scalar.value}",
            ).generate_function(output_dir_src, skip_directory_nesting=True)

        write_cargo_toml(output_dir_base)
        write_lib_rs(output_dir_src)

        self.compare_or_update_directory(output_dir_base, TEST_DATA_DIR)

        # Ideally we would build inside the temporary directory, but cargo build fails there for some reason
        # likely to do with symlinks. Instead we build in the test data directory and make sure that we clean up
        # after ourselves.
        self.cargo_build(TEST_DATA_DIR)

    def cargo_build(self, output_dir: Path) -> None:
        """Run cargo build in the output directory

        Note that this fails when called from within a temporary directory!
        """

        # Check if cargo is installed and skip the test if it isn't.
        cargo = shutil.which("cargo")
        if cargo is None:
            return

        result = subprocess.run(  # pylint: disable=subprocess-run-check
            ["cargo", "build", "--manifest-path", output_dir / "Cargo.toml"],
            capture_output=True,
            text=True,
        )

        # Always clean up after ourselves even if the build fails.
        cargo_clean(output_dir)

        if result.returncode != 0:
            self.fail(f"cargo build failed:\n{result.stderr}")


if __name__ == "__main__":
    SymforceRustCodegenTest.main()