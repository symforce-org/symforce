from pathlib import Path
from dataclasses import dataclass
import subprocess
import re
import os

from symforce.codegen import template_util
from symforce.test_util import TestCase
from symforce import typing as T
from symforce import path_util


class SymforceCCSymStubsCodegenTest(TestCase):
    def cc_sym_stubgen_output(self) -> str:
        """
        Returns the contents of the stub file produced by pybind11-stubgen on module cc_sym
        """
        output_dir = Path(self.make_output_dir("sf_cc_sym_stubgen_output"))

        subprocess.check_call(
            [
                "pybind11-stubgen",
                "--bare-numpy-ndarray",
                "--no-setup-py",
                "-o",
                output_dir,
                "cc_sym",
            ],
            env=dict(
                os.environ,
                PYTHONPATH=os.pathsep.join(
                    [os.environ.get("PYTHONPATH", ""), os.fspath(path_util.cc_sym_install_dir())]
                ),
            ),
        )
        generated_file = output_dir / "cc_sym-stubs" / "__init__.pyi"

        return generated_file.read_text()

    def test_generate_cc_sym_stubs(self) -> None:
        output_dir = Path(self.make_output_dir("sf_cc_sym_stubs_codegen_test"))

        # Change return type of Values.at to be Any rather than object
        stubgen_output = re.sub(
            r"(def at\([^)]*\) ->) object", r"\1 typing.Any", self.cc_sym_stubgen_output()
        )

        @dataclass
        class TypeStubParts:
            lcm_include_type_names: T.List[str]
            sym_include_type_names: T.List[str]
            third_party_includes: T.List[str]
            cleaned_up_stubgen_output: str

        pybind_dir = path_util.symforce_dir() / "symforce" / "pybind"

        template_util.render_template(
            template_path=str(pybind_dir / "cc_sym.pyi.jinja"),
            data={
                "spec": TypeStubParts(
                    lcm_include_type_names=[
                        "key_t",
                        "index_entry_t",
                        "index_t",
                        "linearized_dense_factor_t",
                        "optimization_iteration_t",
                        "optimization_stats_t",
                        "optimizer_params_t",
                        "values_t",
                    ],
                    sym_include_type_names=["Rot2", "Rot3", "Pose2", "Pose3"],
                    third_party_includes=["import scipy"],
                    cleaned_up_stubgen_output=stubgen_output,
                )
            },
            output_path=str(output_dir / "cc_sym.pyi"),
            template_dir=str(pybind_dir),
        )

        self.compare_or_update_file(
            new_file=output_dir / "cc_sym.pyi", path=pybind_dir / "cc_sym.pyi",
        )


if __name__ == "__main__":
    TestCase.main()
