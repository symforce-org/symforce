# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from pathlib import Path

from symforce import codegen
from symforce import geo
from symforce import logger
from symforce import path_util
from symforce import sympy as sm
from symforce import typing as T
from symforce.test_util import TestCase


class SymforceJavascriptCodegenTest(TestCase):
    """
    Simple test for the Javascript codegen backend.
    """

    @staticmethod
    def javascript_codegen_example(
        a: T.Scalar, b: geo.V2, c: geo.M22, epsilon: T.Scalar
    ) -> T.Tuple[geo.V3, geo.M22, T.Scalar]:
        return (
            geo.V3(a + c[0], sm.sin(b[0]) ** a, b[1] ** 2 / (a - b[0] - c[1] - epsilon)),
            geo.M22(
                [[-sm.atan2(b[1], a), (a + b[0]) / c[1, :].norm(epsilon=epsilon)], [1, c[1, 0]]]
            ),
            a ** 2,
        )

    def test_javascript_codegen(self) -> None:
        for config in (codegen.PythonConfig(), codegen.CppConfig(), codegen.JavascriptConfig()):
            cg = codegen.Codegen.function(
                func=self.javascript_codegen_example,
                config=config,
                output_names=["d", "e", "f"],
            )
            out_path = cg.generate_function().generated_files[0]

            logger.debug(Path(out_path).read_text())

            if config.backend_name() == "javascript":
                self.compare_or_update_file(
                    path=path_util.symforce_dir() / "test" / "test_data" / out_path.name,
                    new_file=out_path,
                )


if __name__ == "__main__":
    TestCase.main()
