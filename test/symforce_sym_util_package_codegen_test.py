# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import os

from symforce.test_util import TestCase
from symforce import codegen
from symforce.codegen import sym_util_package_codegen


SYMFORCE_DIR = os.path.dirname(os.path.dirname(__file__))


class SymforceSymUtilCodegenTest(TestCase):
    """
    Generate C++ utils
    """

    def test_codegen_cpp(self) -> None:
        """
        Generate typedefs.h
        """
        output_dir = self.make_output_dir("sf_opt_codegen_test_")

        sym_util_package_codegen.generate(config=codegen.CppConfig(), output_dir=output_dir)

        self.compare_or_update_directory(
            actual_dir=os.path.join(output_dir, "sym", "util"),
            expected_dir=os.path.join(SYMFORCE_DIR, "gen", "cpp", "sym", "util"),
        )


if __name__ == "__main__":
    TestCase.main()
