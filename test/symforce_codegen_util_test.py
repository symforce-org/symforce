# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from pathlib import Path
import sys

from symforce import typing as T
from symforce.test_util import TestCase
from symforce.codegen import codegen_util


class SymforceCodegenUtilTest(TestCase):
    """
    Tests contents of symforce.codegen.codgen_util.
    """

    def test_load_generated_package(self) -> None:
        """
        Tests:
            codegen_util.load_generated_package
        """

        pkg_locations = Path(__file__).parent / "test_data" / "codegen_util_test_data"

        relative_path = Path("example_pkg", "__init__.py")

        package_name = "example_pkg"

        pkg1 = codegen_util.load_generated_package(
            name=package_name, path=pkg_locations / "example_pkg_1" / relative_path
        )

        # Testing that the module was loaded correctly
        self.assertEqual(pkg1.package_id, 1)
        self.assertEqual(pkg1.sub_module.sub_module_id, 1)

        # Testing that sys.modules was not polluted
        self.assertFalse(package_name in sys.modules)

        pkg2 = codegen_util.load_generated_package(
            name=package_name, path=pkg_locations / "example_pkg_2" / relative_path
        )

        # Testing that the module was loaded correctly when a module with the same name has
        # already been loaded
        self.assertEqual(pkg2.package_id, 2)
        self.assertEqual(pkg2.sub_module.sub_module_id, 2)


if __name__ == "__main__":
    TestCase.main()
