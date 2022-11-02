# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import sys
from pathlib import Path

from symforce import typing as T
from symforce.codegen import codegen_util
from symforce.test_util import TestCase

PKG_LOCATIONS = Path(__file__).parent / "test_data" / "codegen_util_test_data"
RELATIVE_PATH = Path("example_pkg", "__init__.py")
PACKAGE_NAME = "example_pkg"


class SymforceCodegenUtilTest(TestCase):
    """
    Tests contents of symforce.codegen.codgen_util.
    """

    def test_load_generated_package(self) -> None:
        """
        Tests:
            codegen_util.load_generated_package
        """

        pkg1 = codegen_util.load_generated_package(
            name=PACKAGE_NAME, path=PKG_LOCATIONS / "example_pkg_1" / RELATIVE_PATH
        )

        # Testing that the module was loaded correctly
        self.assertEqual(pkg1.package_id, 1)
        self.assertEqual(pkg1.sub_module.sub_module_id, 1)

        # Testing that sys.modules was not polluted
        self.assertFalse(PACKAGE_NAME in sys.modules)

        pkg2 = codegen_util.load_generated_package(
            name=PACKAGE_NAME, path=PKG_LOCATIONS / "example_pkg_2" / RELATIVE_PATH
        )

        # Testing that the module was loaded correctly when a module with the same name has
        # already been loaded
        self.assertEqual(pkg2.package_id, 2)
        self.assertEqual(pkg2.sub_module.sub_module_id, 2)

    def test_load_generated_function(self) -> None:
        """
        Tests:
            codegen_util.load_generated_function
        """

        func1 = codegen_util.load_generated_function(
            func_name="func", path_to_package=PKG_LOCATIONS / "example_pkg_1" / RELATIVE_PATH
        )

        # Testing that the function was loaded correctly
        self.assertEqual(func1(), 1)

        # Testing that sys.modules was not polluted
        self.assertFalse(PACKAGE_NAME in sys.modules)

        func2 = codegen_util.load_generated_function(
            func_name="func", path_to_package=PKG_LOCATIONS / "example_pkg_2" / RELATIVE_PATH
        )

        # Testing that the function was loaded correctly when a function of the same name
        # has already been loaded.
        self.assertEqual(func2(), 2)


if __name__ == "__main__":
    TestCase.main()
