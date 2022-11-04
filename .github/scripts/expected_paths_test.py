#!/usr/bin/env python3

# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import importlib
import os
import unittest


class SymforceExpectedPathsTest(unittest.TestCase):
    """
    Regression test that symforce-related libraries importable from expected paths in CI

    Not a symforce unit test, run specifically from test_editable_pip_install.yml
    """

    def check_module_location(self, module_name: str) -> None:
        module = importlib.import_module(module_name)
        self.assertEqual(
            module.__file__, os.environ[f"{module_name.upper().replace('.', '_')}_LOCATION"]
        )

    def test_cc_sym(self) -> None:
        self.check_module_location("cc_sym")

    def test_sym(self) -> None:
        self.check_module_location("sym")

    def test_skymarshal(self) -> None:
        self.check_module_location("skymarshal")

    def test_symengine(self) -> None:
        self.check_module_location("symengine")

    def test_lcmtypes_sym(self) -> None:
        self.check_module_location("lcmtypes.sym")

    def test_lcmtypes_eigen_lcm(self) -> None:
        self.check_module_location("lcmtypes.eigen_lcm")

    def test_sf_sympy(self) -> None:
        import symforce.symbolic as sf

        self.assertEqual(sf.sympy.__file__, os.environ["SF_SYMPY_LOCATION"])

    def test_symforce_api(self) -> None:
        import symforce

        self.assertEqual(symforce.get_symbolic_api(), "symengine")


if __name__ == "__main__":
    unittest.main()
