# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import sys

from symforce import typing as T
from symforce.test_util import TestCase


def clear_symforce() -> None:
    """
    Removes symforce modules from sys.modules. Ensures that all symforce
    code will be (re-)executed upon import.
    """
    module_names = list(sys.modules.keys())
    for module in module_names:
        if module.startswith("symforce"):
            del sys.modules[module]


class SymforceSetEpsilonTest(TestCase):
    """
    Test symforce.set_epsilon functions.
    """

    saved_modules: T.List[T.Tuple[str, T.Any]] = []

    @classmethod
    def setUpClass(cls) -> None:
        # NOTE(brad): This is necessary because while the test classes are run sequentially, it
        # seems they are all created before any are run. That means they can save references to
        # modules which might get deleted and reloaded by clear_symforce.
        #
        # Since some code relies on the original module still being available in sys.modules, we
        # need to save and restore the original modules to keep from breaking other tests.
        cls.saved_modules = []
        for module_name, module in sys.modules.items():
            if module_name.startswith("symforce"):
                cls.saved_modules.append((module_name, module))

    @classmethod
    def tearDownClass(cls) -> None:
        clear_symforce()

        for key, module in cls.saved_modules:
            sys.modules[key] = module

    def test_set_epsilon_to_zero(self) -> None:
        """
        Assumes symforce.set_epsilon_to_number works.

        Tests:
            symforce.set_epsilon_to_zero
        """
        clear_symforce()
        with self.subTest(msg="Test set_epsilon_to_zero()"):
            import symforce

            symforce.set_epsilon_to_number(4.4)
            symforce.set_epsilon_to_zero()
            import symforce.symbolic as sf

            self.assertEqual(0.0, sf.epsilon())

        clear_symforce()
        with self.subTest(
            msg="Test function does not raise on setting epsilon to the current value"
        ):
            import symforce
            import symforce.symbolic as sf

            sf.epsilon()
            symforce.set_epsilon_to_zero()

    def test_set_epsilon_to_symbol(self) -> None:
        """
        Assumes symforce.set_epsilon_to_number works.

        Tests:
            symforce.set_epsilon_to_symbol
        """
        clear_symforce()
        with self.subTest(msg="Test set_epsilon_to_symbol()"):
            import symforce

            symforce.set_epsilon_to_number(4.4)
            symforce.set_epsilon_to_symbol()
            import symforce.symbolic as sf

            self.assertEqual(sf.Symbol("epsilon"), sf.epsilon())

        clear_symforce()
        with self.subTest(msg="Test set_epsilon_to_symbol(name)"):
            import symforce

            symforce.set_epsilon_to_number(4.4)
            symforce.set_epsilon_to_symbol(name="alpha")
            import symforce.symbolic as sf

            self.assertEqual(sf.Symbol("alpha"), sf.epsilon())

        clear_symforce()
        with self.subTest(msg="Test function properly raises AlreadyUsedEpsilon exception"):
            import symforce

            with self.assertRaises(symforce.AlreadyUsedEpsilon):
                import symforce.symbolic as sf

                sf.epsilon()
                symforce.set_epsilon_to_number()

    def test_set_epsilon_to_number(self) -> None:
        """
        Tests:
            symforce.set_epsilon_to_number
        """
        clear_symforce()
        with self.subTest(msg="Test set_epsilon_to_number()"):
            import symforce

            symforce.set_epsilon_to_number()
            import symforce.symbolic as sf

            self.assertEqual(sf.numeric_epsilon, sf.epsilon())

        clear_symforce()
        with self.subTest(msg="Test set_epsilon_to_number(value)"):
            import symforce

            symforce.set_epsilon_to_number(value=4.4)
            import symforce.symbolic as sf

            self.assertEqual(4.4, sf.epsilon())

        clear_symforce()
        with self.subTest(msg="Test function properly raises AlreadyUsedEpsilon exception"):
            import symforce

            with self.assertRaises(symforce.AlreadyUsedEpsilon):
                import symforce.symbolic as sf

                sf.epsilon()
                symforce.set_epsilon_to_symbol()


if __name__ == "__main__":
    TestCase.main()
