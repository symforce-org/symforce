import unittest

from symforce import types as T
from symforce.test_util import TestCase
from symforce import python_util


class SymforceUtilTest(TestCase):
    """
    Tests python_util.py
    """

    def test_getattr_recursive(self) -> None:
        """
        Tests:
            python_util.getattr_recursive
        """

        class Obj:
            class A:
                class B:
                    class C:
                        pass

        with self.subTest(msg="Returns obj when attrs is empty"):
            self.assertEqual(Obj, python_util.getattr_recursive(Obj, []))

        with self.subTest(msg="Returns correct attribute when attrs is a tuple"):
            self.assertEqual(Obj.A.B.C, python_util.getattr_recursive(Obj, ("A", "B", "C")))

    def test_base_and_indices(self) -> None:
        """
        Tests:
            python_util.base_and_indices
        """
        with self.subTest("Correct output for name with no indices"):
            self.assertEqual(("arr", []), python_util.base_and_indices("arr"))

        with self.subTest("Correct output for name with one index"):
            self.assertEqual(("arr", [3]), python_util.base_and_indices("arr[3]"))

        with self.subTest("Correct output for name with multiple indices"):
            self.assertEqual(("arr", [1, 2, 3]), python_util.base_and_indices("arr[1][2][3]"))

        with self.subTest("Correct output for multi-digit indices"):
            self.assertEqual(("arr", [11, 22, 33]), python_util.base_and_indices("arr[11][22][33]"))

        with self.subTest("A string with no base (only indices) is accepted"):
            self.assertEqual(("", [1, 2, 3]), python_util.base_and_indices("[1][2][3]"))

        with self.subTest("Raises ValueError if non-index characters exist between/after indices"):
            with self.assertRaises(ValueError):
                python_util.base_and_indices("arr[1].bad[2]")
            with self.assertRaises(ValueError):
                python_util.base_and_indices("arr[1].bad")

        with self.subTest("Raises ValueError if floating point index"):
            with self.assertRaises(ValueError):
                python_util.base_and_indices("arr[2.0]")

        with self.subTest("Raises ValueError on indices missing/extra brackets"):
            with self.assertRaises(ValueError):
                python_util.base_and_indices("arr[11")
            with self.assertRaises(ValueError):
                python_util.base_and_indices("arr11]")
            with self.assertRaises(ValueError):
                python_util.base_and_indices("arr[[1]")
            with self.assertRaises(ValueError):
                python_util.base_and_indices("arr[1]]")


if __name__ == "__main__":
    TestCase.main()
