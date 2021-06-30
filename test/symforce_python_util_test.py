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


if __name__ == "__main__":
    TestCase.main()
