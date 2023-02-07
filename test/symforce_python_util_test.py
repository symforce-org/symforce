# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce import python_util
from symforce.test_util import TestCase


class SymforceUtilTest(TestCase):
    """
    Tests python_util.py
    """

    def test_snakecase_to_camelcase(self) -> None:
        """
        Tests:
            python_util.snakecase_to_camelcase
        """
        self.assertEqual(python_util.snakecase_to_camelcase("easy_peasy"), "EasyPeasy")
        self.assertEqual(
            python_util.snakecase_to_camelcase("a__little___harder"), "A_Little_Harder"
        )
        self.assertEqual(
            python_util.snakecase_to_camelcase("__why____are_______you_doing__this___"),
            "_Why__Are___YouDoing_This_",
        )

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

        with self.subTest(
            "Raises InvalidKeyError if non-index characters exist between/after indices"
        ):
            with self.assertRaises(python_util.InvalidKeyError):
                python_util.base_and_indices("arr[1].bad[2]")
            with self.assertRaises(python_util.InvalidKeyError):
                python_util.base_and_indices("arr[1].bad")

        with self.subTest("Raises InvalidKeyError if floating point index"):
            with self.assertRaises(python_util.InvalidKeyError):
                python_util.base_and_indices("arr[2.0]")

        with self.subTest("Raises InvalidKeyError on indices missing/extra brackets"):
            for malformed_index in ["arr[11", "arr11]", "arr[[1]", "arr[1]]"]:
                with self.assertRaises(python_util.InvalidKeyError):
                    python_util.base_and_indices(malformed_index)


if __name__ == "__main__":
    TestCase.main()
