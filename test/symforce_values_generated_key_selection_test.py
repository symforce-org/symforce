# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import string

from symforce.test_util import TestCase
from symforce.values import generated_key_selection


class SymforceValuesGeneratedKeySelectionTest(TestCase):
    """
    Test symforce.values.generated_key_selection
    """

    def test_choices_for_name(self) -> None:
        """
        Tests:
            generated_key_selection._choices_for_name
        """
        # pylint: disable=protected-access

        letters_to_try, sub = generated_key_selection._choices_for_name("foo")
        self.assertSequenceEqual(
            letters_to_try, ["f", "o"] + [l for l in string.ascii_lowercase if l not in "foo"]
        )
        self.assertEqual(sub, None)

        letters_to_try, sub = generated_key_selection._choices_for_name("foo2")
        self.assertSequenceEqual(
            letters_to_try, ["f", "o"] + [l for l in string.ascii_lowercase if l not in "foo"]
        )
        self.assertEqual(sub, 2)

        letters_to_try, sub = generated_key_selection._choices_for_name("foo_bar")
        self.assertSequenceEqual(
            letters_to_try,
            ["f", "b", "o", "a", "r"] + [l for l in string.ascii_lowercase if l not in "foobar"],
        )
        self.assertEqual(sub, None)

        letters_to_try, sub = generated_key_selection._choices_for_name("foo203_45")
        self.assertSequenceEqual(
            letters_to_try, ["f", "o"] + [l for l in string.ascii_lowercase if l not in "foo"]
        )
        self.assertEqual(sub, 203)

        letters_to_try, sub = generated_key_selection._choices_for_name("foo_-203_45")
        self.assertSequenceEqual(
            letters_to_try, ["f", "o"] + [l for l in string.ascii_lowercase if l not in "foo"]
        )
        self.assertEqual(sub, -203)


if __name__ == "__main__":
    TestCase.main()
