# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import inspect

# Unused imports here for testing purposes
import symforce.symbolic as sf
from symforce import geo  # pylint: disable=unused-import
from symforce import typing as T  # pylint: disable=unused-import
from symforce.cam import LinearCameraCal  # pylint: disable=unused-import
from symforce.geo import Pose3  # pylint: disable=unused-import
from symforce.geo import Vector3  # pylint: disable=unused-import
from symforce.test_util import TestCase
from symforce.type_helpers import deduce_input_types
from symforce.typing import Scalar  # pylint: disable=unused-import


class SymforceTypeHelpersTest(TestCase):
    """
    Test type_helpers.py
    """

    def test_deduce_input_types(self) -> None:
        # Lots of unused arguments in here
        # pylint: disable=unused-argument

        # Can deduce self for bound method on geo classes
        assert (
            inspect.signature(sf.Rot3.compose).parameters["self"].annotation
            == inspect.Parameter.empty
        ), "Our test function shouldn't have an annotation on self"
        self.assertEqual(deduce_input_types(sf.Rot3.compose), [sf.Rot3, sf.Rot3])

        # Can't deduce types that aren't annotated
        def my_function_partly_typed(a: sf.Pose3, b) -> None:  # type: ignore
            pass

        self.assertRaises(ValueError, deduce_input_types, my_function_partly_typed)

        # Can deduce types annotated as strings
        def my_function_annotated_with_strings(
            a: "float",
            b: "Scalar",
            c: "Pose3",
            d: "Vector3",
            e: "LinearCameraCal",
            f: "sf.Scalar",
            g: "sf.Pose3",
            h: "sf.Vector3",
            i: "sf.LinearCameraCal",
        ) -> None:
            pass

        expected_types = [
            float,
            float,
            sf.Pose3,
            sf.Vector3,
            sf.LinearCameraCal,
            float,
            sf.Pose3,
            sf.Vector3,
            sf.LinearCameraCal,
        ]

        self.assertEqual(deduce_input_types(my_function_annotated_with_strings), expected_types)

        # Fails for nonexistant types annotated as 2-part strings in expected modules
        def my_function_annotated_with_something_that_doesnt_exist(
            a: "sf.Foo",  # type: ignore
        ) -> None:
            pass

        self.assertRaises(
            AttributeError,
            deduce_input_types,
            my_function_annotated_with_something_that_doesnt_exist,
        )

        # Works for a string with more than 2 parts
        def my_function_with_a_multipart_string_annotation(a: "geo.matrix.Vector3") -> None:
            pass

        self.assertEqual(
            deduce_input_types(my_function_with_a_multipart_string_annotation), [sf.V3]
        )


if __name__ == "__main__":
    TestCase.main()
