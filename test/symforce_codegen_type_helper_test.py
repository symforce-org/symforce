import inspect
from symforce import cam
from symforce.cam import LinearCameraCal
from symforce import geo
from symforce.geo import Pose3, Vector3
from symforce import typing as T
from symforce.typing import Scalar
from symforce.codegen.type_helper import deduce_input_types
from symforce.test_util import TestCase


class SymforceCodegenTypeHelperTest(TestCase):
    """
    Test type_helper.py
    """

    def test_deduce_input_types(self) -> None:
        # Can deduce self for bound method on geo classes
        assert (
            inspect.signature(geo.Rot3.compose).parameters["self"].annotation
            == inspect.Parameter.empty
        ), "Our test function shouldn't have an annotation on self"
        self.assertEqual(deduce_input_types(geo.Rot3.compose), [geo.Rot3, geo.Rot3])

        # Can't deduce types that aren't annotated
        def my_function_partly_typed(a: geo.Pose3, b) -> None:  # type: ignore
            pass

        self.assertRaises(ValueError, deduce_input_types, my_function_partly_typed)

        # Can deduce types annotated as strings
        def my_function_annotated_with_strings(
            a: "float",
            b: "Scalar",
            c: "Pose3",
            d: "Vector3",
            e: "LinearCameraCal",
            f: "T.Scalar",
            g: "geo.Pose3",
            h: "geo.Vector3",
            i: "cam.LinearCameraCal",
        ) -> None:
            pass

        expected_types = [
            float,
            float,
            geo.Pose3,
            geo.Vector3,
            cam.LinearCameraCal,
            float,
            geo.Pose3,
            geo.Vector3,
            cam.LinearCameraCal,
        ]

        self.assertEqual(deduce_input_types(my_function_annotated_with_strings), expected_types)

        # Fails for nonexistant types annotated as 2-part strings in expected modules
        def my_function_annotated_with_something_that_doesnt_exist(
            a: "geo.Foo",  # type: ignore
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
            deduce_input_types(my_function_with_a_multipart_string_annotation), [geo.V3]
        )


if __name__ == "__main__":
    TestCase.main()
