import numpy as np
import itertools
import unittest

from symforce import logger
from symforce import geo
from symforce import cam
from symforce import ops
from symforce import sympy as sm
from symforce.python_util import InvalidKeyError
from symforce.test_util import TestCase, slow_on_sympy
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin
from symforce.values import Values


class SymforceValuesTest(LieGroupOpsTestMixin, TestCase):
    """
    Test symforce.values.Values.
    """

    # Our test Values contains a Pose3 for which this is not true
    MANIFOLD_IS_DEFINED_IN_TERMS_OF_GROUP_OPS = False

    @classmethod
    def element(cls) -> Values:
        v = Values()
        v["float"] = 3.0
        v["rot3"] = geo.Rot3.from_tangent(np.random.normal(size=(3,)).tolist())
        v["pose3"] = geo.Pose3.from_tangent(np.random.normal(size=(6,)).tolist())
        other_values = v.copy()
        v["values"] = other_values
        v["vec_values"] = [other_values, other_values]
        v["vec_rot3"] = [
            geo.Rot3.from_tangent(np.random.normal(size=(3,)).tolist()),
            geo.Rot3.from_tangent(np.random.normal(size=(3,)).tolist()),
        ]
        return v

    def test_as_ordered_dict(self) -> None:
        # TODO(nathan): Disallow adding strings as elements? Certain functions break with string elements
        v = Values(z=5, bar="foo")
        self.assertEqual(v["z"], 5)
        self.assertEqual(v["bar"], "foo")

        keys = v.keys()
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], "z")

        values = v.values()
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0], 5)

        items = v.items()
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0], ("z", 5))

        self.assertEqual(v.get("bar"), "foo")
        self.assertEqual(v.get("baz"), None)
        self.assertEqual(v.get("baz", 15), 15)

        blah = dict(apple=33, pear=55)
        v.update(blah)
        self.assertEqual(v["apple"], 33)
        self.assertEqual(v["pear"], 55)
        self.assertEqual(len(v.keys()), 4)

        self.assertTrue("apple" in v)
        self.assertFalse("orange" in v)
        self.assertFalse("orange.sky" in v)
        self.assertFalse("orange.sky.beneath" in v)

        v["apple"] = 42
        self.assertEqual(v["apple"], 42)

        del v["apple"]
        self.assertFalse("apple" in v)
        self.assertEqual(len(v.keys()), 3)

        string = repr(v)
        logger.debug("v:\n" + string)

    def test_name_scope(self) -> None:
        s = sm.Symbol("foo.blah")
        self.assertEqual("foo.blah", s.name)

        with sm.scope("hey"):
            v = sm.Symbol("you")
            with sm.scope("there"):
                w = sm.Symbol("what")
        self.assertEqual("hey.you", v.name)
        self.assertEqual("hey.there.what", w.name)

        with self.subTest(msg="Scopes are cleaned up correctly"):
            try:
                with sm.scope("hey"):
                    raise Exception
            except:
                pass
            x = sm.Symbol("who")
            self.assertEqual("who", x.name)
        sm.__scopes__ = []

    def test_values(self) -> None:
        v = Values()
        self.assertEqual(len(v.keys()), 0)

        # items/keys/values_recursive work well even on empty Values
        self.assertEqual([], v.items_recursive())
        self.assertEqual([], v.keys_recursive())
        self.assertEqual([], v.values_recursive())

        v["foo"] = sm.Symbol("foo")
        v.add(sm.Symbol("bar"))
        self.assertEqual("foo", v["foo"].name)
        self.assertTrue("foo" in v.keys())

        # Add only works with strings and symbols
        self.assertRaises(NameError, lambda: v.add(3))

        v["complex"] = geo.Complex.identity()

        v["other"] = Values(blah=3)
        v["rot"] = geo.Rot2.identity()
        self.assertEqual(geo.Rot2(), v["rot"])

        v["mat"] = geo.Matrix([[1, 2], [3, 4]])

        # Test adding with scope
        with v.scope("states"):
            v["x0"] = sm.Symbol("x0")
            with sm.scope("vel"):
                v.add("v0")
                v.add("v1")
            v.add("y0")
        self.assertEqual("states.x0", v["states.x0"].name)

        self.assertEqual(v["states.x0"], sm.Symbol("states.x0"))

        # Test getting flattened list of elements/keys
        vals = v.values_recursive()
        for i, key in enumerate(v.keys_recursive()):
            self.assertEqual(v[key], vals[i])
        for key, val in v.items_recursive():
            self.assertEqual(v[key], val)

        v2 = v.from_storage_index(v.to_storage(), v.index())
        self.assertEqual(v, v2)

        # Test flattened list of items equal
        self.assertEqual(list(v.items_recursive()), list(v2.items_recursive()))

        # Test attribute access
        x0 = v["states.x0"]
        x0_attr = v.attr.states.x0
        self.assertEqual(x0, x0_attr)

        # Test other containers
        v["arr"] = np.array([1, 2, 3])
        v["list"] = [1, 2, 3]
        v["tuple"] = (4, 5, -6)
        Values.from_storage_index(v.to_storage(), v.index())

        # Unknown type
        class Floop:
            pass

        v["uhoh"] = Floop()
        self.assertRaises(NotImplementedError, v.to_storage)
        del v["uhoh"]

        string = repr(v)
        logger.debug("v:\n" + string)

        # Copy into other values and change element
        new_v = v.copy()
        self.assertEqual(new_v, v)
        v["list"].append(4)
        self.assertNotEqual(new_v, v)

    def test_evalf(self) -> None:
        v = Values()
        v["a"] = sm.S.One / 3
        v["b"] = geo.Rot3.from_angle_axis(angle=sm.pi / 2, axis=geo.V3(1, 0, 0))

        v_evalf = v.evalf()

        logger.debug(v)
        logger.debug(v_evalf)

        self.assertEqual(v["a"].evalf(), v_evalf["a"])
        self.assertNear(v_evalf["a"], 0.3333333, places=6)

    def test_getitem(self) -> None:
        """
        Tests:
            Values.__getitem__
        """
        with self.subTest(msg="Can get values inside nested lists"):
            v = Values(lst=[[1, 2, 3], [4, 5, 6]])
            self.assertEqual(4, v["lst[1][0]"])

        with self.subTest(msg="Can get values insides listed Values"):
            v = Values(lst=[Values(a=1), Values(b=2), Values(c=3)])
            self.assertEqual(2, v["lst[1].b"])

        with self.subTest(msg="Strings which are not python identifiers are not valid keys either"):
            v = Values()
            for invalid_key in ["", "+", "0", "no[1]dot"]:
                with self.assertRaises(InvalidKeyError):
                    v[invalid_key]

    def test_setitem(self) -> None:
        """
        Tests:
            Values.__setitem__
        """
        with self.subTest(msg="Can set an item within a list"):
            v = Values(lst=[1, 2, 3])
            v["lst[1]"] = 0
            v_expected = Values(lst=[1, 0, 3])
            self.assertEqual(v, v_expected)

        with self.subTest(msg="Can set an item within a nested list"):
            v = Values(lst=[[1, 2, 3], [4, 5, 6]])
            v["lst[1][0]"] = 0
            v_expected = Values(lst=[[1, 2, 3], [0, 5, 6]])
            self.assertEqual(v, v_expected)

        with self.subTest(msg="Can set a new item"):
            v = Values()
            v["lst"] = [1, 2, 3]
            v_expected = Values(lst=[1, 2, 3])
            self.assertEqual(v, v_expected)

        with self.subTest(msg="Strings which are not python identifiers are not valid keys either"):
            v = Values()
            for invalid_key in ["", "+", "0", "no[1]dot"]:
                with self.assertRaises(InvalidKeyError):
                    v[invalid_key] = 0

    def test_delitem(self) -> None:
        """
        Tests:
            Values.__delitem__
        """
        with self.subTest(msg="Can delete an element from a list"):
            v = Values(lst=[1, 2, 3])
            del v["lst[1]"]
            v_expected = Values(lst=[1, 3])
            self.assertEqual(v, v_expected)

        with self.subTest(msg="Can delete an element form a nested list"):
            v = Values(lst=[[1, 2, 3], [4, 5, 6]])
            del v["lst[0][2]"]
            v_expected = Values(lst=[[1, 2], [4, 5, 6]])
            self.assertEqual(v, v_expected)

        with self.subTest(msg="Can delete a nested Values"):
            v = Values(nested_v=Values())
            del v["nested_v"]
            v_expected = Values()
            self.assertEqual(v, v_expected)

        with self.subTest(msg="Strings which are not python identifiers are not valid keys either"):
            v = Values()
            for invalid_key in ["", "+", "0", "no[1]dot"]:
                with self.assertRaises(InvalidKeyError):
                    del v[invalid_key]

    def test_contains(self) -> None:
        """
        Tests:
            Values.__contains__
        """
        with self.subTest(msg="Values contains both list keys and their elements"):
            v = Values(lst=[1, 2, 3])
            self.assertTrue("lst" in v)
            self.assertTrue("lst[2]" in v)

        with self.subTest(msg="Values does not contain too large indices"):
            v = Values(lst=[1, 2, 3])
            self.assertFalse("lst[3]" in v)

        with self.subTest(msg="Values contains nested lists"):
            v = Values(nested_lists=[[1, 2, 3], [4, 5, 6]])
            self.assertTrue("nested_lists[1]" in v)
            self.assertTrue("nested_lists[1][2]" in v)

        with self.subTest(msg="Values contains nested Values"):
            v = Values(v_nested=Values(a=1))
            self.assertTrue("v_nested.a" in v)

        with self.subTest(msg="Values contains nested Values in lists"):
            v = Values(v_list=[Values(a=1)])
            self.assertTrue("v_list[0].a" in v)

        with self.subTest(msg="Values does not contain what it does not contain"):
            v = Values(a=1)
            self.assertFalse("b" in v)

        with self.subTest(msg="Strings which are not python identifiers are not valid keys either"):
            v = Values()
            for invalid_key in ["", "+", "0", "no[1]dot"]:
                with self.assertRaises(InvalidKeyError):
                    invalid_key in v

    def test_mixing_scopes(self) -> None:
        v1 = Values()
        v1.add("x")
        with sm.scope("foo"):
            v1.add("x")
        self.assertEqual(v1["foo.x"], sm.Symbol("foo.x"))

        v2 = Values()

        v2.add(sm.Symbol("x"))

        with sm.scope("foo"):
            v2.add("x")
            with v2.scope("bar"):
                v2.add("x")

        v2_expected = Values(
            x=sm.Symbol("x"),
            foo=Values(x=sm.Symbol("foo.x"), bar=Values(x=sm.Symbol("foo.bar.x"))),
        )

        self.assertEqual(v2, v2_expected)

    def test_from_tangent_scalar(self) -> None:
        """
        Ensure `.from_tangent` works with native python types as keys.
        """
        keys = ["a", "b", "c", "d", "e"]
        entries = [2, 1.2, sm.S(3.4), sm.Symbol("x"), 5 * sm.Symbol("x") ** -2]

        v = Values()
        for key, entry in zip(keys, entries):
            # Add correct type but different value
            v[key] = entry + 10

        # Make sure from_tangent and from_storage run and we get correct values back
        self.assertSequenceEqual(list(zip(keys, entries)), list(v.from_tangent(entries).items()))
        self.assertSequenceEqual(list(zip(keys, entries)), list(v.from_storage(entries).items()))

    def test_items_recursive(self) -> None:
        """
        Tests:
            Values.items_recursive
        Ensure that the key item pairs returned by items_recursive are valid key item pairs.
        """

        v = self.element()
        for key, value in v.items_recursive():
            self.assertEqual(v[key], value)

    def test_items_recursive_with_ndarray(self) -> None:
        """
        Tests:
            Values.items_recursive
        Ensure that the keys of ndarrays contain indices for each element
        """
        keys = [key for key, _ in Values(array=np.array([1, 2])).items_recursive()]
        self.assertEqual(["array[0]", "array[1]"], keys)

    def test_values_and_keys_recursive_return_lists(self) -> None:
        """
        Tests:
            Values.values_recursive
            Values.keys_recursive
        Ensure the return types of values_recusive and keys_recursive are both lists
        """
        v = Values(entry=1)
        self.assertIsInstance(v.values_recursive(), list)
        self.assertIsInstance(v.keys_recursive(), list)

    def test_subkeys_recursive(self) -> None:
        """
        Tests:
            Values.subkeys_recursive
        """
        v = Values(
            level1=1, base1a=Values(level2a=2, base2a=Values(level3=3), base1b=Values(level2b=4))
        )
        with self.subTest(msg="Returns the correct values"):
            expected_subkeys = ["level1", "level2a", "level3", "level2b"]
            self.assertEqual(expected_subkeys, v.subkeys_recursive())

        with self.subTest(msg="key order is insertion order of highest level dot seperated key"):
            v = Values()
            v["first_top_level"] = Values()
            v["second_top_level"] = Values(first_inner=2)
            v["first_top_level"]["second_inner"] = 1
            self.assertEqual(["second_inner", "first_inner"], v.subkeys_recursive())

    def test_scalar_keys_recursive(self) -> None:
        """
        Tests:
            Values.scalar_keys_recursive
        """
        with self.subTest(msg="Handles nested scalars"):
            v = Values(base1=Values(base2=Values(val=sm.S(1))))
            self.assertEqual(["base1.base2.val"], v.scalar_keys_recursive())

        with self.subTest(msg="Gets keys for scalar components of non-scalar types"):
            keys = Values(rot=geo.Rot3.identity()).scalar_keys_recursive()
            self.assertEqual(["rot[0]", "rot[1]", "rot[2]", "rot[3]"], keys)

        with self.subTest(msg="Gets keys for scalar components of an ndarray"):
            keys = Values(array=np.array([1, 2])).scalar_keys_recursive()
            self.assertEqual(["array[0]", "array[1]"], keys)

        with self.subTest(msg="Gets keys for scalar components of a 1d ndarray"):
            keys = Values(array=np.array([1])).scalar_keys_recursive()
            self.assertEqual(["array[0]"], keys)

        with self.subTest(msg="Gets keys for scalar components of non-scalar types in lists"):
            keys = Values(
                rot_list=[geo.Rot3.identity(), geo.Rot3.identity()]
            ).scalar_keys_recursive()
            self.assertEqual(
                [f"rot_list[{i}][{j}]" for i, j in itertools.product(range(2), range(4))], keys
            )

        with self.subTest(msg="Gets keys for the scalar components of non-scalar types in ndrrays"):
            keys = Values(
                rot_list=np.array([geo.Rot3.identity(), geo.Rot3.identity()])
            ).scalar_keys_recursive()
            self.assertEqual(
                [f"rot_list[{i}][{j}]" for i, j in itertools.product(range(2), range(4))], keys
            )

        with self.subTest(msg="key order is insertion order of highest level dot seperated key"):
            v = Values()
            v["first_top_level"] = Values()
            v["second_top_level"] = Values(first_inner=2)
            v["first_top_level"]["second_inner"] = 1
            self.assertEqual(
                ["first_top_level.second_inner", "second_top_level.first_inner"],
                v.scalar_keys_recursive(),
            )

    def test_to_from_storage_with_tuples(self) -> None:
        """
        Tests:
            Values.to_storage
            Values.from_storage_index
        Check that tuples are returned from storage properly (as opposed to, say, as lists)
        """
        v_tuple = Values(pair=(1, 2))
        v_after = Values.from_storage_index(v_tuple.to_storage(), v_tuple.index())

        self.assertEqual(v_tuple["pair"], v_after["pair"])

    def test_from_storage_index(self) -> None:
        """
        Tests:
            Values.from_storage_index
        Ensure that from_storage_index works with various value types
        """
        # To test a complex structure
        v_structure = self.element()
        self.assertEqual(
            v_structure, Values.from_storage_index(v_structure.to_storage(), v_structure.index())
        )

        # To check handling of CameraCal subclasses
        v_cam = Values()
        # The particular arguments being used to construct the CameraCals are arbitrary. Just want
        # something their constructors will be happy with
        with v_cam.scope("CameraCals"):
            [f_x, f_y, c_x, c_y] = np.random.uniform(low=0.0, high=1000.0, size=(4,))
            for c in cam.CameraCal.__subclasses__():
                v_cam[c.__name__] = c(
                    focal_length=(f_x, f_y),
                    principal_point=(c_x, c_y),
                    distortion_coeffs=np.random.uniform(
                        low=1.0, high=10.0, size=c.NUM_DISTORTION_COEFFS
                    ).tolist(),
                )

        self.assertEqual(v_cam, Values.from_storage_index(v_cam.to_storage(), v_cam.index()))

        # To check handling of normal geo types
        v_geo = Values()
        with v_geo.scope("geo_types"):
            for geo_type in [
                geo.Rot2,
                geo.Rot3,
                geo.Pose2,
                geo.Pose3,
                geo.Complex,
                geo.Quaternion,
                geo.DualQuaternion,
            ]:
                v_geo[geo_type.__name__] = ops.GroupOps.identity(geo_type)

        self.assertEqual(v_geo, Values.from_storage_index(v_geo.to_storage(), v_geo.index()))

        with self.subTest(msg="Values can handle dynamic matrix types"):
            v = Values(dynamic_mat=geo.Matrix.ones(14, 14))
            self.assertEqual(v, Values.from_storage_index(v.to_storage(), v.index()))

    @slow_on_sympy
    def test_tangent_D_storage(self) -> None:
        super().test_tangent_D_storage()

    def test_eq(self) -> None:
        """
        Tests:
            Values.__eq__
        """
        values45 = Values(i=4, j=5)
        vector45 = geo.V2(4, 5)
        with self.subTest(msg="Should not equal when different type despite same storage"):
            self.assertNotEqual(values45, vector45)

        nested_values45 = Values(root=values45)
        with self.subTest(msg="Should not equal when different structure despite same storage"):
            self.assertNotEqual(values45, nested_values45)

        val_tuple45 = Values(i=(4, 5))
        val_list45 = Values(i=[4, 5])
        with self.subTest(msg="Should distinguish between tuples and lists"):
            self.assertNotEqual(val_tuple45, val_list45)

    def test_subs(self) -> None:
        """
        Tests:
            Values.subs
        """
        with self.subTest(msg="Non-subbed entries should not be modified"):
            v = Values(num=4, sym=sm.S("x"))
            self.assertEqual(int, type(v.subs({"x": 3})["num"]))

        with self.subTest(msg="Subbing works, even with nested Values"):
            v = Values(inner_v=Values(sym=sm.S("x")))
            v_goal = Values(inner_v=Values(sym=3))
            self.assertEqual(v_goal, v.subs({"x": 3}))


if __name__ == "__main__":
    TestCase.main()
