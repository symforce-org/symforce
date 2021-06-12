import numpy as np
import unittest

from symforce import logger
from symforce import geo
from symforce import sympy as sm
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
        v["rot3"] = geo.Rot3.from_tangent(np.random.normal(size=(3,)))
        v["pose3"] = geo.Pose3.from_tangent(np.random.normal(size=(6,)))
        other_values = v.copy()
        v["values"] = other_values
        v["vec_values"] = [other_values, other_values]
        v["vec_rot3"] = [
            geo.Rot3.from_tangent(np.random.normal(size=(3,))),
            geo.Rot3.from_tangent(np.random.normal(size=(3,))),
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
        self.assertFalse("." in v)
        self.assertFalse(".foo" in v)

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
        v["b"] = geo.Rot3.from_axis_angle(axis=geo.V3(1, 0, 0), angle=sm.pi / 2)

        v_evalf = v.evalf()

        logger.debug(v)
        logger.debug(v_evalf)

        self.assertEqual(v["a"].evalf(), v_evalf["a"])
        self.assertNear(v_evalf["a"], 0.3333333, places=6)

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

    @unittest.expectedFailure
    def test_items_recursive(self) -> None:
        """
        TODO(Bradley) Get this passing
        Tests:
            items_recursive
        Ensure that the key item pairs returned by items_recursive are valid key item pairs.
        """

        v = self.element()
        for key, value in v.items_recursive():
            self.assertEqual(v[key], value)

    @slow_on_sympy
    def test_tangent_D_storage(self) -> None:
        super().test_tangent_D_storage()


if __name__ == "__main__":
    TestCase.main()
