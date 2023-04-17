# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

import symforce.symbolic as sf
from symforce import typing as T
from symforce.ops import StorageOps
from symforce.test_util import TestCase

if T.TYPE_CHECKING:
    _Base = TestCase
else:
    _Base = object


class StorageOpsTestMixin(_Base):
    """
    Test helper for the StorageOps concept. Inherit a test case from this.
    """

    @classmethod
    def element(cls) -> T.Any:
        """
        Overriden by child to provide an example non-identity element.
        """
        raise NotImplementedError()

    @classmethod
    def element_type(cls) -> T.Type:
        """
        Returns the type of the StorageOps-compatible class being tested.
        """
        return type(cls.element())

    def test_storage_ops(self) -> None:
        """
        Tests:
            storage_dim
            to_storage
            from_storage
        """
        # Check sane storage dimension
        element = self.element()
        storage_dim = StorageOps.storage_dim(element)
        self.assertGreater(storage_dim, 0)

        # Create from list
        vec = np.random.normal(size=(storage_dim,)).tolist()
        value = StorageOps.from_storage(element, vec)
        self.assertEqual(type(value), self.element_type())

        # Serialize to list
        vec2 = StorageOps.to_storage(value)
        self.assertEqual(len(vec2), storage_dim)
        self.assertListEqual(vec, vec2)

        # Build from list again
        value2 = StorageOps.from_storage(value, vec2)

        # Check equalities
        self.assertEqual(value, value2)
        vec2[0] = 10000.0
        self.assertNotEqual(value, StorageOps.from_storage(value, vec2))

        # Exercise printing
        self.assertGreater(len(str(value)), 0)

        # Test symbolic operations
        sym_element = StorageOps.symbolic(element, "name")
        self.assertEqual(
            sym_element,
            StorageOps.subs(sym_element, {sf.Symbol("var_not_in_element"): sf.Symbol("new_var")}),
        )
        self.assertEqual(sym_element, StorageOps.simplify(sym_element))

        with self.assertRaises(ValueError):
            StorageOps.subs(
                sym_element,
                StorageOps.to_storage(sf.Symbol("var_not_in_element")),
                StorageOps.to_storage(sf.Symbol("new_var")) + [0.0],
            )
