import numpy as np

from symforce import types as T
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
    def element(cls):
        # type: () -> T.Any
        """
        Overriden by child to provide an example non-identity element.
        """
        raise NotImplementedError()

    @classmethod
    def element_type(cls):
        # type: () -> T.Type
        """
        Returns the type of the StorageOps-compatible class being tested.
        """
        return type(cls.element())

    def test_storage_ops(self):
        # type: () -> None
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
        self.assertNotEqual(element, StorageOps.from_storage(value, vec2))

        # Exercise printing
        self.assertGreater(len(str(value)), 0)

        # Test symbolic operations
        sym_element = StorageOps.symbolic(element, "name")
        sym_element.subs({"x": "y"})
        sym_element.simplify()
