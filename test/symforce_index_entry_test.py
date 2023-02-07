# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.symbolic as sf
from symforce.test_util import TestCase
from symforce.values import IndexEntry


# Helper classes for testing type recovery of nested classes
class A:
    class B:
        class C:
            pass


class IndexEntryTest(TestCase):
    """
    Tests IndexEntry
    """

    def test_datatype(self) -> None:
        """
        Tests:
            IndexEntry.__init__
            IndexEntry.datatype
        """
        # Fills in offset and storage_dim (fields irrelevant to this test) with arbitrary values
        entry_helper = lambda dtype: IndexEntry(offset=0, storage_dim=0, stored_datatype=dtype)

        with self.subTest("datatype correctly returns common StorageOps types"):
            for datatype in [int, sf.Rot3, sf.V2, sf.Scalar]:
                self.assertEqual(datatype, entry_helper(datatype).datatype())

        with self.subTest("datatype can handle nested classes"):
            self.assertEqual(A.B.C, entry_helper(A.B.C).datatype())


if __name__ == "__main__":
    TestCase.main()
