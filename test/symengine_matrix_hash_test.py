# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.symbolic as sf
from symforce.test_util import TestCase
from symforce.test_util import symengine_only


class SymEngineMatrixHashTest(TestCase):
    @symengine_only
    def test_matrix_hash(self) -> None:
        """
        Tests:
            sf.sympy.Matrix.__hash__
        """
        hash1 = hash(sf.sympy.Matrix([[0, 1], [2, 3]]))
        hash2 = hash(sf.sympy.Matrix([[0, 1], [2, 4]]))
        hash3 = hash(sf.sympy.Matrix([[0, 1, 2, 3]]))

        self.assertNotEqual(hash1, 0)
        self.assertNotEqual(hash2, 0)
        self.assertNotEqual(hash3, 0)

        self.assertNotEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)
        self.assertNotEqual(hash2, hash3)


if __name__ == "__main__":
    TestCase.main()
