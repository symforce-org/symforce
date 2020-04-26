import numpy as np

from symforce import geo
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class MatrixTest(LieGroupOpsTestMixin, TestCase):
    """
    Test the Matrix geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    @classmethod
    def element(cls):
        return geo.Matrix([-0.2, 5.3, 1.2])

    # TODO(hayk): Full test coverage of Matrix class!

    # TODO(hayk): Test from_storage for matrices - how should shape be preserved?


if __name__ == "__main__":
    TestCase.main()
