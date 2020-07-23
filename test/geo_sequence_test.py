from symforce import geo
from symforce import types as T
from symforce.test_util import TestCase
from symforce.test_util.storage_ops_test_mixin import StorageOpsTestMixin


class GeoSequenceTest(StorageOpsTestMixin, TestCase):
    """
    Test a scalar as a geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    @classmethod
    def element(cls):
        # type: () -> T.List[geo.Rot3]
        element = []
        element.append(geo.Rot3())
        element.append(geo.Rot3.from_euler_ypr(1.0, 0, 0))
        return element


if __name__ == "__main__":
    TestCase.main()
