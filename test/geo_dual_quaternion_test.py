from symforce import geo
from symforce.test_util import TestCase
from symforce.test_util.group_ops_test_mixin import GroupOpsTestMixin


class GeoDualQuaternionTest(GroupOpsTestMixin, TestCase):
    """
    Test the DualQuaternion geometric class.
    Note the mixin that tests all storage and group ops.
    """

    @classmethod
    def element(cls):
        return geo.DualQuaternion(
            real_q=geo.Quaternion(xyz=geo.V3(0.1, -0.3, 1.3), w=3.2),
            inf_q=geo.Quaternion(xyz=geo.V3(1.2, 0.3, 0.7), w=0.1),
        )


if __name__ == "__main__":
    TestCase.main()
