# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from dataclasses import dataclass

from symforce import geo
from symforce import sympy as sm
from symforce import typing as T

from symforce.test_util import TestCase
from symforce.ops import StorageOps
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


@dataclass
class TestSubType:
    rot: geo.Rot3


@dataclass
class TestDynamicSizeType:
    rot: geo.Rot3
    x: T.Scalar
    subtype: TestSubType
    seq: T.Sequence[T.Sequence[TestSubType]]
    optional: T.Optional[T.Scalar] = None


@dataclass
class TestFixedSizeType:
    rot: geo.Rot3
    x: T.Scalar
    subtype: TestSubType
    seq: TestSubType


class SymforceDataclassOpsTest(LieGroupOpsTestMixin, TestCase):
    """
    Tests ops.impl.dataclass_*_ops.py
    """

    @classmethod
    def element(cls) -> T.Dataclass:
        element = TestDynamicSizeType(
            rot=geo.Rot3.identity(),
            x=1.0,
            subtype=TestSubType(rot=geo.Rot3.from_yaw_pitch_roll(1.0, 2.0, 3.0)),
            seq=[
                [TestSubType(rot=geo.Rot3.from_yaw_pitch_roll(0.1, 0.2, 0.3)) for _ in range(3)]
                for _ in range(5)
            ],
        )
        return element

    def test_fixed_size_storage_ops(self) -> None:
        """
        Tests:
            DataclassStorageOps, with fixed size type
        """
        with self.subTest("storage dim"):
            self.assertEqual(StorageOps.storage_dim(TestFixedSizeType), 13)

        with self.subTest("symbolic"):
            instance = StorageOps.symbolic(TestFixedSizeType, "instance")

        with self.subTest("to_storage"):
            storage = StorageOps.to_storage(instance)
            self.assertEqual(len(storage), 13)
            for x in storage:
                self.assertIsInstance(x, sm.Symbol)
                self.assertTrue(x.name.startswith("instance"))

        with self.subTest("from_storage"):
            instance2 = StorageOps.from_storage(TestFixedSizeType, storage)
            self.assertEqual(instance, instance2)

        with self.subTest("evalf"):
            instance.x = sm.S(5)
            instance.rot = geo.Rot3.from_yaw_pitch_roll(instance.x, 0, 0)
            instance.subtype.rot = instance.rot.inverse()
            instance.seq.rot = instance.rot * instance.rot

            instancef = StorageOps.evalf(instance)

            for x in StorageOps.to_storage(instancef):
                xf = float(x)
                self.assertIsInstance(xf, float)

    def test_dynamic_size_storage_ops(self) -> None:
        """
        Tests:
            DataclassStorageOps, with dynamic size type
        """
        empty_instance = self.element()

        expected_size = 1 + (
            2 + len(empty_instance.seq) * len(empty_instance.seq[0])
        ) * StorageOps.storage_dim(geo.Rot3)

        with self.subTest("storage dim"):
            self.assertEqual(StorageOps.storage_dim(empty_instance), expected_size)

        with self.subTest("symbolic"):
            instance = StorageOps.symbolic(empty_instance, "instance")

        with self.subTest("to_storage"):
            storage = StorageOps.to_storage(instance)
            self.assertEqual(len(storage), expected_size)
            for x in storage:
                self.assertIsInstance(x, sm.Symbol)
                self.assertTrue(x.name.startswith("instance"))

        with self.subTest("from_storage"):
            instance2 = StorageOps.from_storage(empty_instance, storage)
            self.assertEqual(instance, instance2)

        with self.subTest("evalf"):
            instance.x = sm.S(5)
            instance.rot = geo.Rot3.from_yaw_pitch_roll(instance.x, 0, 0)
            instance.subtype.rot = instance.rot.inverse()
            for i in range(5):
                for j in range(3):
                    instance.seq[i][j].rot = instance.rot * instance.rot

            instancef = StorageOps.evalf(instance)

            for x in StorageOps.to_storage(instancef):
                xf = float(x)
                self.assertIsInstance(xf, float)


if __name__ == "__main__":
    TestCase.main()
