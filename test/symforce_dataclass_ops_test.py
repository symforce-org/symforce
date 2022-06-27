# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from dataclasses import dataclass, field

import symforce.symbolic as sf
from symforce import typing as T

from symforce.test_util import TestCase
from symforce.ops import StorageOps, GroupOps, LieGroupOps
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


@dataclass
class TestSubType:
    rot: sf.Rot3


@dataclass
class TestDynamicSizeType:
    rot: sf.Rot3
    x: T.Scalar
    subtype: TestSubType
    seq: T.Sequence[T.Sequence[TestSubType]]
    optional: T.Optional[T.Scalar] = None


@dataclass
class TestFixedSizeType:
    rot: sf.Rot3
    x: T.Scalar
    subtype: TestSubType
    seq: T.Sequence[TestSubType] = field(metadata={"length": 2})


class SymforceDataclassOpsTest(LieGroupOpsTestMixin, TestCase):
    """
    Tests ops.impl.dataclass_*_ops.py
    """

    @classmethod
    def element(cls) -> T.Dataclass:
        element = TestDynamicSizeType(
            rot=sf.Rot3.identity(),
            x=1.0,
            subtype=TestSubType(rot=sf.Rot3.from_yaw_pitch_roll(1.0, 2.0, 3.0)),
            seq=[
                [TestSubType(rot=sf.Rot3.from_yaw_pitch_roll(0.1, 0.2, 0.3)) for _ in range(3)]
                for _ in range(5)
            ],
        )
        return element

    def test_fixed_size_ops(self) -> None:
        """
        Tests:
            Dataclass ops, with fixed size type. Also tests ops which take a type (rather than an
            instance) on a dataclass with a sequence of known size.
        """
        with self.subTest("storage dim"):
            self.assertEqual(StorageOps.storage_dim(TestFixedSizeType), 17)

        with self.subTest("symbolic"):
            instance = StorageOps.symbolic(TestFixedSizeType, "instance")

        with self.subTest("to_storage"):
            storage = StorageOps.to_storage(instance)
            self.assertEqual(len(storage), 17)
            for x in storage:
                self.assertIsInstance(x, sf.Symbol)
                self.assertTrue(x.name.startswith("instance"))

        with self.subTest("from_storage"):
            instance2 = StorageOps.from_storage(TestFixedSizeType, storage)
            self.assertEqual(instance, instance2)

        with self.subTest("evalf"):
            instance.x = sf.S(5)
            instance.rot = sf.Rot3.from_yaw_pitch_roll(instance.x, 0, 0)
            instance.subtype.rot = instance.rot.inverse()
            instance.seq[0].rot = instance.rot * instance.rot
            instance.seq[1].rot = instance.rot * instance.rot

            instancef = StorageOps.evalf(instance)

            for x in StorageOps.to_storage(instancef):
                xf = float(x)
                self.assertIsInstance(xf, float)

        with self.subTest("identity"):
            identity = GroupOps.identity(TestFixedSizeType)

        with self.subTest("tangent_dim"):
            self.assertEqual(LieGroupOps.tangent_dim(TestFixedSizeType), 13)

        with self.subTest("from_tangent"):
            tangent = LieGroupOps.to_tangent(instance)
            instance2 = LieGroupOps.from_tangent(TestFixedSizeType, tangent)
            self.assertLieGroupNear(instance, instance2)

    def test_dynamic_size_storage_ops(self) -> None:
        """
        Tests:
            DataclassStorageOps, with dynamic size type
        """
        empty_instance = self.element()

        expected_size = 1 + (
            2 + len(empty_instance.seq) * len(empty_instance.seq[0])
        ) * StorageOps.storage_dim(sf.Rot3)

        with self.subTest("storage dim"):
            self.assertEqual(StorageOps.storage_dim(empty_instance), expected_size)

        with self.subTest("symbolic"):
            instance = StorageOps.symbolic(empty_instance, "instance")

        with self.subTest("to_storage"):
            storage = StorageOps.to_storage(instance)
            self.assertEqual(len(storage), expected_size)
            for x in storage:
                self.assertIsInstance(x, sf.Symbol)
                self.assertTrue(x.name.startswith("instance"))

        with self.subTest("from_storage"):
            instance2 = StorageOps.from_storage(empty_instance, storage)
            self.assertEqual(instance, instance2)

        with self.subTest("evalf"):
            instance.x = sf.S(5)
            instance.rot = sf.Rot3.from_yaw_pitch_roll(instance.x, 0, 0)
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
