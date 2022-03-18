# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import itertools

from symforce import typing as T
from symforce.test_util.cam_test_mixin import CamTestMixin


class CamCalTestMixin(CamTestMixin):
    """
    Test helper for camera calibration objects. Inherit a test case from this.
    """

    def test_storage_order(self) -> None:
        """
        Tests:
            storage_order
        Specifically, that the names of storage_order match those of the argument of the
        class constructors, and the the order of the scalars in the storage matches the
        order specified by storage_order.
        """
        cam_cls = type(self.element())
        order = cam_cls.storage_order()

        counter = itertools.count(0)

        def get_counting_list(size: int) -> T.Union[int, T.List[int]]:
            if size == 1:
                return next(counter)
            return [next(counter) for _ in range(size)]

        ordered_args = {arg: get_counting_list(size) for arg, size in order}

        storage = cam_cls(**ordered_args).to_storage()

        self.assertEqual(storage, list(range(len(storage))))
