# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Package for symbolic geometric types using the Sympy API, focused on 2D and 3D transforms
for use in robotics and optimization.

Implements strict notions of Storage types, Group types, and LieGroup types.

The design of these concepts is similar to those in GTSAM and Sophus:

* https://gtsam.org/
* https://github.com/strasdat/Sophus
"""

from symforce import typing as T
from symforce.ops.interfaces.lie_group import LieGroup

from . import unsupported
from .complex import Complex
from .dual_quaternion import DualQuaternion
from .matrix import *  # noqa: F403
from .pose2 import Pose2
from .pose3 import Pose3
from .quaternion import Quaternion
from .rot2 import Rot2
from .rot3 import Rot3
from .unit3 import Unit3

# Default generated geo types
GEO_TYPES: T.Tuple[T.Type[LieGroup], ...] = (
    Rot2,
    Rot3,
    Pose2,
    Pose3,
    Unit3,
)

GROUP_GEO_TYPES: T.Tuple[T.Type[LieGroup], ...] = (
    Rot2,
    Rot3,
    Pose2,
    Pose3,
)

# All geo types, including GEO_TYPES plus types used for internal representations that are not
# generated
ALL_GEO_TYPES = GEO_TYPES + (
    Quaternion,
    DualQuaternion,
    Complex,
)
